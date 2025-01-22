import torch
import random
import numpy as np
from collections import deque
from model import DQN

class SnakeAITrainer:
    """
    Trainer class for the Snake AI using Deep Q-Learning.

    Args:
        state_size (int): Dimension of the state vector.
        action_size (int): Number of possible actions.
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for future rewards.
        batch_size (int): Size of the minibatch for training.
    """
    def __init__(self, state_size, action_size, lr=0.0005, gamma=0.95, batch_size=128):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr  # Learning rate
        self.gamma = gamma  # Discount factor
        self.batch_size = batch_size  # Batch size for training

        # Replay memory for storing transitions
        self.memory = deque(maxlen=200000)

        # Neural network model and optimizer
        self.model = DQN(state_size, 256, action_size)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.MSELoss()  # Loss function (Mean Squared Error)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay memory.

        Args:
            state (list): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (list): Next state after the action.
            done (bool): Whether the episode ended.
        """
        state_tensor = torch.tensor([state], dtype=torch.float32)
        next_state_tensor = torch.tensor([next_state], dtype=torch.float32)

        # Calculate TD error for priority
        q_value = self.model(state_tensor).max().item()
        next_q_value = self.model(next_state_tensor).max().item() if not done else 0

        # Priority scaling factor
        alpha = 0.6
        priority = (abs(reward + self.gamma * next_q_value - q_value) + 1e-5) ** alpha

        self.memory.append((priority, (state, action, reward, next_state, done)))

    def train_step(self):
        """
        Performs a single training step using a minibatch from the replay memory.
        """
        # Ensure enough samples are in the memory
        if len(self.memory) < self.batch_size:
            return

        # Sample a minibatch with priority weighting
        priorities, minibatch = zip(*random.choices(
            self.memory,
            weights=[m[0] for m in self.memory],  # Use priorities for weighted sampling
            k=self.batch_size
        ))

        # Unpack the minibatch
        states, actions, rewards, next_states, dones = zip(*[m for m in minibatch])

        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Compute Q-values for the current and next states
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]

        # Compute target Q-values using the Bellman equation
        target_q_values = rewards + (1 - dones.float()) * self.gamma * max_next_q_values

        # Get Q-values for the taken actions
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Normalize the priorities for weighted loss
        priorities_tensor = torch.tensor(priorities, dtype=torch.float32)
        normalized_weights = priorities_tensor / priorities_tensor.sum()

        # Compute the weighted loss
        loss = (self.criterion(q_values, target_q_values) * normalized_weights).mean()

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def get_parameters(self):
        """
        Returns the trainer's parameters for logging or debugging.

        Returns:
            dict: A dictionary of key training parameters.
        """
        return {
            "Learning Rate": self.lr,
            "Gamma (Discount Factor)": self.gamma,
            "Batch Size": self.batch_size,
        }
