import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from model import DQN

# Hyperparameters
LR = 0.001            # Learning rate
GAMMA = 0.99           # Discount factor
MEMORY_SIZE = 100_000  # Replay buffer size
BATCH_SIZE = 64       # Mini-batch size


class SnakeAITrainer:
    def __init__(self, state_size, action_size):
        """
        Initializes the trainer with model, optimizer, memory, and loss function.

        Args:
            state_size (int): Size of the input state vector.
            action_size (int): Number of possible actions.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=MEMORY_SIZE)  # Replay buffer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available
        self.model = DQN(state_size, 128, action_size).to(self.device)  # Move model to device
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)  # Optimizer
        self.criterion = nn.MSELoss()  # Mean squared error loss function

    def store_transition(self, state, action, reward, next_state, done):
        """
        Stores a transition in the replay buffer.

        Args:
            state (numpy.ndarray): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (numpy.ndarray): Next state after the action.
            done (bool): Whether the episode ended.
        """
        # Calculate priority based on the absolute TD-error or reward
        if len(self.memory) > 0:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                next_state_tensor = torch.tensor(next_state, dtype=torch.float32).to(self.device)
                q_value = self.model(state_tensor).max().item()
                max_next_q_value = self.model(next_state_tensor).max().item()
                priority = abs(reward + (1 - done) * GAMMA * max_next_q_value - q_value)
        else:
            priority = abs(reward)  # Use reward as priority for initial transitions

        # Store the transition with its priority
        self.memory.append((priority, (state, action, reward, next_state, done)))

    def train_step(self):
        """
        Trains the model using a prioritized sample from the replay buffer.
        """
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample based on priority
        priorities, minibatch = zip(*random.choices(self.memory, weights=[m[0] for m in self.memory], k=BATCH_SIZE))

        # Unpack minibatch
        states, actions, rewards, next_states, dones = zip(*[m for m in minibatch])

        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)

        # Perform the Q-learning update
        q_values = self.model(states)
        next_q_values = self.model(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        target_q_values = rewards + (1 - dones.float()) * GAMMA * max_next_q_values

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate weighted loss using priorities
        weights = torch.tensor(priorities, dtype=torch.float32).to(self.device)
        normalized_weights = weights / weights.sum()  # Normalize weights
        loss = (self.criterion(q_values, target_q_values) * normalized_weights).mean()

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
