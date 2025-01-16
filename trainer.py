import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from model import DQN

# Hyperparameters
LR = 0.01            # Learning rate
GAMMA = 0.9           # Discount factor
MEMORY_SIZE = 100_000  # Replay buffer size
BATCH_SIZE = 256       # Mini-batch size


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
        self.model = DQN(state_size, 128, action_size)  # Neural network model
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
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        """
        Trains the model using a random sample from the replay buffer.
        """
        if len(self.memory) < BATCH_SIZE:
            return

        # Sample a mini-batch
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert lists of NumPy arrays to a single NumPy array
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.bool)

        # Compute Q-values for current states
        q_values = self.model(states)

        # Compute target Q-values for next states
        next_q_values = self.model(next_states)
        target_q_values = q_values.clone()  # Clone Q-values for updates
        for i in range(BATCH_SIZE):
            if dones[i]:  # If the episode is done, use only the reward
                target_q_values[i, actions[i]] = rewards[i]
            else:  # Update using the Bellman equation
                target_q_values[i, actions[i]] = rewards[i] + GAMMA * torch.max(next_q_values[i])

        # Calculate loss
        loss = self.criterion(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
