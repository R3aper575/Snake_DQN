import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Deep Q-Network architecture.

        Args:
            input_size (int): Size of the input layer (state space size).
            hidden_size (int): Number of neurons in the hidden layers.
            output_size (int): Size of the output layer (action space size).
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor representing the game state.

        Returns:
            torch.Tensor: Output tensor with Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
