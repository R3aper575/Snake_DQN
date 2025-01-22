import torch
import torch.nn as nn

class DQN(nn.Module):
    """
    Deep Q-Network (DQN) model for approximating Q-values.

    Args:
        input_size (int): Size of the input state vector.
        hidden_size (int): Number of neurons in each hidden layer.
        output_size (int): Number of possible actions.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()

        # Define the layers of the neural network
        self.fc1 = nn.Linear(input_size, hidden_size * 2)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 2)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)  # Third hidden layer
        self.fc4 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input state vector.

        Returns:
            Tensor: Q-values for each action.
        """
        x = torch.relu(self.fc1(x))  # Activation for the first layer
        x = torch.relu(self.fc2(x))  # Activation for the second layer
        x = torch.relu(self.fc3(x))  # Activation for the third layer
        x = self.fc4(x)  # Linear output layer (no activation function for Q-values)
        return x
