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

        # Define the layers of the neural network with batch normalization and dropout
        self.fc1 = nn.Linear(input_size, hidden_size * 2)
        self.bn1 = nn.BatchNorm1d(hidden_size * 2)  # Batch normalization
        self.dropout1 = nn.Dropout(0.2)  # Dropout layer

        self.fc2 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(hidden_size, output_size)  # Output layer

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (Tensor): Input state vector.

        Returns:
            Tensor: Q-values for each action.
        """
        x = torch.relu(self.bn1(self.fc1(x)))  # Batch norm and activation
        x = self.dropout1(x)  # Dropout after first layer
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)  # Linear output layer
        return x
