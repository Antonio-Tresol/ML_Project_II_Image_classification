from torch import nn


# Multi Layer Perceptron (MLP) modele
class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_layer_count, hidden_layer_size, output_size, device
    ):
        """
        Initializes a MLP model.

        Args:
            output_size (int): The number of output classes.
            hidden"""
        self.input_size = input_size
        self.hidden_layer_count = hidden_layer_count
        self.hidden_layer_size = hidden_layer_size
        self.output_size = output_size

        self.layers = nn.ModuleList().to(device)
        self.layers.append(nn.Linear(input_size, hidden_layer_size).to(device))
        for _ in range(hidden_layer_count - 1):
            self.layers.append(
                nn.Linear(hidden_layer_size, hidden_layer_size).to(device)
            )
        self.layers.append(nn.Linear(hidden_layer_size, output_size).to(device))

    def forward(self, x):
        """
        Forward pass of the MLP model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        for layer in self.layers:
            x = layer(x)
        return x
