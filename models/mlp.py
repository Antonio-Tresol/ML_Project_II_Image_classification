from torch import nn
import torchvision


# Multi Layer Perceptron (MLP) modele
class MLP(nn.Module):
    def __init__(
        self, input_size, hidden_layer_count, hidden_layer_size, output_size, device
    ):
        super(MLP, self).__init__()
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
            self.layers.append(nn.ReLU().to(device))
        self.layers.append(nn.Linear(hidden_layer_size, output_size).to(device))
        self.require_grads()

    def require_grads(self):
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def flatten_image(x):
    return x.view(-1)


def get_mlp_transformations() -> tuple[torchvision.transforms.Compose]:
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.Lambda(flatten_image),  # Flatten the image
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((32, 32)),
            torchvision.transforms.Lambda(flatten_image),  # Flatten the image
        ]
    )
    return train_transform, test_transform
