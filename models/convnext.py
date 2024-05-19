from torch import nn
import torch
import torchvision
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import configuration as conf


class ConvNext(nn.Module):
    def __init__(self, num_classes, device) -> None:
        """
        Initializes a ConvNext model.

        Args:
            num_classes (int): The number of output classes.
            device (torch.device): The device to run the model on.
        """
        super().__init__()
        self.convnext = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT).to(
            device=device
        )
        self.convnext.classifier[2] = nn.Linear(
            in_features=768, out_features=num_classes, bias=True
        ).to(device=device)

    def forward(self, x) -> torch.Tensor:
        """
        Forward pass of the ConvNext model.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor.
        """
        return self.convnext(x)


def get_conv_model_transformations() -> tuple[torchvision.transforms.Compose]:
    train_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(conf.RESIZE),
            torchvision.transforms.CenterCrop(conf.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomRotation(conf.ROTATION),
            torchvision.transforms.Normalize(conf.MEAN, conf.STD),
        ]
    )

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(conf.RESIZE),
            torchvision.transforms.CenterCrop(conf.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(conf.MEAN, conf.STD),
        ]
    )
    return train_transform, test_transform


def get_preprocess_transformation() -> torchvision.transforms.Compose:
    preprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(conf.RESIZE),
            torchvision.transforms.CenterCrop(conf.CROP),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(conf.MEAN, conf.STD),
        ]
    )

    return preprocess


def get_deprocess_transformation() -> torchvision.transforms.Compose:
    deprocess = torchvision.transforms.Compose(
        [
            torchvision.transforms.Normalize(
                mean=[-m / s for m, s in zip(conf.MEAN, conf.STD)],
                std=[1 / s for s in conf.STD],
            ),
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize(conf.ORIGINAL_SIZE),
        ]
    )
    return deprocess
