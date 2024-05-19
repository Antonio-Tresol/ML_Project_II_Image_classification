from abc import ABC, abstractmethod
from PIL import Image


class ImageTransformation(ABC):
    def __init__(self) -> None:
        """Initializes the image transformation"""
        pass

    @abstractmethod
    def fit(image: Image.Image) -> Image.Image:
        """
        Applies the transformation to the image.

        Args:
            image: Image to transform.

        Returns:
            The transformed image.
        """
        pass


import cv2
import numpy as np


class BilateralFilter(ImageTransformation):
    def __init__(self, d: int = 3, sigmaColor: int = 24, sigmaSpace: int = 5) -> None:
        """
        Initializes Bilateral Filter transformation.
        """
        self.d = d
        self.sigmaColor = 75
        self.sigmaSpace = 75
        pass

    def fit(self, image: Image.Image) -> Image.Image:
        """
        Applies the transformation to the image.

        Args:
            image: Image to transform.

        Returns:
            The transformed image.
        """

        # Convert PIL image to NumPy array
        numpy_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Apply bilateral filter
        filtered_image = cv2.bilateralFilter(
            numpy_image,
            d=self.d,
            sigmaColor=self.sigmaColor,
            sigmaSpace=self.sigmaSpace,
        )

        # Convert the filtered NumPy array back to PIL image
        filtered_pil_image = Image.fromarray(
            cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
        )

        return filtered_pil_image
