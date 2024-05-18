import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


from data.transforms.image_transformation import ImageTransformation
from data.transforms.folder_image_converter import FolderImageConverter
from data.transforms.folder_image_converter import try_create_directory

import os
import cv2
from PIL import Image
import numpy as np
import pickle
from sklearn.cluster import KMeans


def load_keypoints_descriptors(filepath):
    with open(filepath, "rb") as f:
        key_points, descriptors = pickle.load(f)
    return key_points, descriptors


def create_feature_vector(descriptors, kmeans):
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(float) / np.sum(hist)
    return hist

def normalize_descriptors(descriptors, output_shape=(32, 32)):
    # Normalize descriptors to the range [0, 255]
    normalized_descriptors = cv2.normalize(descriptors, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    normalized_descriptors = np.uint8(normalized_descriptors)

    # Reshape the descriptors to fit into the desired image shape
    if normalized_descriptors.shape[0] > output_shape[0] * output_shape[1]:
        normalized_descriptors = normalized_descriptors[:output_shape[0] * output_shape[1], :]

    padded_descriptors = np.zeros((output_shape[0] * output_shape[1], normalized_descriptors.shape[1]), dtype=np.uint8)
    padded_descriptors[:normalized_descriptors.shape[0], :] = normalized_descriptors

    image_shape = (output_shape[0], output_shape[1], normalized_descriptors.shape[1])
    descriptor_image = padded_descriptors.reshape(image_shape)
    
    if descriptor_image.shape[2] == 1:
        descriptor_image = descriptor_image.squeeze()
    
    return descriptor_image

def save_descriptors_as_image(descriptors, filepath, output_shape=(32, 32)):
    descriptor_image = normalize_descriptors(descriptors, output_shape=output_shape)

    # Convert to PIL Image
    if descriptor_image.ndim == 2:  # Grayscale image
        descriptor_image_pil = Image.fromarray(descriptor_image, mode='L')
    elif descriptor_image.ndim == 3:  # RGB image
        descriptor_image = np.repeat(descriptor_image, 3, axis=2)  # Ensure 3 channels if not already
        descriptor_image_pil = Image.fromarray(descriptor_image, mode='RGB')
    else:
        raise ValueError("Unsupported descriptor image shape for conversion to PIL Image.")

    descriptor_image_pil.save(filepath)

class MlpFeatureExtractor(ImageTransformation):
    """
    Parameters:
    - n_features (int): Number of features to extract.
    - color (tuple): Color of the keypoints.
    - flags (int): controls how keypoints are visualized. Existing flags are:
        - cv2.DRAW_MATCHES_FLAGS_DEFAULT
        - cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        - cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG
        - cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS
    - score_type (int): The ORB score type. Existing types are:
        - cv2.ORB_HARRIS_SCORE
        - cv2.ORB_FAST_SCORE
    - fast_threshold (int): The threshold for the FAST keypoint detector.
        The lower it is the more widespread the keypoints are going to be when found
    """

    def __init__(
        self,
        n_features: int = 1500,
        color=(0, 0, 255),
        flags=0,
        score_type=cv2.ORB_HARRIS_SCORE,
        fast_threshold=20,
    ):
        super(MlpFeatureExtractor, self).__init__()

        self.n_features = n_features
        self.color = color
        self.flags = flags
        self.score_type = score_type
        self.fast_threshold = fast_threshold

    def fit(self, image: Image.Image) -> Image.Image:
        _, _, image_pil = self.extract_features(image)
        return image_pil

    """
        Retuns
        - key_points: the keypoints found in the image
        - descriptors: the descriptors of the keypoints
        - image_pil: the image with the keypoints drawn on it (for feature visualization purposes)
    """

    def extract_features(self, image: Image.Image):
        """
        PIL.Image format is not compatible, as such we need a few conversions first
        1. Convert PIL.Image to numpy array
        2. Convert numpy array to cv2 image
        """
        image_array = np.array(image)
        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # convert image to grayscale
        grayscale_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        # create ORB object
        orb = cv2.ORB_create(
            nfeatures=self.n_features,
            scoreType=self.score_type,
            fastThreshold=self.fast_threshold,
        )

        # detect and compute the keypoints on image (grayscale)
        key_points = orb.detect(grayscale_image, None)
        key_points, descriptors = orb.compute(grayscale_image, key_points)

        # draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(
            grayscale_image, key_points, None, self.color, self.flags
        )

        # convert the image back to PIL format
        image_array = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_array)

        return key_points, descriptors, image_pil

    # Extracts the features of an image and saves them to a file as well as the picture in order to visualize the features
    def extract_picture_features(self, dest_folder_dir, image) -> None:
        img = Image.open(image.path).convert("RGB")

        key_points, descriptors, new_img = self.extract_features(img)

        new_img.save(os.path.join(dest_folder_dir, image.name))

        features_image_name = image.name.replace(".png", "_features.png")

        if descriptors is not None:
            save_descriptors_as_image(descriptors, os.path.join(dest_folder_dir, features_image_name))

    # before training, reads the data from the features folder and generates the feature vectors
    def generate_features_vector(
        self, features_folder_path: str, num_clusters: int = 50, random_state: int = 42
    ):
        for folder in os.scandir(features_folder_path):
            if folder.is_dir():
                descriptors = []

                for features in os.scandir(folder):
                    _, descriptors = load_keypoints_descriptors(features.path)

                    if descriptors is not None:
                        descriptors.append(descriptors)

        descriptors = np.vstack(descriptors)

        kmeans = KMeans(n_clusters=num_clusters, random_state=random_state)
        kmeans.fit(descriptors)

        feature_vectors = [create_feature_vector(desc, kmeans) for desc in descriptors]
        feature_vectors = np.array(feature_vectors)

        return feature_vectors


class FolderImageFeatureExtractor:
    def __init__(self, root_dir: str, dest_dir: str, check_if_exists: bool) -> None:
        """
        Initialize the FolderImageConverter class.

        Parameters:
        - root_dir (str): The root directory containing the folders with images.
        - dest_dir (str): The destination directory where transformed images will be saved.
        - check_if_exists (bool): Whether to check if the root directory exists.

        Returns:
        - None
        """
        self.root_dir = root_dir
        self.dest_dir = dest_dir
        self.check_if_folder_exists = check_if_exists

    def __transform_folder(self, transformation) -> None:
        for folder in os.scandir(self.root_dir):
            if folder.is_dir():
                dest_folder_dir = os.path.join(self.dest_dir, folder.name)

                # Create the new folder
                try_create_directory(directory=dest_folder_dir, remove_if_exists=False)

                for image in os.scandir(folder):
                    print(image.path)

                    transformation.extract_picture_features(dest_folder_dir, image)

    def convert(self, transformation):
        """
        Convert images in the root directory.

        Parameters:
        - transformation (ImageTransformation): The transformation to apply to the images.

        Returns:
        - None
        """
        if self.check_if_folder_exists:
            print("Checking if folder exists")
            directory_existed_already = try_create_directory(
                directory=self.dest_dir, remove_if_exists=False
            )
            if not directory_existed_already:
                print("Folder did not exist")
                self.__transform_folder(transformation=transformation)
                return

            print("Folder existed")

        else:
            try_create_directory(directory=self.dest_dir, remove_if_exists=True)

            self.__transform_folder(transformation=transformation)


import configuration as config

imageExtractor = FolderImageFeatureExtractor(
    root_dir=config.ROOT_DIR,
    dest_dir=config.MLP_FEATURES_DIR,
    check_if_exists=True,
)

featureExtractor = MlpFeatureExtractor()

print("Extracting features from images...")

imageExtractor.convert(transformation=featureExtractor)

print("Features extracted successfully!")
