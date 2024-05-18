from data.transforms.image_transformation import ImageTransformation

import os
import cv2
from PIL import Image
import numpy as np
import pickle
from sklearn.cluster import KMeans

def load_keypoints_descriptors(filepath):
    with open(filepath, 'rb') as f:
        key_points, descriptors = pickle.load(f)
    return key_points, descriptors  

def create_feature_vector(descriptors, kmeans):
    labels = kmeans.predict(descriptors)
    hist, _ = np.histogram(labels, bins=np.arange(kmeans.n_clusters + 1))
    hist = hist.astype(float) / np.sum(hist)
    return hist    

class MlpFeatureExtractor(ImageTransformation):
    '''
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
    '''
    def __init__(self
                 , n_features: int = 1500
                 , color = (0, 0, 255)
                 , flags = 0
                 , score_type = cv2.ORB_HARRIS_SCORE
                 , fast_threshold = 20):
        super(MlpFeatureExtractor, self).__init__()

        self.n_features = n_features
        self.color = color
        self.flags = flags
        self.score_type = score_type
        self.fast_threshold = fast_threshold

    def fit(self, image: Image.Image) -> Image.Image:
        _, _, image_pil = self.extract_features(image)
        return image_pil
    
    '''
        Retuns
        - key_points: the keypoints found in the image
        - descriptors: the descriptors of the keypoints
        - image_pil: the image with the keypoints drawn on it (for feature visualization purposes)
    '''
    def extract_features(self, image: Image.Image):
        '''
        PIL.Image format is not compatible, as such we need a few conversions first
        1. Convert PIL.Image to numpy array
        2. Convert numpy array to cv2 image
        '''
        image_array = np.array(image)
        image_cv2 = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        #convert image to grayscale
        grayscale_image = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2GRAY)

        # create ORB object
        orb = cv2.ORB_create(nfeatures = self.n_features
                             , scoreType = self.score_type
                             , fastThreshold = self.fast_threshold)

        # detect and compute the keypoints on image (grayscale)
        key_points = orb.detect(grayscale_image, None)
        key_points, descriptors = orb.compute(grayscale_image, key_points)

        # draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(grayscale_image, key_points, None, self.color, self.flags)

        # convert the image back to PIL format
        image_array = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_array)

        return key_points, descriptors, image_pil
    
    # Extracts the features of an image and saves them to a file as well as the picture in order to visualize the features
    def extract_picture_features(self, dest_folder_dir, image_path: str) -> None: 
        img = Image.open(image_path).convert("RGB")

        key_points, descriptors, new_img = self.extract_features(img)

        new_img.save(os.path.join(dest_folder_dir, image_path))

        with open(os.path.join(dest_folder_dir, image_path + ".txt"), "wb") as file:
            pickle.dump((key_points, descriptors), file)

    # before training, reads the data from the features folder and generates the feature vectors
    def generate_features_vector(self, features_folder_path: str, num_clusters: int = 50, random_state: int = 42): 
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
                    
      