import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.transforms.image_transformation import ImageTransformation

import cv2
from PIL import Image
import numpy as np

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
        key_points, des = orb.compute(grayscale_image, key_points)

        # draw keypoints on the image
        image_with_keypoints = cv2.drawKeypoints(grayscale_image, key_points, None, self.color, self.flags)

        # convert the image back to PIL format
        image_array = cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_array)

        return image_pil

def extract_all_features():
    from data.transforms.folder_image_converter import FolderImageConverter
    import configuration as config

    converter = FolderImageConverter(
        root_dir = config.ROOT_DIR,
        dest_dir = config.MLP_FEATURES_DIR,
        check_if_exists = True
    )

    feature_extractor = MlpFeatureExtractor(n_features=5000)
    converter.convert(transformation=feature_extractor)

extract_all_features()