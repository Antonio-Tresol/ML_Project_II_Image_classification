import os
import sys
import inspect

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe()))
)
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
    
import configuration as config
from data.transforms.image_transformation import BilateralFilter
from data.transforms.folder_image_converter import FolderImageConverter

converter = FolderImageConverter(
    root_dir=config.ROOT_DIR,
    dest_dir="dataset/Covid_BilateralFilter",
    check_if_exists=False
)

bilateral = BilateralFilter()
converter.convert(transformation=bilateral)
