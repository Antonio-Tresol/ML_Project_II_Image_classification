import os
from PIL import Image
import shutil


def try_create_directory(directory: str, remove_if_exists: bool):
    existed = False

    try:
        os.makedirs(directory)
    except FileExistsError:
        existed = True
        # Remove folder and contents
        if remove_if_exists:
            shutil.rmtree(directory)
            os.makedirs(directory)

    return existed


class FolderImageConverter:
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

    def __transform_folder(self, transformation):
        """
        Transform images in each folder of the root directory.

        Parameters:
        - transformation (ImageTransformation): The transformation to apply to the images.

        Returns:
        - None
        """

        for folder in os.scandir(self.root_dir):
            if folder.is_dir():
                dest_folder_dir = os.path.join(self.dest_dir, folder.name)

                # Create the new folder
                try_create_directory(directory=dest_folder_dir, remove_if_exists=False)

                for image in os.scandir(folder):
                    print(image.path)
                    img = Image.open(image.path).convert("RGB")

                    # Transform the image
                    new_image = transformation.fit(img)
                    new_image.save(os.path.join(dest_folder_dir, image.name))

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

            print("Folder existed")

        else:
            try_create_directory(directory=self.dest_dir, remove_if_exists=True)

            self.__transform_folder(transformation=transformation)
