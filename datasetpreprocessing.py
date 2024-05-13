import os
import configuration as config
import shutil



for folder in os.scandir(config.ROOT_DIR):
    if folder.is_dir():
        image_folder = os.path.join(config.ROOT_DIR, folder.name, "images")
        masks_folder = os.path.join(config.ROOT_DIR, folder.name, "masks")
        
        for file in os.scandir(image_folder):
            _, extension = os.path.splitext(file)
            new_file = os.path.join(folder, file.name.replace(".png", "-raw") + extension)
            shutil.copyfile(file.path, new_file)
            
        shutil.rmtree(image_folder)
        
        for file in os.scandir(masks_folder):
            _, extension = os.path.splitext(file)
            new_file = os.path.join(folder, file.name.replace(".png", "-mask") + extension)
            shutil.copyfile(file.path, new_file)

        shutil.rmtree(masks_folder)
