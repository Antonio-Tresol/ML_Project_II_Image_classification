import os


def count_classes(directory: str) -> int:
    """
    Get the list of unique labels in the given directory.

    Args:
        directory (str): The directory path.

    Returns:
        list: A list of unique labels found in the directory.
    """
    unique_labels = sum([1 for label in os.listdir(directory)])
    return unique_labels


def count_per_class(directory: str) -> list[int]:
    count_per_class = [
        len(os.listdir(os.path.join(directory, label)))
        for label in os.listdir(directory)
    ]
    return count_per_class


print(str(count_per_class("dataset/COVID-19_Radiography_Dataset")))
