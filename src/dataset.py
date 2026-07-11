"""
dataset.py

Defines a custom PyTorch Dataset that wraps the EuroSAT dataset and
applies our preprocessing pipeline (from utils.py) to every image.
"""

import torchvision
from torch.utils.data import Dataset
from utils import preprocess_image


class EuroSATDataset(Dataset):
    """
    A PyTorch Dataset wrapper around torchvision's EuroSAT dataset,
    applying our custom preprocessing to each image.
    """

    def __init__(self, root="data/raw"):
        # Load the underlying EuroSAT dataset (already downloaded).
        self.base_dataset = torchvision.datasets.EuroSAT(root=root, download=False)
        self.classes = self.base_dataset.classes

    def __len__(self):
        # Total number of samples — required by PyTorch's Dataset interface.
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Fetch one raw (PIL image, label) pair...
        image, label = self.base_dataset[idx]

        # ...and preprocess the image into a normalized tensor.
        tensor = preprocess_image(image)

        return tensor, label