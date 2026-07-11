python src/download_data.py"""
download_data.py

Downloads the EuroSAT dataset (RGB version) using torchvision's built-in
dataset class. This script only needs to be run once.
"""

import torchvision

def download_eurosat(root: str = "data/raw"):
    """
    Downloads the EuroSAT dataset into the given root folder.

    Parameters
    ----------
    root : str
        Directory where the dataset will be stored.
    """
    print("Downloading EuroSAT dataset... this may take a few minutes.")

    dataset = torchvision.datasets.EuroSAT(
        root=root,
        download=True
    )

    print(f"Download complete. Total images: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

if __name__ == "__main__":
    download_eurosat()