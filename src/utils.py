"""
utils.py

Shared preprocessing utilities used across training, evaluation,
and the Streamlit app. Keeping this logic in one place ensures every
image is processed identically no matter where it's used.
"""

import cv2
import numpy as np
import torch

# ImageNet's standard mean/std values — widely used as a default
# normalization baseline, especially useful if we later use
# pretrained models (transfer learning).
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])

# The size our model expects every image to be resized to.
IMAGE_SIZE = 64


def preprocess_image(pil_image):
    """
    Converts a PIL image into a normalized PyTorch tensor ready
    for model input.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        A raw RGB image (e.g., from the dataset or a user upload).

    Returns
    -------
    torch.Tensor
        A tensor of shape (3, IMAGE_SIZE, IMAGE_SIZE), normalized.
    """

    # Step 1: Convert PIL image -> NumPy array so OpenCV can use it.
    # PIL gives us RGB order already, shape (H, W, 3).
    image_np = np.array(pil_image)

    # Step 2: Resize using OpenCV, in case the input isn't 64x64
    # (e.g., a real satellite image uploaded by a user).
    image_resized = cv2.resize(image_np, (IMAGE_SIZE, IMAGE_SIZE))

    # Step 3: Scale pixel values from [0, 255] integers to [0.0, 1.0] floats.
    image_scaled = image_resized.astype(np.float32) / 255.0

    # Step 4: Standardize using ImageNet mean/std (per-channel).
    image_normalized = (image_scaled - IMAGENET_MEAN) / IMAGENET_STD

    # Step 5: Rearrange from (Height, Width, Channels) to (Channels, Height, Width).
    # PyTorch expects channels first — this is called "CHW" format.
    image_transposed = np.transpose(image_normalized, (2, 0, 1))

    # Step 6: Convert to a PyTorch tensor with float32 precision.
    tensor = torch.tensor(image_transposed, dtype=torch.float32)

    return tensor