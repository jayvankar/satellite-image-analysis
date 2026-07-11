"""
model.py

Defines our custom CNN architecture for classifying EuroSAT satellite images
into 10 land-use classes.
"""

import torch.nn as nn

NUM_CLASSES = 10

class SatelliteCNN(nn.Module):
    """
    A simple CNN for 64x64 RGB satellite image classification.
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        # --- Feature extraction: conv + activation + pooling, stacked 3 times ---
        self.features = nn.Sequential(
            # Block 1: 3 input channels (RGB) -> 32 filters
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64x64 -> 32x32

            # Block 2: 32 -> 64 filters
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 32x32 -> 16x16

            # Block 3: 64 -> 128 filters
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 16x16 -> 8x8
        )

        # --- Classification head: flatten, then fully connected layers ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x