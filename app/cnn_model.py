"""The exact CNN architecture specified in Assignment2.pdf."""

import torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ARCHITECTURE = "assignment2_cnn_v1"
CLASS_NAMES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)

# Save this information with the weights so the API can check its preprocessing.
PREPROCESSING = {
    "image_size": [64, 64],
    "color_mode": "RGB",
    "interpolation": "bilinear",
    "antialias": True,
    "mean": [0.5, 0.5, 0.5],
    "std": [0.5, 0.5, 0.5],
}


def make_transform():
    """Use the same resize and normalization during training and inference."""
    return transforms.Compose([
        transforms.Resize(
            (64, 64), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=PREPROCESSING["mean"], std=PREPROCESSING["std"]),
    ])


class AssignmentCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                 # 32 * 16 * 16 = 8,192 features
            nn.Linear(32 * 16 * 16, 100),
            nn.ReLU(),
            nn.Linear(100, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # Return raw logits. CrossEntropyLoss applies the needed log-softmax.
        return self.classifier(self.features(images))
