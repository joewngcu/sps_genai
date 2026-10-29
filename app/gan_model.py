"""The exact 28x28 GAN architecture in Assignment 3."""

import torch
from torch import nn


ARCHITECTURE = "assignment3_mnist_gan_v1"
LATENT_DIM = 100
PREPROCESSING = {"image_size": [28, 28], "channels": 1, "range": [-1.0, 1.0]}


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(LATENT_DIM, 7 * 7 * 128)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        features = self.project(noise).reshape(noise.shape[0], 128, 7, 7)
        return self.layers(features)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 1),
            nn.Sigmoid(),  # Convert the final scalar into a real/fake probability.
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.layers(images)


def initialize_weights(module: nn.Module) -> None:
    """Small initial weights, as commonly used for convolutional GANs."""
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
