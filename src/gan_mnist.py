"""
GAN-Based Synthetic Image Generation, Diversity, Visual Quality and Training Stability Analysis
Dataset: MNIST handwritten digits
Baseline: Fully Connected Generative Adversarial Network (GAN)
"""

import os
import random
import argparse
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure standard deterministic behavior without triggering unsupported operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Automatically select CUDA GPU if available, otherwise CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


def get_mnist_dataloader(
    data_dir: str = "./dataset",
    batch_size: int = 128,
    num_workers: int = 0,
    shuffle: bool = True
) -> DataLoader:
    """
    Load the MNIST training dataset and return a PyTorch DataLoader.
    
    Transforms:
      - Converts PIL images to PyTorch tensors in range [0, 1].
      - Normalizes pixel values to [-1, 1] via Normalize((0.5,), (0.5,)).
    """
    os.makedirs(data_dir, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True
    )
    
    return dataloader


class Generator(nn.Module):
    """
    Fully connected Generator network for MNIST digit generation.
    
    Architecture:
      - Input: Latent noise vector z in R^{latent_dim} (default: 100)
      - Linear(latent_dim, 256) -> LeakyReLU(0.2)
      - Linear(256, 512) -> LeakyReLU(0.2)
      - Linear(512, 1024) -> LeakyReLU(0.2)
      - Linear(1024, 784) -> Tanh()
      - Output: Reshaped tensor of shape [batch_size, 1, 28, 28] with values in [-1, 1]
    """
    def __init__(self, latent_dim: int = 100, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape
        out_features = int(np.prod(img_shape))  # 1 * 28 * 28 = 784
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, out_features),
            nn.Tanh()
        )
        self.apply(init_weights)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Forward pass generating images from latent noise vectors."""
        img_flat = self.model(z)
        return img_flat.view(img_flat.size(0), *self.img_shape)


def init_weights(m: nn.Module) -> None:
    """Initialize linear layer weights with normal distribution and zero biases."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    device = get_device()
    print(f"Active Device: {device}")
    set_seed(42)
    loader = get_mnist_dataloader(batch_size=64)
    images, labels = next(iter(loader))
    print(f"Loaded batch shape: {images.shape}")
    print(f"Value range: min={images.min().item():.3f}, max={images.max().item():.3f}")
    
    # Test Generator
    gen = Generator(latent_dim=100).to(device)
    z = torch.randn(64, 100, device=device)
    gen_imgs = gen(z)
    print(f"Generated images shape: {gen_imgs.shape}")
    print(f"Generator value range: min={gen_imgs.min().item():.3f}, max={gen_imgs.max().item():.3f}")

