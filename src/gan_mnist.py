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


if __name__ == "__main__":
    device = get_device()
    print(f"Active Device: {device}")
    set_seed(42)
    loader = get_mnist_dataloader(batch_size=64)
    images, labels = next(iter(loader))
    print(f"Loaded batch shape: {images.shape}")
    print(f"Value range: min={images.min().item():.3f}, max={images.max().item():.3f}")
