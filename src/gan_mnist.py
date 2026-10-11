"""
GAN-Based Synthetic Image Generation, Diversity, Visual Quality and Training Stability Analysis
Dataset: MNIST Handwritten Digits
Architecture: Fully Connected Generative Adversarial Network (GAN)
"""

import os
import csv
import time
import random
import argparse
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# =====================================================================
# 1. Configuration Dataclass
# =====================================================================
@dataclass
class GANConfig:
    latent_dim: int = 100
    batch_size: int = 128
    epochs: int = 30
    lr: float = 0.0002
    beta1: float = 0.5
    beta2: float = 0.999
    seed: int = 42
    dataset_dir: str = "./dataset"
    results_dir: str = "./results"
    num_eval_samples: int = 500
    checkpoint_epochs: Tuple[int, ...] = (1, 5, 10, 20, 30, 40, 50)


# =====================================================================
# 2. Reproducibility & Device Configuration
# =====================================================================
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Automatically select CUDA GPU if available, otherwise fall back to CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    return device


# =====================================================================
# 3. Dataset Loading and Normalization
# =====================================================================
def denormalize_images(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert image tensors from [-1, 1] back to [0, 1] range for visualization/evaluation.
    """
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


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


# =====================================================================
# 4. Generator Architecture
# =====================================================================
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


# =====================================================================
# 5. Discriminator Architecture
# =====================================================================
class Discriminator(nn.Module):
    """
    Fully connected Discriminator network for MNIST digit classification.
    
    Architecture:
      - Input: Image tensor [batch_size, 1, 28, 28] -> Flatten to 784 features
      - Linear(784, 512) -> LeakyReLU(0.2) -> Dropout(0.3)
      - Linear(512, 256) -> LeakyReLU(0.2) -> Dropout(0.3)
      - Linear(256, 1) -> Sigmoid()
      - Output: Real/Fake probability score in [0, 1] of shape [batch_size, 1]
    """
    def __init__(self, img_shape: Tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        self.img_shape = img_shape
        in_features = int(np.prod(img_shape))  # 784
        
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.apply(init_weights)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """Forward pass returning probability that input image is real."""
        return self.model(img)


# =====================================================================
# 6. Weight Initialization
# =====================================================================
def init_weights(m: nn.Module) -> None:
    """Initialize linear layer weights with normal distribution and zero biases."""
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)


# =====================================================================
# 7. Diversity and Quality Metrics
# =====================================================================
def compute_pixel_diversity(images: torch.Tensor) -> float:
    """
    Mean per-pixel standard deviation across a batch of generated images.
    
    1. Denormalize images to [0, 1].
    2. Compute standard deviation across the batch dimension for every pixel.
    3. Return the average standard deviation over all 784 pixels.
    """
    imgs_01 = denormalize_images(images)
    # std across batch dimension (dim 0)
    std_per_pixel = imgs_01.std(dim=0)
    return float(std_per_pixel.mean().item())


def compute_pairwise_rms_distance(images: torch.Tensor, max_samples: int = 200) -> float:
    """
    Mean pairwise Root-Mean-Square (RMS) pixel distance on a subset of generated samples.
    
    Formula for image pair (x_i, x_j):
        RMS(x_i, x_j) = sqrt( (1 / 784) * sum_{k=1}^{784} (x_{i,k} - x_{j,k})^2 )
    
    Computational Complexity: O(N^2 * D) where N=max_samples, D=784.
    A subset (default 200 samples => 19,900 distinct pairs) gives a statistically robust
    estimate while computing in under a second.
    """
    imgs_01 = denormalize_images(images)
    num_samples = min(len(imgs_01), max_samples)
    flat_imgs = imgs_01[:num_samples].view(num_samples, -1)  # [N, 784]
    
    # Compute pairwise differences: [N, 1, 784] - [1, N, 784] -> [N, N, 784]
    diff = flat_imgs.unsqueeze(1) - flat_imgs.unsqueeze(0)
    # Pairwise RMS: [N, N]
    rms_matrix = torch.sqrt(torch.mean(diff ** 2, dim=-1))
    
    # Extract upper triangular matrix (excluding diagonal i == j)
    triu_indices = torch.triu_indices(num_samples, num_samples, offset=1)
    pairwise_distances = rms_matrix[triu_indices[0], triu_indices[1]]
    
    return float(pairwise_distances.mean().item())


def compute_nearest_neighbor_distance(
    generated_images: torch.Tensor,
    real_images: torch.Tensor,
    max_eval: int = 100
) -> float:
    """
    Proxy metric for visual quality / memorization:
    Computes the average minimum Euclidean distance from generated samples to real samples.
    """
    gen_01 = denormalize_images(generated_images)[:max_eval].view(-1, 784)
    real_01 = denormalize_images(real_images)[:1000].view(-1, 784)
    
    # Pairwise Euclidean distances: [num_gen, num_real]
    dists = torch.cdist(gen_01, real_01, p=2.0)
    min_dists, _ = dists.min(dim=1)
    return float(min_dists.mean().item())


# =====================================================================
# 8. Checkpoint Image Visualization & Plotting
# =====================================================================
def save_image_grid(
    images: torch.Tensor,
    output_path: str,
    grid_size: Tuple[int, int] = (5, 5),
    title: Optional[str] = None
) -> None:
    """
    Save a grid of generated images to disk with hidden axes and clear title.
    Images are denormalized from [-1, 1] to [0, 1].
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    imgs_01 = denormalize_images(images).cpu().detach().numpy()
    
    nrows, ncols = grid_size
    num_to_plot = min(len(imgs_01), nrows * ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)
    
    for idx in range(nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = axes[row, col] if nrows > 1 else axes[col]
        if idx < num_to_plot:
            ax.imshow(imgs_01[idx, 0], cmap="gray", interpolation="nearest")
        ax.axis("off")
        
    if title:
        fig.suptitle(title, fontsize=12, y=0.96)
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_loss_curves(metrics_csv_path: str, output_path: str) -> None:
    """Generate and save Generator vs Discriminator loss curve across epochs."""
    epochs, g_losses, d_losses = [], [], []
    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            g_losses.append(float(row["generator_loss"]))
            d_losses.append(float(row["discriminator_loss"]))
            
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, g_losses, label="Generator Loss", color="#1f77b4", linewidth=2)
    plt.plot(epochs, d_losses, label="Discriminator Loss", color="#ff7f0e", linewidth=2)
    plt.title("GAN Training Stability: Generator vs Discriminator Loss", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("Binary Cross-Entropy Loss", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_diversity_curves(metrics_csv_path: str, output_path: str) -> None:
    """Generate and save image diversity curves across training epochs."""
    epochs, pixel_divs, pairwise_dists = [], [], []
    with open(metrics_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            pixel_divs.append(float(row["pixel_diversity"]))
            pairwise_dists.append(float(row["mean_pairwise_distance"]))
            
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = "#2ca02c"
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Mean Pixel Std (Per-Pixel Diversity)", color=color1, fontsize=11)
    line1 = ax1.plot(epochs, pixel_divs, color=color1, marker="o", label="Per-Pixel Std Diversity", linewidth=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.5)
    
    ax2 = ax1.twinx()
    color2 = "#d62728"
    ax2.set_ylabel("Mean Pairwise RMS Distance", color=color2, fontsize=11)
    line2 = ax2.plot(epochs, pairwise_dists, color=color2, marker="s", linestyle="--", label="Pairwise RMS Distance", linewidth=2)
    ax2.tick_params(axis="y", labelcolor=color2)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", frameon=True, facecolor="white")
    
    plt.title("Generated Image Diversity Evolution Across Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


# =====================================================================
# 9. Training Loop
# =====================================================================
def train_gan(config: GANConfig) -> Dict[str, List[float]]:
    """
    Execute full alternating GAN training with checkpoint evaluation and metric logging.
    """
    set_seed(config.seed)
    device = get_device()
    print(f"=== Starting GAN Training on {device.type.upper()} ===")
    print(f"Config: Epochs={config.epochs}, Batch Size={config.batch_size}, LR={config.lr}, Seed={config.seed}")
    
    os.makedirs(config.results_dir, exist_ok=True)
    metrics_csv_path = os.path.join(config.results_dir, "training_metrics.csv")
    
    # Fixed latent vectors for consistent checkpoint visualization across epochs (5x5 grid = 25 images)
    fixed_z = torch.randn(25, config.latent_dim, device=device)
    
    # Dataloader
    dataloader = get_mnist_dataloader(
        data_dir=config.dataset_dir,
        batch_size=config.batch_size,
        num_workers=0,
        shuffle=True
    )
    
    # Models
    generator = Generator(latent_dim=config.latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    # Loss & Optimizers
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
    
    # Initialize CSV metrics log
    with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "generator_loss",
            "discriminator_loss",
            "pixel_diversity",
            "mean_pairwise_distance",
            "d_real_score",
            "d_fake_score"
        ])
        
    start_time = time.time()
    
    for epoch in range(1, config.epochs + 1):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        epoch_d_real = 0.0
        epoch_d_fake = 0.0
        total_batches = len(dataloader)
        
        generator.train()
        discriminator.train()
        
        for batch_idx, (real_imgs, _) in enumerate(dataloader):
            current_batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            
            # Ground truth labels
            real_labels = torch.ones(current_batch_size, 1, device=device)
            fake_labels = torch.zeros(current_batch_size, 1, device=device)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real loss
            output_real = discriminator(real_imgs)
            loss_d_real = criterion(output_real, real_labels)
            
            # Fake loss
            z = torch.randn(current_batch_size, config.latent_dim, device=device)
            fake_imgs = generator(z)
            output_fake = discriminator(fake_imgs.detach())
            loss_d_fake = criterion(output_fake, fake_labels)
            
            # Total Discriminator loss
            loss_d = loss_d_real + loss_d_fake
            loss_d.backward()
            optimizer_D.step()
            
            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()
            
            # Generator wants Discriminator to classify fake images as real
            output_g = discriminator(fake_imgs)
            loss_g = criterion(output_g, real_labels)
            loss_g.backward()
            optimizer_G.step()
            
            # Accumulate metrics
            epoch_d_loss += loss_d.item()
            epoch_g_loss += loss_g.item()
            epoch_d_real += output_real.mean().item()
            epoch_d_fake += output_fake.mean().item()
            
        avg_d_loss = epoch_d_loss / total_batches
        avg_g_loss = epoch_g_loss / total_batches
        avg_d_real = epoch_d_real / total_batches
        avg_d_fake = epoch_d_fake / total_batches
        
        # -----------------------------
        # Diversity & Quality Evaluation
        # -----------------------------
        generator.eval()
        with torch.no_grad():
            eval_z = torch.randn(config.num_eval_samples, config.latent_dim, device=device)
            eval_fake = generator(eval_z)
            pixel_div = compute_pixel_diversity(eval_fake)
            pairwise_dist = compute_pairwise_rms_distance(eval_fake, max_samples=200)
            
            # Checkpoint Visualization
            if epoch in config.checkpoint_epochs or epoch == config.epochs:
                chkpt_path = os.path.join(config.results_dir, f"epoch_{epoch:03d}.png")
                chkpt_samples = generator(fixed_z)
                save_image_grid(
                    chkpt_samples,
                    chkpt_path,
                    grid_size=(5, 5),
                    title=f"MNIST GAN - Epoch {epoch} (D_loss={avg_d_loss:.3f}, G_loss={avg_g_loss:.3f})"
                )
                print(f"  [Checkpoint Saved] {chkpt_path}", flush=True)
                
        # Append metrics to CSV
        with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                f"{avg_g_loss:.4f}",
                f"{avg_d_loss:.4f}",
                f"{pixel_div:.4f}",
                f"{pairwise_dist:.4f}",
                f"{avg_d_real:.4f}",
                f"{avg_d_fake:.4f}"
            ])
            
        print(
            f"Epoch [{epoch:02d}/{config.epochs:02d}] "
            f"Loss_D: {avg_d_loss:.4f} | Loss_G: {avg_g_loss:.4f} | "
            f"D(x): {avg_d_real:.3f} | D(G(z)): {avg_d_fake:.3f} | "
            f"Pixel_Div: {pixel_div:.4f} | Pairwise_RMS: {pairwise_dist:.4f}",
            flush=True
        )
        
    elapsed_time = time.time() - start_time
    print(f"=== Training Completed in {elapsed_time:.1f}s ({elapsed_time/60:.2f} min) ===")
    
    # -----------------------------
    # Final Visualizations & Plots
    # -----------------------------
    generator.eval()
    with torch.no_grad():
        final_samples = generator(fixed_z)
        final_sample_path = os.path.join(config.results_dir, "generated_samples.png")
        save_image_grid(
            final_samples,
            final_sample_path,
            grid_size=(5, 5),
            title=f"Final Generated MNIST Samples (Epoch {config.epochs})"
        )
        print(f"Saved final samples to {final_sample_path}")
        
    loss_curve_path = os.path.join(config.results_dir, "loss_curve.png")
    plot_loss_curves(metrics_csv_path, loss_curve_path)
    print(f"Saved loss curve to {loss_curve_path}")
    
    diversity_curve_path = os.path.join(config.results_dir, "diversity_analysis.png")
    plot_diversity_curves(metrics_csv_path, diversity_curve_path)
    print(f"Saved diversity analysis plot to {diversity_curve_path}")
    
    # Save model weights
    torch.save(generator.state_dict(), os.path.join(config.results_dir, "generator.pth"))
    torch.save(discriminator.state_dict(), os.path.join(config.results_dir, "discriminator.pth"))
    print("Saved model checkpoints (generator.pth, discriminator.pth)")
    
    return {
        "metrics_csv": metrics_csv_path,
        "final_samples": final_sample_path,
        "loss_curve": loss_curve_path,
        "diversity_curve": diversity_curve_path
    }


# =====================================================================
# 10. Command-Line Entry Point
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Train a fully connected GAN on MNIST.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (default: 128)")
    parser.add_argument("--latent-dim", type=int, default=100, help="Latent noise vector dimension (default: 100)")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for Adam optimizer (default: 0.0002)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--dataset-dir", type=str, default="./dataset", help="Directory where MNIST is stored")
    parser.add_argument("--results-dir", type=str, default="./results", help="Directory where results and plots are saved")
    
    args = parser.parse_args()
    
    config = GANConfig(
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        seed=args.seed,
        dataset_dir=args.dataset_dir,
        results_dir=args.results_dir
    )
    
    train_gan(config)


if __name__ == "__main__":
    main()
