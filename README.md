# GAN-Based Synthetic Image Generation, Diversity, Visual Quality and Training Stability Analysis

## 1. Problem Statement & Objective
In real-world machine learning systems, data scarcity and class imbalances are common bottlenecks. Generative Adversarial Networks (GANs) provide an unsupervised framework to generate synthetic samples mimicking true data distributions. However, GAN training is notoriously susceptible to:
- **Mode Collapse:** The generator outputs only a narrow subset of modes (e.g., repeating the same digit) rather than exploring the complete data distribution.
- **Adversarial Imbalance:** If either the Discriminator or Generator becomes overpowered, gradients vanish or explode, causing instability or divergent loss trajectories.

**Objective:** Develop an end-to-end fully connected Generative Adversarial Network (GAN) on the MNIST handwritten digits dataset and rigorously analyze:
1. **Generated-Image Diversity:** Quantitative intra-batch variability using Mean Per-Pixel Standard Deviation and Pairwise Root-Mean-Square (RMS) pixel distance.
2. **Visual Quality Progression:** Qualitative and proxy metric tracking across training checkpoints (Epochs 1, 5, 10, 20, 30).
3. **Training Stability:** Minimax adversarial loss curves, discriminator probability outputs ($D(x)$ vs $D(G(z))$), and training dynamics.

---

## 2. Technology Stack & Environment
- **Operating System:** Windows 11 (tested on Windows PowerShell)
- **Language:** Python 3.10.10
- **Deep Learning Framework:** PyTorch 2.14.1 (CPU / CUDA compatible)
- **Computer Vision:** Torchvision 0.29.1
- **Numerical Computing:** NumPy 2.2.6
- **Plotting & Visualization:** Matplotlib 3.10.9, Pillow 12.3.0
- **Interactive Notebooks:** JupyterLab 4.6.4 / Notebook 7.6.3
- **Version Control:** Git & GitHub

---

## 3. Dataset Source & Preprocessing
The **MNIST** (Modified National Institute of Standards and Technology) dataset contains 60,000 training images of handwritten digits $0-9$ of dimension $28 \times 28$ grayscale pixels (Y. LeCun, C. Cortes, C.J. Burges, 1998).

- **Data Download:** Automatically fetched into `dataset/` via `torchvision.datasets.MNIST`.
- **Normalization:** Converted to PyTorch tensors and scaled to range $[-1, 1]$ via `transforms.Normalize((0.5,), (0.5,))`.
- **Batching:** Mini-batches of 128 samples with random shuffling and `num_workers=0` (ensuring safe, reliable execution on Windows).

---

## 4. Repository Structure

```
gan-synthetic-data-analysis/
├── README.md                  # Comprehensive academic & technical report
├── requirements.txt            # Direct dependencies
├── .gitignore                  # Clean exclusion for .venv, dataset caches, bytecode
├── dataset/
│   └── .gitkeep               # Tracked directory anchor for MNIST download
├── src/
│   └── gan_mnist.py           # Complete GAN architecture, training, and metrics CLI
├── notebooks/
│   └── GAN_Analysis.ipynb     # Interactive Jupyter analysis with tables, images & plots
├── results/
│   ├── epoch_001.png          # Checkpoint grid at Epoch 1
│   ├── epoch_005.png          # Checkpoint grid at Epoch 5
│   ├── epoch_010.png          # Checkpoint grid at Epoch 10
│   ├── epoch_020.png          # Checkpoint grid at Epoch 20
│   ├── epoch_030.png          # Checkpoint grid at Epoch 30
│   ├── generated_samples.png  # Final 5x5 grid of generated synthetic digits
│   ├── diversity_analysis.png # Per-pixel std & pairwise RMS distance curves
│   ├── loss_curve.png         # Generator vs Discriminator BCE loss trajectories
│   └── training_metrics.csv   # Full epoch-by-epoch numeric metric logs
├── screenshots/
│   └── output.png             # Application/terminal execution documentation
└── tests/
    └── test_gan.py            # Unit test suite for shapes, ranges, gradients & metrics
```

---

## 5. GAN Architecture & Mathematical Formulation

The minimax two-player game is defined by Goodfellow et al. (2014):
$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1 - D(G(z)))]$$

### Generator Architecture ($G$)
- **Input:** Latent Gaussian noise $z \sim \mathcal{N}(0, I) \in \mathbb{R}^{100}$
- **Layer 1:** Linear(100, 256) + LeakyReLU(slope=0.2)
- **Layer 2:** Linear(256, 512) + LeakyReLU(slope=0.2)
- **Layer 3:** Linear(512, 1024) + LeakyReLU(slope=0.2)
- **Layer 4:** Linear(1024, 784) + Tanh()
- **Output:** Reshaped tensor of shape $[B, 1, 28, 28]$ with pixel intensities in $[-1, 1]$

### Discriminator Architecture ($D$)
- **Input:** Image $x \in \mathbb{R}^{1 \times 28 \times 28}$ flattened to $\mathbb{R}^{784}$
- **Layer 1:** Linear(784, 512) + LeakyReLU(slope=0.2) + Dropout(0.3)
- **Layer 2:** Linear(512, 256) + LeakyReLU(slope=0.2) + Dropout(0.3)
- **Layer 3:** Linear(256, 1) + Sigmoid()
- **Output:** Probability scalar $D(x) \in [0, 1]$ indicating whether the image is real

---

## 6. Training Methodology

1. **Alternating Optimization:**
   - **Step 1 (Discriminator Update):** Train $D$ on real images ($y=1.0$) and generated images ($y=0.0$) with gradient detachment (`fake_imgs.detach()`).
   - **Step 2 (Generator Update):** Generate new fake images, propagate through $D$, and compute loss against real labels ($y=1.0$) to incentivize $G$ to fool $D$.
2. **Optimizers:** Adam optimizer ($\text{lr}=0.0002, \beta_1=0.5, \beta_2=0.999$) for both networks.
3. **Weight Initialization:** Linear weights initialized from $\mathcal{N}(0.0, 0.02^2)$ and biases set to $0.0$.
4. **Reproducibility:** Global random seeds (Python, NumPy, PyTorch) set to `42`. Fixed latent vector $z_{\text{fixed}} \in \mathbb{R}^{25 \times 100}$ tracked at every checkpoint.

---

## 7. Diversity, Quality, and Stability Evaluation Methodology

### 7.1 Quantitative Diversity Evaluation
- **Mean Per-Pixel Standard Deviation:**
  $$\bar{\sigma}_{\text{pixel}} = \frac{1}{784} \sum_{k=1}^{784} \sigma(x_{\cdot, k})$$
  Measures variation per pixel across 500 generated samples. A collapse toward 0 indicates deterministic output repetition.
- **Mean Pairwise RMS Pixel Distance:**
  $$\bar{D}_{\text{RMS}} = \frac{2}{N(N-1)} \sum_{i < j} \sqrt{\frac{1}{784} \sum_{k=1}^{784} (x_{i,k} - x_{j,k})^2}$$
  Computed over $N=200$ samples (19,900 pairs). Measures pairwise Euclidean dissimilarity.

### 7.2 Visual Quality Analysis & Nearest-Neighbor Proxy
- Fixed latent vector image grids saved at Epochs 1, 5, 10, 20, 30.
- Nearest-neighbor minimum Euclidean distance between generated samples and real training digits to distinguish generalization from verbatim memorization.

### 7.3 Training Stability
- Epoch-by-epoch tracking of $L_G$, $L_D$, $D(x)$ (real score), and $D(G(z))$ (fake score) in `results/training_metrics.csv`.
- Plotted loss trajectory in `results/loss_curve.png`.

---

## 8. Installation & Execution Guide

### 8.1 Setup Virtual Environment
```powershell
# Open Windows PowerShell in project directory
cd D:\Projects\gan-synthetic-data-analysis

# Create and activate virtual environment (if not already existing)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install direct dependencies
pip install -r requirements.txt
```

### 8.2 Run Unit Tests
```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests
```

### 8.3 Run 1-Epoch Smoke Test
```powershell
& .\.venv\Scripts\python.exe src/gan_mnist.py --epochs 1 --results-dir ./results_smoke
```

### 8.4 Run Full Experiment (30 Epochs)
```powershell
& .\.venv\Scripts\python.exe src/gan_mnist.py --epochs 30 --results-dir ./results
```

### 8.5 Launch Jupyter Analysis Notebook
```powershell
& .\.venv\Scripts\jupyter-lab.exe notebooks/GAN_Analysis.ipynb
```

---

## 9. Limitations & Failure Modes
1. **Fully Connected Architecture:** MLPs lack translational invariance compared to Convolutional GANs (DCGAN), leading to occasional jagged boundary artifacts.
2. **Unconditional Generation:** Digit class cannot be explicitly requested at inference without conditioning labels (as in cGAN).
3. **Resolution:** Limited to $28 \times 28$ grayscale digits.

---

## 10. References & Acknowledgements
- Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS.
- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.
- LeCun, Y., Cortes, C., & Burges, C. J. (1998). *The MNIST Database of Handwritten Digits*.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*.
