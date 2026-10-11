"""
Unit tests for GAN components:
- Generator and Discriminator shapes and value bounds
- Gradient flow for both networks
- Diversity metrics calculation
- Denormalization utility
"""

import unittest
import torch
import torch.nn as nn
from src.gan_mnist import (
    Generator,
    Discriminator,
    compute_pixel_diversity,
    compute_pairwise_rms_distance,
    denormalize_images
)


class TestGANComponents(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.latent_dim = 100
        self.batch_size = 16
        self.generator = Generator(latent_dim=self.latent_dim)
        self.discriminator = Discriminator()

    def test_generator_output_shape_and_range(self):
        z = torch.randn(self.batch_size, self.latent_dim)
        gen_imgs = self.generator(z)
        self.assertEqual(gen_imgs.shape, (self.batch_size, 1, 28, 28))
        self.assertTrue(torch.all(gen_imgs >= -1.0) and torch.all(gen_imgs <= 1.0))
        self.assertFalse(torch.isnan(gen_imgs).any())

    def test_discriminator_output_shape_and_range(self):
        imgs = torch.randn(self.batch_size, 1, 28, 28).clamp(-1.0, 1.0)
        preds = self.discriminator(imgs)
        self.assertEqual(preds.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(preds >= 0.0) and torch.all(preds <= 1.0))
        self.assertFalse(torch.isnan(preds).any())

    def test_gradient_flow(self):
        z = torch.randn(self.batch_size, self.latent_dim)
        real_imgs = torch.randn(self.batch_size, 1, 28, 28).clamp(-1.0, 1.0)
        criterion = nn.BCELoss()
        
        # Test Discriminator backward
        real_labels = torch.ones(self.batch_size, 1)
        fake_labels = torch.zeros(self.batch_size, 1)
        
        gen_imgs = self.generator(z)
        d_real = self.discriminator(real_imgs)
        d_fake = self.discriminator(gen_imgs.detach())
        d_loss = criterion(d_real, real_labels) + criterion(d_fake, fake_labels)
        
        d_loss.backward()
        for param in self.discriminator.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.any(param.grad != 0.0))
            
        # Test Generator backward
        g_preds = self.discriminator(gen_imgs)
        g_loss = criterion(g_preds, real_labels)
        g_loss.backward()
        for param in self.generator.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.any(param.grad != 0.0))

    def test_diversity_metrics(self):
        # Generate varied synthetic images
        images = torch.randn(50, 1, 28, 28).clamp(-1.0, 1.0)
        pixel_div = compute_pixel_diversity(images)
        pairwise_dist = compute_pairwise_rms_distance(images, max_samples=30)
        
        self.assertIsInstance(pixel_div, float)
        self.assertIsInstance(pairwise_dist, float)
        self.assertGreater(pixel_div, 0.0)
        self.assertGreater(pairwise_dist, 0.0)

    def test_denormalize_images(self):
        norm_imgs = torch.tensor([-1.0, 0.0, 1.0])
        denorm = denormalize_images(norm_imgs)
        self.assertTrue(torch.allclose(denorm, torch.tensor([0.0, 0.5, 1.0]), atol=1e-5))


if __name__ == "__main__":
    unittest.main()
