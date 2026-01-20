"""Loss functions for sparse dictionary learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReconstructionLoss(nn.Module):
    """Reconstruction loss (MSE)"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, reconstruction, target):
        """
        Args:
            reconstruction: (batch, 1, H, W)
            target: (batch, 1, H, W)
            
        Returns:
            MSE loss
        """
        return F.mse_loss(reconstruction, target)


class SparsityLoss(nn.Module):
    """L1 sparsity penalty on codes"""
    
    def __init__(self, weight: float = 0.01):
        super().__init__()
        self.weight = weight
    
    def forward(self, codes):
        """
        Args:
            codes: (batch, code_dim) sparse codes
            
        Returns:
            L1 norm of codes scaled by weight
        """
        return self.weight * codes.abs().mean()


class DiversityLoss(nn.Module):
    """Diversity penalty to prevent dictionary collapse
    
    Penalizes high coherence (correlation) between dictionary atoms.
    We want atoms to be as orthogonal as possible.
    """
    
    def __init__(self, weight: float = 0.001):
        super().__init__()
        self.weight = weight
    
    def forward(self, dictionary_weight):
        """
        Args:
            dictionary_weight: (H*W, K) dictionary matrix
                Each column is an atom
            
        Returns:
            Coherence penalty
        """
        # Normalize atoms
        atoms = dictionary_weight.t()  # (K, H*W)
        atoms_normalized = F.normalize(atoms, p=2, dim=1)
        
        # Compute gram matrix (atom similarities)
        gram = atoms_normalized @ atoms_normalized.t()  # (K, K)
        
        # Penalize off-diagonal elements (want atoms orthogonal)
        # Subtract identity to ignore diagonal
        coherence = (gram - torch.eye(gram.size(0), device=gram.device)).abs().mean()
        
        return self.weight * coherence


class CombinedLoss(nn.Module):
    """Combined loss for sparse autoencoder training"""
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        sparsity_weight: float = 0.01,
        diversity_weight: float = 0.001
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        
        self.reconstruction_loss = ReconstructionLoss()
        self.sparsity_loss = SparsityLoss(weight=1.0)  # weight applied externally
        self.diversity_loss = DiversityLoss(weight=1.0)
    
    def forward(self, reconstruction, target, codes, dictionary_weight):
        """Compute total loss
        
        Args:
            reconstruction: (batch, 1, H, W) reconstructed images
            target: (batch, 1, H, W) target images
            codes: (batch, K) sparse codes
            dictionary_weight: (H*W, K) dictionary matrix
            
        Returns:
            dict with 'total' and component losses
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(reconstruction, target)
        
        # Sparsity loss
        sparse_loss = self.sparsity_loss(codes)
        
        # Diversity loss
        div_loss = self.diversity_loss(dictionary_weight)
        
        # Total weighted loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.sparsity_weight * sparse_loss +
            self.diversity_weight * div_loss
        )
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss.item(),
            'sparsity': sparse_loss.item(),
            'diversity': div_loss.item()
        }


class SSIMLoss(nn.Module):
    """Structural Similarity Index loss"""

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.register_buffer('window', self._create_window(window_size, sigma))

    def _create_window(self, window_size: int, sigma: float) -> torch.Tensor:
        """Create Gaussian window for SSIM"""
        coords = torch.arange(window_size).float() - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, reconstruction, target):
        """
        Args:
            reconstruction: (batch, 1, H, W)
            target: (batch, 1, H, W)

        Returns:
            1 - SSIM (so lower is better)
        """
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        window = self.window.to(reconstruction.device)
        padding = self.window_size // 2

        mu1 = F.conv2d(reconstruction, window, padding=padding)
        mu2 = F.conv2d(target, window, padding=padding)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(reconstruction ** 2, window, padding=padding) - mu1_sq
        sigma2_sq = F.conv2d(target ** 2, window, padding=padding) - mu2_sq
        sigma12 = F.conv2d(reconstruction * target, window, padding=padding) - mu1_mu2

        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return 1 - ssim.mean()


class ConvSparseLoss(nn.Module):
    """Combined loss for convolutional sparse coding"""

    def __init__(
        self,
        mse_weight: float = 1.0,
        ssim_weight: float = 0.1,
        sparsity_weight: float = 0.01
    ):
        super().__init__()

        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.sparsity_weight = sparsity_weight

        self.mse_loss = nn.MSELoss()
        self.ssim_loss = SSIMLoss()

    def forward(self, reconstruction, target, codes):
        """Compute total loss

        Args:
            reconstruction: (batch, 1, H, W) reconstructed images
            target: (batch, 1, H, W) target images
            codes: (batch, n_atoms, H, W) spatial sparse codes

        Returns:
            dict with 'total' and component losses
        """
        mse = self.mse_loss(reconstruction, target)
        ssim = self.ssim_loss(reconstruction, target)
        sparsity = codes.abs().mean()

        total_loss = (
            self.mse_weight * mse +
            self.ssim_weight * ssim +
            self.sparsity_weight * sparsity
        )

        return {
            'total': total_loss,
            'mse': mse.item(),
            'ssim': ssim.item(),
            'sparsity': sparsity.item()
        }
