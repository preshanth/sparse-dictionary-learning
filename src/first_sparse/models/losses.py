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
