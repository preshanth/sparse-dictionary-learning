"""Metrics for evaluating sparse dictionary learning"""

from typing import Dict

import numpy as np
import torch


def compute_reconstruction_metrics(
    reconstruction: torch.Tensor,
    target: torch.Tensor
) -> Dict[str, float]:
    """Compute reconstruction quality metrics
    
    Args:
        reconstruction: (batch, 1, H, W) reconstructed images
        target: (batch, 1, H, W) target images
        
    Returns:
        Dictionary with metrics:
            - mse: Mean squared error
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - psnr: Peak signal-to-noise ratio
    """
    with torch.no_grad():
        # MSE
        mse = torch.mean((reconstruction - target) ** 2).item()
        
        # RMSE
        rmse = np.sqrt(mse)
        
        # MAE
        mae = torch.mean(torch.abs(reconstruction - target)).item()
        
        # PSNR (assume data in [0, 1] range)
        if mse > 0:
            psnr = 10 * np.log10(1.0 / mse)
        else:
            psnr = float('inf')
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'psnr': psnr
    }


def compute_sparsity_metrics(codes: torch.Tensor) -> Dict[str, float]:
    """Compute sparsity metrics for codes
    
    Args:
        codes: (batch, K) sparse code matrix
        
    Returns:
        Dictionary with metrics:
            - l0_norm: Average number of non-zero elements
            - l1_norm: Average L1 norm
            - l2_norm: Average L2 norm
            - active_percent: Percentage of non-zero elements
            - max_activation: Maximum activation value
            - mean_activation: Mean of non-zero activations
    """
    with torch.no_grad():
        # L0 norm (number of non-zero elements)
        threshold = 1e-6
        l0 = (codes.abs() > threshold).float().sum(dim=1).mean().item()
        
        # L1 norm
        l1 = codes.abs().sum(dim=1).mean().item()
        
        # L2 norm
        l2 = torch.norm(codes, p=2, dim=1).mean().item()
        
        # Active percentage
        active_percent = (codes.abs() > threshold).float().mean().item() * 100
        
        # Activation statistics
        nonzero_codes = codes[codes.abs() > threshold]
        if len(nonzero_codes) > 0:
            max_activation = nonzero_codes.abs().max().item()
            mean_activation = nonzero_codes.abs().mean().item()
        else:
            max_activation = 0.0
            mean_activation = 0.0
    
    return {
        'l0_norm': l0,
        'l1_norm': l1,
        'l2_norm': l2,
        'active_percent': active_percent,
        'max_activation': max_activation,
        'mean_activation': mean_activation
    }


def compute_atom_usage(codes: torch.Tensor) -> Dict[str, np.ndarray]:
    """Compute per-atom usage statistics
    
    Args:
        codes: (batch, K) sparse code matrix
        
    Returns:
        Dictionary with:
            - usage_count: (K,) number of times each atom is used
            - usage_fraction: (K,) fraction of samples using each atom
            - mean_activation: (K,) mean activation when atom is used
    """
    with torch.no_grad():
        threshold = 1e-6
        K = codes.size(1)
        
        # Usage count
        usage_mask = codes.abs() > threshold  # (batch, K)
        usage_count = usage_mask.sum(dim=0).cpu().numpy()  # (K,)
        
        # Usage fraction
        usage_fraction = usage_count / codes.size(0)
        
        # Mean activation when used
        mean_activation = np.zeros(K)
        for k in range(K):
            active_codes = codes[usage_mask[:, k], k]
            if len(active_codes) > 0:
                mean_activation[k] = active_codes.abs().mean().item()
    
    return {
        'usage_count': usage_count,
        'usage_fraction': usage_fraction,
        'mean_activation': mean_activation
    }
