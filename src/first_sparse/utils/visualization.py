"""Visualization utilities for dictionary atoms and reconstructions"""

from pathlib import Path
from typing import Optional, List

import numpy as np
import matplotlib.pyplot as plt
import torch


def plot_dictionary_atoms(
    atoms: torch.Tensor,
    n_atoms: int = 64,
    n_cols: int = 8,
    figsize: tuple = (16, 16),
    save_path: Optional[str] = None,
    title: str = "Dictionary Atoms",
    cmap: str = 'viridis'
):
    """Plot grid of dictionary atoms
    
    Args:
        atoms: (K, 1, H, W) dictionary atoms
        n_atoms: Number of atoms to plot (up to K)
        n_cols: Number of columns in grid
        figsize: Figure size
        save_path: Path to save figure
        title: Figure title
        cmap: Colormap
    """
    atoms = atoms.detach().cpu()
    K = atoms.size(0)
    n_atoms = min(n_atoms, K)
    
    n_rows = int(np.ceil(n_atoms / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i in range(n_atoms):
        atom = atoms[i, 0].numpy()
        
        axes[i].imshow(atom, cmap=cmap, interpolation='nearest')
        axes[i].set_title(f"Atom {i}", fontsize=8)
        axes[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_atoms, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved dictionary atoms to {save_path}")
    
    return fig


def plot_reconstructions(
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    n_samples: int = 8,
    figsize: tuple = (16, 8),
    save_path: Optional[str] = None,
    title: str = "Reconstructions",
    cmap: str = 'viridis'
):
    """Plot original images vs reconstructions
    
    Args:
        originals: (batch, 1, H, W) original images
        reconstructions: (batch, 1, H, W) reconstructed images
        n_samples: Number of samples to plot
        figsize: Figure size
        save_path: Path to save figure
        title: Figure title
        cmap: Colormap
    """
    originals = originals.detach().cpu()
    reconstructions = reconstructions.detach().cpu()
    
    n_samples = min(n_samples, originals.size(0))
    
    fig, axes = plt.subplots(3, n_samples, figsize=figsize)
    
    for i in range(n_samples):
        orig = originals[i, 0].numpy()
        recon = reconstructions[i, 0].numpy()
        residual = orig - recon
        
        # Original
        axes[0, i].imshow(orig, cmap=cmap, interpolation='nearest')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)
        axes[0, i].axis('off')
        
        # Reconstruction
        axes[1, i].imshow(recon, cmap=cmap, interpolation='nearest')
        if i == 0:
            axes[1, i].set_ylabel('Reconstruction', fontsize=10)
        axes[1, i].axis('off')
        
        # Residual
        im = axes[2, i].imshow(residual, cmap='RdBu_r', interpolation='nearest',
                               vmin=-np.abs(residual).max(), 
                               vmax=np.abs(residual).max())
        if i == 0:
            axes[2, i].set_ylabel('Residual', fontsize=10)
        axes[2, i].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved reconstructions to {save_path}")
    
    return fig


def plot_training_curves(
    train_losses: List[dict],
    val_losses: List[dict],
    save_path: Optional[str] = None,
    figsize: tuple = (15, 5)
):
    """Plot training curves
    
    Args:
        train_losses: List of training loss dictionaries per epoch
        val_losses: List of validation loss dictionaries per epoch
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    epochs = range(1, len(train_losses) + 1)
    val_epochs = range(1, len(val_losses) + 1)
    
    # Total loss
    axes[0].plot(epochs, [l['total'] for l in train_losses], 
                label='Train', linewidth=2)
    axes[0].plot(val_epochs, [l['total'] for l in val_losses],
                label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, [l['reconstruction'] for l in train_losses],
                label='Train', linewidth=2)
    axes[1].plot(val_epochs, [l['reconstruction'] for l in val_losses],
                label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Sparsity (L0 norm)
    if 'l0' in train_losses[0]:
        axes[2].plot(epochs, [l['l0'] for l in train_losses],
                    label='Train', linewidth=2)
        axes[2].plot(val_epochs, [l['l0'] for l in val_losses],
                    label='Val', linewidth=2)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('L0 Norm (Active Atoms)')
        axes[2].set_title('Sparsity')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training curves to {save_path}")
    
    return fig


def plot_atom_usage_histogram(
    usage_fraction: np.ndarray,
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """Plot histogram of atom usage
    
    Args:
        usage_fraction: (K,) fraction of samples using each atom
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.hist(usage_fraction * 100, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Usage Percentage (%)')
    ax.set_ylabel('Number of Atoms')
    ax.set_title('Dictionary Atom Usage Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    stats_text = f"Mean: {usage_fraction.mean()*100:.1f}%\n"
    stats_text += f"Median: {np.median(usage_fraction)*100:.1f}%\n"
    stats_text += f"Unused: {(usage_fraction == 0).sum()}/{len(usage_fraction)}"
    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
           verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
           fontsize=10)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved atom usage histogram to {save_path}")

    return fig


def save_atoms_by_usage(
    atoms: torch.Tensor,
    usage: np.ndarray,
    output_dir: str,
    usage_threshold: float = 0.01
):
    """Save active and dead atoms to separate files

    Args:
        atoms: (K, 1, H, W) dictionary atoms
        usage: (K,) usage fraction per atom
        output_dir: Directory to save plots
        usage_threshold: Atoms with usage < threshold are "dead"
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    active_mask = usage >= usage_threshold
    active_idx = np.where(active_mask)[0]
    dead_idx = np.where(~active_mask)[0]

    # Plot active atoms
    if len(active_idx) > 0:
        plot_dictionary_atoms(
            atoms[active_idx],
            n_atoms=len(active_idx),
            save_path=str(output_dir / "atoms_active.png"),
            title=f"Active Atoms ({len(active_idx)})"
        )

    # Plot dead atoms
    if len(dead_idx) > 0:
        plot_dictionary_atoms(
            atoms[dead_idx],
            n_atoms=len(dead_idx),
            save_path=str(output_dir / "atoms_dead.png"),
            title=f"Dead Atoms ({len(dead_idx)})"
        )
