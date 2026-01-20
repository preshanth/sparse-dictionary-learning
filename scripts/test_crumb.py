#!/usr/bin/env python
"""Test trained conv sparse model on held-out images"""

import sys
sys.path.insert(0, '/home/pjaganna/Software/dictionary_learning/mirabest')

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt

from CRUMB import CRUMB_MB
from first_sparse.models import ConvSparseAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./results/crumb_conv_sparse/best_model.pt')
    parser.add_argument('--output-dir', type=str, default='./results/crumb_conv_sparse/test')
    parser.add_argument('--n-samples', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']

    model = ConvSparseAutoencoder(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Model: {config['n_atoms']} atoms of size {config['atom_size']}x{config['atom_size']}")

    # Load test set
    def transform(img):
        img = torch.from_numpy(np.array(img)).float() / 255.0
        return img.unsqueeze(0)

    test_dataset = CRUMB_MB(
        root='/home/pjaganna/Software/dictionary_learning/mirabest',
        train=False,
        download=True,
        transform=transform
    )

    print(f"Test set: {len(test_dataset)} images")

    # Get random samples
    indices = np.random.choice(len(test_dataset), min(args.n_samples, len(test_dataset)), replace=False)

    images = []
    labels = []
    for i in indices:
        img, label = test_dataset[i]
        images.append(img)
        labels.append(label)

    images = torch.stack(images).to(device)

    # Run inference
    with torch.no_grad():
        reconstructions, codes = model(images, return_codes=True)

    images = images.cpu()
    reconstructions = reconstructions.cpu()
    codes = codes.cpu()

    # Compute metrics
    mse = ((images - reconstructions) ** 2).mean().item()
    print(f"\nTest MSE: {mse:.6f}")
    print(f"Test RMSE: {np.sqrt(mse):.6f}")

    # Sparsity stats
    sparsity_stats = model.compute_sparsity_stats(codes.to(device))
    print(f"L0 norm (active locations): {sparsity_stats['l0_norm']:.1f}")
    print(f"Active percent: {sparsity_stats['active_percent']:.2f}%")
    print(f"Atom usage - min: {sparsity_stats['min_atom_usage']:.3f}, max: {sparsity_stats['max_atom_usage']:.3f}, mean: {sparsity_stats['mean_atom_usage']:.3f}")

    # Plot reconstructions
    n_cols = min(8, args.n_samples)
    n_rows = int(np.ceil(args.n_samples / n_cols))

    fig, axes = plt.subplots(n_rows * 3, n_cols, figsize=(2*n_cols, 2*n_rows*3))

    for i in range(args.n_samples):
        row = (i // n_cols) * 3
        col = i % n_cols

        orig = images[i, 0].numpy()
        recon = reconstructions[i, 0].numpy()
        diff = orig - recon

        # Original
        axes[row, col].imshow(orig, cmap='gray', origin='lower')
        axes[row, col].set_title(f'L:{labels[i]}', fontsize=7)
        axes[row, col].axis('off')

        # Reconstruction
        axes[row+1, col].imshow(recon, cmap='gray', origin='lower')
        axes[row+1, col].axis('off')

        # Residual as fraction of source flux
        orig_max = max(orig.max(), 1e-6)
        frac_residual = diff / orig_max * 100  # percentage
        axes[row+2, col].imshow(frac_residual, cmap='RdBu_r', vmin=-10, vmax=10, origin='lower')
        axes[row+2, col].set_title(f'{np.abs(frac_residual).mean():.1f}%', fontsize=7)
        axes[row+2, col].axis('off')

    # Add row labels
    for row in range(n_rows):
        axes[row*3, 0].set_ylabel('Original', fontsize=8)
        axes[row*3+1, 0].set_ylabel('Recon', fontsize=8)
        axes[row*3+2, 0].set_ylabel('Residual', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'test_reconstructions.jpg', dpi=150)
    print(f"\nSaved: {output_dir / 'test_reconstructions.jpg'}")

    # Plot atoms
    atoms = model.get_dictionary().cpu()
    n_atoms = atoms.size(0)

    fig, axes = plt.subplots(2, n_atoms // 2 + n_atoms % 2, figsize=(2*n_atoms//2, 4))
    axes = axes.flatten()

    for i in range(n_atoms):
        atom = atoms[i, 0].numpy()
        vmax = np.abs(atom).max()
        axes[i].imshow(atom, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Atom {i}', fontsize=8)
        axes[i].axis('off')

    for i in range(n_atoms, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / 'learned_atoms.jpg', dpi=150)
    print(f"Saved: {output_dir / 'learned_atoms.jpg'}")

    # Plot spatial codes for first image
    fig, axes = plt.subplots(2, n_atoms // 2 + n_atoms % 2, figsize=(2*n_atoms//2, 4))
    axes = axes.flatten()

    code_map = codes[0].numpy()  # (n_atoms, H, W)
    for i in range(n_atoms):
        axes[i].imshow(code_map[i], cmap='hot', origin='lower')
        axes[i].set_title(f'Code {i}', fontsize=8)
        axes[i].axis('off')

    for i in range(n_atoms, len(axes)):
        axes[i].axis('off')

    plt.suptitle('Spatial activation maps for first test image', fontsize=10)
    plt.tight_layout()
    plt.savefig(output_dir / 'spatial_codes.jpg', dpi=150)
    print(f"Saved: {output_dir / 'spatial_codes.jpg'}")


if __name__ == '__main__':
    main()
