#!/usr/bin/env python
"""Test trained conv sparse model on arbitrary images (FITS or PNG/JPG)"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt

from first_sparse.models import ConvSparseAutoencoder


def load_image(path):
    """Load image from FITS or standard image format"""
    path = Path(path)

    if path.suffix.lower() in ['.fits', '.fit', '.fts']:
        from astropy.io import fits
        with fits.open(path) as hdul:
            data = hdul[0].data
            # Handle multi-dimensional data
            while data.ndim > 2:
                data = data[0]
            # Handle NaN
            data = np.nan_to_num(data, nan=0.0)
        return data.astype(np.float32)
    else:
        from PIL import Image
        img = Image.open(path).convert('L')
        return np.array(img).astype(np.float32)


def normalize_image(img):
    """Normalize to [0, 1]"""
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to image (FITS, PNG, JPG)')
    parser.add_argument('--checkpoint', type=str, default='./results/crumb_conv_sparse/best_model.pt')
    parser.add_argument('--output-dir', type=str, default='./results/arbitrary_test')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max-size', type=int, default=1024, help='Max image size (resize if larger)')
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

    # Load image
    print(f"Loading image: {args.image}")
    img = load_image(args.image)
    orig_shape = img.shape
    print(f"Original size: {orig_shape}")

    # Resize if too large
    if max(img.shape) > args.max_size:
        from scipy.ndimage import zoom
        scale = args.max_size / max(img.shape)
        img = zoom(img, scale, order=1)
        print(f"Resized to: {img.shape}")

    # Normalize
    img = normalize_image(img)

    # Convert to tensor
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(device)
    print(f"Input tensor shape: {img_tensor.shape}")

    # Run inference
    with torch.no_grad():
        reconstruction, codes = model(img_tensor, return_codes=True)

    img_np = img_tensor[0, 0].cpu().numpy()
    recon_np = reconstruction[0, 0].cpu().numpy()
    codes_np = codes[0].cpu().numpy()

    # Metrics
    mse = ((img_np - recon_np) ** 2).mean()
    residual = img_np - recon_np
    orig_max = max(img_np.max(), 1e-6)
    frac_residual = residual / orig_max * 100

    print(f"\nResults:")
    print(f"  MSE: {mse:.6f}")
    print(f"  RMSE: {np.sqrt(mse):.6f}")
    print(f"  Mean |residual|: {np.abs(frac_residual).mean():.2f}% of peak")
    print(f"  Max |residual|: {np.abs(frac_residual).max():.2f}% of peak")

    # Sparsity
    active = (np.abs(codes_np) > 1e-6).sum()
    total = codes_np.size
    print(f"  Active codes: {active} / {total} ({100*active/total:.2f}%)")

    # Plot results
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Original
    im0 = axes[0, 0].imshow(img_np, cmap='gray', origin='lower')
    axes[0, 0].set_title(f'Original ({img_np.shape[0]}x{img_np.shape[1]})')
    axes[0, 0].axis('off')
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    # Reconstruction
    im1 = axes[0, 1].imshow(recon_np, cmap='gray', origin='lower')
    axes[0, 1].set_title('Reconstruction')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # Residual (fraction)
    im2 = axes[0, 2].imshow(frac_residual, cmap='RdBu_r', vmin=-10, vmax=10, origin='lower')
    axes[0, 2].set_title(f'Residual (mean: {np.abs(frac_residual).mean():.1f}% of peak)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, label='%')

    # Atoms
    atoms = model.get_dictionary().cpu().numpy()
    n_atoms = atoms.shape[0]
    atom_grid = np.zeros((config['atom_size'], config['atom_size'] * n_atoms))
    for i in range(n_atoms):
        atom = atoms[i, 0]
        atom_grid[:, i*config['atom_size']:(i+1)*config['atom_size']] = atom / max(np.abs(atom).max(), 1e-6)

    im3 = axes[1, 0].imshow(atom_grid, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    axes[1, 0].set_title(f'Learned Atoms ({n_atoms})')
    axes[1, 0].axis('off')

    # Total activation (sum across atoms)
    total_activation = np.abs(codes_np).sum(axis=0)
    im4 = axes[1, 1].imshow(total_activation, cmap='hot', origin='lower')
    axes[1, 1].set_title('Total Activation Map')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    # Most active atom per location
    most_active = np.argmax(np.abs(codes_np), axis=0)
    im5 = axes[1, 2].imshow(most_active, cmap='tab20', origin='lower', vmin=0, vmax=n_atoms-1)
    axes[1, 2].set_title('Dominant Atom per Location')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()

    out_path = output_dir / f'test_{Path(args.image).stem}.jpg'
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
