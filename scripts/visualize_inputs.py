#!/usr/bin/env python
"""Visualize input sources from manifest and run PCA"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from first_sparse.data import FIRSTCutoutDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./input_samples")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--n-cols", type=int, default=10)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = FIRSTCutoutDataset(
        manifest_path=args.manifest,
        split='train',
        augment=False
    )

    n_samples = min(args.n_samples, len(dataset))
    print(f"Dataset: {len(dataset)} sources")
    print(f"Plotting {n_samples} samples")

    # Load manifest for metadata
    with open(args.manifest) as f:
        manifest = json.load(f)

    # Collect samples
    images = []
    majors = []
    for i in range(n_samples):
        batch = dataset[i]
        img = batch['image'][0].numpy()
        images.append(img)
        majors.append(batch['metadata'].get('major', 0))

    # Plot grid
    n_cols = args.n_cols
    n_rows = int(np.ceil(n_samples / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten()

    for i, (img, major) in enumerate(zip(images, majors)):
        axes[i].imshow(img, cmap='gray', origin='lower')
        axes[i].set_title(f"{major:.1f}\"", fontsize=7)
        axes[i].axis('off')

    for i in range(n_samples, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "input_grid.jpg", dpi=150)
    print(f"Saved: {output_dir / 'input_grid.jpg'}")

    # Stats
    majors = np.array(majors)
    print(f"\nSize stats (arcsec):")
    print(f"  min: {majors.min():.1f}")
    print(f"  max: {majors.max():.1f}")
    print(f"  mean: {majors.mean():.1f}")
    print(f"  median: {np.median(majors):.1f}")

    # Value stats
    all_vals = np.concatenate([img.flatten() for img in images])
    print(f"\nPixel value stats (after normalization):")
    print(f"  min: {all_vals.min():.4f}")
    print(f"  max: {all_vals.max():.4f}")
    print(f"  mean: {all_vals.mean():.4f}")
    print(f"  std: {all_vals.std():.4f}")

    # PCA on all loaded images
    print(f"\nRunning PCA on {len(images)} images...")
    img_size = images[0].shape[0]
    X = np.array([img.flatten() for img in images])

    n_components = min(32, len(images), img_size * img_size)
    pca = PCA(n_components=n_components)
    pca.fit(X)

    print(f"Explained variance (top 10): {pca.explained_variance_ratio_[:10]}")
    print(f"Cumulative variance (10 components): {pca.explained_variance_ratio_[:10].sum():.2%}")
    print(f"Cumulative variance (all {n_components}): {pca.explained_variance_ratio_.sum():.2%}")

    # Plot PCA components
    n_pca_cols = 8
    n_pca_rows = int(np.ceil(n_components / n_pca_cols))
    fig, axes = plt.subplots(n_pca_rows, n_pca_cols, figsize=(2*n_pca_cols, 2*n_pca_rows))
    axes = axes.flatten()

    for i in range(n_components):
        component = pca.components_[i].reshape(img_size, img_size)
        vmax = np.abs(component).max()
        axes[i].imshow(component, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        axes[i].set_title(f"PC{i+1}: {pca.explained_variance_ratio_[i]:.1%}", fontsize=7)
        axes[i].axis('off')

    for i in range(n_components, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "pca_components.jpg", dpi=150)
    print(f"Saved: {output_dir / 'pca_components.jpg'}")


if __name__ == "__main__":
    main()
