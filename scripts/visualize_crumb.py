#!/usr/bin/env python
"""Visualize CRUMB dataset and run PCA"""

import sys
sys.path.insert(0, '/home/pjaganna/Software/dictionary_learning/mirabest')

import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from pathlib import Path

from CRUMB import CRUMB_MB


def main():
    output_dir = Path('./results/crumb_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load CRUMB-MB (MiraBest subset)
    print("Loading CRUMB dataset...")
    dataset = CRUMB_MB(
        root='/home/pjaganna/Software/dictionary_learning/mirabest',
        train=True,
        download=True
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # Collect all images
    images = []
    labels = []
    for i in range(len(dataset)):
        img, label = dataset[i]
        img_np = np.array(img)
        images.append(img_np)
        labels.append(label)

    images = np.array(images)
    labels = np.array(labels)
    print(f"Image shape: {images[0].shape}")
    print(f"Label distribution: {np.bincount(labels)}")

    # Plot grid of samples
    n_samples = min(100, len(images))
    n_cols = 10
    n_rows = n_samples // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten()

    for i in range(n_samples):
        axes[i].imshow(images[i], cmap='gray', origin='lower')
        axes[i].set_title(f"L:{labels[i]}", fontsize=7)
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / "crumb_samples.jpg", dpi=150)
    print(f"Saved: {output_dir / 'crumb_samples.jpg'}")

    # Pixel stats
    print(f"\nPixel value stats:")
    print(f"  min: {images.min()}")
    print(f"  max: {images.max()}")
    print(f"  mean: {images.mean():.2f}")
    print(f"  std: {images.std():.2f}")

    # PCA
    print(f"\nRunning PCA on {len(images)} images...")
    X = images.reshape(len(images), -1).astype(np.float32)
    X = X / 255.0  # normalize to [0,1]

    n_components = min(32, len(images))
    pca = PCA(n_components=n_components)
    pca.fit(X)

    print(f"Explained variance (top 10): {pca.explained_variance_ratio_[:10]}")
    print(f"Cumulative variance (10 components): {pca.explained_variance_ratio_[:10].sum():.2%}")
    print(f"Cumulative variance (all {n_components}): {pca.explained_variance_ratio_.sum():.2%}")

    # Plot PCA components
    img_size = images[0].shape[0]
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
    plt.savefig(output_dir / "crumb_pca.jpg", dpi=150)
    print(f"Saved: {output_dir / 'crumb_pca.jpg'}")


if __name__ == "__main__":
    main()
