#!/usr/bin/env python
"""Train convolutional sparse coding on CRUMB dataset"""

import sys
sys.path.insert(0, '/home/pjaganna/Software/dictionary_learning/mirabest')

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from CRUMB import CRUMB_MB
from first_sparse.models import ConvSparseAutoencoder, ConvSparseLoss


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_mse = 0
    total_ssim = 0
    total_sparsity = 0

    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)

        optimizer.zero_grad()
        reconstruction, codes = model(images, return_codes=True)

        losses = criterion(reconstruction, images, codes)
        losses['total'].backward()
        optimizer.step()

        # Normalize atoms
        model.normalize_dictionary()

        total_loss += losses['total'].item()
        total_mse += losses['mse']
        total_ssim += losses['ssim']
        total_sparsity += losses['sparsity']

    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches,
        'ssim': total_ssim / n_batches,
        'sparsity': total_sparsity / n_batches
    }


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_mse = 0

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            reconstruction, codes = model(images, return_codes=True)
            losses = criterion(reconstruction, images, codes)
            total_loss += losses['total'].item()
            total_mse += losses['mse']

    n_batches = len(loader)
    return {
        'loss': total_loss / n_batches,
        'mse': total_mse / n_batches
    }


def save_atoms(model, output_dir, epoch):
    """Save visualization of learned atoms"""
    atoms = model.get_dictionary().cpu()
    n_atoms = atoms.size(0)

    n_cols = min(8, n_atoms)
    n_rows = int(np.ceil(n_atoms / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2*n_cols, 2*n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes]

    for i in range(n_atoms):
        atom = atoms[i, 0].numpy()
        vmax = np.abs(atom).max()
        axes[i].imshow(atom, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
        axes[i].set_title(f'Atom {i}', fontsize=8)
        axes[i].axis('off')

    for i in range(n_atoms, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_dir / f'atoms_epoch_{epoch:03d}.jpg', dpi=150)
    plt.close()


def save_reconstructions(model, loader, output_dir, epoch, device, n_samples=8):
    """Save sample reconstructions"""
    model.eval()

    images, _ = next(iter(loader))
    images = images[:n_samples].to(device)

    with torch.no_grad():
        reconstructions = model(images)

    images = images.cpu()
    reconstructions = reconstructions.cpu()

    fig, axes = plt.subplots(3, n_samples, figsize=(2*n_samples, 6))

    for i in range(n_samples):
        orig = images[i, 0].numpy()
        recon = reconstructions[i, 0].numpy()
        diff = orig - recon

        axes[0, i].imshow(orig, cmap='gray', origin='lower')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Original', fontsize=10)

        axes[1, i].imshow(recon, cmap='gray', origin='lower')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('Recon', fontsize=10)

        vmax = np.abs(diff).max()
        axes[2, i].imshow(diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax, origin='lower')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Residual', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / f'recon_epoch_{epoch:03d}.jpg', dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-atoms', type=int, default=16)
    parser.add_argument('--atom-size', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sparsity-weight', type=float, default=0.01)
    parser.add_argument('--ssim-weight', type=float, default=0.1)
    parser.add_argument('--output-dir', type=str, default='./results/crumb_conv_sparse')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained-atoms', type=str, default=None, help='Initialize atoms from checkpoint (train everything)')
    parser.add_argument('--freeze-atoms', action='store_true', help='Freeze atoms, only train encoder')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Load CRUMB dataset
    print("Loading CRUMB dataset...")

    # Transform to tensor and normalize to [0, 1]
    def transform(img):
        img = torch.from_numpy(np.array(img)).float() / 255.0
        return img.unsqueeze(0)  # Add channel dim

    train_dataset = CRUMB_MB(
        root='/home/pjaganna/Software/dictionary_learning/mirabest',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = CRUMB_MB(
        root='/home/pjaganna/Software/dictionary_learning/mirabest',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    print(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Create model
    config = {
        'n_atoms': args.n_atoms,
        'atom_size': args.atom_size,
        'encoder': {
            'hidden_channels': [32, 64],
            'activation': 'relu',
            'use_batch_norm': True
        },
        'sparsity': {
            'initial_threshold': 0.1,
            'learnable': True
        },
        'decoder': {
            'normalize_atoms': True,
            'init_method': 'normal'
        }
    }

    model = ConvSparseAutoencoder(config).to(device)
    print(f"Model: {args.n_atoms} atoms of size {args.atom_size}x{args.atom_size}")

    start_epoch = 1

    # Resume from checkpoint
    resume_optimizer = None
    if args.resume:
        print(f"Resuming from: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        resume_optimizer = checkpoint.get('optimizer_state_dict')
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # Initialize atoms from pretrained (but train everything)
    if args.pretrained_atoms and not args.resume:
        print(f"Loading pretrained atoms from: {args.pretrained_atoms}")
        pretrained = torch.load(args.pretrained_atoms, map_location=device)
        decoder_state = {
            k.replace('decoder.', ''): v
            for k, v in pretrained['model_state_dict'].items()
            if k.startswith('decoder.')
        }
        model.decoder.load_state_dict(decoder_state)
        print("Atoms initialized from pretrained model")

    # Freeze atoms if requested
    if args.freeze_atoms:
        for param in model.decoder.parameters():
            param.requires_grad = False
        print("Atoms frozen - only training encoder")

    # Loss and optimizer
    criterion = ConvSparseLoss(
        mse_weight=1.0,
        ssim_weight=args.ssim_weight,
        sparsity_weight=args.sparsity_weight
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    if resume_optimizer:
        optimizer.load_state_dict(resume_optimizer)
        print("Optimizer state restored")

    # Training loop
    best_loss = float('inf')
    train_losses = []
    test_losses = []

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        test_metrics = evaluate(model, test_loader, criterion, device)

        train_losses.append(train_metrics['loss'])
        test_losses.append(test_metrics['loss'])

        print(f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} (MSE: {train_metrics['mse']:.4f}, SSIM: {train_metrics['ssim']:.4f}) | Test Loss: {test_metrics['loss']:.4f}")

        # Save atoms and reconstructions periodically
        if epoch % 10 == 0 or epoch == 1:
            save_atoms(model, output_dir, epoch)
            save_reconstructions(model, test_loader, output_dir, epoch, device)

        # Save best model
        if test_metrics['loss'] < best_loss:
            best_loss = test_metrics['loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'loss': best_loss
            }, output_dir / 'best_model.pt')

    # Final save
    save_atoms(model, output_dir, args.epochs)
    save_reconstructions(model, test_loader, output_dir, args.epochs, device)

    # Plot training curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(output_dir / 'training_curve.jpg', dpi=150)
    plt.close()

    print(f"\nTraining complete. Results saved to {output_dir}")
    print(f"Best test loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
