#!/usr/bin/env python
"""Evaluate trained model and visualize dictionary atoms"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from first_sparse.data import FIRSTCutoutDataset
from first_sparse.models import SparseAutoencoder
from first_sparse.utils import (
    compute_reconstruction_metrics,
    compute_sparsity_metrics,
    compute_atom_usage,
    plot_dictionary_atoms,
    plot_reconstructions,
    plot_atom_usage_histogram
)


def evaluate_model(model, dataset, device, n_samples=None):
    """Evaluate model on dataset"""
    
    model.eval()
    
    all_reconstructions = []
    all_targets = []
    all_codes = []
    
    if n_samples is None:
        n_samples = len(dataset)
    else:
        n_samples = min(n_samples, len(dataset))
    
    with torch.no_grad():
        for i in tqdm(range(n_samples), desc="Evaluating"):
            batch = dataset[i]
            image = batch['image'].unsqueeze(0).to(device)
            
            reconstruction, codes = model(image, return_codes=True)
            
            all_reconstructions.append(reconstruction.cpu())
            all_targets.append(image.cpu())
            all_codes.append(codes.cpu())
    
    # Stack all results
    reconstructions = torch.cat(all_reconstructions, dim=0)
    targets = torch.cat(all_targets, dim=0)
    codes = torch.cat(all_codes, dim=0)
    
    # Compute metrics
    recon_metrics = compute_reconstruction_metrics(reconstructions, targets)
    sparsity_metrics = compute_sparsity_metrics(codes)
    atom_usage = compute_atom_usage(codes)
    
    return {
        'reconstruction': recon_metrics,
        'sparsity': sparsity_metrics,
        'atom_usage': atom_usage,
        'reconstructions': reconstructions,
        'targets': targets,
        'codes': codes
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sparse dictionary model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        required=True,
        help="Path to manifest JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=['train', 'val', 'test'],
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (None = all)"
    )
    parser.add_argument(
        "--n-atoms-plot",
        type=int,
        default=64,
        help="Number of dictionary atoms to plot"
    )
    parser.add_argument(
        "--n-reconstructions",
        type=int,
        default=8,
        help="Number of reconstruction examples to plot"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FIRST Sparse Dictionary Learning - Evaluation")
    print("=" * 70)
    
    # Setup device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract model config from checkpoint
    checkpoint_config = checkpoint.get('config', {})
    if 'model' in checkpoint_config:
        model_config = checkpoint_config['model']
    elif 'dictionary_size' in checkpoint_config:
        # Old format where config was the model config
        model_config = checkpoint_config
    else:
        # Try to load from default config files
        try:
            from first_sparse.utils import load_all_configs
            configs = load_all_configs(Path('./config'))
            model_config = configs['model']
            print("Warning: Using default model config from files")
        except:
            raise ValueError("Could not find model configuration in checkpoint")
    
    # Create model
    print("Creating model...")
    model = SparseAutoencoder(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded from epoch {checkpoint['epoch']}")
    print(f"Dictionary size: {model.dictionary_size}")
    
    # Load dataset
    print(f"\nLoading {args.split} dataset...")
    dataset = FIRSTCutoutDataset(
        manifest_path=args.manifest,
        split=args.split,
        augment=False
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, dataset, device, args.n_samples)
    
    # Print metrics
    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    
    print("\nReconstruction Metrics:")
    for key, value in results['reconstruction'].items():
        print(f"  {key.upper()}: {value:.6f}")
    
    print("\nSparsity Metrics:")
    for key, value in results['sparsity'].items():
        print(f"  {key}: {value:.4f}")
    
    print("\nAtom Usage:")
    usage = results['atom_usage']['usage_fraction']
    print(f"  Mean usage: {usage.mean()*100:.2f}%")
    print(f"  Min usage: {usage.min()*100:.2f}%")
    print(f"  Max usage: {usage.max()*100:.2f}%")
    print(f"  Unused atoms: {(usage == 0).sum()}/{len(usage)}")
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    metrics_to_save = {
        'reconstruction': results['reconstruction'],
        'sparsity': results['sparsity'],
        'atom_usage': {
            'mean': float(usage.mean()),
            'std': float(usage.std()),
            'min': float(usage.min()),
            'max': float(usage.max()),
            'unused': int((usage == 0).sum())
        }
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    
    print(f"\nMetrics saved to: {metrics_path}")
    
    # Visualize dictionary atoms
    print("\nGenerating visualizations...")
    
    atoms = model.get_dictionary()
    
    fig = plot_dictionary_atoms(
        atoms,
        n_atoms=min(args.n_atoms_plot, model.dictionary_size),
        save_path=output_dir / "dictionary_atoms.png"
    )
    
    # Plot reconstructions
    n_recon = min(args.n_reconstructions, len(results['targets']))
    fig = plot_reconstructions(
        results['targets'][:n_recon],
        results['reconstructions'][:n_recon],
        n_samples=n_recon,
        save_path=output_dir / "reconstructions.png"
    )
    
    # Plot atom usage histogram
    fig = plot_atom_usage_histogram(
        results['atom_usage']['usage_fraction'],
        save_path=output_dir / "atom_usage.png"
    )
    
    print(f"\nAll results saved to: {output_dir.absolute()}")
    print("\nFiles created:")
    print(f"  - metrics.json")
    print(f"  - dictionary_atoms.png")
    print(f"  - reconstructions.png")
    print(f"  - atom_usage.png")


if __name__ == "__main__":
    main()
