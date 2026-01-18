#!/usr/bin/env python
"""Train sparse autoencoder on FIRST sources"""

import argparse
from pathlib import Path
import torch

from first_sparse.data import create_dataloaders
from first_sparse.models import SparseAutoencoder
from first_sparse.training import Trainer
from first_sparse.utils import load_all_configs


def main():
    parser = argparse.ArgumentParser(
        description="Train sparse dictionary learning model"
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="config/data_config.yaml",
        help="Path to data config"
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model config"
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="config/training_config.yaml",
        help="Path to training config"
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to manifest JSON (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu, overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FIRST Sparse Dictionary Learning - Training")
    print("=" * 70)
    
    # Load configs
    print("\nLoading configurations...")
    configs = load_all_configs(Path(args.data_config).parent)
    
    data_config = configs['data']
    model_config = configs['model']
    training_config = configs['training']
    
    # Override with command line args
    if args.manifest:
        data_config['manifest_path'] = args.manifest
    
    if args.device:
        training_config['device'] = args.device
    
    # Set device
    device = training_config.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"Device: {device}")
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    manifest_path = data_config['manifest_path']
    
    if not Path(manifest_path).exists():
        print(f"Error: Manifest not found: {manifest_path}")
        print("Run build_manifest.py first!")
        return
    
    loaders = create_dataloaders(
        manifest_path=manifest_path,
        batch_size=training_config['batch_size'],
        num_workers=training_config.get('num_workers', 4),
        pin_memory=training_config.get('pin_memory', True),
        normalize=data_config.get('normalization', 'integrated_flux')
    )
    
    print(f"Train batches: {len(loaders['train'])}")
    print(f"Val batches: {len(loaders['val'])}")
    print(f"Test batches: {len(loaders['test'])}")
    
    # Create model
    print("\nCreating model...")
    
    # Add input size to model config
    model_config['encoder']['input_size'] = data_config['cutout_size']
    
    model = SparseAutoencoder(model_config)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    print(f"Dictionary size: {model.dictionary_size}")
    print(f"Encoder type: {model_config['encoder']['type']}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nLoading checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Create trainer
    print("\nInitializing trainer...")
    full_config = {
        'data': data_config,
        'model': model_config,
        'training': training_config
    }
    trainer = Trainer(
        model=model,
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        config=training_config,
        full_config=full_config,
        device=device
    )
    
    # Start training
    print("\n" + "=" * 70)
    print("Starting training")
    print("=" * 70 + "\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint(
            epoch=trainer.current_epoch,
            metrics={'interrupted': True},
            is_best=False
        )
    
    print("\nTraining complete!")
    print(f"Checkpoints saved to: {trainer.checkpoint_dir.absolute()}")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
