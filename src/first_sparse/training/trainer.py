"""Training loop for sparse autoencoder"""

import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.losses import CombinedLoss


class Trainer:
    """Handles training loop, validation, checkpointing"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        full_config: dict = None,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.full_config = full_config or config
        self.device = device
        
        # Setup optimizer
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Setup loss function
        loss_config = config['loss']
        self.criterion = CombinedLoss(
            reconstruction_weight=loss_config['reconstruction_weight'],
            sparsity_weight=loss_config['sparsity_weight'],
            diversity_weight=loss_config['diversity_weight']
        ).to(device)
        
        # Mixed precision scaler
        self.use_amp = config.get('mixed_precision', True) and device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.epochs_since_improvement = 0
        
        # Setup directories
        self.checkpoint_dir = Path(config['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics history
        self.train_losses = []
        self.val_losses = []
    
    def _build_optimizer(self):
        """Build optimizer from config"""
        opt_config = self.config['optimizer']
        opt_type = opt_config.get('type', 'adam').lower()
        
        if opt_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'adamw':
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=opt_config['learning_rate'],
                weight_decay=opt_config.get('weight_decay', 0),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_type}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler from config"""
        sched_config = self.config.get('lr_scheduler', {})
        sched_type = sched_config.get('type', 'none').lower()
        
        if sched_type == 'reduce_on_plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                patience=sched_config.get('patience', 10),
                factor=sched_config.get('factor', 0.5),
                min_lr=sched_config.get('min_lr', 1e-6),
                verbose=True
            )
        elif sched_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=sched_config.get('min_lr', 1e-6)
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0
        }
        epoch_sparsity = {'l0': 0.0, 'l1': 0.0, 'active_percent': 0.0}
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}")
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            
            # Forward pass with mixed precision
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                reconstruction, codes = self.model(images, return_codes=True)
                
                # Compute loss
                loss_dict = self.criterion(
                    reconstruction, images, codes,
                    self.model.decoder.dictionary.weight
                )
                loss = loss_dict['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config.get('gradient_clip', 0) > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.config.get('gradient_clip', 0) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Normalize dictionary atoms
            self.model.normalize_dictionary()
            
            # Update metrics
            for key in epoch_losses:
                epoch_losses[key] += loss_dict[key] if key in loss_dict else loss_dict['total'].item()
            
            # Compute sparsity stats
            with torch.no_grad():
                stats = self.model.compute_sparsity_stats(codes)
                for key in epoch_sparsity:
                    epoch_sparsity[key] += stats.get(f"{key}_norm" if key in ['l0', 'l1'] else key, 0)
            
            # Update progress bar
            if batch_idx % self.config.get('log_every_n_steps', 50) == 0:
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'recon': f"{loss_dict['reconstruction']:.4f}",
                    'l0': f"{stats['l0_norm']:.1f}"
                })
        
        # Average metrics
        n_batches = len(self.train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches
        for key in epoch_sparsity:
            epoch_sparsity[key] /= n_batches
        
        return {**epoch_losses, **epoch_sparsity}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set"""
        self.model.eval()
        
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0
        }
        val_sparsity = {'l0': 0.0, 'l1': 0.0, 'active_percent': 0.0}
        
        for batch in tqdm(self.val_loader, desc="Validating", leave=False):
            images = batch['image'].to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                reconstruction, codes = self.model(images, return_codes=True)
                
                loss_dict = self.criterion(
                    reconstruction, images, codes,
                    self.model.decoder.dictionary.weight
                )
            
            for key in val_losses:
                val_losses[key] += loss_dict[key] if key in loss_dict else loss_dict['total'].item()
            
            stats = self.model.compute_sparsity_stats(codes)
            for key in val_sparsity:
                val_sparsity[key] += stats.get(f"{key}_norm" if key in ['l0', 'l1'] else key, 0)
        
        # Average
        n_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= n_batches
        for key in val_sparsity:
            val_sparsity[key] /= n_batches
        
        return {**val_losses, **val_sparsity}
    
    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.full_config
        }
        
        # Save regular checkpoint
        if self.config.get('save_every_n_epochs', 5) > 0:
            if epoch % self.config['save_every_n_epochs'] == 0:
                path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
                torch.save(checkpoint, path)
                print(f"Saved checkpoint: {path}")
        
        # Save best checkpoint
        if is_best or self.config.get('save_best_only', True):
            if is_best:
                path = self.checkpoint_dir / "best.pt"
                torch.save(checkpoint, path)
                print(f"Saved best checkpoint: {path}")
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        keep_n = self.config.get('keep_last_n', 3)
        if keep_n <= 0:
            return
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.split('_')[-1])
        )
        
        if len(checkpoints) > keep_n:
            for ckpt in checkpoints[:-keep_n]:
                ckpt.unlink()
    
    def train(self):
        """Main training loop"""
        print(f"Starting training for {self.config['epochs']} epochs")
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        
        for epoch in range(self.config['epochs']):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics)
            
            # Validate
            if epoch % self.config.get('validate_every_n_epochs', 1) == 0:
                val_metrics = self.validate()
                self.val_losses.append(val_metrics)
                
                # Check for improvement
                val_loss = val_metrics['total']
                is_best = val_loss < self.best_val_loss
                
                if is_best:
                    self.best_val_loss = val_loss
                    self.epochs_since_improvement = 0
                else:
                    self.epochs_since_improvement += 1
                
                # Print metrics
                print(f"\nEpoch {epoch+1}/{self.config['epochs']}")
                print(f"Train Loss: {train_metrics['total']:.4f} | "
                      f"Val Loss: {val_loss:.4f} | "
                      f"Sparsity (L0): {val_metrics['l0']:.1f}")
                
                # Learning rate scheduling
                if self.scheduler is not None:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
                
                # Save checkpoint
                self.save_checkpoint(epoch, val_metrics, is_best)
                
                # Early stopping
                patience = self.config.get('early_stopping_patience', 20)
                if self.epochs_since_improvement >= patience:
                    print(f"\nEarly stopping after {patience} epochs without improvement")
                    break
        
        print("\nTraining complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
