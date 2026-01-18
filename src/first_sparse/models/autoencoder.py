"""Sparse autoencoder combining encoder, sparsity, and decoder"""

import torch
import torch.nn as nn

from .encoder import create_encoder, LearnedThreshold
from .decoder import DictionaryDecoder


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder: image -> sparse codes -> reconstructed image"""
    
    def __init__(self, config: dict):
        """
        Args:
            config: Model configuration dictionary with keys:
                - dictionary_size: Number of atoms K
                - encoder: Encoder configuration
                - sparsity: Sparsity mechanism configuration  
                - decoder: Decoder configuration
        """
        super().__init__()
        
        self.config = config
        self.dictionary_size = config['dictionary_size']
        
        # Build encoder
        encoder_config = config['encoder'].copy()
        encoder_config['code_dim'] = self.dictionary_size
        
        # Get input size from encoder config or default
        self.input_size = encoder_config.get('input_size', 64)
        encoder_config['input_size'] = self.input_size
        
        self.encoder = create_encoder(encoder_config)
        
        # Build sparsity layer
        sparsity_config = config['sparsity']
        sparsity_method = sparsity_config.get('method', 'learned_threshold')
        
        if sparsity_method == 'learned_threshold':
            threshold_config = sparsity_config.get('learned_threshold', {})
            self.sparsity = LearnedThreshold(
                code_dim=self.dictionary_size,
                initial_threshold=threshold_config.get('initial_threshold', 0.1),
                learnable=threshold_config.get('learnable', True)
            )
        else:
            # No explicit sparsity layer (handled in loss)
            self.sparsity = None
        
        # Build decoder
        decoder_config = config['decoder']
        self.decoder = DictionaryDecoder(
            code_dim=self.dictionary_size,
            image_size=self.input_size,
            normalize_atoms=decoder_config.get('normalize_atoms', True),
            init_method=decoder_config.get('init_method', 'orthogonal')
        )
    
    def forward(self, x, return_codes=False):
        """Forward pass
        
        Args:
            x: Input images (batch, 1, H, W)
            return_codes: If True, return both reconstruction and codes
            
        Returns:
            If return_codes=False: reconstructed images (batch, 1, H, W)
            If return_codes=True: (reconstructed images, sparse codes)
        """
        # Encode
        codes = self.encoder(x)
        
        # Apply sparsity
        if self.sparsity is not None:
            sparse_codes = self.sparsity(codes)
        else:
            sparse_codes = codes
        
        # Decode
        reconstruction = self.decoder(sparse_codes)
        
        if return_codes:
            return reconstruction, sparse_codes
        else:
            return reconstruction
    
    def encode(self, x):
        """Encode images to sparse codes
        
        Args:
            x: Input images (batch, 1, H, W)
            
        Returns:
            codes: (batch, dictionary_size) sparse activation vectors
        """
        codes = self.encoder(x)
        
        if self.sparsity is not None:
            codes = self.sparsity(codes)
        
        return codes
    
    def decode(self, codes):
        """Decode sparse codes to images
        
        Args:
            codes: (batch, dictionary_size) sparse activation vectors
            
        Returns:
            images: (batch, 1, H, W) reconstructed images
        """
        return self.decoder(codes)
    
    def get_dictionary(self):
        """Get dictionary atoms as images
        
        Returns:
            atoms: (dictionary_size, 1, H, W) tensor of atoms
        """
        return self.decoder.get_atoms()
    
    def normalize_dictionary(self):
        """Normalize dictionary atoms (call after optimizer step)"""
        self.decoder.normalize_atoms_during_training()
    
    def compute_sparsity_stats(self, codes):
        """Compute sparsity statistics
        
        Args:
            codes: (batch, dictionary_size) sparse codes
            
        Returns:
            dict with sparsity metrics
        """
        with torch.no_grad():
            # L0 norm (number of non-zero elements)
            l0 = (codes.abs() > 1e-6).float().sum(dim=1).mean().item()
            
            # L1 norm (sum of absolute values)
            l1 = codes.abs().sum(dim=1).mean().item()
            
            # Percentage of active codes
            active_percent = (codes.abs() > 1e-6).float().mean().item() * 100
            
            # Per-atom usage (how often each atom is active)
            atom_usage = (codes.abs() > 1e-6).float().mean(dim=0)
            
        return {
            'l0_norm': l0,
            'l1_norm': l1,
            'active_percent': active_percent,
            'min_atom_usage': atom_usage.min().item(),
            'max_atom_usage': atom_usage.max().item(),
            'mean_atom_usage': atom_usage.mean().item()
        }
