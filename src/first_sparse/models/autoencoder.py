"""Sparse autoencoder combining encoder, sparsity, and decoder"""

import torch
import torch.nn as nn

from .encoder import create_encoder, LearnedThreshold, ConvSparseEncoder, SpatialSoftThreshold
from .decoder import DictionaryDecoder, ConvDictionaryDecoder


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


class ConvSparseAutoencoder(nn.Module):
    """Convolutional sparse autoencoder with small dictionary atoms

    Uses spatial sparse codes and convolutional dictionary.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: Model configuration dictionary with keys:
                - n_atoms: Number of dictionary atoms
                - atom_size: Size of each atom (e.g., 11)
                - encoder: Encoder configuration
                - sparsity: Sparsity configuration
                - decoder: Decoder configuration
        """
        super().__init__()

        self.config = config
        self.n_atoms = config.get('n_atoms', 16)
        self.atom_size = config.get('atom_size', 11)

        # Build encoder
        encoder_config = config.get('encoder', {}).copy()
        encoder_config['n_atoms'] = self.n_atoms
        self.encoder = ConvSparseEncoder(**encoder_config)

        # Build sparsity layer
        sparsity_config = config.get('sparsity', {})
        initial_threshold = sparsity_config.get('initial_threshold', 0.1)
        learnable = sparsity_config.get('learnable', True)
        self.sparsity = SpatialSoftThreshold(
            n_atoms=self.n_atoms,
            initial_threshold=initial_threshold,
            learnable=learnable
        )

        # Build decoder
        decoder_config = config.get('decoder', {})
        self.decoder = ConvDictionaryDecoder(
            n_atoms=self.n_atoms,
            atom_size=self.atom_size,
            normalize_atoms=decoder_config.get('normalize_atoms', True),
            init_method=decoder_config.get('init_method', 'normal')
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
        # Encode to spatial codes
        codes = self.encoder(x)

        # Apply sparsity
        sparse_codes = self.sparsity(codes)

        # Decode
        reconstruction = self.decoder(sparse_codes)

        if return_codes:
            return reconstruction, sparse_codes
        else:
            return reconstruction

    def encode(self, x):
        """Encode images to spatial sparse codes"""
        codes = self.encoder(x)
        return self.sparsity(codes)

    def decode(self, codes):
        """Decode spatial sparse codes to images"""
        return self.decoder(codes)

    def get_dictionary(self):
        """Get dictionary atoms

        Returns:
            atoms: (n_atoms, 1, atom_size, atom_size) tensor of atoms
        """
        return self.decoder.get_atoms()

    def normalize_dictionary(self):
        """Normalize dictionary atoms (call after optimizer step)"""
        self.decoder.normalize_atoms_during_training()

    def compute_sparsity_stats(self, codes):
        """Compute sparsity statistics for spatial codes

        Args:
            codes: (batch, n_atoms, H, W) spatial sparse codes

        Returns:
            dict with sparsity metrics
        """
        with torch.no_grad():
            # Flatten spatial dimensions
            batch_size = codes.size(0)
            codes_flat = codes.view(batch_size, self.n_atoms, -1)

            # L0 norm per image (number of non-zero activations)
            l0 = (codes_flat.abs() > 1e-6).float().sum(dim=(1, 2)).mean().item()

            # L1 norm
            l1 = codes_flat.abs().sum(dim=(1, 2)).mean().item()

            # Percentage of active codes
            active_percent = (codes_flat.abs() > 1e-6).float().mean().item() * 100

            # Per-atom usage (how often each atom is active anywhere)
            atom_usage = (codes.abs() > 1e-6).float().mean(dim=(0, 2, 3))

        return {
            'l0_norm': l0,
            'l1_norm': l1,
            'active_percent': active_percent,
            'min_atom_usage': atom_usage.min().item(),
            'max_atom_usage': atom_usage.max().item(),
            'mean_atom_usage': atom_usage.mean().item()
        }
