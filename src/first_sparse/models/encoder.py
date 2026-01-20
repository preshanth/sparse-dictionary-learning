"""Encoder architectures for sparse dictionary learning"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleEncoder(nn.Module):
    """Simple convolutional encoder
    
    Lightweight encoder for proof-of-concept.
    ~100k parameters for 64x64 images, K=256 codes
    """
    
    def __init__(
        self,
        input_size: int = 64,
        code_dim: int = 256,
        channels: list = None,
        kernel_sizes: list = None,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()
        
        if channels is None:
            channels = [16, 32, 64]
        if kernel_sizes is None:
            kernel_sizes = [5, 3, 3]
        
        self.input_size = input_size
        self.code_dim = code_dim
        
        # Build convolutional layers
        layers = []
        in_channels = 1
        
        for out_channels, kernel_size in zip(channels, kernel_sizes):
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2
            ))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            
            if dropout > 0:
                layers.append(nn.Dropout2d(dropout))
            
            in_channels = out_channels
        
        self.features = nn.Sequential(*layers)
        
        # Calculate feature map size after convolutions
        # Each stride=2 conv halves the spatial dimensions
        feature_size = input_size // (2 ** len(channels))
        feature_dim = channels[-1] * feature_size * feature_size
        
        # Fully connected layer to code dimension
        self.fc = nn.Linear(feature_dim, code_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input images (batch, 1, H, W)
            
        Returns:
            codes: (batch, code_dim)
        """
        features = self.features(x)
        features = features.view(features.size(0), -1)
        codes = self.fc(features)
        return codes


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    
    def __init__(self, channels: int, activation: str = 'relu'):
        super().__init__()
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out = out + residual
        out = self.activation(out)
        
        return out


class ProductionEncoder(nn.Module):
    """Production-quality residual encoder
    
    More robust encoder with skip connections.
    ~500k parameters for 64x64 images, K=256 codes
    """
    
    def __init__(
        self,
        input_size: int = 64,
        code_dim: int = 256,
        initial_channels: int = 32,
        channel_progression: list = None,
        blocks_per_stage: int = 2,
        activation: str = 'relu',
        use_batch_norm: bool = True,
        dropout: float = 0.2
    ):
        super().__init__()
        
        if channel_progression is None:
            channel_progression = [32, 64, 128]
        
        self.input_size = input_size
        self.code_dim = code_dim
        
        # Initial convolution
        self.conv1 = nn.Conv2d(1, initial_channels, kernel_size=7, 
                               stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(initial_channels)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Build residual stages
        self.stages = nn.ModuleList()
        in_channels = initial_channels
        
        for stage_channels in channel_progression:
            stage = []
            
            # Downsample
            stage.append(nn.Conv2d(in_channels, stage_channels, 3, 
                                  stride=2, padding=1))
            stage.append(nn.BatchNorm2d(stage_channels))
            stage.append(self.activation)
            
            # Residual blocks
            for _ in range(blocks_per_stage):
                stage.append(ResidualBlock(stage_channels, activation))
            
            self.stages.append(nn.Sequential(*stage))
            in_channels = stage_channels
        
        # Adaptive pooling to fixed size
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Fully connected layers
        fc_input_dim = channel_progression[-1] * 4 * 4
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.fc = nn.Linear(fc_input_dim, code_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input images (batch, 1, H, W)
            
        Returns:
            codes: (batch, code_dim)
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Residual stages
        for stage in self.stages:
            x = stage(x)
        
        # Pool and flatten
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # FC to codes
        if self.dropout is not None:
            x = self.dropout(x)
        codes = self.fc(x)
        
        return codes


class LearnedThreshold(nn.Module):
    """Learnable soft thresholding for sparsity"""
    
    def __init__(self, code_dim: int, initial_threshold: float = 0.1, 
                 learnable: bool = True):
        super().__init__()
        
        if learnable:
            self.threshold = nn.Parameter(
                torch.ones(code_dim) * initial_threshold
            )
        else:
            self.register_buffer('threshold', 
                               torch.ones(code_dim) * initial_threshold)
    
    def forward(self, x):
        """Soft thresholding: max(0, |x| - threshold) * sign(x)"""
        return torch.sign(x) * F.relu(torch.abs(x) - torch.abs(self.threshold))


class ConvSparseEncoder(nn.Module):
    """Convolutional encoder that outputs spatial sparse codes

    Produces (batch, n_atoms, H, W) sparse activation maps.
    """

    def __init__(
        self,
        n_atoms: int = 16,
        hidden_channels: list = None,
        activation: str = 'relu',
        use_batch_norm: bool = True
    ):
        """
        Args:
            n_atoms: Number of dictionary atoms (output channels)
            hidden_channels: List of hidden channel sizes
            activation: Activation function
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64]

        self.n_atoms = n_atoms

        # Build encoder: input -> hidden features -> atom activations
        layers = []
        in_channels = 1

        for out_channels in hidden_channels:
            layers.append(nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            ))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if activation == 'relu':
                layers.append(nn.ReLU(inplace=True))
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.2, inplace=True))
            in_channels = out_channels

        # Final conv to n_atoms channels (no activation - sparsity applied separately)
        layers.append(nn.Conv2d(in_channels, n_atoms, kernel_size=3, stride=1, padding=1))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Input images (batch, 1, H, W)

        Returns:
            codes: (batch, n_atoms, H, W) spatial activation maps
        """
        return self.encoder(x)


class SpatialSoftThreshold(nn.Module):
    """Learnable soft thresholding for spatial sparse codes"""

    def __init__(self, n_atoms: int, initial_threshold: float = 0.1,
                 learnable: bool = True):
        super().__init__()

        if learnable:
            # One threshold per atom
            self.threshold = nn.Parameter(
                torch.ones(1, n_atoms, 1, 1) * initial_threshold
            )
        else:
            self.register_buffer('threshold',
                               torch.ones(1, n_atoms, 1, 1) * initial_threshold)

    def forward(self, x):
        """Soft thresholding: max(0, |x| - threshold) * sign(x)

        Args:
            x: (batch, n_atoms, H, W) spatial codes

        Returns:
            sparse_x: (batch, n_atoms, H, W) thresholded codes
        """
        return torch.sign(x) * F.relu(torch.abs(x) - torch.abs(self.threshold))


def create_encoder(config: dict) -> nn.Module:
    """Factory function to create encoder from config

    Args:
        config: Dictionary with encoder configuration

    Returns:
        Encoder instance
    """
    encoder_type = config.get('type', 'simple')

    if encoder_type == 'simple':
        params = config.get('simple', {}).copy()
        params['code_dim'] = config.get('code_dim', 256)
        params['input_size'] = config.get('input_size', 64)
        return SimpleEncoder(**params)
    elif encoder_type == 'production':
        params = config.get('production', {}).copy()
        params['code_dim'] = config.get('code_dim', 256)
        params['input_size'] = config.get('input_size', 64)
        return ProductionEncoder(**params)
    elif encoder_type == 'conv_sparse':
        params = config.get('conv_sparse', {}).copy()
        params['n_atoms'] = config.get('n_atoms', 16)
        return ConvSparseEncoder(**params)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
