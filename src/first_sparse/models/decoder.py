"""Dictionary-based decoder for sparse codes"""

import torch
import torch.nn as nn


class DictionaryDecoder(nn.Module):
    """Dictionary decoder: sparse codes -> reconstructed image
    
    The dictionary D is a learnable weight matrix where each column
    is an atom (basis function). Reconstruction is simply x = D @ α
    """
    
    def __init__(
        self,
        code_dim: int,
        image_size: int,
        normalize_atoms: bool = True,
        init_method: str = 'orthogonal'
    ):
        """
        Args:
            code_dim: Dictionary size (number of atoms K)
            image_size: Output image size (assumed square)
            normalize_atoms: Project dictionary atoms to unit norm
            init_method: 'orthogonal', 'normal', or 'uniform'
        """
        super().__init__()
        
        self.code_dim = code_dim
        self.image_size = image_size
        self.normalize_atoms = normalize_atoms
        
        # Dictionary is a linear layer: codes -> flattened image
        # Weight matrix shape: (image_size^2, code_dim)
        # Each column is a dictionary atom
        self.dictionary = nn.Linear(code_dim, image_size * image_size, 
                                     bias=False)
        
        # Initialize dictionary
        self._initialize_dictionary(init_method)
        
        # Normalize atoms to unit norm
        if normalize_atoms:
            self._normalize_atoms()
    
    def _initialize_dictionary(self, method: str):
        """Initialize dictionary weights"""
        if method == 'orthogonal':
            nn.init.orthogonal_(self.dictionary.weight)
        elif method == 'normal':
            nn.init.normal_(self.dictionary.weight, mean=0, std=0.01)
        elif method == 'uniform':
            nn.init.uniform_(self.dictionary.weight, -0.1, 0.1)
        else:
            raise ValueError(f"Unknown init method: {method}")
    
    def _normalize_atoms(self):
        """Project dictionary atoms to unit L2 norm"""
        with torch.no_grad():
            # Normalize each row (atom) to unit norm
            norms = self.dictionary.weight.norm(dim=1, keepdim=True)
            self.dictionary.weight.div_(norms + 1e-8)
    
    def forward(self, codes):
        """Reconstruct images from sparse codes
        
        Args:
            codes: (batch, code_dim) sparse activation vectors
            
        Returns:
            images: (batch, 1, H, W) reconstructed images
        """
        # Linear combination of dictionary atoms
        flat = self.dictionary(codes)  # (batch, H*W)
        
        # Reshape to image
        batch_size = codes.size(0)
        images = flat.view(batch_size, 1, self.image_size, self.image_size)
        
        return images
    
    def get_atoms(self) -> torch.Tensor:
        """Get dictionary atoms as images
        
        Returns:
            atoms: (code_dim, 1, H, W) dictionary atoms
        """
        # Dictionary weight: (H*W, code_dim)
        # Transpose and reshape to get atoms as images
        atoms = self.dictionary.weight.t()  # (code_dim, H*W)
        atoms = atoms.view(self.code_dim, 1, self.image_size, self.image_size)
        return atoms
    
    def normalize_atoms_during_training(self):
        """Call this after each optimizer step to maintain unit norm"""
        if self.normalize_atoms:
            self._normalize_atoms()


class ConvDictionaryDecoder(nn.Module):
    """Convolutional dictionary decoder: spatial sparse codes -> reconstructed image

    Uses small convolutional atoms instead of full image-sized atoms.
    Reconstruction is sum of convolutions: x = sum_k (atom_k * codes_k)
    """

    def __init__(
        self,
        n_atoms: int,
        atom_size: int = 11,
        normalize_atoms: bool = True,
        init_method: str = 'normal'
    ):
        """
        Args:
            n_atoms: Number of dictionary atoms K
            atom_size: Size of each atom (assumed square, should be odd)
            normalize_atoms: Project dictionary atoms to unit norm
            init_method: 'normal' or 'uniform'
        """
        super().__init__()

        self.n_atoms = n_atoms
        self.atom_size = atom_size
        self.normalize_atoms = normalize_atoms
        self.padding = atom_size // 2

        # Dictionary atoms as conv weights: (n_atoms, 1, atom_size, atom_size)
        # Using transposed conv: sparse_codes (B, n_atoms, H, W) -> image (B, 1, H, W)
        self.atoms = nn.ConvTranspose2d(
            in_channels=n_atoms,
            out_channels=1,
            kernel_size=atom_size,
            padding=self.padding,
            bias=False
        )

        self._initialize_atoms(init_method)

        if normalize_atoms:
            self._normalize_atoms()

    def _initialize_atoms(self, method: str):
        """Initialize atom weights"""
        if method == 'normal':
            nn.init.normal_(self.atoms.weight, mean=0, std=0.1)
        elif method == 'uniform':
            nn.init.uniform_(self.atoms.weight, -0.1, 0.1)
        else:
            raise ValueError(f"Unknown init method: {method}")

    def _normalize_atoms(self):
        """Project each atom to unit L2 norm"""
        with torch.no_grad():
            # atoms.weight shape: (n_atoms, 1, atom_size, atom_size)
            weight = self.atoms.weight
            norms = weight.view(self.n_atoms, -1).norm(dim=1, keepdim=True)
            norms = norms.view(self.n_atoms, 1, 1, 1)
            self.atoms.weight.div_(norms + 1e-8)

    def forward(self, codes):
        """Reconstruct image from spatial sparse codes

        Args:
            codes: (batch, n_atoms, H, W) spatial sparse activations

        Returns:
            images: (batch, 1, H, W) reconstructed images
        """
        return self.atoms(codes)

    def get_atoms(self) -> torch.Tensor:
        """Get dictionary atoms as images

        Returns:
            atoms: (n_atoms, 1, atom_size, atom_size) dictionary atoms
        """
        return self.atoms.weight.detach()

    def normalize_atoms_during_training(self):
        """Call this after each optimizer step to maintain unit norm"""
        if self.normalize_atoms:
            self._normalize_atoms()
