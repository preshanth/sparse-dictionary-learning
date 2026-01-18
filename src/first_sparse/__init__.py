"""FIRST Sparse Dictionary Learning Package"""

__version__ = "0.1.0"

from . import data
from . import models
from . import training
from . import utils

# Convenience imports
from .data import FIRSTCutoutDataset, create_dataloaders
from .models import SparseAutoencoder
from .training import Trainer
from .utils import load_all_configs

__all__ = [
    "data", 
    "models", 
    "training", 
    "utils",
    "FIRSTCutoutDataset",
    "create_dataloaders", 
    "SparseAutoencoder",
    "Trainer",
    "load_all_configs"
]
