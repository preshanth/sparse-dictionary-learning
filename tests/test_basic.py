"""Basic tests for package functionality"""

import torch
import numpy as np


def test_imports():
    """Test that all modules can be imported"""
    try:
        from first_sparse.data import FIRSTCutoutDataset, FITSCutoutExtractor
        from first_sparse.models import SparseAutoencoder
        from first_sparse.training import Trainer
        from first_sparse.utils import load_config
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    try:
        from first_sparse.models import SparseAutoencoder
        
        config = {
            'dictionary_size': 256,
            'encoder': {
                'type': 'simple',
                'input_size': 64,
                'simple': {
                    'channels': [16, 32, 64],
                    'kernel_sizes': [5, 3, 3],
                    'activation': 'relu',
                    'use_batch_norm': True,
                    'dropout': 0.1
                }
            },
            'sparsity': {
                'method': 'learned_threshold',
                'learned_threshold': {
                    'initial_threshold': 0.1,
                    'learnable': True
                }
            },
            'decoder': {
                'normalize_atoms': True,
                'init_method': 'orthogonal'
            }
        }
        
        model = SparseAutoencoder(config)
        print(f"✓ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        return True
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return False


def test_forward_pass():
    """Test forward pass"""
    try:
        from first_sparse.models import SparseAutoencoder
        
        config = {
            'dictionary_size': 256,
            'encoder': {
                'type': 'simple',
                'input_size': 64,
                'simple': {
                    'channels': [16, 32, 64],
                    'kernel_sizes': [5, 3, 3],
                    'activation': 'relu',
                    'use_batch_norm': True,
                    'dropout': 0.1
                }
            },
            'sparsity': {
                'method': 'learned_threshold',
                'learned_threshold': {
                    'initial_threshold': 0.1,
                    'learnable': True
                }
            },
            'decoder': {
                'normalize_atoms': True,
                'init_method': 'orthogonal'
            }
        }
        
        model = SparseAutoencoder(config)
        
        # Create dummy input
        x = torch.randn(4, 1, 64, 64)
        
        # Forward pass
        reconstruction, codes = model(x, return_codes=True)
        
        assert reconstruction.shape == x.shape
        assert codes.shape == (4, 256)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {reconstruction.shape}")
        print(f"  Codes shape: {codes.shape}")
        print(f"  Sparsity (L0): {(codes.abs() > 1e-6).float().sum(dim=1).mean().item():.1f}")
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False


def test_dictionary_atoms():
    """Test dictionary atom extraction"""
    try:
        from first_sparse.models import SparseAutoencoder
        
        config = {
            'dictionary_size': 256,
            'encoder': {
                'type': 'simple',
                'input_size': 64,
                'simple': {
                    'channels': [16, 32, 64],
                    'kernel_sizes': [5, 3, 3],
                    'activation': 'relu',
                    'use_batch_norm': True,
                    'dropout': 0.1
                }
            },
            'sparsity': {
                'method': 'learned_threshold',
                'learned_threshold': {
                    'initial_threshold': 0.1,
                    'learnable': True
                }
            },
            'decoder': {
                'normalize_atoms': True,
                'init_method': 'orthogonal'
            }
        }
        
        model = SparseAutoencoder(config)
        atoms = model.get_dictionary()
        
        assert atoms.shape == (256, 1, 64, 64)
        
        print(f"✓ Dictionary atoms extracted")
        print(f"  Shape: {atoms.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Dictionary extraction failed: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=" * 70)
    print("Running Package Tests")
    print("=" * 70)
    print()
    
    tests = [
        ("Import test", test_imports),
        ("Model creation", test_model_creation),
        ("Forward pass", test_forward_pass),
        ("Dictionary atoms", test_dictionary_atoms),
    ]
    
    results = []
    for name, test_func in tests:
        print(f"\n{name}:")
        print("-" * 40)
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 70)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print("=" * 70)
    
    return all(results)


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
