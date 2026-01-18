"""Configuration loading and merging utilities"""

from pathlib import Path
from typing import Dict, Union

import yaml


def load_config(config_path: Union[str, Path]) -> Dict:
    """Load YAML configuration file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(*configs: Dict) -> Dict:
    """Merge multiple configuration dictionaries
    
    Later configs override earlier ones.
    
    Args:
        *configs: Variable number of config dicts
        
    Returns:
        Merged configuration dictionary
    """
    merged = {}
    
    for config in configs:
        _deep_update(merged, config)
    
    return merged


def _deep_update(base: Dict, update: Dict) -> Dict:
    """Recursively update nested dictionary
    
    Args:
        base: Base dictionary to update
        update: Dictionary with updates
        
    Returns:
        Updated base dictionary (modified in-place)
    """
    for key, value in update.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    
    return base


def load_all_configs(config_dir: Union[str, Path]) -> Dict:
    """Load and merge all config files from directory
    
    Expects files: data_config.yaml, model_config.yaml, training_config.yaml
    
    Args:
        config_dir: Directory containing config files
        
    Returns:
        Merged configuration dictionary with keys: 'data', 'model', 'training'
    """
    config_dir = Path(config_dir)
    
    configs = {}
    
    for config_file in ['data_config.yaml', 'model_config.yaml', 'training_config.yaml']:
        path = config_dir / config_file
        if path.exists():
            config = load_config(path)
            configs.update(config)
    
    return configs
