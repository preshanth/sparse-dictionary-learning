# Quick Start Guide

## Installation

```bash
# Clone or download the package
cd first-sparse-dict

# Run setup script
./setup_env.sh

# Or manually:
python -m venv venv
source venv/bin/activate
pip install -e .

# Optional: Install PyBDSF
pip install -e .[pybdsf]
```

## Testing the Installation

```bash
python tests/test_basic.py
```

Expected output:
- ✓ All imports successful
- ✓ Model created with ~XXX,XXX parameters
- ✓ Forward pass successful
- ✓ Dictionary atoms extracted

## Data Preparation

### Option 1: Download FIRST Data

```bash
# Download 10 FIRST fields (~10GB)
python scripts/download_first.py \
    --output-dir ./data/first \
    --max-fields 10

# This will take 30-60 minutes depending on connection speed
```

### Option 2: Use Existing FIRST Data

If you already have FIRST FITS files, just point to them in the next step.

## Build Source Manifest

### Using FIRST Catalog (Recommended)

```bash
# First, download FIRST catalog
wget http://sundog.stsci.edu/first/catalogs/first_14dec17.fits.gz
gunzip first_14dec17.fits.gz
mv first_14dec17.fits ./data/

# Build manifest
python scripts/build_manifest.py \
    --fits-dir ./data/first \
    --catalog ./data/first_14dec17.fits \
    --output ./data/manifest.json \
    --cutout-size 64 \
    --min-snr 10.0

# Expected: ~5,000-10,000 sources from 10 fields
```

### Using PyBDSF (If Installed)

```bash
python scripts/build_manifest.py \
    --fits-dir ./data/first \
    --output ./data/manifest.json \
    --cutout-size 64 \
    --use-pybdsf
```

## Train Model

```bash
# Basic training (uses config defaults)
python scripts/train.py

# With custom config
python scripts/train.py \
    --data-config config/data_config.yaml \
    --model-config config/model_config.yaml \
    --training-config config/training_config.yaml

# Resume from checkpoint
python scripts/train.py --resume ./checkpoints/checkpoint_epoch_20.pt
```

### Expected Training Time
- Simple encoder, 64x64, K=256: ~30-60 min per epoch on GPU
- 100 epochs: ~50-100 hours
- Early stopping typically kicks in around 30-50 epochs

### Monitoring Training
- Checkpoints saved to: `./checkpoints/`
- Logs saved to: `./logs/`
- Watch for:
  - Decreasing reconstruction loss
  - Sparsity (L0) around 10-30 active atoms
  - Validation loss improving

## Evaluate Model

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best.pt \
    --manifest ./data/manifest.json \
    --output-dir ./results

# Evaluate on validation set instead of test
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best.pt \
    --manifest ./data/manifest.json \
    --split val \
    --output-dir ./results
```

## Interpreting Results

### Metrics (results/metrics.json)
- **RMSE**: Root mean squared error (lower is better, <0.1 is good)
- **PSNR**: Peak signal-to-noise ratio (higher is better, >20 is good)
- **L0 norm**: Number of active atoms per source (target: 10-30)
- **Atom usage**: Check if all atoms are being used (>50% usage is good)

### Visualizations
- **dictionary_atoms.png**: Grid showing learned basis functions
  - Look for: Distinct patterns, not random noise
  - Good signs: Jets, cores, lobes, extended structures
  - Bad signs: All atoms look similar, or pure noise
  
- **reconstructions.png**: Original vs reconstruction vs residual
  - Residuals should be noise-like, not structured
  - Good reconstruction captures source morphology
  
- **atom_usage.png**: Histogram of how often each atom is used
  - Uniform distribution: All atoms contribute
  - Heavy tail: Some atoms dominate (might need more diversity penalty)

## Troubleshooting

### Out of Memory
Edit `config/training_config.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
  mixed_precision: true  # Ensure enabled
```

### FITS Files Not Found
- Check `--fits-dir` path is correct
- Verify .fits files exist in subdirectories
- Try absolute paths instead of relative

### No Sources in Manifest
- Lower `--min-snr` threshold
- Increase `--max-angular-size`
- Check FITS files have sources (view in DS9)

### Training Loss Not Decreasing
- Check learning rate (try 1e-4 instead of 1e-3)
- Verify data normalization is working
- Ensure augmentation is enabled for training

### All Atoms Look Similar
- Increase diversity penalty weight
- Try different initialization method
- Check if dataset has enough morphological variety

## Configuration Tips

### For Quick Prototyping
```yaml
# training_config.yaml
epochs: 20
batch_size: 64
validate_every_n_epochs: 5
```

### For Production Run
```yaml
# training_config.yaml
epochs: 100
batch_size: 32
early_stopping_patience: 20

# model_config.yaml
encoder:
  type: "production"  # Use residual encoder
```

### For Memory-Constrained Systems
```yaml
# training_config.yaml
batch_size: 16
num_workers: 2

# data_config.yaml
cutout_size: 64  # Don't increase

# model_config.yaml
dictionary_size: 256  # Don't increase
```

## Next Steps

After successful training:

1. **Analyze atoms**: Are they physically interpretable?
2. **Check sparsity**: Is L0 in the target range?
3. **Test generalization**: Evaluate on different fields
4. **Scale up**: Try 128x128 cutouts, K=512
5. **Iterate**: Adjust sparsity weight, diversity penalty
6. **Research questions**: 
   - Do FRI/FRII sources use different atoms?
   - Can you predict source type from sparse codes?
   - Does dictionary transfer to other surveys?

## Contact & Issues

For questions or issues, check:
- README.md for full documentation
- Config files for parameter descriptions
- Code comments for implementation details
