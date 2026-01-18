# FIRST Sparse Dictionary Learning

Sparse dictionary learning for radio astronomy sources using FIRST survey data.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install core dependencies
pip install -e .

# Optional: Install PyBDSF for source extraction
pip install -e .[pybdsf]
```

## Quick Start

### 1. Download FIRST Data

```bash
python scripts/download_first.py \
    --output-dir ./data/first \
    --max-fields 10
```

Downloads 10 FIRST fields (~10GB). Adjust `--max-fields` as needed.

### 2. Build Source Manifest

Using FIRST catalog (recommended for initial test):
```bash
# First download FIRST catalog
wget http://sundog.stsci.edu/first/catalogs/first_14dec17.fits.gz
gunzip first_14dec17.fits.gz
mv first_14dec17.fits ./data/

python scripts/build_manifest.py \
    --fits-dir ./data/first \
    --catalog ./data/first_14dec17.fits \
    --output ./data/manifest.json \
    --cutout-size 64
```

Using PyBDSF (requires optional dependency):
```bash
python scripts/build_manifest.py \
    --fits-dir ./data/first \
    --output ./data/manifest.json \
    --cutout-size 64 \
    --use-pybdsf
```

### 3. Train Model

```bash
python scripts/train.py \
    --data-config config/data_config.yaml \
    --model-config config/model_config.yaml \
    --training-config config/training_config.yaml
```

Monitor training in `./logs/` directory.

### 4. Evaluate and Visualize

```bash
python scripts/evaluate.py \
    --checkpoint ./checkpoints/best.pt \
    --manifest ./data/manifest.json \
    --output-dir ./results
```

Generates:
- `dictionary_atoms.png` - Grid of learned basis functions
- `reconstructions.png` - Example reconstructions
- `metrics.json` - Quantitative evaluation

## Configuration

Edit files in `config/` to adjust:
- `data_config.yaml` - Cutout size, normalization, filtering
- `model_config.yaml` - Dictionary size, encoder architecture
- `training_config.yaml` - Learning rate, batch size, epochs

### Encoder Selection

Switch between simple and production encoder in `config/model_config.yaml`:

```yaml
encoder:
  type: "simple"  # or "production"
```

## Memory Requirements

- GPU VRAM: ~4GB for training (batch_size=32, cutout_size=64, dict_size=256)
- RAM: ~8GB during data loading
- Storage: ~10GB for 10 fields + checkpoints

## Project Structure

```
first-sparse-dict/
├── config/          # YAML configuration files
├── src/             # Source code
│   └── first_sparse/
│       ├── data/    # Data loading and preprocessing
│       ├── models/  # Neural network architectures
│       ├── training/# Training loop
│       └── utils/   # Utilities and visualization
├── scripts/         # Executable scripts
└── tests/           # Unit tests
```

## Troubleshooting

**Out of memory errors:**
- Reduce `batch_size` in `config/training_config.yaml`
- Reduce `cutout_size` in `config/data_config.yaml`
- Enable mixed precision (already default)

**Download fails:**
- Check internet connection
- Some archive files may be unavailable - script will skip and continue

**FITS files not found:**
- Verify `--fits-dir` path in build_manifest.py
- Check that .fits files exist in subdirectories
