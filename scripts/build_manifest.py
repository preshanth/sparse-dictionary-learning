#!/usr/bin/env python
"""Build manifest of source cutouts from FIRST FITS files"""
import os
import argparse
import json
from pathlib import Path
from typing import List, Dict
import numpy as np
import requests
from tqdm import tqdm
from astropy.io import fits

from first_sparse.data import FIRSTCatalog, FITSCutoutExtractor


def find_fits_files(fits_dir: Path) -> List[Path]:
    """Recursively find all FITS files"""
    fits_files = []
    if pattern := '*.gz':
        #uncompress the files
        fits_files.extend(fits_dir.glob(pattern))
        for file in fits_files:
            os.system(f'gunzip {file}')
        
    for pattern in ['**/*.fits', '**/*.fits.gz', '**/F*']:
        fits_files.extend(fits_dir.glob(pattern))
    
    return sorted(fits_files)


def build_manifest_from_catalog(
    fits_dir: Path,
    catalog_path: Path,
    cutout_size: int,
    min_snr: float,
    max_angular_size: float,
    min_angular_size: float
) -> List[Dict]:
    """Build manifest using FIRST catalog for source positions"""
    
    print("Loading FIRST catalog...")
    catalog = FIRSTCatalog(str(catalog_path))
    print(f"Catalog contains {len(catalog)} sources")
    
    print("\nScanning FITS files...")
    fits_files = find_fits_files(fits_dir)
    print(f"Found {len(fits_files)} FITS files")
    
    sources = []
    source_id = 0
    
    for fits_path in tqdm(fits_files, desc="Processing FITS files"):
        try:
            # Get image metadata
            with fits.open(fits_path) as hdul:
                header = hdul[0].header
                
                # Get image center and size
                if 'CRVAL1' in header and 'CRVAL2' in header:
                    center_ra = header['CRVAL1']
                    center_dec = header['CRVAL2']
                else:
                    continue
                
                # Estimate image size from NAXIS
                try:
                    naxis1 = header['NAXIS1']
                except KeyError:
                    naxis1 = 2048
                
                try:
                    cdelt1 = abs(header['CDELT1'])
                except KeyError:
                    cdelt1 = 0.00055  # degrees/pixel
                
                image_size_deg = naxis1 * cdelt1

                print(f"Processing {fits_path.name}: center=({center_ra}, {center_dec}), size={image_size_deg} deg")
            # Find catalog sources in this image
            image_sources = catalog.get_sources_in_image(
                center_ra, center_dec,
                image_size_deg,
                min_flux=1.0,
                min_snr=min_snr
            )
            # Filter by angular size and reject point-like sources
            for src in image_sources:
                major = src.get('major', 0)
                peak = src.get('peak', 0)
                flux = src.get('flux', 0)

                # Reject point-like sources (peak/integrated ratio > 0.8)
                if flux > 0:
                    peak_to_int = peak / flux
                    if peak_to_int > 0.8:
                        continue

                if min_angular_size <= major <= max_angular_size:
                    sources.append({
                        'source_id': f"FIRST_{source_id:06d}",
                        'fits_path': str(fits_path),
                        'ra': src['ra'],
                        'dec': src['dec'],
                        'cutout_size': cutout_size,
                        'integrated_flux': src.get('flux', 0),
                        'peak_flux': src.get('peak', 0),
                        'rms': src.get('rms', 0),
                        'major': major,
                        'minor': src.get('minor', 0),
                    })
                    source_id += 1
        
        except Exception as e:
            print(f"Error processing {fits_path}: {e}")
            continue
    
    return sources


def build_manifest_from_pybdsf(
    fits_dir: Path,
    cutout_size: int,
    min_snr: float,
    min_angular_size: float = 5.0,
    max_angular_size: float = 100.0
) -> List[Dict]:
    """Build manifest using PyBDSF for source detection"""

    try:
        import bdsf
    except ImportError:
        raise ImportError(
            "PyBDSF not installed. Install with: pip install -e .[pybdsf]"
        )

    print("\nScanning FITS files...")
    fits_files = find_fits_files(fits_dir)
    print(f"Found {len(fits_files)} FITS files")

    sources = []
    source_id = 0

    for fits_path in tqdm(fits_files, desc="Running PyBDSF"):
        try:
            img = bdsf.process_image(
                str(fits_path),
                thresh_isl=3.0,
                thresh_pix=5.0,
                quiet=True
            )

            for src in img.sources:
                # PyBDSF uses attributes, not dict keys
                snr = src.peak_flux_max / src.rms_isl
                major_arcsec = src.size_sky[0] * 3600  # deg to arcsec
                minor_arcsec = src.size_sky[1] * 3600

                if snr >= min_snr and min_angular_size <= major_arcsec <= max_angular_size:
                    sources.append({
                        'source_id': f"FIRST_{source_id:06d}",
                        'fits_path': str(fits_path),
                        'ra': float(src.posn_sky[0]),
                        'dec': float(src.posn_sky[1]),
                        'cutout_size': cutout_size,
                        'integrated_flux': float(src.total_flux) * 1000,
                        'peak_flux': float(src.peak_flux_max) * 1000,
                        'rms': float(src.rms_isl) * 1000,
                        'major': major_arcsec,
                        'minor': minor_arcsec,
                    })
                    source_id += 1

        except Exception as e:
            print(f"Error processing {fits_path}: {e}")
            continue

    return sources


def create_splits(sources: List[Dict], train_frac: float, val_frac: float, 
                 test_frac: float, seed: int) -> Dict[str, List[int]]:
    """Create train/val/test splits"""
    
    n_sources = len(sources)
    indices = np.arange(n_sources)
    
    # Shuffle with fixed seed
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)
    
    # Split indices
    n_train = int(n_sources * train_frac)
    n_val = int(n_sources * val_frac)
    
    splits = {
        'train': indices[:n_train].tolist(),
        'val': indices[n_train:n_train+n_val].tolist(),
        'test': indices[n_train+n_val:].tolist()
    }
    
    return splits


def main():
    parser = argparse.ArgumentParser(
        description="Build source manifest from FIRST FITS files"
    )
    parser.add_argument(
        "--fits-dir",
        type=str,
        required=True,
        help="Directory containing FIRST FITS files"
    )
    parser.add_argument(
        "--catalog",
        type=str,
        default=None,
        help="Path to FIRST catalog FITS file (if not using PyBDSF)"
    )
    parser.add_argument(
        "--use-pybdsf",
        action="store_true",
        help="Use PyBDSF for source detection instead of catalog"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/manifest.json",
        help="Output manifest JSON file"
    )
    parser.add_argument(
        "--cutout-size",
        type=int,
        default=64,
        help="Cutout size in pixels"
    )
    parser.add_argument(
        "--min-snr",
        type=float,
        default=10.0,
        help="Minimum SNR for source selection"
    )
    parser.add_argument(
        "--max-angular-size",
        type=float,
        default=100.0,
        help="Maximum source size in arcsec"
    )
    parser.add_argument(
        "--min-angular-size",
        type=float,
        default=5.0,
        help="Minimum source size in arcsec"
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.8,
        help="Fraction for training set"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction for validation set"
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction for test set"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits"
    )
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("FIRST Manifest Builder")
    print("=" * 70)
    print(f"FITS directory: {args.fits_dir}")
    print(f"Cutout size: {args.cutout_size}x{args.cutout_size} pixels")
    print(f"Min SNR: {args.min_snr}")
    print(f"Angular size range: {args.min_angular_size}-{args.max_angular_size} arcsec")
    print()
    
    fits_dir = Path(args.fits_dir)
    if not fits_dir.exists():
        print(f"Error: FITS directory not found: {fits_dir}")
        return
    
    # Build manifest
    if args.use_pybdsf:
        print("Using PyBDSF for source detection...")
        sources = build_manifest_from_pybdsf(
            fits_dir,
            args.cutout_size,
            args.min_snr,
            args.min_angular_size,
            args.max_angular_size
        )
    else:
        if args.catalog is None:
            print("Error: Must provide --catalog or use --use-pybdsf")
            return
        
        catalog_path = Path(args.catalog)
        if not catalog_path.exists():
            print(f"Error: Catalog not found: {catalog_path}")
            print(f"Will attempt to download from https://sundog.stsci.edu/first/catalogs/first_14dec17.fits.gz")
            # Download catalog
            url = "https://sundog.stsci.edu/first/catalogs/first_14dec17.fits.gz"
            response = requests.get(url)
            if response.status_code == 200:
                with open(catalog_path, 'wb') as f:
                    f.write(response.content)
                print(f"Catalog downloaded to {catalog_path}")
            else:
                print("Error downloading catalog")
            return
        
        print("Using FIRST catalog for source positions...")
        sources = build_manifest_from_catalog(
            fits_dir,
            catalog_path,
            args.cutout_size,
            args.min_snr,
            args.max_angular_size,
            args.min_angular_size
        )
    
    print(f"\nFound {len(sources)} sources")
    
    if len(sources) == 0:
        print("No sources found! Check your filtering criteria.")
        return
    
    # Create splits
    splits = create_splits(
        sources,
        args.train_frac,
        args.val_frac,
        args.test_frac,
        args.seed
    )
    
    print(f"\nData splits:")
    print(f"  Train: {len(splits['train'])} sources")
    print(f"  Val:   {len(splits['val'])} sources")
    print(f"  Test:  {len(splits['test'])} sources")
    
    # Save manifest
    manifest = {
        'sources': sources,
        'splits': splits,
        'metadata': {
            'cutout_size': args.cutout_size,
            'min_snr': args.min_snr,
            'max_angular_size': args.max_angular_size,
            'min_angular_size': args.min_angular_size,
            'n_sources': len(sources),
        }
    }
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\nManifest saved to: {output_path.absolute()}")


if __name__ == "__main__":
    main()
