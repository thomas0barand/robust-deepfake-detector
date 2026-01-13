"""
Download script for FMA and SONICS datasets used in the Deezer AI-Music Detection pipeline.

Usage:
    python scripts/download.py --datasets fma sonics --data-dir data

FMA dataset will be downloaded automatically (~22GB).
SONICS dataset requires manual download - instructions will be provided.
"""

import os
import sys
import argparse
import requests
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm


def download_file(url, dest_path, desc=None):
    """Download a file with progress bar."""
    dest_path = Path(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists():
        print(f"✓ File already exists: {dest_path}")
        return dest_path
    
    print(f"Downloading from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(dest_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
            for chunk in response.iter_content(block_size):
                f.write(chunk)
                pbar.update(len(chunk))
    
    print(f"✓ Downloaded: {dest_path}")
    return dest_path


def extract_zip(zip_path, extract_to):
    """Extract a zip file with progress."""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        with tqdm(total=len(members), desc="Extracting") as pbar:
            for member in members:
                zip_ref.extract(member, extract_to)
                pbar.update(1)
    print(f"✓ Extracted to: {extract_to}")


def download_fma_dataset(data_dir):
    """Download FMA Medium dataset and split file."""
    print("\n" + "="*60)
    print("FMA DATASET DOWNLOAD")
    print("="*60)
    
    fma_dir = Path(data_dir) / "fma"
    fma_dir.mkdir(parents=True, exist_ok=True)
    
    # Download metadata (required for dataset)
    print("\n[1/3] Downloading FMA metadata...")
    metadata_url = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
    metadata_zip = fma_dir / "fma_metadata.zip"
    download_file(metadata_url, metadata_zip, desc="FMA metadata")
    
    # Extract metadata
    extract_zip(metadata_zip, fma_dir)
    metadata_zip.unlink()
    
    # Download FMA Medium dataset
    print("\n[2/3] Downloading FMA Medium dataset (this will take a while, ~22GB)...")
    fma_medium_url = "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
    fma_medium_zip = fma_dir / "fma_medium.zip"
    
    try:
        download_file(fma_medium_url, fma_medium_zip, desc="FMA Medium")
        extract_zip(fma_medium_zip, fma_dir)
        fma_medium_zip.unlink()
    except Exception as e:
        print(f"⚠ Warning: Could not download FMA Medium dataset: {e}")
        print("You may need to download it manually from:")
        print("https://github.com/mdeff/fma")
    
    # Download split file from deezer/deepfake-detector
    print("\n[3/3] Downloading FMA split file...")
    split_url = "https://github.com/deezer/deepfake-detector/raw/main/data/dataset_medium_split.npy"
    split_path = fma_dir / "dataset_medium_split.npy"
    
    try:
        download_file(split_url, split_path, desc="FMA split")
    except Exception as e:
        print(f"⚠ Warning: Could not download split file: {e}")
        print("You can download it manually from:")
        print(split_url)
    
    print("\n✓ FMA dataset setup complete!")
    print(f"  Location: {fma_dir}")
    print(f"  Files expected:")
    print(f"    - fma_medium/ (audio files)")
    print(f"    - fma_metadata/ (metadata)")
    print(f"    - dataset_medium_split.npy (train/test split)")


def download_sonics_dataset(data_dir):
    """Provide instructions for downloading SONICS dataset."""
    print("\n" + "="*60)
    print("SONICS DATASET DOWNLOAD")
    print("="*60)
    
    sonics_dir = Path(data_dir) / "sonics"
    sonics_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n⚠  SONICS dataset requires MANUAL DOWNLOAD")
    print("\nThe SONICS dataset contains AI-generated music from Suno and Udio.")
    print("It is hosted on Hugging Face and requires manual download.\n")
    
    print("📋 INSTRUCTIONS:")
    print("-" * 60)
    print("1. Visit the SONICS dataset page:")
    print("   https://huggingface.co/datasets/awsaf49/SONICS")
    print("\n2. Create a Hugging Face account if you don't have one")
    print("\n3. Download the dataset files:")
    print("   - fake_songs.csv (metadata)")
    print("   - fake_songs/ (directory with MP3 files)")
    print("\n4. Place the downloaded files in:")
    print(f"   {sonics_dir.absolute()}/")
    print("\n5. Expected structure:")
    print(f"   {sonics_dir}/")
    print("     ├── fake_songs.csv")
    print("     └── fake_songs/")
    print("         ├── fake_00000_suno_0.mp3")
    print("         ├── fake_00000_suno_1.mp3")
    print("         └── ...")
    print("-" * 60)
    
    # Check if split file already exists in deezer/sonics
    project_root = Path(__file__).parent.parent
    existing_split = project_root / "deezer" / "sonics" / "sonics_split.npy"
    
    if existing_split.exists():
        print(f"\n✓ Split file already exists: {existing_split}")
    else:
        print("\n⚠  Note: Split file not found in deezer/sonics/")
        print("   You'll need to run create_splits.py after downloading the data")
    
    print("\n💡 Alternative download method using Python:")
    print("-" * 60)
    print("from huggingface_hub import snapshot_download")
    print(f"snapshot_download(repo_id='awsaf49/SONICS', repo_type='dataset', local_dir='{sonics_dir}')")
    print("-" * 60)


def verify_downloads(data_dir):
    """Verify downloaded datasets."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    data_dir = Path(data_dir)
    
    # Check FMA
    fma_dir = data_dir / "fma"
    fma_complete = all([
        (fma_dir / "fma_medium").exists(),
        (fma_dir / "fma_metadata").exists(),
        (fma_dir / "dataset_medium_split.npy").exists(),
    ])
    
    print(f"\nFMA Dataset: {'✓ Complete' if fma_complete else '⚠ Incomplete'}")
    print(f"  Location: {fma_dir}")
    print(f"  - fma_medium/: {'✓' if (fma_dir / 'fma_medium').exists() else '✗'}")
    print(f"  - fma_metadata/: {'✓' if (fma_dir / 'fma_metadata').exists() else '✗'}")
    print(f"  - dataset_medium_split.npy: {'✓' if (fma_dir / 'dataset_medium_split.npy').exists() else '✗'}")
    
    # Check SONICS
    sonics_dir = data_dir / "sonics"
    sonics_complete = all([
        (sonics_dir / "fake_songs.csv").exists(),
        (sonics_dir / "fake_songs").exists(),
    ])
    
    print(f"\nSONICS Dataset: {'✓ Complete' if sonics_complete else '⚠ Incomplete (manual download required)'}")
    print(f"  Location: {sonics_dir}")
    print(f"  - fake_songs.csv: {'✓' if (sonics_dir / 'fake_songs.csv').exists() else '✗'}")
    print(f"  - fake_songs/: {'✓' if (sonics_dir / 'fake_songs').exists() else '✗'}")


def main():
    parser = argparse.ArgumentParser(
        description="Download FMA and SONICS datasets for Deezer AI-Music Detection"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["fma", "sonics", "all"],
        default=["all"],
        help="Which datasets to download (default: all)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store datasets (default: data)"
    )
    
    args = parser.parse_args()
    
    # Determine which datasets to download
    datasets = set(args.datasets)
    if "all" in datasets:
        datasets = {"fma", "sonics"}
    
    print("="*60)
    print("DEEZER AI-MUSIC DETECTION - DATASET DOWNLOADER")
    print("="*60)
    print(f"\nDatasets to download: {', '.join(datasets)}")
    print(f"Data directory: {args.data_dir}")
    
    # Create data directory
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download datasets
    if "fma" in datasets:
        try:
            download_fma_dataset(args.data_dir)
        except KeyboardInterrupt:
            print("\n\n⚠ Download interrupted by user")
            sys.exit(1)
        except Exception as e:
            print(f"\n⚠ Error downloading FMA dataset: {e}")
    
    if "sonics" in datasets:
        download_sonics_dataset(args.data_dir)
    
    # Verify downloads
    verify_downloads(args.data_dir)
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. After downloading SONICS data, run:")
    print("   cd deezer/sonics")
    print("   python create_splits.py")
    print("\n2. Compute fakeprints for FMA:")
    print("   python deezer/compute_fakeprints.py --save fp_fma.npy --path data/fma/fma_medium --sr 44100")
    print("\n3. Compute fakeprints for SONICS:")
    print("   python deezer/compute_fakeprints.py --save fp_sonics.npy --path data/sonics/fake_songs --sr 16000")
    print("\n4. Train the detector:")
    print("   python deezer/train_test_regressor.py --synth fp_sonics.npy --real fp_fma.npy")
    print("="*60)


if __name__ == "__main__":
    main()
