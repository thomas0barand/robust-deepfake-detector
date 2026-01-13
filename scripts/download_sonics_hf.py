"""
Helper script to download SONICS dataset from Hugging Face.

This script uses the huggingface_hub library to download the SONICS dataset.
You need to have a Hugging Face account and be logged in.

Usage:
    # Login first (one time)
    huggingface-cli login
    
    # Then run this script
    python scripts/download_sonics_hf.py --data-dir data
"""

import argparse
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("Error: huggingface_hub not installed")
    print("Install it with: pip install huggingface-hub")
    exit(1)


def download_sonics(data_dir):
    """Download SONICS dataset from Hugging Face."""
    sonics_dir = Path(data_dir) / "sonics"
    sonics_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DOWNLOADING SONICS DATASET FROM HUGGING FACE")
    print("="*60)
    print(f"\nDestination: {sonics_dir.absolute()}")
    print("\nThis may take a while depending on your connection...")
    print("The dataset contains AI-generated music from Suno and Udio.\n")
    
    try:
        snapshot_download(
            repo_id="awsaf49/SONICS",
            repo_type="dataset",
            local_dir=str(sonics_dir),
            resume_download=True
        )
        
        print("\n" + "="*60)
        print("✓ DOWNLOAD COMPLETE!")
        print("="*60)
        print(f"\nDataset location: {sonics_dir.absolute()}")
        print("\nVerifying files...")
        
        # Verify expected files
        fake_songs_csv = sonics_dir / "fake_songs.csv"
        fake_songs_dir = sonics_dir / "fake_songs"
        
        if fake_songs_csv.exists():
            print(f"  ✓ fake_songs.csv found")
        else:
            print(f"  ⚠ fake_songs.csv not found")
        
        if fake_songs_dir.exists() and fake_songs_dir.is_dir():
            mp3_files = list(fake_songs_dir.glob("*.mp3"))
            print(f"  ✓ fake_songs/ directory found ({len(mp3_files)} MP3 files)")
        else:
            print(f"  ⚠ fake_songs/ directory not found")
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Create splits file:")
        print("   cd deezer/sonics")
        print("   python create_splits.py")
        print("\n2. Compute fakeprints:")
        print("   python deezer/compute_fakeprints.py --save fp_sonics.npy --path data/sonics/fake_songs --sr 16000")
        print("="*60)
        
    except Exception as e:
        print(f"\n⚠ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure you're logged in: huggingface-cli login")
        print("2. Check your internet connection")
        print("3. Visit https://huggingface.co/datasets/awsaf49/SONICS to verify access")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download SONICS dataset from Hugging Face"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store the dataset (default: data)"
    )
    
    args = parser.parse_args()
    
    download_sonics(args.data_dir)


if __name__ == "__main__":
    main()
