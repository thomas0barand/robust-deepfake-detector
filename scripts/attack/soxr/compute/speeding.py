#!/usr/bin/env python3
"""
Resample audio to new sample rate

Run:
python scripts/attack/soxr/compute/speeding.py data/signals/originals/signal1.mp3 --s 1.02 -n signal1_s.mp3
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Modify audio speed using sox")
    parser.add_argument("input", help="Input audio file path")
    parser.add_argument("-s", "--speed", type=float, required=True, help="Speed factor (e.g., 1.02)")
    parser.add_argument("-n", "--name", required=True, help="Output filename")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{input_path}' not found", file=sys.stderr)
        sys.exit(1)
    
    output_path = "data/signals/spedup/" + args.name
    
    cmd = ["sox", str(input_path), str(output_path), "speed", str(args.speed)]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Created: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error: sox command failed with code {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print("Error: sox not found. Install it with 'brew install sox'", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()