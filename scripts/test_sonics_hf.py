"""
Test the SONICS SpecTTTra HuggingFace model on a dataset of .mp3 files.

Expected folder structure:
    <data_dir>/
        ai/       <- fake/synthetic songs
        human/    <- real songs

Usage:
    python scripts/test_sonics_hf.py --data_dir data/sonics_test/
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report

from sonics import HFAudioClassifier

SAMPLE_RATE = 16_000
MODEL_ID = "awsaf49/sonics-spectttra-alpha-120s"


def load_audio(path: Path, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    # mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.squeeze(0)  # (T,)


def predict(model: torch.nn.Module, waveform: torch.Tensor, device: torch.device) -> int:
    with torch.no_grad():
        audio = waveform.unsqueeze(0).to(device)  # (1, T)
        logits = model(audio)
        return logits.argmax(dim=-1).item()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading model {MODEL_ID}...")
    model = HFAudioClassifier.from_pretrained(MODEL_ID, map_location=str(device))
    model.to(device).eval()

    data_dir = Path(args.data_dir)
    # label 0 = human (real), label 1 = ai (fake) — matches SONICS convention
    splits = {"human": 0, "ai": 1}

    all_files, all_labels = [], []
    for folder, label in splits.items():
        folder_path = data_dir / folder
        if not folder_path.exists():
            print(f"  Warning: {folder_path} not found, skipping.")
            continue
        mp3s = sorted(folder_path.glob("*.mp3"))
        all_files.extend(mp3s)
        all_labels.extend([label] * len(mp3s))
        print(f"  {folder}: {len(mp3s)} files (label={label})")

    if not all_files:
        print("No .mp3 files found. Check --data_dir.")
        return

    preds, labels = [], []
    errors = []
    for path, label in tqdm(zip(all_files, all_labels), total=len(all_files), desc="Evaluating"):
        try:
            waveform = load_audio(path)
            pred = predict(model, waveform, device)
            preds.append(pred)
            labels.append(label)
        except Exception as e:
            errors.append((path.name, str(e)))

    if errors:
        print(f"\n{len(errors)} file(s) failed to load:")
        for name, err in errors:
            print(f"  {name}: {err}")

    print("\n--- Results ---")
    print(classification_report(labels, preds, target_names=["human", "ai"], digits=4))
    print(f"Accuracy : {accuracy_score(labels, preds):.4f}")
    print(f"F1 (macro): {f1_score(labels, preds, average='macro'):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root folder containing 'ai/' and 'human/' subfolders with .mp3 files")
    run(parser.parse_args())
