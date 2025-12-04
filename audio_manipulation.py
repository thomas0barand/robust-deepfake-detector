#!/usr/bin/env python3
"""
Script pour appliquer des manipulations audio (pitch shift, time stretch, noise addition)
sur le fichier satellites.mp3
"""

import librosa
import soundfile as sf
import numpy as np
import os
from pathlib import Path


def pitch_shift_audio(y, sr, n_steps=2):
    """
    Applique un pitch shift à l'audio.
    
    Args:
        y: Signal audio (numpy array)
        sr: Sample rate
        n_steps: Nombre de demi-tons à décaler (positif = plus aigu, négatif = plus grave)
    
    Returns:
        Signal audio avec pitch shift
    """
    return librosa.effects.pitch_shift(y=y, sr=sr, n_steps=n_steps)


def time_stretch_audio(y, rate=1.2):
    """
    Applique un time stretch à l'audio (sans changer le pitch).
    
    Args:
        y: Signal audio (numpy array)
        rate: Facteur de vitesse (1.2 = 20% plus rapide, 0.8 = 20% plus lent)
    
    Returns:
        Signal audio avec time stretch
    """
    return librosa.effects.time_stretch(y=y, rate=rate)


def add_noise(y, noise_factor=0.01):
    """
    Ajoute du bruit blanc à l'audio.
    
    Args:
        y: Signal audio (numpy array)
        noise_factor: Intensité du bruit (0.01 = 1% du signal max)
    
    Returns:
        Signal audio avec bruit ajouté
    """
    noise = np.random.normal(0, noise_factor * np.max(np.abs(y)), len(y))
    return y + noise


def main():
    # Chemin du fichier audio
    input_file = "audio/satellites.mp3"
    output_dir = Path("audio/processed")
    output_dir.mkdir(exist_ok=True)
    
    print(f"Chargement de {input_file}...")
    # Charger l'audio
    y, sr = librosa.load(input_file, sr=None)
    
    print(f"Audio chargé: {len(y)/sr:.2f} secondes, sample rate: {sr} Hz")
    
    # 1. Pitch shift (plus aigu)
    print("\n1. Application du pitch shift (+2 demi-tons)...")
    y_pitch_up = pitch_shift_audio(y, sr, n_steps=2)
    output_file = output_dir / "satellites_pitch_up.wav"
    sf.write(str(output_file), y_pitch_up, sr)
    print(f"   Sauvegardé: {output_file}")
    
    # 2. Pitch shift (plus grave)
    print("\n2. Application du pitch shift (-2 demi-tons)...")
    y_pitch_down = pitch_shift_audio(y, sr, n_steps=-2)
    output_file = output_dir / "satellites_pitch_down.wav"
    sf.write(str(output_file), y_pitch_down, sr)
    print(f"   Sauvegardé: {output_file}")
    
    # 3. Time stretch (plus rapide)
    print("\n3. Application du time stretch (1.2x plus rapide)...")
    y_faster = time_stretch_audio(y, rate=1.2)
    output_file = output_dir / "satellites_faster.wav"
    sf.write(str(output_file), y_faster, sr)
    print(f"   Sauvegardé: {output_file}")
    
    # 4. Time stretch (plus lent)
    print("\n4. Application du time stretch (0.8x plus lent)...")
    y_slower = time_stretch_audio(y, rate=0.8)
    output_file = output_dir / "satellites_slower.wav"
    sf.write(str(output_file), y_slower, sr)
    print(f"   Sauvegardé: {output_file}")
    
    # 5. Ajout de bruit
    print("\n5. Ajout de bruit (facteur 0.01)...")
    y_noisy = add_noise(y, noise_factor=0.01)
    output_file = output_dir / "satellites_noisy.wav"
    sf.write(str(output_file), y_noisy, sr)
    print(f"   Sauvegardé: {output_file}")
    
    # 6. Combinaison: pitch shift + time stretch + noise
    print("\n6. Application combinée (pitch + time stretch + noise)...")
    y_combined = pitch_shift_audio(y, sr, n_steps=1)
    y_combined = time_stretch_audio(y_combined, rate=1.1)
    y_combined = add_noise(y_combined, noise_factor=0.005)
    output_file = output_dir / "satellites_combined.wav"
    sf.write(str(output_file), y_combined, sr)
    print(f"   Sauvegardé: {output_file}")
    
    print("\n✓ Toutes les manipulations ont été appliquées et sauvegardées!")
    print(f"   Fichiers sauvegardés dans: {output_dir}")


if __name__ == "__main__":
    main()

