# Fakeprint Detector

Binary classifier that distinguishes AI-generated music from human-made music using spectral fingerprints (fakeprints) extracted via STFT or CQT, and made robust through cross-correlation-based convolutional layer.

---

## Pipeline Overview

```
Raw Audio Dataset          Preprocessing              PyTorch Dataset
(AI / Human .mp3 files) → (extract fakeprints    → (FakeprintDataset loads
                           → save as .npz)            .npz files + labels)
                                                            ↓
                                                   Lightning Training
                                                   (RobustDetector module)
                                                            ↓
                                                   Best Checkpoint (.ckpt)
                                                   monitored by val_auroc
```

### 1. Raw Dataset

Audio files organized by label:
```
data/
├── ai/
│   ├── track_001.mp3
│   └── ...
└── human/
    ├── track_001.mp3
    └── ...
```

### 2. Preprocessing — Extract Fakeprints

Converts raw audio into checkerboard artifacts and saves them as compressed `.npz` files. Supports two transforms:

- **STFT** — Short-Time Fourier Transform, standard spectrogram representation
- **CQT** — Constant-Q Transform, better frequency resolution for music. Computed using nnAudio, a PyTorch-based audio processing library that runs transforms directly on the GPU, avoiding CPU bottlenecks during preprocessing and inference.

The notebook `generate_data.ipynb` performs this step, outputting files like:
```
src/checkpoints/fp/ai/
├── fakeprints_01.npz
├── fakeprints_02.npz
└── ...
```

Each `.npz` stores the fakeprint arrays for each transform (STFT or CQT).

### 3. Dataset

`FakeprintDataset` reads the preprocessed `.npz` files from the output directory and serves `(fakeprint_tensor, label)` pairs to the DataLoader. A random train/val split is applied at training time.

### 4. Training

`RobustDetector` is a PyTorch Lightning module wrapping the classifier. Training is managed by a `Trainer` with:
- **ModelCheckpoint** — saves the best model by `val_auroc`
- **EarlyStopping** — halts training if `val_auroc` stops improving
- **TensorBoardLogger** — logs metrics to `logs/`

### 5. Checkpoint
The best `.ckpt` is saved to `--ckpt_dir`. It can be used directly for inference or fine-tuning.

---

## Training

```bash
python -m src.script.train \
    --data_dir src/checkpoints/fp/ \
    --mode stft \
    --batch_size 64 \
    --max_epochs 50 \
    --patience 5 \
    --lr 1e-3 \
    --weight_decay 1e-5 \
    --log_dir logs \
    --ckpt_dir src/checkpoints/models/
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `src/checkpoints/fp/` | Path to preprocessed `.npz` fakeprints |
| `--mode` | `stft` | Transform type: `stft` or `cqt` |
| `--val_split` | `0.2` | Fraction of data held out for validation |
| `--use_convolution` / `--no-use_convolution` | `True` | Enable convolutional layers |
| `--use_norm` / `--no-use_norm` | `True` | Enable batch normalization |
| `--use_bias` / `--no-use_bias` | `True` | Enable bias terms |
| `--n_fft` | `16384` | FFT size |
| `--sampling_rate` | `48000` | Audio sample rate (Hz) |
| `--bins_per_octave` | `96` | CQT frequency resolution (CQT mode only) |
| `--freq_range` | `200 6000` | Frequency range in Hz, e.g. `--freq_range 200 6000` |
| `--batch_size` | `64` | Training batch size |
| `--max_epochs` | `50` | Maximum training epochs |
| `--patience` | `5` | Early stopping patience (epochs) |
| `--lr` | `1e-3` | Learning rate |
| `--weight_decay` | `1e-5` | AdamW weight decay |
| `--seed` | `42` | Random seed for reproducibility |
| `--log_dir` | `logs/` | TensorBoard log directory |
| `--ckpt_dir` | `src/checkpoints/models/` | Directory to save checkpoints |

### Monitoring training

```bash
tensorboard --logdir logs/
```

---

## Requirements

```bash
pip install torch torchaudio nnAudio lightning
```