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
                                                   monitored by f1 score
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

To process to .mp3 files into fakeprints, run:
```bash
export PYTHONPATH=$PYTHONPATH:.
python /scripts/preprocess/pipeline.py \
    --data_dir path/to/sunov5 \
    --out_dir data/train/attack/ai \
    --num_shards 10 \
    --shard_size 500 \
    --speed_up discrete \
```

This will save 500 fakeprints per shard, with a total of 10 shards (adjustable via `--num_shards` and `--shard_size`). The `--speed_up discrete` flag applies discrete speed changes to augment the dataset with speed variations, which can help improve model robustness.

You will find the output fakeprints in `data/train/attack/ai/` with both `stft` and `cqt` in a `.npz` format:
```
data/train/attack/ai/
├── fakeprints_01.npz
├── fakeprints_02.npz
└── ...
```

### 3. Dataset

`FakeprintDataset` reads the preprocessed `.npz` files from the output directory and serves `(fakeprint_tensor, label, speed_factor)` pairs to the DataLoader. A random train/val split is applied at training time.

### 4. Training

`RobustDetector` is a PyTorch Lightning module wrapping the classifier. Training is managed by a `Trainer` with:
- **ModelCheckpoint** — saves the best model by `f1_score`
- **EarlyStopping** — halts training if `val_f1_score` stops improving
- **TensorBoardLogger** — logs metrics to `logs/`

### 5. Checkpoint

The best `.ckpt` is saved to `--ckpt_dir`. It can be used directly for inference or fine-tuning.

---

## Training

To train the model with the resampled log-STFT fakeprints and convolution, run:

```bash
export PYTHONPATH=$PYTHONPATH:.
python scripts/training/train.py \
    --data_dir data/train/ \
    --mode stft \
    --use_convolution \
    --use_bias \
    --log_stft \
```

## Testing

To evaluate the best checkpoint on the test set, run:

```bash
export PYTHONPATH=$PYTHONPATH:.
python scripts/testing/test.py \
    --data_dir data/test/attack/ \
    --ckpt_path checkpoints/robustdetector-log_stft-use_conv.ckpt \
    --output_dir results/attack/ \
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--data_dir` | `data/train/attack/` | Path to train dataset containing `.npz` fakeprints |
| `--mode` | `stft` | Transform type: `stft` or `cqt` |
| `--log_stft` | `False` | Apply log transformation to STFT |
| `--val_split` | `0.1` | Fraction of data held out for validation |
| `--use_convolution` / `--no-use_convolution` | `True` | Enable convolutional layers |
| `--use_norm` / `--no-use_norm` | `True` | Enable batch normalization |
| `--use_bias` / `--no-use_bias` | `True` | Enable bias terms |
| `--n_fft` | `16384` | FFT size |
| `--sampling_rate` | `44100` | Audio sample rate (Hz) |
| `--bins_per_octave` | `192` | CQT frequency resolution (CQT mode only) |
| `--bins_per_octave_stft` | `1920` | log-STFT frequency resolution (STFT mode only) |
| `--freq_range` | `5000 16000` | Frequency range in Hz, e.g. `--freq_range 5000 16000` |
| `--batch_size` | `64` | Training batch size |
| `--max_epochs` | `50` | Maximum training epochs |
| `--patience` | `5` | Early stopping patience (epochs) |
| `--lr` | `1e-3` | Learning rate |
| `--weight_decay` | `1e-5` | AdamW weight decay |
| `--seed` | `42` | Random seed for reproducibility |
| `--log_dir` | `logs/` | TensorBoard log directory |
| `--ckpt_dir` | `checkpoints/` | Directory to save checkpoints |

### Monitoring training

```bash
tensorboard --logdir logs/
```

---

## Requirements

```bash
pip install torch nnAudio lightning
```