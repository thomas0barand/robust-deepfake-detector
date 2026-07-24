# robust-deepfake-detector

## Setup


### Simple setup

Setup a Python 3.13 virtual environment, activate it and install packages listed in `pyproject.toml`:

```bash
pip install -e .
```

To ensure reproducibility, we kept the lockfile `requirements.lock` with all the pinned dependencies.

### Dev setup
If you need to run the preprocessing pipeline for the pitch shift attack, you will need to install the following dependency:
- rubberband [https://breakfastquay.com/rubberband/]: software library for audio time-stretching and pitch-shifting

## Project structure

```
robust-deepfake-detector/
├── checkpoints/
│   ├── attack/
│   └── noattack/
├── data/
│   ├── test/
│   └── train/
├── research/
│   ├── collect/
│   ├── notebooks/
│   └── preprocess/
├── scripts/
│   ├── pipeline.py
│   ├── preprocessing_pipeline.ipynb
│   ├── test.py
│   ├── train.py
│   ├── visualize_weights.py
│   └── utils.py
└── src/
    ├── data/
    │   └── dataset.py
    ├── models/
    │   ├── detector.py
    │   ├── linear.py
    │   └── metrics.py
    └── utils/
        └── fakeprints.py
```

## Scripts

See the [readme](/scripts/README.md) in the `scripts/` directory for details on the training, testing, and visualization scripts.

## Roles

- **checkpoints**: Model checkpoints for attack and no-attack settings.
- **data**: Raw and processed manifests, fakeprint datasets (train/test splits), audio files, and research signals.
- **references**: Reference code from Deezer.
- **research**: Exploratory scripts and notebooks (cross-correlation, CQT tests, pipeline tests).
- **scripts**: Data collection (FMA, Sonics, Suno scraping), preprocessing pipelines, attack pipelines (soxr resampling/speeding), training, testing, and visualization.
- **src**: Core source code — data loaders, model definitions (detector, linear, metrics), and fakeprint utilities.

