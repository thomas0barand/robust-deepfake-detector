# robust-deepfake-detector

## Project structure

```
robust-deepfake-detector/
├── data/
│   ├── ai/
│   ├── human/
│   └── signals/
├── deezer/
│   ├── sonics/
│   ├── compute_fakeprints.py
│   ├── train_test_regressor.py
│   └── encodec_latent_visualisation.ipynb
├── outputs/
│   └── figures/
├── scripts/
│   ├── attack/
│   │   ├── create_simple_signals.py
│   │   ├── resampling/
│   │   └── soxr/
│   ├── data/
│   ├── scraping/
│   ├── training/
│   └── utils.py
└── src/
    ├── fp/
    └── models/
```

## Roles

- **data**: Raw and processed audio: AI vs human tracks, plus synthetic signals for resampling/speed attacks.
- **deezer**: Fakeprint extraction (EnCodec), dataset splits (Sonics), regressor training and evaluation; notebook for latent viz.
- **scripts**: Data download (FMA, Sonics), attack pipelines (soxr resampling/speeding, visualisations), training helpers; scraping pipeline (planned).
- **src**: Persisted fakeprints and trained model artifacts (weights, sonics-vs-fma).
