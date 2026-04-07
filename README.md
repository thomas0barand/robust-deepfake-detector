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
```


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

## Dataset

### Download from Google Drive (local)

1. Make sure you have **~60 Go** of free disk space
2. Create a Google Cloud API key at [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)
3. Enable the Drive API at [console.cloud.google.com/apis/library/drive.googleapis.com](https://console.cloud.google.com/apis/library/drive.googleapis.com)
4. Set your API key (pick one):

```bash
# option A: env var
export GDRIVE_API_KEY=your_key_here

# option B: .env file
echo "GDRIVE_API_KEY=your_key_here" > .env
source .env

# option C: inline
python research/collect/download_suno_5_gdrive.py --api-key your_key_here
```

5. Run the download script:

```bash
python research/collect/download_suno_5_gdrive.py -o data/suno_v5

# or a subset
python research/collect/download_suno_5_gdrive.py -o data/suno_v5 -n 1000
```

### Upload to Google Drive (Colab)

Use a Colab notebook to download Suno songs directly into your Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/thomas0barand/robust-deepfake-detector.git
%cd robust-deepfake-detector/
!git checkout THOMAS/dataset
!pip install aiohttp tqdm

!python research/collect/suno_scraping/download.py \
  -i src/dataset/v5/suno_urls_v5.json \
  -o "/content/drive/MyDrive/Robust deepfake detector/data/suno_v5" \
  --limit 10000
```

The dataset is stored in this shared Drive folder:
[suno_v5 — Google Drive](https://drive.google.com/drive/folders/1jMrO05xSY4q9vDRHjjXh2iJcl8-H7g15?usp=sharing)
## Roles

- **checkpoints**: Model checkpoints for attack and no-attack settings.
- **data**: Raw and processed manifests, fakeprint datasets (train/test splits), audio files, and research signals.
- **references**: Reference code from Deezer.
- **research**: Exploratory scripts and notebooks (cross-correlation, CQT tests, pipeline tests).
- **scripts**: Data collection (FMA, Sonics, Suno scraping), preprocessing pipelines, attack pipelines (soxr resampling/speeding), training, testing, and visualization.
- **src**: Core source code — data loaders, model definitions (detector, linear, metrics), and fakeprint utilities.

