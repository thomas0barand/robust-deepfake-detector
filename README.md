# robust-deepfake-detector

## Project structure

```
robust-deepfake-detector/
├── checkpoints/
│   ├── attack/
│   └── noattack/
├── data/
│   ├── research/
│   │   └── signals/
│   ├── suno_v5/
│   ├── test/
│   │   ├── attack/
│   │   │   ├── ai/
│   │   │   └── human/
│   │   └── noattack/
│   │       ├── ai/
│   │       └── human/
│   └── train/
│       ├── attack/
│       │   ├── ai/
│       │   └── human/
│       └── noattack/
│           ├── ai/
│           └── human/
├── outputs/
│   └── figures/
│       └── signals/
├── references/
│   └── deezer/
├── research/
│   ├── batch_crosscorr.py
│   ├── crosscorr.py
│   ├── show_lag.ipynb
│   ├── test_cqt.ipynb
│   └── test_pipeline.ipynb
├── scripts/
│   ├── attack/
│   │   ├── create_simple_signals.py
│   │   ├── resampling/
│   │   └── soxr/
│   │       ├── compute/
│   │       └── visualise/
│   ├── collect/
│   │   ├── download_fma.py
│   │   ├── download_sonics.py
│   │   ├── download_suno_5_gdrive.py
│   │   └── suno_scraping/
│   ├── preprocess/
│   │   ├── dataset_schema_v1.json
│   │   ├── manifest_pipeline.py
│   │   └── pipeline.py
│   ├── testing/
│   ├── training/
│   ├── visualize/
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
python scripts/collect/download_suno_5_gdrive.py --api-key your_key_here
```

5. Run the download script:

```bash
python scripts/collect/download_suno_5_gdrive.py -o data/suno_v5

# or a subset
python scripts/collect/download_suno_5_gdrive.py -o data/suno_v5 -n 1000
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

!python scripts/collect/suno_scraping/download.py \
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

