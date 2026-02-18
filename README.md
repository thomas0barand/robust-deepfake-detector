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
python scripts/data/get_dataset.py --api-key your_key_here
```

5. Run the download script:

```bash
python scripts/data/get_dataset.py -o data/suno_v5

# or a subset
python scripts/data/get_dataset.py -o data/suno_v5 -n 1000
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

!python scripts/data/scraping/download.py \
  -i src/dataset/v5/suno_urls_v5.json \
  -o "/content/drive/MyDrive/Robust deepfake detector/data/suno_v5" \
  --limit 10000
```

The dataset is stored in this shared Drive folder:
[suno_v5 — Google Drive](https://drive.google.com/drive/folders/1jMrO05xSY4q9vDRHjjXh2iJcl8-H7g15?usp=sharing)
## Roles

- **data**: Raw and processed audio: AI vs human tracks, plus synthetic signals for resampling/speed attacks.
- **deezer**: Fakeprint extraction (EnCodec), dataset splits (Sonics), regressor training and evaluation; notebook for latent viz.
- **scripts**: Data download (FMA, Sonics), attack pipelines (soxr resampling/speeding, visualisations), training helpers; scraping pipeline (planned).
- **src**: Persisted fakeprints and trained model artifacts (weights, sonics-vs-fma).
