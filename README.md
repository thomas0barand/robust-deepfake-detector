# robust-deepfake-detector

```
robust-deepfake-detector/
├── data/
│   ├── ai/
│   └── human/
├── deezer/
│   ├── compute_fakeprints.py
│   ├── encodec_latent_visualisation.ipynb
│   ├── train_test_regressor.py
│   └── sonics/
│       ├── create_splits.py
│       └── sonics_split.npy
├── outputs/
│   └── figures/
├── scripts/
│   ├── attack/
│   ├── data/
│   │   ├── download_fma.py
│   │   └── download_sonics.py
│   ├── training/
│   │   └── visualize_weights.py
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