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

## Data Collection Strategy

Chosen approach:
- ingestion happens in notebooks
- notebooks export source-specific raw CSV files in a shared raw format
- `scripts/data/collection/manifest_pipeline.py` merges, normalizes, and deduplicates raw CSVs into one processed manifest

### Canonical schema

- JSON Schema: `scripts/data/collection/dataset_schema_v1.json`

### Collection components

- `scripts/data/collection/ingest_hf_dataset.ipynb`
- `scripts/data/collection/ingest_json_dataset.ipynb`
- `scripts/data/collection/manifest_pipeline.py`

### Workflow

```bash
# 1) In notebooks, export each source to a raw CSV.
#    - HF source -> ingest_hf_dataset.ipynb
#    - JSON source -> ingest_json_dataset.ipynb

# 2) Build the processed manifest from raw CSVs:
python scripts/data/collection/manifest_pipeline.py \
  --raw-input data/raw/hf_suno.csv \
  --raw-input data/raw/hf_udio.csv \
  --raw-input data/raw/suno_v5.csv \
  --output data/processed/manifest_v1.csv
```

The output manifest is deduplicated and normalized to schema version `v1`.

### Example notebook mapping (`nyuuzyou/suno`)

```python
from datasets import load_dataset
import json
import pandas as pd

ds = load_dataset("nyuuzyou/suno", split="train")
df = ds.to_pandas()

norm = pd.DataFrame(
    {
        "source": "hf_suno",
        "source_track_id": df["id"],
        "audio_uri": df["audio_url"],
        "title": df.get("title"),
        "artist": df.get("display_name"),
        "username": df.get("handle"),
        "generator": "suno",
        "generator_version": df.get("major_model_version"),
        "metadata": df.apply(
            lambda r: json.dumps(
                {
                    "model_name": r.get("model_name"),
                    "metadata_prompt": r.get("metadata_prompt"),
                    "metadata_tags": r.get("metadata_tags"),
                    "metadata_duration": r.get("metadata_duration"),
                },
                default=str,
            ),
            axis=1,
        ),
    }
)

norm.to_csv("data/raw/hf_suno.csv", index=False)
```



