from huggingface_hub import snapshot_download
import zipfile
import os
import urllib.request


# Download FMA Small dataset
fma_dir = "data/fma_small"
os.makedirs(fma_dir, exist_ok=True)

fma_url = "https://os.unil.cloud.switch.ch/fma/fma_small.zip"
fma_zip_path = os.path.join("data", "fma_small.zip")

print("Downloading FMA Small dataset...")
urllib.request.urlretrieve(fma_url, fma_zip_path)

print("Extracting FMA Small dataset...")
with zipfile.ZipFile(fma_zip_path, 'r') as zip_ref:
    zip_ref.extractall("data")

print("Cleaning up...")
os.remove(fma_zip_path)
print("FMA Small dataset downloaded successfully!")
