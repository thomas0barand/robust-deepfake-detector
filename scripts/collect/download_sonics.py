from huggingface_hub import snapshot_download
import zipfile
import os
import urllib.request



# Download SONICS dataset
local_dir = "data/sonics"
snapshot_download(repo_id="awsaf49/sonics", repo_type="dataset", local_dir=local_dir)


zip_path = os.path.join(local_dir, "part_01.zip")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(local_dir)

