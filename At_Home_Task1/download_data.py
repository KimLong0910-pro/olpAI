import os
import random
import subprocess
import zipfile
from pathlib import Path

# Install gdown if not available
try:
    import gdown
except ImportError:
    subprocess.run(["pip", "install", "gdown"])
    import gdown

# dataset.zip mirrors in your IOAI-2026/AudioClassifier/data folder. Each file ID has its
# own Google Drive download quota, so we try them in random order to survive heavy load.
MIRRORS = [
    "12XaRWYDsHq3eBwrgZ0QnEOffflS_dlff", # dataset.zip
    "1iUYDtKxXM1GSG5VNJ3r9847hqhWEiQRq", # dataset_m2.zip
    "112F9IFfn3_UtOPwSM1cGbuAL7Ci2aBP_", # dataset_m3.zip
]
random.shuffle(MIRRORS)
ZIP = "dataset.zip"

if not (os.path.exists(ZIP) and os.path.getsize(ZIP) > 10**9):
    for fid in MIRRORS:
        try:
            gdown.download(id=fid, output=ZIP, quiet=False)
        except Exception as e:
            print("mirror failed:", fid, "->", e)
        if os.path.exists(ZIP) and os.path.getsize(ZIP) > 10**9:
            break
    if os.path.exists(ZIP) and os.path.getsize(ZIP) > 10**9:
        pass
    else:
        raise RuntimeError("All dataset mirrors are rate-limited right now. "
                           "Wait ~30 min and re-run.")

# Extract dataset locally to ./dataset
print("Extracting dataset...")
os.makedirs("./dataset", exist_ok=True)
with zipfile.ZipFile(ZIP, 'r') as zip_ref:
    zip_ref.extractall("./dataset")
print("Extraction complete!")

LOCAL_DATA_DIR = Path("./dataset")

if not LOCAL_DATA_DIR.exists():
    raise FileNotFoundError(
        "Dataset not found. Please run the dataset download/unzip cell first."
    )

print("Dataset found:")
for i in LOCAL_DATA_DIR.iterdir():
    print(i)