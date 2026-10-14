"""Set up the course in Google Colab: install ml4floods and download the course data and models.

Called from the first cell of every notebook. Safe to re-run: steps already done are skipped, so after a
Colab disconnection just run the first cell again.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RELEASE = "https://github.com/kipoju/geomundus-ml4floods/releases/download/data-v2"
DATA = {  # zip in the release -> a file it contains (if that file exists, the zip is not downloaded again)
    "geomundus_data.zip": "data/worldfloods_subset/train/floodmaps/EMSR429_AOI09_DEL_MONIT01.geojson",
}
OUR_FLOODS = "geomundus_our_floods.zip"   # images of the floods labelled in Hands-on 1 (notebook 02a)
ML4FLOODS = "ml4floods==1.1.1"


def pip_install(*args: str) -> None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)


def install() -> None:
    """ml4floods without its dependencies (they would pull Earth Engine, GCS... and touch Colab's PyTorch)
    + the few packages the course needs."""
    if importlib.util.find_spec("ml4floods") is not None:
        return
    print("Installing ml4floods and dependencies (~1 min)...")
    pip_install("-r", str(ROOT / "requirements.txt"))
    pip_install("--no-deps", ML4FLOODS)


def download_zip(url: str) -> None:
    """Download a zip and unzip it in the course folder."""
    zip_file = ROOT / url.rsplit("/", 1)[-1]
    print(f"Downloading {url} ...")
    try:
        urllib.request.urlretrieve(url, zip_file)
    except Exception as e:
        raise RuntimeError(f"Could not download {url} ({e}). Ask the instructor for the zip and "
                           f"unzip it in {ROOT}.") from e
    with zipfile.ZipFile(zip_file) as z:
        z.extractall(ROOT)
    zip_file.unlink()


def download_data() -> None:
    """Course data (data/) and models (models/), unzipped next to the notebooks."""
    for name, marker in DATA.items():
        if not (ROOT / marker).exists():
            download_zip(f"{RELEASE}/{name}")


def download_our_floods(labels_url: str) -> Path:
    """The floods we labelled in Hands-on 1: their images (once, from the course release) and our labels
    (every time: the last training set frozen in the labelling app). Returns the dataset folder."""
    data_dir = ROOT / "data" / "our_floods"
    if not (data_dir / "train" / "S2").exists():
        download_zip(f"{RELEASE}/{OUR_FLOODS}")
    shutil.rmtree(data_dir / "train" / "gt", ignore_errors=True)     # labels of a previous freeze
    shutil.rmtree(data_dir / "train" / "meta", ignore_errors=True)
    download_zip(f"{labels_url.rstrip('/')}/dataset/labels.zip")
    return data_dir


def setup() -> None:
    install()
    download_data()
    os.chdir(ROOT / "notebooks")
    if str(ROOT / "notebooks") not in sys.path:
        sys.path.insert(0, str(ROOT / "notebooks"))
    print(f"Ready. Course folder: {ROOT}")


if __name__ == "__main__":
    setup()
