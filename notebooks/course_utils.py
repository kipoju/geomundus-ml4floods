"""Shared helpers for the GeoMundus flood-mapping notebooks.

Figures for the slides are exported with `save_fig(fig, "fig_id")` only when the environment
variable EXPORT_FIGS=1 is set (so participants never write files by accident):

    EXPORT_FIGS=1 jupyter nbconvert --execute --to notebook --inplace notebooks/01_*.ipynb
"""
import json
import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from ml4floods.data.worldfloods.configs import BANDS_S2
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.morphology import dilation, footprint_rectangle, remove_small_objects
from skimage.util import view_as_blocks
from sklearn.metrics import confusion_matrix

IN_COLAB = "google.colab" in sys.modules
ROOT = Path(os.environ.get("COURSE_ROOT", Path(__file__).resolve().parents[1]))
FIG_DIR = ROOT / "figures"
EXPORT_FIGS = os.environ.get("EXPORT_FIGS", "0") == "1"
# Headless execution (nbconvert/nbclient) hangs on ipywidgets.interact -> `interact` below skips the widgets
HEADLESS = EXPORT_FIGS or os.environ.get("NB_HEADLESS", "0") == "1"

PRODUCTS = ("S2", "gt", "PERMANENTWATERJRC", "meta", "floodmaps", "S1", "S1meta", "HAND")
INDEX_CMAP = "BrBG"   # diverging: brown = dry, blue-green = water


def save_fig(fig, fig_id: str, dpi: int = 200) -> None:
    """Export a figure to figures/<fig_id>.png (only if EXPORT_FIGS=1)."""
    if not EXPORT_FIGS:
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{fig_id}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    print(f"saved figures/{fig_id}.png")


def interact(fn, **sliders):
    """ipywidgets.interact(fn, **sliders); when HEADLESS just runs fn with the sliders' initial values."""
    if HEADLESS:
        fn(**{k: getattr(w, "value", w) for k, w in sliders.items()})
        plt.show()
        return
    import ipywidgets
    ipywidgets.interact(fn, **sliders)


def list_events(data_dir) -> dict[str, list[str]]:
    """{split: [event, ...]} for a folder with the ml4floods layout."""
    data_dir = Path(data_dir)
    return {split.name: sorted(p.stem for p in (split / "S2").glob("*.tif"))
            for split in sorted(data_dir.iterdir()) if (split / "S2").exists()}


def read_product(data_dir, split: str, product: str, event: str, bands=None) -> GeoTensor:
    """Read one product of an event as a GeoTensor (bands: 0-based indexes, None = all)."""
    reader = RasterioReader(str(Path(data_dir) / split / product / f"{event}.tif"))
    if bands is not None:
        reader = reader.isel({"band": list(bands)})
    return reader.load()


def read_event(data_dir, split: str, event: str) -> dict:
    """Read S2 (13 bands, reflectance*1e4, float32), gt (2 bands), JRC permanent water and metadata.
    Rasters are GeoTensors sharing the S2 grid (S1 and HAND are added if they have been downloaded)."""
    base = Path(data_dir) / split
    s2 = read_product(data_dir, split, "S2", event, bands=range(len(BANDS_S2)))
    s2 = GeoTensor(s2.values.astype(np.float32), transform=s2.transform, crs=s2.crs,
                   fill_value_default=0)
    out = {"s2": s2,
           "gt": read_product(data_dir, split, "gt", event),
           "jrc": read_product(data_dir, split, "PERMANENTWATERJRC", event)[0],
           "meta": json.loads((base / "meta" / f"{event}.json").read_text()),
           "event": event, "split": split}
    if (base / "S1" / f"{event}.tif").exists():
        out["s1"] = read_product(data_dir, split, "S1", event)  # (VV, VH) in dB
        out["s1meta"] = json.loads((base / "S1meta" / f"{event}.json").read_text())
    if (base / "CLOUDSEN12" / f"{event}.tif").exists():  # CloudSEN12 cloud mask
        out["cloudsen12"] = read_product(data_dir, split, "CLOUDSEN12", event)[0]
    if (base / "HAND" / f"{event}.tif").exists():
        out["hand"] = read_product(data_dir, split, "HAND", event)[0]  # metres (MERIT Hydro)
    return out


def band(s2: np.ndarray, name: str) -> np.ndarray:
    return s2[BANDS_S2.index(name)]


def composite(s2: np.ndarray, bands=("B4", "B3", "B2"), max_val: float = 3000.) -> np.ndarray:
    """(H, W, 3) image in [0, 1] for plt.imshow. Use bands=("B11", "B8", "B4") for SWIR-NIR-Red."""
    return np.clip(np.stack([np.asarray(band(s2, b)) for b in bands], axis=-1) / max_val, 0, 1)


def mndwi(s2: np.ndarray) -> np.ndarray:
    """MNDWI = (Green - SWIR1) / (Green + SWIR1)."""
    g, swir = band(s2, "B3"), band(s2, "B11")
    return (g - swir) / (g + swir + 1e-6)


def awei_sh(s2: np.ndarray) -> np.ndarray:
    """AWEI_sh (Feyisa et al., 2014) = Blue + 2.5 Green - 1.5 (NIR + SWIR1) - 0.25 SWIR2 (reflectance)."""
    b, g, nir, sw1, sw2 = (band(s2, x) / 10_000 for x in ("B2", "B3", "B8", "B11", "B12"))
    return b + 2.5 * g - 1.5 * (nir + sw1) - 0.25 * sw2


def ndwi(s2: np.ndarray) -> np.ndarray:
    """NDWI (McFeeters) = (Green - NIR) / (Green + NIR)."""
    g, nir = band(s2, "B3"), band(s2, "B8")
    return (g - nir) / (g + nir + 1e-6)


CONFUSION_COLORS = {"TN (land)": (0.78, 0.78, 0.78), "TP (water)": (0.0, 0.13, 0.42),
                    "FN (missed water)": (0.10, 0.65, 0.20), "FP (false alarm)": (0.85, 0.10, 0.10),
                    "invalid / cloud": (1.0, 1.0, 1.0)}


def confusion_overlay(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """RGB image: TN grey, TP navy, FN green, FP red, invalid white (legend: `confusion_legend`)."""
    pred, ref, valid = np.asarray(pred), np.asarray(ref), np.asarray(valid)
    c = list(CONFUSION_COLORS.values())
    out = np.empty(pred.shape + (3,), dtype=np.float32)
    out[...] = c[0]
    out[pred & ref] = c[1]
    out[~pred & ref] = c[2]
    out[pred & ~ref] = c[3]
    out[~valid] = c[4]
    return out


def confusion_legend(ax=None, **kwargs):
    """Legend with the colours of `confusion_overlay`."""
    from matplotlib.patches import Patch
    ax = ax or plt.gca()
    handles = [Patch(facecolor=col, edgecolor="k", lw=0.5, label=k) for k, col in CONFUSION_COLORS.items()]
    return ax.legend(handles=handles, **{"loc": "lower left", "fontsize": 8, "framealpha": 0.9, **kwargs})


def binary_metrics(pred: np.ndarray, ref: np.ndarray, valid: np.ndarray) -> dict:
    """Precision / recall / IoU / overall accuracy of the water class on valid pixels (sklearn.metrics)."""
    valid = np.asarray(valid)
    tn, fp, fn, tp = confusion_matrix(np.asarray(ref)[valid], np.asarray(pred)[valid], labels=[False, True]).ravel()
    return {"precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / (tp + fn) if tp + fn else 0.,
            "IoU": tp / (tp + fp + fn) if tp + fp + fn else 0.,
            "OA": (tp + tn) / (tp + tn + fp + fn)}


# ----------------------------------------------------------------------------------------------
# Events, S1 and HAND
# ----------------------------------------------------------------------------------------------
EVENT = "EMSR271_02FARKADONA_DEL_v1"  # common thread of the workshop (Thessaly, Greece, Feb 2018)
# Red box (UTM 34N): small area around the flooded river where VV is bimodal
BOX_BOUNDS = (588725, 4377599, 598783, 4383196)
BOX_CRS = "EPSG:32634"
S2_WAVELENGTHS = [443, 490, 560, 665, 705, 740, 783, 842, 865, 940, 1375, 1610, 2190]  # nm, order of BANDS_S2


def ml4floods_split(data_dir, dst_dir=None, windows: bool = True, split_file: str = "traintestval.json") -> Path:
    """Make the course split usable by ml4floods wherever the data were unzipped (local or Colab).

    data_dir holds `traintestval.json` and `windows_traintestval.json` with paths relative to data_dir.
    ml4floods needs absolute paths, so both are rewritten in dst_dir
    (default: .cache/ml4floods_split). Returns the split file for `data_params.train_test_split_file`;
    set `data_params.path_to_splits` to its folder so that `get_dataset` loads the precomputed windows."""
    data_dir = Path(data_dir).resolve()
    dst_dir = Path(dst_dir) if dst_dir is not None else ROOT / ".cache" / "ml4floods_split"
    dst_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads((data_dir / split_file).read_text())
    split = {s: {folder: [str(data_dir / f) for f in files] for folder, files in folders.items()}
             for s, folders in split.items()}
    (dst_dir / split_file).write_text(json.dumps(split, indent=2))
    if windows:
        win = json.loads((data_dir / f"windows_{split_file}").read_text())
        for w in win["slices"]:
            w["file_name"] = str(data_dir / w["file_name"])
        (dst_dir / f"windows_{split_file}").write_text(json.dumps(win))
    return dst_dir / split_file


def find_event(event: str = EVENT) -> tuple[Path, str]:
    """(data_dir, split) of an event: the course subset if it exists, else the candidate events."""
    for data_dir in (ROOT / "data" / "worldfloods_subset", ROOT / "data" / "worldfloods_candidates"):
        for split in ("test", "val", "train"):
            if (data_dir / split / "S2" / f"{event}.tif").exists():
                return data_dir, split
    raise FileNotFoundError(f"{event} not found under {ROOT / 'data'}")


# ----------------------------------------------------------------------------------------------
# SAR thresholding: Edge Otsu, BMax Otsu (plain Otsu = skimage.filters.threshold_otsu)
# Edge Otsu and BMax Otsu follow the reference implementation of Markert et al. (2020),
# https://doi.org/10.3390/rs12152469 (hydrafloods `thresholding.py`), ported from Earth Engine to numpy.
# ----------------------------------------------------------------------------------------------
def bmax(values: np.ndarray, t0: float) -> float:
    """Normalised between-class variance with the split at t0: p1·p2·(m1-m2)² / σ²  (0..1).
    High (> ~0.75) = the two classes are well separated = bimodal histogram."""
    values = np.asarray(values).ravel()
    w = values < t0
    p1 = w.mean()
    if p1 == 0 or p1 == 1:
        return 0.
    return float(p1 * (1 - p1) * (values[w].mean() - values[~w].mean()) ** 2 / values.var())


def edge_otsu(img: np.ndarray, t0: float = -16., canny_sigma: float = 1., edge_length: int = 50,
              buffer_px: int = 10) -> tuple[float, np.ndarray]:
    """Edge Otsu: Otsu only on pixels close to water/land edges.
    1. initial water mask img < t0;  2. Canny edges of that mask;  3. drop edges shorter than
    `edge_length` px;  4. buffer the edges `buffer_px` px per side (100 m at 10 m);  5. Otsu there.
    Returns (threshold, buffered-edges mask)."""
    img = np.asarray(img)
    edges = canny((img < t0).astype(np.float32), sigma=canny_sigma)
    # keep the edges with >= edge_length connected pixels (max_size = largest size that is removed)
    edges = remove_small_objects(edges, max_size=edge_length - 1, connectivity=2)
    buffered = dilation(edges, footprint_rectangle((2 * buffer_px + 1, 2 * buffer_px + 1)))
    return float(threshold_otsu(img[buffered])), buffered


def bmax_otsu(img: np.ndarray, t0: float = -16., tile: int = 128, bmax_threshold: float = 0.75
              ) -> tuple[float, np.ndarray, np.ndarray]:
    """BMax Otsu: split the image in tiles, keep the tiles whose histogram is bimodal
    (BMax > bmax_threshold) and run Otsu on the pixels of those tiles.
    Returns (threshold, per-tile BMax array, selected-pixels mask)."""
    img = np.asarray(img)
    nr, nc = img.shape[0] // tile, img.shape[1] // tile
    tiles = view_as_blocks(img[:nr * tile, :nc * tile], (tile, tile))   # (nr, nc, tile, tile); border dropped
    bm = np.array([[bmax(t, t0) for t in row] for row in tiles])
    if not (bm > bmax_threshold).any():
        raise ValueError(f"no tile with BMax > {bmax_threshold}: try smaller tiles or another t0")
    selected = np.zeros(img.shape, dtype=bool)
    selected[:nr * tile, :nc * tile] = np.kron(bm > bmax_threshold, np.ones((tile, tile), dtype=bool))  # tile -> px
    return float(threshold_otsu(img[selected])), bm, selected


def hist_by_class(values: np.ndarray, water: np.ndarray, land: np.ndarray, t: float | None = None,
                  ax=None, bins=np.linspace(-30, 5, 141), xlabel: str = r"VV backscatter $\sigma^0$ [dB]"):
    """Overlaid histograms of the reference water (blue) and land (orange) pixels + threshold line."""
    ax = ax or plt.gca()
    values = np.asarray(values)
    ax.hist(values[np.asarray(land)], bins=bins, color="tab:orange", alpha=0.6, label="land (reference)")
    ax.hist(values[np.asarray(water)], bins=bins, color="tab:blue", alpha=0.6, label="water (reference)")
    if t is not None:
        ax.axvline(t, color="k", ls="--", lw=1.5, label=f"threshold = {t:.1f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("pixels")
    ax.legend()
    return ax


# ----------------------------------------------------------------------------- learning curves
# ml4floods logs the training loss only every 100 batches (weighted with pos_weight) and the validation loss
# without weights → the two numbers are not comparable. This callback computes the same unweighted loss on
# every training batch (model in eval mode, so BatchNorm statistics are not touched) and stores it next to
# the validation loss, one row per epoch, in <csv_path>.
def learning_curve_callback(csv_path):
    import pandas as pd
    import torch
    from ml4floods.models.utils import losses
    from pytorch_lightning import Callback

    class LearningCurve(Callback):
        def __init__(self):
            self.rows, self.sum, self.n = [], 0.0, 0

        def on_train_epoch_start(self, trainer, pl_module):
            self.sum, self.n = 0.0, 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            pl_module.eval()
            with torch.no_grad():
                logits = pl_module(batch["image"])
                self.sum += float(losses.calc_loss_multioutput_logistic_mask_invalid(logits, batch["mask"]))
                self.n += 1
            pl_module.train()

        def on_train_epoch_end(self, trainer, pl_module):   # runs after the validation of the epoch
            val = trainer.callback_metrics.get("val_bce_loss")
            self.rows.append({"epoch": trainer.current_epoch + 1, "train_loss": self.sum / max(self.n, 1),
                              "val_loss": float(val) if val is not None else float("nan")})
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(self.rows).to_csv(csv_path, index=False)

    return LearningCurve()


def plot_learning_curve(csv_path, ax=None, title="", val_label="validation loss"):
    """Training vs validation loss per epoch (CSV written by learning_curve_callback)."""
    import pandas as pd
    log = pd.read_csv(csv_path)
    ax = ax or plt.gca()
    ax.plot(log["epoch"], log["train_loss"], "o-", color="tab:blue", label="training loss (training floods)")
    ax.plot(log["epoch"], log["val_loss"], "s-", color="tab:orange", label=val_label)
    ax.set(xlabel="epoch", ylabel="loss (binary cross-entropy)", title=title)
    ax.set_xticks(log["epoch"])
    ax.grid(alpha=0.3); ax.legend()
    return ax
