# Flood mapping from satellite images with WorldFloods and ml4floods

Materials of the GeoMundus 2026 workshop *Geospatial Intelligence for Disaster Resilience* (16 October 2026).

Floods are frequent and devastating, and remote sensing has been used for decades to map them with classical
approaches. In the last decade, deep learning has opened possibilities for automatic, data-driven flood mapping.
This tutorial explores **WorldFloods**, a dataset for flood mapping and cloud masking in Sentinel-2 images, and
**ml4floods**, a Python package to process floods into ML-ready data, train deep learning models, evaluate them
and run inference.

## Notebooks

| Notebook | Content | Colab |
|---|---|---|
| `01a_optical` | Hands-on 1 · Sentinel-2: spectral signatures, water indices and thresholds | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kipoju/geomundus-ml4floods/blob/main/notebooks/01a_optical.ipynb) |
| `01b_sar` | Hands-on 1 · Sentinel-1: backscatter, Otsu thresholding and HAND | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kipoju/geomundus-ml4floods/blob/main/notebooks/01b_sar.ipynb) |
| `02a_training` | Hands-on 2 · Train your own flood segmentation model with ml4floods | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kipoju/geomundus-ml4floods/blob/main/notebooks/02a_training.ipynb) |
| `02b_inference` | Hands-on 2 · Map new floods with the WorldFloods pretrained model, export to GIS | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kipoju/geomundus-ml4floods/blob/main/notebooks/02b_inference.ipynb) |

## Running the notebooks

**Google Colab (recommended).** Click a badge above and run the first cell: it downloads the course code, data
and models and installs ml4floods (about 1 minute). If Colab disconnects, run the first cell again.
For training (`02a_training`), select a GPU: *Runtime → Change runtime type → T4 GPU*. Everything also runs on CPU,
only slower.

**Your own computer.** Unzip the course package (or clone this repository and unzip `geomundus_data.zip` in its
root folder) and create the environment with [conda](https://conda-forge.org/download/):

```bash
conda env create -f environment.yml
conda activate geomundus-floods
pip install --no-deps ml4floods==1.1.1
jupyter lab notebooks/
```

On macOS with Apple Silicon, set `accelerator="cpu"` when training.

## Contents

```
notebooks/      the four notebooks + course_utils.py (helper functions) + colab_setup.py
data/           WorldFloods v2 events (Sentinel-2, reference flood maps, permanent water), plus Sentinel-1 and HAND
                for some of them, organised as an ml4floods dataset (train / val / test)
models/         WF2_unetv2_bgriswirs (pretrained on WorldFloods v2) and course_unet (trained on the course data)
slides/         workshop slides (PDF)
```

## Data and credits

- **WorldFloods v2** — Sentinel-2 images and flood maps derived from Copernicus EMS, available at
  [huggingface.co/datasets/isp-uv-es/WorldFloodsv2](https://huggingface.co/datasets/isp-uv-es/WorldFloodsv2)
  (CC BY-NC 4.0). Portalés-Julià, E., Mateo-García, G., Purcell, C. & Gómez-Chova, L. *Global flood extent
  segmentation in optical satellite images.* Scientific Reports (2023).
- **ml4floods** — [github.com/spaceml-org/ml4floods](https://github.com/spaceml-org/ml4floods) (LGPL-3.0).
  Pretrained models at [huggingface.co/isp-uv-es/ml4floods](https://huggingface.co/isp-uv-es/ml4floods)
  (CC BY-NC 4.0). Mateo-Garcia, G. et al. *Towards global flood mapping onboard low cost satellites with machine
  learning.* Scientific Reports 11 (2021).
- **Sentinel-1 and Sentinel-2** — Copernicus data, Sentinel-1 GRD as preprocessed in Google Earth Engine.
- **HAND** — MERIT Hydro (Yamazaki et al., 2019).
- **Permanent water** — JRC Global Surface Water (Pekel et al., 2016).
- **Cloud masks** — CloudSEN12 (Aybar et al., 2022).
- **georeader** — [github.com/spaceml-org/georeader](https://github.com/spaceml-org/georeader).

Instructor: Enrique Portalés-Julià (Image Processing Laboratory, Universitat de València).
