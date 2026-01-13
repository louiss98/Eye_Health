# Eye Health (Image Processing + Metrics Dashboard)

This repository contains a research-oriented pipeline for analyzing centered eye images and a companion static dashboard for reviewing the outputs.

The project has two parts:
- **Python pipeline**: detects the pupil and iris, computes basic quality metrics (including blur), and generates masks/overlays plus a consolidated `metrics.json` per image.
- **Static frontend** (`frontend/`): reads the generated files from `output/` and provides a lightweight UI to browse images and inspect metrics and overlays.

## Repository Layout

- `image_preprocess.py` - End-to-end processing pipeline (input → detection → metrics → output files).
- `blur_detection.py` - Blur scoring utilities (Laplacian/Tenegrad/Fourier + combined score).
- `iris_detection.py` - Iris boundary detection plus mask/overlay helpers.
- `data/images/` - Input images.
- `output/` - Generated outputs (created by running the pipeline).
- `frontend/` - Static web dashboard.

## Prerequisites

- Python 3.x
- Python packages used by the pipeline (notably `opencv-python` and `numpy`)

If dependencies are missing, install them in your environment (example):

```powershell
pip install opencv-python numpy
```

## Run the Pipeline

From the repository root (`C:\Development\Eye_Health`):

```powershell
python image_preprocess.py --input data/images --output output
```

Optional flags:
- `--max-dimension <px>` resizes the longest edge (default is 800).
- `--show` displays interactive overlay windows while processing.

## View Results in the Frontend

The frontend is file-based and expects to fetch assets from `data/` and `output/`, so serve the repository root.

```powershell
python -m http.server 8000
```

Open:
- `http://localhost:8000/frontend/`

Quick verification (optional):
- `http://localhost:8000/output/<image-stem>/metrics.json` should return JSON (not 404).

## Outputs

For each input image, the pipeline writes a folder under `output/<image-stem>/` containing:
- `metrics.json` - consolidated pupil/iris/blur metrics used by the UI
- `*_mask.png` - binary masks
- `*_overlay.png` - visualization overlays for qualitative review

## Notes

- The frontend is intentionally static (no backend) to support offline review and reproducible research runs.
- If you add new images to `data/images/`, update the `imageFiles` list in `frontend/app.js` so they appear in the UI.
