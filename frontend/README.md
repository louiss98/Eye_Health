# Eye Health Frontend (Static Dashboard)

This directory contains a lightweight, static web dashboard for reviewing eye-image metrics produced by the preprocessing pipeline in this repository. The UI presents a collection of eye images, a metrics panel (pupil/iris/blur), and an overlay preview when available.

## Contents

- `index.html` - Main page
- `styles.css` - Styling
- `app.js` - Client-side logic (loads images, fetches `metrics.json`, renders overlays)

## Run Locally

Because the frontend fetches assets from `data/` and `output/`, serve the repository root (not just the `frontend/` directory).

From the repository root (`C:\Development\Eye_Health`):

```powershell
python -m http.server 8000
```

Then open:

- `http://localhost:8000/frontend/`

Quick verification (optional):

- `http://localhost:8000/output/<image-stem>/metrics.json` should return JSON (not 404).

## Generate Pipeline Outputs

The UI expects preprocessing outputs under `output/<image-stem>/`, including `metrics.json` and overlay images (e.g., `combined_overlay.png`).

To (re)generate outputs:

```powershell
python image_preprocess.py --input data/images --output output
```

If you add new images to `data/images/`, update the `imageFiles` list in `frontend/app.js` to include them.

## Notes

- This frontend is intentionally static (no backend). It is suitable for local review, demonstrations, and research workflows where outputs are generated offline and served as files.
- For integration into a larger research system, consider adding an API layer that serves metrics and assets from a study database or storage service.
