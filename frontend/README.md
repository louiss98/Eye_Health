# Eye Health — Frontend (static demo)

This is a minimal static frontend showing a left-hand list of eye health metrics with circular icons and a right-side preview area.

Files:
- `index.html` — main page
- `styles.css` — styling
- `app.js` — simple metric population

Run locally (from repository root) using a simple HTTP server. Using `cmd.exe` on Windows:

```cmd
cd frontend
python -m http.server 8000
```

Then open `http://localhost:8000` in your browser.

Notes:
- This is a static demo; to integrate with the Python processing pipeline, add an API endpoint or serve generated overlay images into the `preview-area`.
- I can add a small file-picker UI to preview images from the `output/` folder if you'd like.