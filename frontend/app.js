// Image sources (sync with data/images)
const imageFiles = ['mobile-eye-damage.jpg'];

// Where to probe for assets depending on where the static server is run
const imageBaseCandidates = ['../data/images', 'images'];
const metricsBaseCandidates = ['../output', 'output'];

// Expected iris ratio range (mirrors defaults in iris_detection.IrisConfig)
const irisRatioBounds = { min: 1.5, max: 5.0 };

const state = {
  imageBase: imageBaseCandidates[imageBaseCandidates.length - 1],
  metricsBase: metricsBaseCandidates[metricsBaseCandidates.length - 1],
  metricsCache: new Map(),
  colorCache: new Map(),
  activeImage: null,
  activeMetricId: null,
};

window.addEventListener('DOMContentLoaded', async () => {
  setPreviewMessage('Select an eye to view metrics.');

  const basePath = imageFiles.length
    ? await resolveBasePath(imageBaseCandidates, imageFiles[0])
    : imageBaseCandidates[imageBaseCandidates.length - 1];

  state.imageBase = basePath;
  buildImageCollection(imageFiles, basePath);
});

/**
 * Changes the background image with a fade transition.
 * @param {string} url - The URL of the new image.
 */
function setBackgroundImage(url) {
  const layer = document.getElementById('bg-image-layer');

  if (!layer) return;
  if (!url) {
    layer.style.opacity = 0;
    layer.style.backgroundImage = 'none';
    return;
  }

  layer.style.opacity = 0;
  setTimeout(() => {
    layer.style.backgroundImage = `url('${url}')`;
    requestAnimationFrame(() => {
      layer.style.opacity = 1;
    });
  }, 1000);
}

/**
 * Updates the background blur amount.
 * @param {number} px - Blur radius in pixels.
 */
function setBackgroundBlur(px) {
  document.documentElement.style.setProperty('--bg-blur', `${px}px`);
}

/**
 * Updates the background tint color.
 * @param {string} color - CSS color string (e.g., 'rgba(0, 50, 0, 0.5)').
 */
function setBackgroundTint(color) {
  document.documentElement.style.setProperty('--bg-tint', color);
}

/**
 * Resolve a base path by checking which candidate can serve a test file.
 * @param {string[]} candidates - Base paths to probe.
 * @param {string} testPath - File to request under each base.
 * @returns {Promise<string>} Resolved base path.
 */
async function resolveBasePath(candidates, testPath) {
  if (!window.fetch) return candidates[candidates.length - 1];

  for (const base of candidates) {
    const url = `${base}/${testPath}`;
    try {
      const response = await fetch(url, { method: 'HEAD' });
      if (response.ok) return base;
    } catch (err) {
      // Ignore and try the next candidate
    }
  }
  return candidates[candidates.length - 1];
}

/**
 * Builds the horizontal image collection strip.
 * @param {string[]} files - Image file names.
 * @param {string} basePath - Base path for image URLs.
 */
function buildImageCollection(files, basePath) {
  const track = document.getElementById('image-collection');
  if (!track) return;

  track.innerHTML = '';

  for (const file of files) {
    const button = document.createElement('button');
    const label = filenameToLabel(file);
    const src = `${basePath}/${file}`;
    const stem = getStem(file);

    button.type = 'button';
    button.className = 'collection-item';
    button.style.backgroundImage = `url('${src}')`;
    button.setAttribute('role', 'option');
    button.setAttribute('aria-selected', 'false');
    button.setAttribute('aria-label', label);
    button.title = label;
    button.dataset.src = src;
    button.dataset.stem = stem;
    button.dataset.label = label;

    button.addEventListener('click', () => {
      selectCollectionItem(button, track);
    });

    track.appendChild(button);
  }

  const first = track.querySelector('.collection-item');
  if (first) {
    selectCollectionItem(first, track, false);
  }
}

/**
 * Selects a collection item, updates background, and loads metrics.
 * @param {HTMLButtonElement} button - The selected button.
 * @param {HTMLElement} track - The collection container.
 * @param {boolean} focus - Whether to focus the button.
 */
function selectCollectionItem(button, track, focus = true) {
  const items = track.querySelectorAll('.collection-item');
  items.forEach((item) => {
    item.classList.remove('is-active');
    item.setAttribute('aria-selected', 'false');
  });

  button.classList.add('is-active');
  button.setAttribute('aria-selected', 'true');
  if (focus) {
    button.focus({ preventScroll: true });
  }

  const stem = button.dataset.stem;
  const label = button.dataset.label;
  state.activeImage = stem;

  setBackgroundImage(button.dataset.src);
  loadMetricsForImage(stem, label).catch((err) => {
    setPreviewMessage(`Could not load metrics for ${label || stem}: ${err.message || err}`);
  });
}

/**
 * Fetch and render metrics for a given image stem.
 * @param {string} stem - Image name without extension.
 * @param {string} label - Human-friendly label.
 */
async function loadMetricsForImage(stem, label) {
  setMetricsLoading(label);
  setPreviewMessage(`Loading metrics for ${label || stem}...`);

  const cachedMetrics = state.metricsCache.get(stem) || null;
  const cachedColors = state.colorCache.get(stem) || null;
  if (cachedMetrics) {
    renderMetrics(stem, cachedMetrics, cachedColors);
    if (!cachedColors) {
      computeColorsForImage(stem, cachedMetrics)
        .then((colors) => {
          if (colors) {
            state.colorCache.set(stem, colors);
            renderMetrics(stem, cachedMetrics, colors);
          }
        })
        .catch(() => {});
    }
    return;
  }

  const metricsBase = await resolveBasePath(metricsBaseCandidates, `${stem}/metrics.json`);
  const url = `${metricsBase}/${stem}/metrics.json`;

  let payload;
  try {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    payload = await response.json();
  } catch (err) {
    renderMetrics(stem, null, null, `Could not load metrics (${err.message || err}). Looked for ${url}`);
    return;
  }

  state.metricsBase = metricsBase;
  state.metricsCache.set(stem, payload);

  let colors = null;
  try {
    colors = await computeColorsForImage(stem, payload);
    if (colors) state.colorCache.set(stem, colors);
  } catch (err) {
    colors = null;
  }

  renderMetrics(stem, payload, colors);
}

/**
 * Populate the metrics list UI.
 * @param {string} stem - Image name without extension.
 * @param {object|null} data - Metrics payload.
 * @param {object|null} colors - Sampled colors.
 * @param {string} [errorMessage] - Optional error text.
 */
function renderMetrics(stem, data, colors, errorMessage) {
  const list = document.getElementById('metrics-list');
  if (!list) return;

  list.innerHTML = '';

  if (errorMessage || !data) {
    const item = document.createElement('li');
    item.className = 'metric-item';
    item.innerHTML = `<div class="metric-body"><div class="metric-title">No metrics</div><div class="metric-sub">${errorMessage || 'Run the preprocessing pipeline first.'}</div></div>`;
    list.appendChild(item);
    setPreviewMessage(errorMessage || 'No metrics available.');
    return;
  }

  const items = buildMetricItems(data, colors);
  if (!items.length) {
    const empty = document.createElement('li');
    empty.className = 'metric-item';
    empty.innerHTML = `<div class="metric-body"><div class="metric-title">No metrics</div><div class="metric-sub">Metrics file is missing expected fields.</div></div>`;
    list.appendChild(empty);
    setPreviewMessage('Metrics file is missing expected fields.');
    return;
  }

  let targetMetric = items[0];
  if (state.activeMetricId) {
    const found = items.find((m) => m.id === state.activeMetricId);
    if (found) targetMetric = found;
  }

  for (const item of items) {
    const li = document.createElement('li');
    li.className = 'metric-item';
    if (item.id === targetMetric.id) {
      li.classList.add('is-active');
    }

    const icon = document.createElement('div');
    icon.className = 'metric-icon';
    if (item.color) {
      icon.style.background = item.color;
    } else {
      if (item.status === 'warn') icon.classList.add('icon-warn');
      if (item.status === 'neutral') icon.classList.add('icon-neutral');
    }

    const body = document.createElement('div');
    body.className = 'metric-body';

    const title = document.createElement('div');
    title.className = 'metric-title';
    title.textContent = item.title;

    const sub = document.createElement('div');
    sub.className = 'metric-sub';
    sub.textContent = item.sub;

    body.appendChild(title);
    body.appendChild(sub);

    li.appendChild(icon);
    li.appendChild(body);
    li.addEventListener('click', () => {
      state.activeMetricId = item.id;
      list.querySelectorAll('.metric-item').forEach((node) => node.classList.remove('is-active'));
      li.classList.add('is-active');
      renderPreview(item, data, colors, stem);
    });

    list.appendChild(li);
  }

  renderPreview(targetMetric, data, colors, stem);
}

/**
 * Build metric entries from a metrics.json payload.
 * @param {object} data - Parsed metrics JSON.
 * @param {object|null} colors - Sampled colors.
 * @returns {Array} Metric item definitions.
 */
function buildMetricItems(data, colors) {
  const items = [];
  const blur = data.blur || {};
  const pupil = data.pupil || {};
  const iris = data.iris || null;
  const colorsSafe = colors || {};

  if (isNumber(blur.combined_score)) {
    items.push({
      id: 'blur',
      title: 'Blurriness',
      sub: `${formatNumber(blur.combined_score, 1)} / ${formatNumber(blur.blur_threshold || 100, 0)}`,
      status: blur.is_blurry ? 'warn' : undefined,
      color: 'linear-gradient(180deg,#fbbf24,#f59e0b)',
      colorKey: 'blur',
      details: [
        ['Status', blur.is_blurry ? 'Blurry' : 'Clear'],
        ['Combined score', formatNumber(blur.combined_score, 2)],
        ['Threshold', formatNumber(blur.blur_threshold, 1)],
        ['Laplacian variance', formatNumber(blur.laplacian_variance, 2)],
        ['Tenegrad energy', formatNumber(blur.tenegrad_score, 2)],
        ['Fourier score', formatNumber(blur.fourier_score, 4)],
      ],
      extra: {
        label: 'Blur sharpness',
        value: blur.combined_score,
        threshold: blur.blur_threshold,
        status: blur.is_blurry ? 'Blurry' : 'Clear',
      },
    });
  }

  // Pupil component
  if (isNumber(pupil.radius_px)) {
    items.push({
      id: 'pupil',
      title: 'Pupil',
      sub: `${formatNumber(pupil.radius_px, 0)} px radius`,
      status: blur.is_blurry ? 'warn' : undefined,
      color: colorsSafe.pupil || 'linear-gradient(180deg,var(--success),#059669)',
      colorKey: 'pupil',
      details: [
        ['Centre', isArray(pupil.centre) ? `(${pupil.centre[0]}, ${pupil.centre[1]})` : '—'],
        ['Radius (px)', formatNumber(pupil.radius_px, 0)],
        ['Detection score', formatNumber(pupil.score, 2)],
      ],
      extra: blur && isNumber(blur.combined_score)
        ? {
            label: 'Blur sharpness',
            value: blur.combined_score,
            threshold: blur.blur_threshold,
            status: blur.is_blurry ? 'Blurry' : 'Clear',
          }
        : null,
    });
  }

  // Iris component
  if (iris && isNumber(iris.radius_px)) {
    const warn = iris.pupil_ratio < irisRatioBounds.min || iris.pupil_ratio > irisRatioBounds.max;
    items.push({
      id: 'iris',
      title: 'Iris',
      sub: `${formatNumber(iris.radius_px, 0)} px radius`,
      status: warn ? 'warn' : undefined,
      color: colorsSafe.iris || 'linear-gradient(180deg,#60a5fa,#2563eb)',
      colorKey: 'iris',
      details: [
        ['Centre', isArray(iris.centre) ? `(${iris.centre[0]}, ${iris.centre[1]})` : '—'],
        ['Radius (px)', formatNumber(iris.radius_px, 0)],
        ['Fit score', formatNumber(iris.score, 2)],
        ['Edge points', isNumber(iris.edge_points) ? iris.edge_points : '—'],
        ['Iris/pupil ratio', isNumber(iris.pupil_ratio) ? `${formatNumber(iris.pupil_ratio, 2)} : 1` : '—'],
      ],
      ratio: isNumber(iris.pupil_ratio)
        ? {
            value: iris.pupil_ratio,
            min: irisRatioBounds.min,
            max: irisRatioBounds.max,
          }
        : null,
    });
  }

  // Sclera component (color only for now)
  items.push({
    id: 'sclera',
    title: 'Sclera',
    sub: colorsSafe.sclera ? 'Color sampled' : 'No metrics yet',
    color: colorsSafe.sclera || 'linear-gradient(180deg,#f5f5f5,#d1d5db)',
    colorKey: 'sclera',
    details: [
      ['Notes', 'Sclera metrics not computed in pipeline yet.'],
    ],
  });

  return items;
}

/**
 * Render detailed preview for the selected metric.
 * @param {object} metric - Selected metric item.
 * @param {object} data - Metrics payload.
 * @param {object|null} colors - Sampled colors.
 * @param {string} stem - Image name without extension.
 */
function renderPreview(metric, data, colors, stem) {
  const preview = document.getElementById('preview-area');
  if (!preview) return;

  if (!metric) {
    setPreviewMessage('Select a metric to view details.');
    return;
  }

  const overlaySrc = resolveOverlayPath(data.files, stem);
  const details = metric.details || [];
  const imageLabel = filenameToLabel(data.image || stem);
  const colorKey = metric.colorKey || metric.id;
  const chipColor = colors && colors[colorKey] ? colors[colorKey] : metric.color;

  const detailRows = details
    .filter((row) => Array.isArray(row) && row.length === 2)
    .map(
      ([label, value]) =>
        `<div class="preview-field"><span class="preview-key">${label}</span><span class="preview-value">${value ?? '—'}</span></div>`
    )
    .join('');

  const colorRow = chipColor
    ? `<div class="preview-color">
        <span class="preview-key">Color</span>
        <span class="preview-swatch" style="background:${chipColor}"></span>
        <span class="preview-value">${chipColor}</span>
      </div>`
    : '';

  const ratioRow =
    metric.ratio && metric.ratio.value
      ? buildProgressRow(
          'Iris/Pupil Ratio',
          metric.ratio.value,
          metric.ratio.min,
          metric.ratio.max,
          chipColor || '#60a5fa'
        )
      : '';

  const blurRow = metric.extra
    ? buildProgressRow(
        metric.extra.label,
        metric.extra.value,
        0,
        metric.extra.threshold || 100,
        chipColor || '#34d399',
        metric.extra.status
      )
    : '';

  preview.innerHTML = `
    <div class="preview-body">
      <div class="preview-head">
        <div class="preview-title">${metric.title}</div>
        <div class="preview-sub">${imageLabel}</div>
      </div>
      <div class="preview-highlight">${metric.sub || ''}</div>
      ${colorRow}
      ${ratioRow}
      ${blurRow}
      <div class="preview-grid">${detailRows}</div>
      ${overlaySrc ? `<div class="preview-image-wrap"><img src="${overlaySrc}" alt="Overlay for ${imageLabel}" loading="lazy"></div>` : ''}
    </div>
  `;
}

/**
 * Show a loading row while metrics are fetched.
 * @param {string} label - Label for the image.
 */
function setMetricsLoading(label) {
  const list = document.getElementById('metrics-list');
  if (!list) return;
  list.innerHTML = `<li class="metric-item"><div class="metric-body"><div class="metric-title">Loading metrics</div><div class="metric-sub">${label || ''}</div></div></li>`;
}

/**
 * Show a message in the preview area.
 * @param {string} text - Message to display.
 */
function setPreviewMessage(text) {
  const preview = document.getElementById('preview-area');
  if (!preview) return;
  preview.innerHTML = `<div class="preview-message">${text}</div>`;
}

/**
 * Resolve which overlay image to display in the preview.
 * @param {object} files - Files block from metrics.json.
 * @param {string} stem - Image name without extension.
 * @returns {string|null} Overlay URL.
 */
function resolveOverlayPath(files, stem) {
  if (!files) return null;
  const preferred = files.combined_overlay || files.iris_overlay || files.pupil_overlay;
  if (!preferred) return null;
  return `${state.metricsBase}/${stem}/${preferred}`;
}

/**
 * Turns a filename into a human-readable label.
 * @param {string} file - File name.
 * @returns {string} Label.
 */
function filenameToLabel(file) {
  const base = getStem(file);
  return base.replace(/[-_]+/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
}

/**
 * Extracts filename without extension.
 * @param {string} file - File name.
 * @returns {string} Stem.
 */
function getStem(file) {
  return file.replace(/\.[^/.]+$/, '');
}

function isNumber(value) {
  return typeof value === 'number' && Number.isFinite(value);
}

function isArray(value) {
  return Array.isArray(value);
}

function formatNumber(value, digits = 1) {
  if (!isNumber(value)) return '—';
  return Number(value).toFixed(digits);
}

/**
 * Build a horizontal progress row for the preview panel.
 * @param {string} label - Row label.
 * @param {number} value - Current value.
 * @param {number} min - Minimum expected.
 * @param {number} max - Maximum expected.
 * @param {string} color - Fill color.
 * @param {string} [status] - Optional status text.
 * @returns {string} HTML string.
 */
function buildProgressRow(label, value, min, max, color, status) {
  if (!isNumber(value) || !isNumber(max) || max <= 0) return '';
  const clamped = Math.max(min || 0, Math.min(max, value));
  const percent = Math.min(100, Math.max(0, (clamped / max) * 100));
  const statusText = status ? `<span class="preview-progress-status">${status}</span>` : '';

  return `
    <div class="preview-progress-row">
      <div class="preview-progress-head">
        <span class="preview-key">${label}</span>
        <span class="preview-value">${formatNumber(value, 2)}</span>
      </div>
      <div class="preview-progress">
        <div class="preview-progress-fill" style="width:${percent}%;background:${color || '#34d399'}"></div>
      </div>
      ${statusText}
    </div>
  `;
}

/**
 * Sample eye colors for pupil, iris, and sclera regions.
 * @param {string} url - Image URL.
 * @param {object} data - Metrics payload for geometry.
 * @returns {Promise<object|null>} Sampled colors.
 */
function sampleEyeColors(url, data) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas');
        const ctx = canvas.getContext('2d', { willReadFrequently: true });
        const maxDim = Math.max(img.naturalWidth, img.naturalHeight);
        const target = Math.min(400, maxDim);
        const scale = maxDim > 0 ? target / maxDim : 1;
        const w = Math.max(1, Math.round(img.naturalWidth * scale));
        const h = Math.max(1, Math.round(img.naturalHeight * scale));
        canvas.width = w;
        canvas.height = h;
        ctx.drawImage(img, 0, 0, w, h);

        const pupil = data.pupil || {};
        const iris = data.iris || {};
        if (!pupil.centre || !isNumber(pupil.radius_px)) {
          resolve(null);
          return;
        }

        const cx = (pupil.centre[0] || 0) * scale;
        const cy = (pupil.centre[1] || 0) * scale;
        const pupilRadius = Math.max(1, (pupil.radius_px || 1) * scale);
        const irisRadius = iris.radius_px ? Math.max(pupilRadius * 1.5, iris.radius_px * scale) : pupilRadius * 2;

        const colors = {
          pupil: avgCircle(ctx, cx, cy, pupilRadius * 0.8),
          iris: avgRing(ctx, cx, cy, pupilRadius * 1.1, irisRadius * 0.95),
          sclera: avgRing(ctx, cx, cy, irisRadius * 1.1, irisRadius * 1.6),
        };

        resolve({
          pupil: toCss(colors.pupil),
          iris: toCss(colors.iris),
          sclera: toCss(colors.sclera),
        });
      } catch (err) {
        reject(err);
      }
    };
    img.onerror = () => reject(new Error('Failed to load image for color sampling'));
    img.src = url;
  });
}

function toCss(rgb) {
  if (!rgb) return null;
  const [r, g, b] = rgb.map((v) => Math.max(0, Math.min(255, Math.round(v))));
  return `rgb(${r}, ${g}, ${b})`;
}

function avgCircle(ctx, cx, cy, r) {
  return avgRegion(ctx, cx, cy, 0, r);
}

function avgRing(ctx, cx, cy, rInner, rOuter) {
  return avgRegion(ctx, cx, cy, rInner, rOuter);
}

function avgRegion(ctx, cx, cy, rInner, rOuter) {
  const x0 = Math.max(0, Math.floor(cx - rOuter));
  const y0 = Math.max(0, Math.floor(cy - rOuter));
  const x1 = Math.min(ctx.canvas.width - 1, Math.ceil(cx + rOuter));
  const y1 = Math.min(ctx.canvas.height - 1, Math.ceil(cy + rOuter));

  let rSum = 0;
  let gSum = 0;
  let bSum = 0;
  let count = 0;
  const rInnerSq = rInner * rInner;
  const rOuterSq = rOuter * rOuter;

  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const distSq = dx * dx + dy * dy;
      if (distSq < rInnerSq || distSq > rOuterSq) continue;
      const [r, g, b] = ctx.getImageData(x, y, 1, 1).data;
      rSum += r;
      gSum += g;
      bSum += b;
      count += 1;
    }
  }

  if (!count) return null;
  return [rSum / count, gSum / count, bSum / count];
}

/**
 * Compute colors for an image (pupil, iris, sclera) based on geometry.
 * @param {string} stem - Image stem.
 * @param {object} data - Metrics payload.
 * @returns {Promise<object|null>}
 */
async function computeColorsForImage(stem, data) {
  const imageName = data.image || `${stem}.jpg`;
  const imageUrl = `${state.imageBase}/${imageName}`;
  return sampleEyeColors(imageUrl, data);
}
