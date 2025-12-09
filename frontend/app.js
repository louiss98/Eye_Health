// Minimal demo JS to populate the metrics list
const metrics = [
  { id: 'pupil_center', title: 'Iris', sub: 'x: 124, y: 98' },
  { id: 'pupil_size', title: 'Pupil', sub: '3.2 mm' },
  { id: 'iris_integrity', title: 'Sclera', sub: 'No abnormalities' },
  { id: 'sclera', title: 'Sclera', sub: 'Mild redness', status: 'warn' },
  { id: 'tear_film', title: 'Tear Film', sub: 'Stable', status: 'neutral' },
  { id: 'use_camera', title: 'Use Camera', sub: 'Capture new image', camera: true }
];

function buildMetrics() {
  const list = document.getElementById('metrics-list');
  list.innerHTML = '';
  for (const m of metrics) {
    const li = document.createElement('li');
    li.className = 'metric-item';
    if (m.camera) li.classList.add('camera-item');

    const icon = document.createElement('div');
    icon.className = 'metric-icon';
    if (m.status === 'warn') icon.classList.add('icon-warn');
    if (m.status === 'neutral') icon.classList.add('icon-neutral');
    // default is now icon-ok style, no need to add class

    const body = document.createElement('div');
    body.className = 'metric-body';
    const title = document.createElement('div');
    title.className = 'metric-title';
    title.textContent = m.title;
    const sub = document.createElement('div');
    sub.className = 'metric-sub';
    sub.textContent = m.sub;

    body.appendChild(title);
    body.appendChild(sub);

    li.appendChild(icon);
    li.appendChild(body);
    list.appendChild(li);
  }
}

window.addEventListener('DOMContentLoaded', () => {
  buildMetrics();
  
  // Initialize background image
  setBackgroundImage('images/mobile-eye-damage.jpg');
});

/**
 * Changes the background image with a fade transition.
 * @param {string} url - The URL of the new image.
 */
function setBackgroundImage(url) {
  const layer = document.getElementById('bg-image-layer');
  
  // Fade out
  layer.style.opacity = 0;
  
  // Wait for fade out, then swap and fade in
  setTimeout(() => {
    layer.style.backgroundImage = `url('${url}')`;
    // Small delay to ensure DOM update before fading back in
    requestAnimationFrame(() => {
      layer.style.opacity = 1;
    });
  }, 1000); // Match CSS transition time
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