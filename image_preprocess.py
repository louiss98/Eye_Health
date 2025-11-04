"""Starburst-based pupil segmentation for centred eye photos."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


@dataclass
class StarburstConfig:
    """Configuration for the Starburst ray-casting detector."""

    ray_count: int = 48
    ray_length_ratio: float = 0.45
    min_radius_ratio: float = 0.02
    intensity_jump: float = 12.0
    gradient_threshold: float = 25.0
    sample_step: int = 1
    max_iterations: int = 5
    min_edge_points: int = 16
    convergence_tol: float = 1.5
    radius_tol: float = 1.2


@dataclass
class PipelineConfig:
    """Configuration driving the preprocessing pipeline."""

    max_dimension: int = 800
    median_blur_kernel: int = 5
    display_poll_delay_ms: int = 50
    initial_centre_window_ratio: float = 0.6
    initial_gaussian_kernel: int = 15
    dark_intensity_threshold: int = 110
    starburst: StarburstConfig = field(default_factory=StarburstConfig)


@dataclass
class DetectionResult:
    centre: Tuple[float, float]
    radius: float
    score: float
    meta: Dict[str, np.ndarray] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Util
# ---------------------------------------------------------------------------


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def normalize_size(image: np.ndarray, max_dimension: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= max_dimension:
        return image

    scale = max_dimension / float(longest_edge)
    new_size = (int(width * scale), int(height * scale))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def ensure_odd(value: int) -> int:
    return value if value % 2 == 1 else value + 1


def prepare_grayscale(image: np.ndarray, kernel_size: int) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = max(1, ensure_odd(kernel_size))
    if kernel > 1:
        gray = cv2.medianBlur(gray, kernel)
    return gray


def estimate_initial_centre(gray: np.ndarray, config: PipelineConfig) -> Tuple[int, int]:
    height, width = gray.shape
    ratio = np.clip(config.initial_centre_window_ratio, 0.2, 1.0)
    half_w = int(width * ratio / 2)
    half_h = int(height * ratio / 2)
    cx, cy = width // 2, height // 2

    x0 = max(0, cx - half_w)
    x1 = min(width, cx + half_w)
    y0 = max(0, cy - half_h)
    y1 = min(height, cy + half_h)

    if x1 <= x0 or y1 <= y0:
        return cx, cy

    kernel = ensure_odd(config.initial_gaussian_kernel)
    blurred = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    region = blurred[y0:y1, x0:x1]
    min_idx = int(np.argmin(region))
    dy, dx = divmod(min_idx, region.shape[1])
    candidate = (int(x0 + dx), int(y0 + dy))

    if int(region[dy, dx]) > config.dark_intensity_threshold:
        return cx, cy

    return candidate


def max_radius_from_centre(height: int, width: int, centre: Tuple[int, int]) -> int:
    cx, cy = centre
    distances = (cx, cy, width - cx - 1, height - cy - 1)
    return int(max(0, min(distances)))


def create_pupil_mask(shape: Tuple[int, int], centre: Tuple[int, int], radius: int) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.circle(mask, centre, radius, 255, thickness=-1)
    return mask


def create_overlay(image: np.ndarray, centre: Tuple[int, int], radius: int, color: Tuple[int, int, int]) -> np.ndarray:
    overlay = image.copy()
    cv2.circle(overlay, centre, radius, color, thickness=2)
    cv2.circle(overlay, centre, max(1, radius // 12), (0, 0, 255), thickness=-1)
    return overlay


def display_overlay(title: str, image: np.ndarray, poll_delay_ms: int) -> None:
    cv2.imshow(title, image)
    try:
        while True:
            key = cv2.waitKey(poll_delay_ms)
            visible = cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE)
            if key in (27, ord("q"), ord("Q")) or visible < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyWindow(title)


# ---------------------------------------------------------------------------
# Starburst (Li et al.)
# ---------------------------------------------------------------------------


def fit_circle_least_squares(points: np.ndarray) -> Tuple[Tuple[float, float], float]:
    if points.shape[0] < 3:
        raise ValueError("At least three points are required for circle fitting.")

    A = np.column_stack((points[:, 0], points[:, 1], np.ones(points.shape[0])))
    b = -(points[:, 0] ** 2 + points[:, 1] ** 2)
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = params
    cx = -0.5 * D
    cy = -0.5 * E
    radius_sq = cx * cx + cy * cy - F
    if radius_sq <= 0:
        raise ValueError("Fitted circle has non-positive radius.")
    return (float(cx), float(cy)), float(np.sqrt(radius_sq))


def run_starburst(gray: np.ndarray, config: PipelineConfig) -> DetectionResult:
    scfg = config.starburst
    height, width = gray.shape
    centre = np.array(estimate_initial_centre(gray, config), dtype=np.float32)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)

    span = min(height, width)
    ray_length = max(12, int(scfg.ray_length_ratio * span))
    angles = np.linspace(0, 2 * np.pi, scfg.ray_count, endpoint=False)
    radius = max(3, int(scfg.min_radius_ratio * span))

    for iteration in range(scfg.max_iterations):
        edge_points: List[Tuple[int, int]] = []
        intensity_jump = scfg.intensity_jump * (0.85 ** iteration)
        gradient_threshold = scfg.gradient_threshold * (0.9 ** iteration)

        centre_intensity = float(
            gray[int(np.clip(round(centre[1]), 0, height - 1)), int(np.clip(round(centre[0]), 0, width - 1))]
        )

        for angle in angles:
            sin_val = np.sin(angle)
            cos_val = np.cos(angle)
            prev_intensity = centre_intensity

            for dist in range(scfg.sample_step, ray_length, scfg.sample_step):
                x = int(round(centre[0] + dist * cos_val))
                y = int(round(centre[1] + dist * sin_val))

                if x < 0 or x >= width or y < 0 or y >= height:
                    break

                intensity = float(gray[y, x])
                gradient = float(grad_mag[y, x])

                if (
                    ((intensity - prev_intensity) >= intensity_jump)
                    or ((intensity - centre_intensity) >= intensity_jump)
                ) and gradient >= gradient_threshold:
                    edge_points.append((x, y))
                    break

                prev_intensity = intensity

        if len(edge_points) < scfg.min_edge_points:
            continue

        points_np = np.asarray(edge_points, dtype=np.float32)
        try:
            new_centre, new_radius = fit_circle_least_squares(points_np)
        except ValueError:
            continue

        shift = np.linalg.norm(points_np.mean(axis=0) - new_centre)
        radius_change = abs(new_radius - radius)

        centre = np.array(new_centre, dtype=np.float32)
        radius = float(new_radius)

        if shift < scfg.convergence_tol and radius_change < scfg.radius_tol:
            edge_std = float(
                np.std(
                    np.sqrt(
                        ((points_np[:, 0] - centre[0]) ** 2)
                        + ((points_np[:, 1] - centre[1]) ** 2)
                    )
                )
            )
            mask = create_pupil_mask(gray.shape, (int(round(centre[0])), int(round(centre[1]))), max(1, int(round(radius)) - 1))
            inside_pixels = gray[mask == 255]
            if inside_pixels.size == 0:
                continue

            mean_inside = float(np.mean(inside_pixels))
            if mean_inside > config.dark_intensity_threshold:
                continue

            score = (config.dark_intensity_threshold - mean_inside) / (edge_std + 1e-6)
            return DetectionResult(centre=tuple(centre), radius=radius, score=score, meta={"edge_points": points_np})

    raise ValueError("Starburst failed to converge to a pupil boundary.")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def process_image_file(path: Path, output_dir: Path, config: PipelineConfig, show: bool) -> None:
    image = load_image(path)
    image = normalize_size(image, config.max_dimension)
    gray = prepare_grayscale(image, config.median_blur_kernel)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem

    try:
        result = run_starburst(gray, config)
    except Exception as exc:
        print(f"{path.name}: starburst detector failed -> {exc}")
        return

    centre_int = (int(round(result.centre[0])), int(round(result.centre[1])))
    radius_int = max(1, int(round(result.radius)))
    centre_int = (
        int(np.clip(centre_int[0], 0, gray.shape[1] - 1)),
        int(np.clip(centre_int[1], 0, gray.shape[0] - 1)),
    )
    radius_int = min(radius_int, max_radius_from_centre(gray.shape[0], gray.shape[1], centre_int))

    mask = create_pupil_mask(gray.shape, centre_int, radius_int)
    overlay = create_overlay(image, centre_int, radius_int, (255, 0, 255))

    cv2.imwrite(str(output_dir / f"{stem}_starburst_mask.png"), mask)
    cv2.imwrite(str(output_dir / f"{stem}_starburst_overlay.png"), overlay)

    print(
        f"{path.name}: starburst centre=({centre_int[0]}, {centre_int[1]}) radius={radius_int}px score={result.score:.2f}"
    )

    if show:
        display_overlay(f"{stem} - starburst", overlay, config.display_poll_delay_ms)


def list_image_files(directory: Path) -> Iterable[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in extensions and path.is_file():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Starburst pupil detector on eye photos."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/images"),
        help="Folder containing input eye images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output"),
        help="Folder to write detector masks and overlays.",
    )
    parser.add_argument(
        "--max-dimension",
        type=int,
        default=800,
        help="Resize so the longest edge does not exceed this many pixels.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display overlays in interactive windows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = PipelineConfig(max_dimension=args.max_dimension)

    if not args.input.exists() or not any(args.input.iterdir()):
        raise FileNotFoundError(
            f"Input folder '{args.input}' does not exist or contains no image files."
        )

    processed_any = False
    for image_path in list_image_files(args.input):
        process_image_file(image_path, args.output, config, show=args.show)
        processed_any = True

    if not processed_any:
        raise FileNotFoundError(
            "No supported image files were found. Supported extensions: .jpg, .jpeg, .png, .bmp, .tif, .tiff"
        )


if __name__ == "__main__":
    main()
