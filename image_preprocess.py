"""Starburst-based pupil segmentation for centred eye photos."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np

from blur_detection import BlurConfig, BlurResult, analyze_pupil_blur
from iris_detection import IrisConfig, create_combined_overlay, create_iris_mask, detect_iris_boundary


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
    iris: IrisConfig = field(default_factory=IrisConfig)
    blur: BlurConfig = field(default_factory=BlurConfig)
    blur_threshold: float = 50.0  # Combined score below which pupil is considered blurry
    blur_padding: int = 5  # Extra pixels around pupil to include in blur analysis


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


def _save_blur_results(output_dir: Path, blur_result: BlurResult) -> None:
    """Save blur analysis results to JSON file."""
    results = {
        "laplacian_variance": float(blur_result.laplacian_score),
        "tenegrad_score": float(blur_result.tenegrad_score),
        "fourier_score": float(blur_result.fourier_score),
        "combined_score": float(blur_result.combined_score),
        "is_blurry": bool(blur_result.is_blurry),
        "blur_threshold": float(blur_result.blur_threshold),
    }
    with open(output_dir / "blur_analysis.json", "w") as f:
        json.dump(results, f, indent=2)


def _save_metrics_summary(
    output_dir: Path,
    source_image: Path,
    pupil_result: DetectionResult,
    pupil_centre: Tuple[int, int],
    pupil_radius: int,
    blur_result: BlurResult,
    pupil_crop_exists: bool,
    iris_result: IrisResult | None = None,
    iris_centre: Tuple[int, int] | None = None,
    iris_radius: int | None = None,
) -> None:
    """Save consolidated metrics (pupil, iris, blur) to JSON for the UI."""
    metrics: Dict[str, object] = {
        "image": source_image.name,
        "pupil": {
            "centre": [int(pupil_centre[0]), int(pupil_centre[1])],
            "radius_px": int(pupil_radius),
            "score": float(pupil_result.score),
        },
        "blur": {
            "laplacian_variance": float(blur_result.laplacian_score),
            "tenegrad_score": float(blur_result.tenegrad_score),
            "fourier_score": float(blur_result.fourier_score),
            "combined_score": float(blur_result.combined_score),
            "is_blurry": bool(blur_result.is_blurry),
            "blur_threshold": float(blur_result.blur_threshold),
        },
        "files": {
            "pupil_mask": "pupil_mask.png",
            "iris_mask": None,
            "pupil_overlay": "pupil_overlay.png",
            "iris_overlay": None,
            "combined_overlay": None,
            "pupil_crop": "pupil_crop.png" if pupil_crop_exists else None,
            "blur_analysis": "blur_analysis.json",
        },
    }

    if iris_result is not None and iris_centre is not None and iris_radius is not None:
        metrics["iris"] = {
            "centre": [int(iris_centre[0]), int(iris_centre[1])],
            "radius_px": int(iris_radius),
            "score": float(iris_result.score),
            "edge_points": int(len(iris_result.edge_points)),
            "pupil_ratio": float(iris_radius / max(1e-6, pupil_radius)),
        }
        metrics["files"].update(
            {
                "iris_mask": "iris_mask.png",
                "iris_overlay": "iris_overlay.png",
                "combined_overlay": "combined_overlay.png",
            }
        )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


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

    stem = path.stem
    image_output_dir = output_dir / stem
    image_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect pupil
    try:
        pupil_result = run_starburst(gray, config)
    except Exception as exc:
        print(f"{path.name}: starburst pupil detector failed -> {exc}")
        return

    pupil_centre_int = (int(round(pupil_result.centre[0])), int(round(pupil_result.centre[1])))
    pupil_radius_int = max(1, int(round(pupil_result.radius)))
    pupil_centre_int = (
        int(np.clip(pupil_centre_int[0], 0, gray.shape[1] - 1)),
        int(np.clip(pupil_centre_int[1], 0, gray.shape[0] - 1)),
    )
    pupil_radius_int = min(pupil_radius_int, max_radius_from_centre(gray.shape[0], gray.shape[1], pupil_centre_int))

    # Step 2: Extract pupil region and analyze blur
    blur_result, pupil_crop, pupil_crop_mask = analyze_pupil_blur(
        gray,
        pupil_centre_int,
        pupil_radius_int,
        config.blur,
        config.blur_threshold,
        config.blur_padding,
    )

    pupil_crop_exists = bool(pupil_crop.size > 0)

    # Save pupil crop if valid
    if pupil_crop_exists:
        cv2.imwrite(str(image_output_dir / "pupil_crop.png"), pupil_crop)

    # Step 3: Detect iris
    try:
        iris_result = detect_iris_boundary(
            image, gray, pupil_result.centre, pupil_result.radius, config.iris
        )
    except Exception as exc:
        print(f"{path.name}: iris detector failed -> {exc}")
        # Still save pupil results even if iris fails
        pupil_mask = create_pupil_mask(gray.shape, pupil_centre_int, pupil_radius_int)
        pupil_overlay = create_overlay(image, pupil_centre_int, pupil_radius_int, (255, 0, 255))
        cv2.imwrite(str(image_output_dir / "pupil_mask.png"), pupil_mask)
        cv2.imwrite(str(image_output_dir / "pupil_overlay.png"), pupil_overlay)
        _save_blur_results(image_output_dir, blur_result)
        _save_metrics_summary(
            image_output_dir,
            path,
            pupil_result,
            pupil_centre_int,
            pupil_radius_int,
            blur_result,
            pupil_crop_exists,
        )
        print(
            f"{path.name}: pupil centre=({pupil_centre_int[0]}, {pupil_centre_int[1]}) "
            f"radius={pupil_radius_int}px score={pupil_result.score:.2f} | "
            f"blur: combined={blur_result.combined_score:.2f} "
            f"(laplacian={blur_result.laplacian_score:.2f}, "
            f"tenegrad={blur_result.tenegrad_score:.2f}, "
            f"fourier={blur_result.fourier_score:.4f}) "
            f"{'BLURRY' if blur_result.is_blurry else 'CLEAR'}"
        )
        if show:
            display_overlay(f"{stem} - pupil only", pupil_overlay, config.display_poll_delay_ms)
        return

    iris_centre_int = (int(round(iris_result.centre[0])), int(round(iris_result.centre[1])))
    iris_radius_int = max(1, int(round(iris_result.radius)))
    iris_centre_int = (
        int(np.clip(iris_centre_int[0], 0, gray.shape[1] - 1)),
        int(np.clip(iris_centre_int[1], 0, gray.shape[0] - 1)),
    )
    iris_radius_int = min(iris_radius_int, max_radius_from_centre(gray.shape[0], gray.shape[1], iris_centre_int))

    # Step 4: Create and save masks and overlays
    pupil_mask = create_pupil_mask(gray.shape, pupil_centre_int, pupil_radius_int)
    iris_mask = create_iris_mask(gray.shape, pupil_centre_int, pupil_radius_int, iris_centre_int, iris_radius_int)
    
    pupil_overlay = create_overlay(image, pupil_centre_int, pupil_radius_int, (255, 0, 255))
    iris_overlay = create_overlay(image, iris_centre_int, iris_radius_int, (0, 255, 0))
    combined_overlay = create_combined_overlay(
        image, pupil_centre_int, pupil_radius_int, iris_centre_int, iris_radius_int
    )

    cv2.imwrite(str(image_output_dir / "pupil_mask.png"), pupil_mask)
    cv2.imwrite(str(image_output_dir / "iris_mask.png"), iris_mask)
    cv2.imwrite(str(image_output_dir / "pupil_overlay.png"), pupil_overlay)
    cv2.imwrite(str(image_output_dir / "iris_overlay.png"), iris_overlay)
    cv2.imwrite(str(image_output_dir / "combined_overlay.png"), combined_overlay)
    _save_blur_results(image_output_dir, blur_result)
    _save_metrics_summary(
        image_output_dir,
        path,
        pupil_result,
        pupil_centre_int,
        pupil_radius_int,
        blur_result,
        pupil_crop_exists,
        iris_result=iris_result,
        iris_centre=iris_centre_int,
        iris_radius=iris_radius_int,
    )

    print(
        f"{path.name}: pupil centre=({pupil_centre_int[0]}, {pupil_centre_int[1]}) "
        f"radius={pupil_radius_int}px score={pupil_result.score:.2f} | "
        f"iris centre=({iris_centre_int[0]}, {iris_centre_int[1]}) "
        f"radius={iris_radius_int}px score={iris_result.score:.2f} | "
        f"blur: combined={blur_result.combined_score:.2f} "
        f"(laplacian={blur_result.laplacian_score:.2f}, "
        f"tenegrad={blur_result.tenegrad_score:.2f}, "
        f"fourier={blur_result.fourier_score:.4f}) "
        f"{'BLURRY' if blur_result.is_blurry else 'CLEAR'}"
    )

    if show:
        display_overlay(f"{stem} - combined", combined_overlay, config.display_poll_delay_ms)


def list_image_files(directory: Path) -> Iterable[Path]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    for path in sorted(directory.glob("*")):
        if path.suffix.lower() in extensions and path.is_file():
            yield path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Starburst pupil detector and iris boundary detection on eye photos."
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
        help="Base folder for output; each image gets its own subfolder with labeled masks.",
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
