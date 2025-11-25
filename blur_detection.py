"""Blur detection methods for pupil region analysis.

This module provides three blur detection techniques:
1. Variance of Laplacian - measures how rapidly pixel values change
2. Tenegrad Gradient Energy - measures edge contrast/sharpness
3. Frequency Domain (Fourier) - analyzes frequency content dampening

These methods are combined with weighted averaging to produce a final blur score.
Higher scores indicate sharper/clearer images; lower scores indicate more blur.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class BlurConfig:
    """Configuration for blur detection weighting."""

    laplacian_weight: float = 0.4
    tenegrad_weight: float = 0.35
    fourier_weight: float = 0.25
    # Minimum region size for reliable detection
    min_region_pixels: int = 100
    # Fourier analysis parameters
    fourier_low_freq_ratio: float = 0.1  # Fraction of spectrum considered low freq


@dataclass
class BlurResult:
    """Results from blur detection analysis."""

    laplacian_score: float
    tenegrad_score: float
    fourier_score: float
    combined_score: float
    is_blurry: bool
    blur_threshold: float


def extract_pupil_region(
    image: np.ndarray,
    centre: Tuple[int, int],
    radius: int,
    padding: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the circular pupil region from an image.

    Args:
        image: Source image (grayscale or color)
        centre: Pupil center coordinates (x, y)
        radius: Pupil radius in pixels
        padding: Additional pixels to include around the pupil (default 0)

    Returns:
        Tuple of (cropped_region, mask):
            - cropped_region: Rectangular crop containing the pupil
            - mask: Circular mask for the pupil area within the crop
    """
    height, width = image.shape[:2]
    cx, cy = centre
    effective_radius = radius + padding

    # Calculate bounding box
    x0 = max(0, cx - effective_radius)
    y0 = max(0, cy - effective_radius)
    x1 = min(width, cx + effective_radius)
    y1 = min(height, cy + effective_radius)

    # Extract the region
    if len(image.shape) == 3:
        cropped = image[y0:y1, x0:x1, :].copy()
    else:
        cropped = image[y0:y1, x0:x1].copy()

    # Create circular mask within the cropped region
    crop_height, crop_width = cropped.shape[:2]
    mask = np.zeros((crop_height, crop_width), dtype=np.uint8)

    # Mask center relative to the crop
    mask_cx = cx - x0
    mask_cy = cy - y0

    cv2.circle(mask, (mask_cx, mask_cy), effective_radius, 255, thickness=-1)

    # Clip mask to valid region
    mask[mask_cy + effective_radius + 1:, :] = 0
    mask[:max(0, mask_cy - effective_radius), :] = 0
    mask[:, mask_cx + effective_radius + 1:] = 0
    mask[:, :max(0, mask_cx - effective_radius)] = 0

    return cropped, mask


def _ensure_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert to grayscale if needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_laplacian_variance(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Compute Variance of Laplacian (VoL) for blur detection.

    The Laplacian operator measures how rapidly pixel values change.
    A sharp image will have high variance in the Laplacian response,
    while a blurry image will have low variance.

    Args:
        image: Input image (grayscale or color)
        mask: Optional mask to restrict analysis to specific region

    Returns:
        Variance of Laplacian (higher = sharper image)
    """
    gray = _ensure_grayscale(image)

    # Apply Laplacian operator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    if mask is not None:
        # Only consider pixels within the mask
        masked_laplacian = laplacian[mask > 0]
        if masked_laplacian.size == 0:
            return 0.0
        return float(np.var(masked_laplacian))

    return float(np.var(laplacian))


def compute_tenegrad_score(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """
    Compute Tenegrad Gradient Energy Focus Measure.

    Tenegrad uses the Sobel operator to detect gradients (edges).
    It's particularly good at detecting edge contrast which is
    expected at the boundary of eye features.

    Args:
        image: Input image (grayscale or color)
        mask: Optional mask to restrict analysis to specific region

    Returns:
        Tenegrad score (higher = sharper edges/more contrast)
    """
    gray = _ensure_grayscale(image)

    # Compute Sobel gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Tenegrad is the sum of squared gradient magnitudes
    gradient_energy = grad_x**2 + grad_y**2

    if mask is not None:
        masked_energy = gradient_energy[mask > 0]
        if masked_energy.size == 0:
            return 0.0
        # Return mean energy for normalization
        return float(np.mean(masked_energy))

    return float(np.mean(gradient_energy))


def compute_fourier_blur_score(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    low_freq_ratio: float = 0.1,
) -> float:
    """
    Compute Frequency Domain Blur Score using Fourier analysis.

    Cataracts and blur dampen high frequencies, resulting in a
    "milkier" or more uniform color. This method analyzes the
    ratio of high-frequency to low-frequency content.

    Args:
        image: Input image (grayscale or color)
        mask: Optional mask to restrict analysis to specific region
        low_freq_ratio: Fraction of spectrum considered low frequency

    Returns:
        Fourier blur score (higher = more high-frequency content = sharper)
    """
    gray = _ensure_grayscale(image)

    if mask is not None:
        # Apply mask - set non-masked pixels to mean to reduce edge artifacts
        masked_gray = gray.copy().astype(np.float64)
        mean_val = np.mean(gray[mask > 0]) if np.any(mask > 0) else 0
        masked_gray[mask == 0] = mean_val
        gray = masked_gray.astype(np.uint8)

    # Compute 2D FFT
    f_transform = np.fft.fft2(gray.astype(np.float64))
    f_shift = np.fft.fftshift(f_transform)

    # Get magnitude spectrum
    magnitude = np.abs(f_shift)

    # Create low-frequency mask (center of spectrum)
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2

    # Low frequency radius
    low_freq_radius = int(min(rows, cols) * low_freq_ratio)

    # Create masks for low and high frequency regions
    y, x = np.ogrid[:rows, :cols]
    distance_from_center = np.sqrt((x - ccol)**2 + (y - crow)**2)

    low_freq_mask = distance_from_center <= low_freq_radius
    high_freq_mask = ~low_freq_mask

    # Calculate energy in each band
    low_freq_energy = np.sum(magnitude[low_freq_mask]**2)
    high_freq_energy = np.sum(magnitude[high_freq_mask]**2)

    # Avoid division by zero
    total_energy = low_freq_energy + high_freq_energy
    if total_energy < 1e-10:
        return 0.0

    # Return ratio of high frequency energy (higher = sharper)
    return float(high_freq_energy / total_energy)


def analyze_blur(
    image: np.ndarray,
    mask: np.ndarray | None = None,
    config: BlurConfig | None = None,
    blur_threshold: float = 50.0,
) -> BlurResult:
    """
    Analyze image blur using weighted combination of three methods.

    Args:
        image: Input image (grayscale or color)
        mask: Optional mask to restrict analysis
        config: Blur detection configuration
        blur_threshold: Combined score below which image is considered blurry

    Returns:
        BlurResult with individual and combined scores
    """
    if config is None:
        config = BlurConfig()

    # Compute individual scores
    laplacian_score = compute_laplacian_variance(image, mask)
    tenegrad_score = compute_tenegrad_score(image, mask)
    fourier_score = compute_fourier_blur_score(
        image, mask, config.fourier_low_freq_ratio
    )

    # Normalize scores for combination
    # Laplacian variance can be very high, normalize to ~0-100 range
    normalized_laplacian = min(100.0, laplacian_score / 10.0)

    # Tenegrad is already in a reasonable range, normalize similarly
    normalized_tenegrad = min(100.0, np.sqrt(tenegrad_score) / 10.0)

    # Fourier score is 0-1, scale to 0-100
    normalized_fourier = fourier_score * 100.0

    # Compute weighted average
    combined_score = (
        config.laplacian_weight * normalized_laplacian
        + config.tenegrad_weight * normalized_tenegrad
        + config.fourier_weight * normalized_fourier
    )

    return BlurResult(
        laplacian_score=laplacian_score,
        tenegrad_score=tenegrad_score,
        fourier_score=fourier_score,
        combined_score=combined_score,
        is_blurry=combined_score < blur_threshold,
        blur_threshold=blur_threshold,
    )


def analyze_pupil_blur(
    image: np.ndarray,
    pupil_centre: Tuple[int, int],
    pupil_radius: int,
    config: BlurConfig | None = None,
    blur_threshold: float = 50.0,
    include_padding: int = 5,
) -> Tuple[BlurResult, np.ndarray, np.ndarray]:
    """
    Extract pupil region and analyze it for blur.

    This is the main entry point for analyzing pupil blur.

    Args:
        image: Full eye image (grayscale or color)
        pupil_centre: Detected pupil center (x, y)
        pupil_radius: Detected pupil radius in pixels
        config: Blur detection configuration
        blur_threshold: Score below which pupil is considered blurry
        include_padding: Extra pixels around pupil to include

    Returns:
        Tuple of (BlurResult, cropped_region, mask)
    """
    if config is None:
        config = BlurConfig()

    # Verify minimum region size
    region_area = np.pi * (pupil_radius + include_padding) ** 2
    if region_area < config.min_region_pixels:
        # Return a result indicating insufficient data
        return (
            BlurResult(
                laplacian_score=0.0,
                tenegrad_score=0.0,
                fourier_score=0.0,
                combined_score=0.0,
                is_blurry=True,
                blur_threshold=blur_threshold,
            ),
            np.array([]),
            np.array([]),
        )

    # Extract pupil region
    cropped, mask = extract_pupil_region(
        image, pupil_centre, pupil_radius, padding=include_padding
    )

    # Analyze blur
    result = analyze_blur(cropped, mask, config, blur_threshold)

    return result, cropped, mask
