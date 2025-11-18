"""Iris boundary detection using radial color/gradient analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np


@dataclass
class IrisConfig:
    """Configuration for iris boundary detection."""

    ray_count: int = 64
    search_start_ratio: float = 1.2  # Start searching at pupil_radius * this ratio
    search_end_ratio: float = 4.0    # Stop searching at pupil_radius * this ratio
    min_iris_pupil_ratio: float = 1.5  # Iris must be at least this much larger than pupil
    max_iris_pupil_ratio: float = 5.0  # Iris cannot be more than this much larger than pupil
    sample_step: int = 1
    color_jump_threshold: float = 15.0  # Threshold for detecting color change
    gradient_threshold: float = 20.0    # Minimum gradient magnitude
    min_edge_points: int = 32           # Minimum rays that must find an edge


@dataclass
class IrisResult:
    centre: Tuple[float, float]
    radius: float
    score: float
    edge_points: np.ndarray


def detect_iris_boundary(
    image: np.ndarray,
    gray: np.ndarray,
    pupil_centre: Tuple[float, float],
    pupil_radius: float,
    config: IrisConfig,
) -> IrisResult:
    """
    Detect iris boundary by casting rays outward from pupil center.
    
    Args:
        image: Color image (BGR)
        gray: Grayscale version of the image
        pupil_centre: Detected pupil center (x, y)
        pupil_radius: Detected pupil radius
        config: Iris detection configuration
        
    Returns:
        IrisResult with iris center, radius, and quality score
    """
    height, width = gray.shape[:2]
    cx, cy = pupil_centre
    
    # Compute gradients for edge detection
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    
    # Convert to LAB color space for better color distance
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Define search range based on pupil radius
    search_start = int(pupil_radius * config.search_start_ratio)
    search_end = int(pupil_radius * config.search_end_ratio)
    search_end = min(search_end, min(height, width) // 2)  # Don't search beyond image bounds
    
    angles = np.linspace(0, 2 * np.pi, config.ray_count, endpoint=False)
    edge_points = []
    
    for angle in angles:
        cos_val = np.cos(angle)
        sin_val = np.sin(angle)
        
        # Sample initial color just outside the pupil
        start_x = int(round(cx + search_start * cos_val))
        start_y = int(round(cy + search_start * sin_val))
        
        if start_x < 0 or start_x >= width or start_y < 0 or start_y >= height:
            continue
            
        prev_color = lab_image[start_y, start_x].astype(np.float32)
        best_edge_dist = None
        best_edge_score = 0.0
        
        # Cast ray outward looking for iris-sclera boundary
        for dist in range(search_start, search_end, config.sample_step):
            x = int(round(cx + dist * cos_val))
            y = int(round(cy + dist * sin_val))
            
            if x < 0 or x >= width or y < 0 or y >= height:
                break
            
            curr_color = lab_image[y, x].astype(np.float32)
            gradient = float(grad_mag[y, x])
            
            # Calculate color distance in LAB space
            color_diff = float(np.linalg.norm(curr_color - prev_color))
            
            # Look for significant color change with strong gradient
            if color_diff >= config.color_jump_threshold and gradient >= config.gradient_threshold:
                # Score this edge candidate
                edge_score = color_diff * gradient
                
                if edge_score > best_edge_score:
                    best_edge_dist = dist
                    best_edge_score = edge_score
            
            prev_color = curr_color
        
        # If we found a valid edge on this ray, record it
        if best_edge_dist is not None:
            edge_x = int(round(cx + best_edge_dist * cos_val))
            edge_y = int(round(cy + best_edge_dist * sin_val))
            edge_points.append((edge_x, edge_y))
    
    if len(edge_points) < config.min_edge_points:
        raise ValueError(
            f"Insufficient iris edge points detected: {len(edge_points)} < {config.min_edge_points}"
        )
    
    # Fit circle to edge points
    edge_array = np.array(edge_points, dtype=np.float32)
    
    # Use least squares circle fitting
    A = np.column_stack((edge_array[:, 0], edge_array[:, 1], np.ones(len(edge_array))))
    b = -(edge_array[:, 0] ** 2 + edge_array[:, 1] ** 2)
    params, *_ = np.linalg.lstsq(A, b, rcond=None)
    D, E, F = params
    
    iris_cx = -0.5 * D
    iris_cy = -0.5 * E
    radius_sq = iris_cx * iris_cx + iris_cy * iris_cy - F
    
    if radius_sq <= 0:
        raise ValueError("Fitted iris circle has non-positive radius")
    
    iris_radius = float(np.sqrt(radius_sq))
    
    # Validate iris size relative to pupil
    iris_pupil_ratio = iris_radius / pupil_radius
    if iris_pupil_ratio < config.min_iris_pupil_ratio:
        raise ValueError(
            f"Detected iris too small: ratio {iris_pupil_ratio:.2f} < {config.min_iris_pupil_ratio}"
        )
    if iris_pupil_ratio > config.max_iris_pupil_ratio:
        raise ValueError(
            f"Detected iris too large: ratio {iris_pupil_ratio:.2f} > {config.max_iris_pupil_ratio}"
        )
    
    # Calculate fit quality score
    distances = np.sqrt(
        (edge_array[:, 0] - iris_cx) ** 2 + (edge_array[:, 1] - iris_cy) ** 2
    )
    std_dev = float(np.std(distances))
    score = len(edge_points) / (std_dev + 1e-6)
    
    return IrisResult(
        centre=(float(iris_cx), float(iris_cy)),
        radius=iris_radius,
        score=score,
        edge_points=edge_array,
    )


def create_iris_mask(
    shape: Tuple[int, int],
    pupil_centre: Tuple[int, int],
    pupil_radius: int,
    iris_centre: Tuple[int, int],
    iris_radius: int,
) -> np.ndarray:
    """Create a mask showing only the iris region (excluding pupil)."""
    mask = np.zeros(shape, dtype=np.uint8)
    # Draw outer iris circle
    cv2.circle(mask, iris_centre, iris_radius, 255, thickness=-1)
    # Subtract pupil
    cv2.circle(mask, pupil_centre, pupil_radius, 0, thickness=-1)
    return mask


def create_combined_overlay(
    image: np.ndarray,
    pupil_centre: Tuple[int, int],
    pupil_radius: int,
    iris_centre: Tuple[int, int],
    iris_radius: int,
) -> np.ndarray:
    """Create an overlay showing both pupil and iris boundaries."""
    overlay = image.copy()
    # Draw iris boundary in green
    cv2.circle(overlay, iris_centre, iris_radius, (0, 255, 0), thickness=2)
    # Draw pupil boundary in magenta
    cv2.circle(overlay, pupil_centre, pupil_radius, (255, 0, 255), thickness=2)
    # Draw pupil center
    cv2.circle(overlay, pupil_centre, max(1, pupil_radius // 12), (0, 0, 255), thickness=-1)
    return overlay
