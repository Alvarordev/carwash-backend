"""Dominant color detection from vehicle image via HSV analysis."""

import cv2
import numpy as np

from app.constants import (
    BLACK_V_MAX,
    COLOR_RANGES,
    GRAY_S_MAX,
    GRAY_V_RANGE,
    SILVER_S_MAX,
    SILVER_V_RANGE,
    WHITE_S_MAX,
    WHITE_V_MIN,
)


def detect_color(
    image_bgr: np.ndarray,
    plate_bbox: tuple[int, int, int, int] | None = None,
) -> str:
    """Return the dominant vehicle color name in Spanish.

    When a plate bounding box is available, focuses on the car body region
    around the plate (above/beside it), which avoids background contamination.

    Filters out light sources (headlights, tail-lights, reflections) that
    have high saturation + brightness, which would otherwise skew detection.
    """
    roi = _get_color_roi(image_bgr, plate_bbox)

    # Down-sample for speed (max 200 px wide)
    scale = min(1.0, 200.0 / roi.shape[1])
    if scale < 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    filtered = _filter_light_sources(pixels)

    if len(filtered) < 50:
        filtered = pixels  # fallback if too aggressively filtered

    # ---- K-means (k=3) on the filtered pixels ----
    samples = filtered.astype(np.float32)
    if len(samples) > 3000:
        indices = np.random.default_rng(42).choice(len(samples), 3000, replace=False)
        samples = samples[indices]

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        samples, 3, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    # Pick the best cluster.
    # When cluster sizes are close (within 10% of each other), prefer the
    # most saturated one — background (sky, road, buildings) is always low-S,
    # while the car body is the colorful part of the image.
    counts = np.bincount(labels.flatten(), minlength=3)
    total_pts = counts.sum()
    max_count = counts.max()

    candidates = []
    for i in range(3):
        # Consider clusters that are at least 90% the size of the largest
        if counts[i] >= max_count * 0.90:
            c = centers[i]
            candidates.append((i, float(c[1])))  # (index, saturation)

    if len(candidates) > 1:
        # Among similarly-sized clusters, pick the most saturated
        best_idx = max(candidates, key=lambda x: x[1])[0]
    else:
        best_idx = counts.argmax()

    dominant_center = centers[best_idx].astype(int)
    h_val, s_val, v_val = (
        int(dominant_center[0]),
        int(dominant_center[1]),
        int(dominant_center[2]),
    )

    return _classify_hsv(h_val, s_val, v_val)


def _filter_light_sources(pixels: np.ndarray) -> np.ndarray:
    """Remove pixels that are likely from light sources, not car paint.

    Applies fixed filters for obvious light artifacts, then an adaptive
    saturation cap when a large fraction of the image has unnaturally high
    saturation (indicates coloured ambient light contamination).
    """
    s_vals = pixels[:, 1]
    v_vals = pixels[:, 2]

    # Fixed filters:
    # 1. Very dark pixels (shadows, tyres)
    # 2. Near-white glare (low S, very high V)
    mask = (
        (v_vals >= 25)
        & ~((s_vals < 30) & (v_vals > 230))
    )
    filtered = pixels[mask]

    if len(filtered) < 50:
        return filtered

    # Adaptive saturation cap:
    # If > 30% of pixels have S > 80, it could be either:
    #   a) A genuinely colorful car (red, blue, etc.) — mean S is high (>160)
    #   b) Colored ambient light contamination — mean S is moderate (100-150)
    # We only filter out high-sat pixels when they look like light bleed,
    # not real paint.
    high_sat_mask = filtered[:, 1] > 80
    high_sat_ratio = high_sat_mask.sum() / len(filtered)
    if high_sat_ratio > 0.30:
        high_sat_pixels = filtered[high_sat_mask]
        mean_sat = float(high_sat_pixels[:, 1].mean())

        # Real car paint has very high saturation (>160); light
        # contamination sits in the 100-150 range.
        if mean_sat < 160:
            body_pixels = filtered[~high_sat_mask]
            if len(body_pixels) >= 50:
                return body_pixels

    return filtered


def _get_color_roi(
    image_bgr: np.ndarray,
    plate_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    """Extract the region of interest for color analysis.

    If a plate bbox is available, sample from the car body area around the
    plate (above and to the sides), which is mostly car paint. Otherwise
    fall back to the central 60% of the image.
    """
    h, w = image_bgr.shape[:2]

    if plate_bbox is not None:
        px1, py1, px2, py2 = plate_bbox
        plate_h = py2 - py1
        plate_w = px2 - px1

        # Body region: above the plate, extending to the sides
        # This is typically the trunk/tailgate — mostly car paint
        body_y1 = max(0, py1 - 4 * plate_h)
        body_y2 = py1  # stop at the top of the plate
        body_x1 = max(0, px1 - plate_w)
        body_x2 = min(w, px2 + plate_w)

        # Ensure the ROI is large enough to be useful
        if (body_y2 - body_y1) > 20 and (body_x2 - body_x1) > 20:
            return image_bgr[body_y1:body_y2, body_x1:body_x2]

    # Fallback: central 60% of the image
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    return image_bgr[y1:y2, x1:x2]


def _classify_hsv(h: int, s: int, v: int) -> str:
    """Map an HSV triplet to a Spanish color name."""
    # Achromatic checks first
    if v < BLACK_V_MAX:
        return "NEGRO"
    if s < WHITE_S_MAX and v >= WHITE_V_MIN:
        return "BLANCO"
    if s < SILVER_S_MAX and SILVER_V_RANGE[0] <= v <= SILVER_V_RANGE[1]:
        return "PLATEADO"
    if s < GRAY_S_MAX and GRAY_V_RANGE[0] <= v <= GRAY_V_RANGE[1]:
        return "GRIS"

    # Chromatic lookup
    for h_lo, h_hi, s_lo, s_hi, v_lo, v_hi, name in COLOR_RANGES:
        if h_lo <= h <= h_hi and s_lo <= s and v_lo <= v:
            return name

    # Low saturation fallback
    if s < 50:
        if v >= WHITE_V_MIN:
            return "BLANCO"
        if v >= SILVER_V_RANGE[0]:
            return "PLATEADO"
        return "GRIS"

    return "OTRO"
