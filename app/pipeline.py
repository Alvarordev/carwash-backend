"""Detection-first analysis pipeline: YOLO plate detect -> OCR -> badge search -> color."""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from app.color import detect_color
from app.constants import MAX_TIME
from app.detector import detect_plates, get_badge_search_regions
from app.ocr import (
    extract_plate_fallback,
    extract_texts_fallback,
    match_brand,
    match_model,
    read_badge_regions,
    read_plate_region,
)

logger = logging.getLogger(__name__)


def _is_over_budget(start: float) -> bool:
    return time.perf_counter() - start > MAX_TIME


def analyze(image_bytes: bytes) -> dict:
    """Run the full vehicle rear-image analysis pipeline.

    Pipeline flow:
      1. Decode & resize
      2. YOLO plate detection
      3. OCR plate region
      4. Badge search near plate
      5. Fallback if no plate detected
      6. Model/brand matching
      7. Color detection
    """
    start = time.perf_counter()

    plate: str | None = None
    brand: str | None = None
    model: str | None = None
    color: str | None = None
    partial = False

    # -- Step 1: decode & preprocess ----------------------------------------
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        return _result(plate, brand, model, color, True, start)

    # Resize to max 1280 px wide
    h, w = image_bgr.shape[:2]
    if w > 1280:
        scale = 1280 / w
        image_bgr = cv2.resize(
            image_bgr,
            (1280, int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # -- Step 2: YOLO plate detection --------------------------------------
    plate_bbox = None
    tokens: list[str] = []

    if not _is_over_budget(start):
        try:
            detections = detect_plates(image_rgb)
            if detections:
                plate_bbox = detections[0]["bbox"]
                logger.info(
                    "Plate detected: bbox=%s conf=%.2f",
                    plate_bbox, detections[0]["confidence"],
                )
        except Exception:
            logger.warning("YOLO plate detection failed, using fallback", exc_info=True)

    # -- Step 3: OCR plate region ------------------------------------------
    if plate_bbox and not _is_over_budget(start):
        x1, y1, x2, y2 = plate_bbox
        plate_crop = image_rgb[y1:y2, x1:x2]
        if plate_crop.size > 0:
            plate = read_plate_region(plate_crop)

    # -- Step 4: Badge search near plate -----------------------------------
    if plate_bbox and not _is_over_budget(start):
        badge_crops = get_badge_search_regions(image_rgb, plate_bbox)
        if badge_crops:
            tokens = read_badge_regions(badge_crops)

    # -- Step 5: Fallback if no plate detected -----------------------------
    if plate is None and not _is_over_budget(start):
        plate = extract_plate_fallback(image_rgb)

    if not tokens and not _is_over_budget(start):
        tokens = extract_texts_fallback(image_rgb)

    if _is_over_budget(start):
        color = detect_color(image_bgr, plate_bbox)
        return _result(plate, brand, model, color, True, start)

    # -- Step 6: model/brand matching --------------------------------------
    if tokens:
        model, brand = match_model(tokens)

    if model is None and tokens:
        brand = match_brand(tokens)

    if _is_over_budget(start):
        color = detect_color(image_bgr, plate_bbox)
        return _result(plate, brand, model, color, True, start)

    # -- Step 7: color detection -------------------------------------------
    color = detect_color(image_bgr, plate_bbox)

    partial = plate is None and brand is None and model is None and color is None
    return _result(plate, brand, model, color, partial, start)


def _result(
    plate: str | None,
    brand: str | None,
    model: str | None,
    color: str | None,
    partial: bool,
    start: float,
) -> dict:
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return {
        "plate": plate,
        "brand": brand,
        "model": model,
        "color": color,
        "partial": partial,
        "processing_time_ms": elapsed_ms,
    }
