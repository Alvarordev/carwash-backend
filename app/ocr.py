from __future__ import annotations

import re

import cv2
import numpy as np
from paddleocr import PaddleOCR
from rapidfuzz import fuzz

from app.constants import BADGE_DET_THRESH, BRANDS, MODEL_TO_BRAND, PLATE_RE

_ocr: PaddleOCR | None = None


def get_ocr() -> PaddleOCR:
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
        )
    return _ocr


def warmup() -> None:
    """Pre-load PaddleOCR so the first request isn't slow."""
    get_ocr()


_STRIP_RE = re.compile(r"[^A-Z0-9]")


def _normalise(text: str) -> str:
    return _STRIP_RE.sub("", text.upper())

def read_plate_region(plate_crop: np.ndarray) -> str | None:
    """Read license plate text from a cropped plate image.

    Tries both the original crop and a CLAHE-preprocessed version,
    returns the best match against the plate regex.
    """
    ocr = get_ocr()
    candidates: list[str] = []

    # Attempt 1: original crop (as BGR for PaddleOCR)
    crop_bgr = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2BGR)
    candidates.extend(_ocr_texts(ocr, crop_bgr))

    # Attempt 2: grayscale + CLAHE enhanced
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    # Convert back to 3-channel for PaddleOCR
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    candidates.extend(_ocr_texts(ocr, enhanced_bgr))

    return _find_plate(candidates)


def read_badge_regions(badge_crops: list[np.ndarray]) -> list[str]:
    """Run PaddleOCR on badge region crops with lower detection threshold.

    Returns normalised text tokens found across all crops.
    """
    if not badge_crops:
        return []

    # Create a separate OCR instance with lower detection threshold
    # to catch chrome/metallic text
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="en",
        use_gpu=False,
        show_log=False,
        det_db_thresh=BADGE_DET_THRESH,
    )

    tokens: list[str] = []
    for crop in badge_crops:
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        texts = _ocr_texts(ocr, crop_bgr)
        tokens.extend(_normalise(t) for t in texts if len(t) >= 2)

    return tokens

def extract_plate_fallback(image_rgb: np.ndarray) -> str | None:
    """Fallback plate extraction using bottom-centre crop (no YOLO)."""
    ocr = get_ocr()
    h, w = image_rgb.shape[:2]

    # Bottom-centre crop: bottom 35%, middle 60%
    crop = image_rgb[int(h * 0.65):h, int(w * 0.2):int(w * 0.8)]
    crop_bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)

    texts = _ocr_texts(ocr, crop_bgr)
    plate = _find_plate(texts)
    if plate:
        return plate

    # Try full image
    full_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    texts = _ocr_texts(ocr, full_bgr)
    return _find_plate(texts)


def extract_texts_fallback(image_rgb: np.ndarray) -> list[str]:
    """Fallback full-image OCR for model/brand tokens."""
    ocr = get_ocr()
    full_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    raw = _ocr_texts(ocr, full_bgr)
    return [_normalise(t) for t in raw if len(t) >= 2]

def match_model(tokens: list[str]) -> tuple[str | None, str | None]:
    """Fuzzy-match tokens against MODEL_TO_BRAND. Returns (model, brand)."""
    for token in tokens:
        if len(token) < 2:
            continue
        for model_key, brand_val in MODEL_TO_BRAND.items():
            score = fuzz.ratio(token, model_key)
            if score >= 88:
                return model_key, brand_val
    return None, None


def match_brand(tokens: list[str]) -> str | None:
    """Fuzzy-match tokens against BRANDS list."""
    for token in tokens:
        if len(token) < 2:
            continue
        for brand in BRANDS:
            normalised_brand = _normalise(brand)
            score = fuzz.ratio(token, normalised_brand)
            if score >= 90:
                return brand
    return None

def _ocr_texts(ocr: PaddleOCR, image_bgr: np.ndarray) -> list[str]:
    """Run PaddleOCR and return a flat list of detected text strings."""
    result = ocr.ocr(image_bgr, cls=True)
    texts: list[str] = []
    if result and result[0]:
        for line in result[0]:
            # Each line: [bbox, (text, confidence)]
            if line and len(line) >= 2:
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 1:
                    texts.append(str(text_info[0]))
    return texts


def _find_plate(texts: list[str]) -> str | None:
    for t in texts:
        cleaned = _normalise(t)
        m = PLATE_RE.search(cleaned)
        if m:
            raw = m.group(0).replace("-", "")
            return f"{raw[:3]}-{raw[3:]}"
    return None
