"""YOLO ONNX plate detector and badge region extraction."""

from __future__ import annotations

import os

import cv2
import numpy as np
import onnxruntime as ort

from app.constants import PLATE_DETECT_CONF, PLATE_MODEL_PATH

_session: ort.InferenceSession | None = None
_INPUT_SIZE = 640  # YOLOv8 expects 640x640


def _get_session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not os.path.exists(PLATE_MODEL_PATH):
            raise FileNotFoundError(
                f"Plate detection model not found at {PLATE_MODEL_PATH}. "
                "Download the YOLOv8n plate detector ONNX model first."
            )
        _session = ort.InferenceSession(
            PLATE_MODEL_PATH,
            providers=["CPUExecutionProvider"],
        )
    return _session


def warmup() -> None:
    """Load the ONNX model so the first request isn't slow."""
    _get_session()

def detect_plates(image_rgb: np.ndarray) -> list[dict]:
    """Detect license plates in an image using YOLOv8n ONNX.

    Returns list of {'bbox': (x1, y1, x2, y2), 'confidence': float}
    sorted by confidence descending.
    """
    session = _get_session()
    h_orig, w_orig = image_rgb.shape[:2]

    # Preprocess: letterbox resize to 640x640
    img, ratio, (pad_w, pad_h) = _letterbox(image_rgb, _INPUT_SIZE)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW

    # Inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: img})

    # Parse YOLOv8 output: shape (1, 5, N) — [x_center, y_center, w, h, conf]
    preds = outputs[0]  # (1, 5, N) or (1, N, 5)

    # Handle both possible output shapes
    if preds.shape[1] == 5 and preds.shape[2] > 5:
        # Shape (1, 5, N) — transpose to (1, N, 5)
        preds = preds.transpose(0, 2, 1)
    elif preds.shape[2] == 5:
        pass  # Already (1, N, 5)
    else:
        # For models with class scores: (1, 4+num_classes, N)
        preds = preds.transpose(0, 2, 1)

    preds = preds[0]  # (N, 5+)

    results = []
    for det in preds:
        # For YOLOv8 with 1 class: columns are [cx, cy, w, h, class_conf]
        if len(det) > 5:
            conf = float(np.max(det[4:]))
        else:
            conf = float(det[4])

        if conf < PLATE_DETECT_CONF:
            continue

        cx, cy, bw, bh = det[0], det[1], det[2], det[3]
        x1 = (cx - bw / 2 - pad_w) / ratio
        y1 = (cy - bh / 2 - pad_h) / ratio
        x2 = (cx + bw / 2 - pad_w) / ratio
        y2 = (cy + bh / 2 - pad_h) / ratio

        # Clamp to image bounds
        x1 = max(0, min(int(x1), w_orig))
        y1 = max(0, min(int(y1), h_orig))
        x2 = max(0, min(int(x2), w_orig))
        y2 = max(0, min(int(y2), h_orig))

        if x2 <= x1 or y2 <= y1:
            continue

        # Filter out false positives: a real plate is a small region,
        # typically < 15% of the image area
        box_area = (x2 - x1) * (y2 - y1)
        img_area = w_orig * h_orig
        if box_area / img_area > 0.15:
            continue

        results.append({"bbox": (x1, y1, x2, y2), "confidence": conf})

    results.sort(key=lambda d: d["confidence"], reverse=True)
    return results


def get_badge_search_regions(
    image_rgb: np.ndarray, plate_bbox: tuple[int, int, int, int]
) -> list[np.ndarray]:
    """Return cropped regions where model badges typically appear.

    Badges are usually above or beside the plate on the trunk lid.
    Returns up to 2 crops: focused region above plate, and wider scan.
    """
    h_img, w_img = image_rgb.shape[:2]
    x1, y1, x2, y2 = plate_bbox
    plate_h = y2 - y1
    plate_w = x2 - x1
    plate_cx = (x1 + x2) // 2

    crops = []

    # Region 1: Above plate, with horizontal padding (focused)
    pad_x = plate_w
    r1_x1 = max(0, plate_cx - plate_w - pad_x)
    r1_x2 = min(w_img, plate_cx + plate_w + pad_x)
    r1_y1 = max(0, y1 - 4 * plate_h)
    r1_y2 = y1
    if r1_y2 > r1_y1 and r1_x2 > r1_x1:
        crops.append(image_rgb[r1_y1:r1_y2, r1_x1:r1_x2])

    # Region 2: Wider scan — full width, above plate
    r2_y1 = max(0, y1 - 5 * plate_h)
    r2_y2 = y1
    if r2_y2 > r2_y1:
        crops.append(image_rgb[r2_y1:r2_y2, 0:w_img])

    return crops


def _letterbox(
    img: np.ndarray, target_size: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with aspect ratio preserved, pad to square."""
    h, w = img.shape[:2]
    ratio = target_size / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (target_size - new_w) // 2
    pad_h = (target_size - new_h) // 2

    padded = cv2.copyMakeBorder(
        resized,
        pad_h, target_size - new_h - pad_h,
        pad_w, target_size - new_w - pad_w,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return padded, ratio, (pad_w, pad_h)
