from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from functools import partial

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile

from app.auth import verify_jwt
from app.detector import warmup as warmup_detector
from app.ocr import warmup as warmup_ocr
from app.pipeline import analyze
from app.whatsapp.router import router as whatsapp_router

load_dotenv()

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
logger = logging.getLogger(__name__)

_REQUIRED_ENV_VARS = [
    "SUPABASE_URL",
    "SUPABASE_SERVICE_ROLE_KEY",
    "WEBHOOK_SECRET",
]


def _check_env_vars() -> None:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")


def _warmup_all() -> None:
    logger.info("Warming up PaddleOCR...")
    warmup_ocr()
    logger.info("Warming up YOLO plate detector...")
    try:
        warmup_detector()
    except FileNotFoundError:
        logger.warning(
            "Plate detector ONNX model not found — "
            "YOLO detection disabled, falling back to crop-based OCR"
        )
    logger.info("Warmup complete.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    _check_env_vars()
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _warmup_all)
    yield


app = FastAPI(
    title="Vehicle Rear-Image Analyzer",
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(whatsapp_router)


@app.post("/analyze-vehicle")
async def analyze_vehicle(image: UploadFile = File(...), _: dict = Depends(verify_jwt)):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await image.read()
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Image exceeds 10 MB limit.")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, partial(analyze, data))
    return result


@app.get("/health")
async def health():
    return {"status": "ok"}
