FROM python:3.11-slim

# System deps for opencv-headless and ONNX Runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Note: PaddleOCR models are downloaded on first startup (requires ~500MB)
# Commenting out pre-download to avoid OOM during build

EXPOSE 8000
# Use 1 worker to avoid concurrent PaddleOCR model downloads
# Can scale to more workers after models are cached
CMD ["gunicorn", "app.main:app", "--workers", "1", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
