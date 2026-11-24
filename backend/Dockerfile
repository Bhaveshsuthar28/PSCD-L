# CPU-compatible TRINETRA Backend Dockerfile
# Run from repository root so both backend/ and src/ are included:
#   docker build -f backend/Dockerfile -t trinetra-backend .
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY backend/requirements.txt /app/backend/requirements.txt

RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.1" \
    "torchvision>=0.17" && \
    pip install --no-cache-dir -r /app/requirements.txt -r /app/backend/requirements.txt

COPY src /app/src
COPY backend /app/backend

EXPOSE 8000

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
