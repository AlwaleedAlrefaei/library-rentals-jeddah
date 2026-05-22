# syntax=docker/dockerfile:1
FROM python:3.10-slim

# System deps for numpy/pandas/tensorflow wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the source
COPY . .

# Default command — run the training pipeline
CMD ["python", "-m", "src.train", "--data", "data/jeddah_library_rentals.csv"]
