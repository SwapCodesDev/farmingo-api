# Hugging Face Spaces-friendly Dockerfile for your FastAPI app
FROM python:3.9-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install small set of system deps needed for wheels, TF and image libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        ca-certificates \
        libjpeg-dev \
        zlib1g-dev \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy and install Python deps (as root so install can write to system)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Ensure the non-root user owns the app files
RUN chown -R appuser:appuser /app

# Run as non-root user
USER appuser
ENV PATH="/home/appuser/.local/bin:$PATH"

# Expose HF Spaces default port
EXPOSE 7860

# Entrypoint: run the FastAPI app located at api/api.py
# Note: uvicorn import path is "api.api:app"
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers"]
