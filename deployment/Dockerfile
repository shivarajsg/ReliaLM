FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for bitsandbytes/torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project
COPY . .

# Expose the API port
EXPOSE 8000

# Optional: set model directory via env var
# ENV RELIALM_MODEL_DIR=/app/experiments/phase2_qlora_r16
# ENV RELIALM_BASE_MODEL=Qwen/Qwen2.5-7B-Instruct

# Start the FastAPI server from the deployment directory
CMD ["uvicorn", "deployment.app:app", "--host", "0.0.0.0", "--port", "8000"]
