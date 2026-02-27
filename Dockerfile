# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and other ML libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install core Python dependencies
RUN pip install --no-cache-dir fastapi uvicorn pydantic python-multipart \
    torch torchvision timm facenet-pytorch \
    opencv-python-headless pyyaml redis

# Copy the current directory contents into the container at /app
COPY . /app

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Add backend and inference engine to PYTHONPATH so modules map correctly
ENV PYTHONPATH=/app/backend-api:/app/inference-engine

# Run the application
CMD ["uvicorn", "backend-api.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
