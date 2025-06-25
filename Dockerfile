# AutoMix Docker Image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    libgomp1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY README.md ./
COPY automix/ ./automix/

# Install Python dependencies
RUN uv pip install --system .

# Optional: Install boto3 for S3 storage support
RUN uv pip install --system boto3

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=automix.web.app
ENV PYTHONUNBUFFERED=1

# Create upload directory
RUN mkdir -p /app/uploads

# Use PORT environment variable for Cloud Run compatibility
CMD python -m automix.web.app --host 0.0.0.0 --port ${PORT:-5000}