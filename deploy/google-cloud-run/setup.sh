#!/bin/bash

# Google Cloud Run deployment script for AutoMix

echo "Setting up AutoMix on Google Cloud Run..."

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud CLI is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Set variables
PROJECT_ID=$(gcloud config get-value project)
REGION="us-central1"
SERVICE_NAME="automix"

echo "Project ID: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"

# Enable required APIs
echo "Enabling required APIs..."
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Build and deploy
echo "Building and deploying..."
gcloud builds submit --config=cloudbuild.yaml

# Get the service URL
echo "Getting service URL..."
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)')
echo "Your AutoMix service is deployed at: $SERVICE_URL"

# Create a lightweight version for free tier
echo "Creating lightweight Dockerfile for free tier..."
cat > Dockerfile.lightweight << 'EOF'
FROM python:3.11-slim

# Install only essential dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN pip install uv

# Copy and install Python dependencies
COPY pyproject.toml ./
COPY README.md ./
COPY automix/ ./automix/

# Install with minimal dependencies
RUN uv pip install --system . --no-cache-dir

# Use gunicorn for production
RUN uv pip install --system gunicorn

# Cloud Run port
ENV PORT 8080
EXPOSE 8080

# Run with gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 300 automix.web.app:app
EOF

echo "Setup complete!"
echo ""
echo "To deploy the lightweight version (recommended for free tier):"
echo "1. Replace Dockerfile with Dockerfile.lightweight"
echo "2. Run: gcloud builds submit --config=cloudbuild.yaml"
echo ""
echo "Free tier limits:"
echo "- 2 million requests/month"
echo "- 180,000 vCPU-seconds/month" 
echo "- 360,000 GiB-seconds/month"