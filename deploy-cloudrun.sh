#!/usr/bin/env bash
set -euo pipefail

# ── Configuration ──
PROJECT_ID="${GCP_PROJECT_ID:?Set GCP_PROJECT_ID environment variable}"
REGION="${GCP_REGION:-us-central1}"
SERVICE_NAME="pet-lab-api"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "==> Building Docker image..."
docker build --platform linux/amd64 -t "${IMAGE}" ./backend

echo "==> Pushing image to Google Container Registry..."
docker push "${IMAGE}"

echo "==> Deploying to Cloud Run..."
gcloud run deploy "${SERVICE_NAME}" \
  --image "${IMAGE}" \
  --platform managed \
  --region "${REGION}" \
  --memory 2Gi \
  --cpu 1 \
  --timeout 300 \
  --concurrency 10 \
  --min-instances 0 \
  --max-instances 2 \
  --port 8000 \
  --allow-unauthenticated

echo ""
echo "==> Deployed! Getting service URL..."
URL=$(gcloud run services describe "${SERVICE_NAME}" --region "${REGION}" --format 'value(status.url)')
echo "==> Your API is live at: ${URL}"
echo ""
echo "==> Next: update PET_Lab_Website/index.html to use this URL:"
echo "    const API = '${URL}';"
