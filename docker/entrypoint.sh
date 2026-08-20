#!/bin/sh
set -e

# FastAPI runs in the background on 8000 (internal/programmatic use, not the
# Space's public port). Streamlit is the foreground process HF Spaces health-
# checks and serves publicly on 7860.
(cd /app/api && uvicorn main:app --host 0.0.0.0 --port 8000) &

exec streamlit run /app/ui/app.py \
  --server.port 7860 \
  --server.address 0.0.0.0 \
  --server.headless true \
  --browser.gatherUsageStats false
