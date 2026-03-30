#!/bin/bash
cd /workspaces/genbi/Genbi-main
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
streamlit run app.py \
  --server.port 8501 \
  --server.headless true \
  --server.enableXsrfProtection false \
  --client.showErrorDetails true
