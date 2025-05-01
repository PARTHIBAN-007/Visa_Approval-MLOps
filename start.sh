#!/bin/bash
mlflow server \
  --backend-store-uri ./mlruns \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000 &
  

python main.py 

uvicorn app:app --host 0.0.0.0 --port 8000 &

streamlit run streamlit.py --server.port 8501
