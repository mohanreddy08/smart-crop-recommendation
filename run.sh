#!/usr/bin/env bash
set -e

echo "========================================================"
echo "      Starting Smart Crop Recommendation Web App"
echo "========================================================"
echo ""

if [ ! -f "model/crop_model.pkl" ]; then
    echo "[INFO] Trained model not found. Running train.py first..."
    python3 train.py
fi

echo "[INFO] Launching Streamlit web application..."
python3 -m streamlit run app.py
