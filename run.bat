@echo off
echo ========================================================
echo       Starting Smart Crop Recommendation Web App
echo ========================================================
echo.

IF NOT EXIST "model\crop_model.pkl" (
    echo [INFO] Trained model not found. Running train.py first...
    python train.py
)

echo [INFO] Launching Streamlit web application...
python -m streamlit run app.py
pause
