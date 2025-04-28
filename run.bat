@echo off
echo Starting Stock Price Prediction System...
echo.
echo Make sure you have installed all required dependencies:
echo pip install -r requirements.txt
echo pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
echo.

REM Create necessary directories if they don't exist
if not exist "data" mkdir data
if not exist "models" mkdir models

REM Run the Streamlit app
streamlit run app.py

pause

