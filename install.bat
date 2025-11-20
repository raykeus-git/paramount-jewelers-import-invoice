@echo off
echo Installing dependencies...
python -m pip install --upgrade pip
python -m pip install streamlit pandas openpyxl pdfplumber
echo.
echo Installation complete!
echo.
pause
