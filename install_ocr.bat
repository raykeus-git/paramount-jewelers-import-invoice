@echo off
echo Installing OCR dependencies...
echo.
pip install pytesseract pdf2image Pillow
echo.
echo Python packages installed!
echo.
echo IMPORTANT: You still need to install Tesseract OCR manually:
echo 1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
echo 2. Run the installer
echo 3. Add to PATH or configure in code
echo.
echo See INSTALL_OCR.md for detailed instructions
pause
