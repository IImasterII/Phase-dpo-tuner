@echo off
cd /d "%~dp0"
if exist "%~dp0venv\Scripts\activate.bat" (
    call "%~dp0venv\Scripts\activate.bat"
    python "%~dp0phase_trainer_gui.py"
) else (
    echo venv not found. Creating...
    python -m venv venv
    call "%~dp0venv\Scripts\activate.bat"
    echo.
    echo Installing dependencies...
    pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
    pip install "git+https://github.com/huggingface/transformers.git@main"
    pip install peft==0.7.1 trl bitsandbytes accelerate customtkinter datasets
    echo.
    echo Done! Run again to start.
    pause
    exit /b
)
pause
