@echo off
REM Movie RAG System Setup Script for Windows

echo 🎬 Movie RAG System Setup
echo ==========================

REM Check if UV is installed
uv --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ UV is not installed. Please install UV first:
    echo    Visit: https://docs.astral.sh/uv/getting-started/installation/
    exit /b 1
)

echo ✅ UV found

REM Install dependencies
echo 📦 Installing dependencies...
uv sync

if %errorlevel% neq 0 (
    echo ❌ Failed to install dependencies
    exit /b 1
)

echo ✅ Dependencies installed

REM Create data directory
echo 📁 Creating data directory...
if not exist "data\chroma_db" mkdir data\chroma_db

REM Copy environment file
if not exist ".env" (
    echo 📋 Creating .env file...
    copy .env.example .env
    echo ⚠️  Please edit .env file with your API keys
) else (
    echo ✅ .env file already exists
)

REM Run tests
echo 🧪 Running tests...
uv run pytest -v

if %errorlevel% neq 0 (
    echo ⚠️  Some tests failed, but installation can continue
) else (
    echo ✅ All tests passed
)

echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your OpenAI API key
echo 2. Run the Streamlit app: uv run streamlit run src/app.py
echo 3. Or use the CLI: uv run python src/cli.py --help
echo.
