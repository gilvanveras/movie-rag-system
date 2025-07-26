#!/bin/bash

# Movie RAG System Setup Script

echo "🎬 Movie RAG System Setup"
echo "=========================="

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV is not installed. Please install UV first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✅ UV found"

# Install dependencies
echo "📦 Installing dependencies..."
uv sync

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo "✅ Dependencies installed"

# Create data directory
echo "📁 Creating data directory..."
mkdir -p data/chroma_db

# Copy environment file
if [ ! -f .env ]; then
    echo "📋 Creating .env file..."
    cp .env.example .env
    echo "⚠️  Please edit .env file with your API keys"
else
    echo "✅ .env file already exists"
fi

# Run tests
echo "🧪 Running tests..."
uv run pytest -v

if [ $? -ne 0 ]; then
    echo "⚠️  Some tests failed, but installation can continue"
else
    echo "✅ All tests passed"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your OpenAI API key"
echo "2. Run the Streamlit app: uv run streamlit run src/app.py"
echo "3. Or use the CLI: uv run python src/cli.py --help"
echo ""
