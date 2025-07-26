# Testing Guide

## Test Structure

### Official Test Suite
Located in `tests/` directory:
- `test_models.py` - Data model tests
- `test_scrapers.py` - Web scraper tests  
- `test_rag_system.py` - RAG system tests
- `test_crews.py` - CrewAI analysis tests
- `conftest.py` - Test configuration and fixtures

### Manual Testing
- `test_system.py` - Comprehensive manual test script

## Running Tests

### Run Official Test Suite
```bash
# Run all tests
uv run python -m pytest tests/ -v

# Run specific test file
uv run python -m pytest tests/test_models.py -v

# Run with coverage
uv run python -m pytest tests/ --cov=src --cov-report=html
```

### Run Manual Tests
```bash
# Basic system tests only
uv run python test_system.py --basic-only

# Full tests with default movie (The Matrix)
uv run python test_system.py

# Full tests with specific movie
uv run python test_system.py --movie "The Fantastic Four: First Steps"
```

## Test Examples

### Basic Test (No movie collection)
```bash
uv run python test_system.py --basic-only
```
Tests:
- Environment setup
- Groq API connection
- Database operations

### Full Test with Movie
```bash
uv run python test_system.py --movie "The Matrix"
```
Tests:
- All basic tests
- Movie data collection from Rotten Tomatoes
- RAG query functionality
- Sentiment analysis

## Environment Requirements

Make sure your `.env` file contains:
```
GROQ_API_KEY=your_groq_api_key_here
CHROMA_DB_PATH=./data/chroma_db
SCRAPING_DELAY=1
MAX_RETRIES=3
TIMEOUT=30
```

## Troubleshooting

- If imports fail, ensure you're running from the project root
- For web scraping tests, ensure you have internet connectivity
- For LLM tests, verify your Groq API key is valid
