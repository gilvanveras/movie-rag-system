"""Test configuration and fixtures."""

import os
import shutil
import sys
import tempfile
from typing import Generator
from unittest.mock import Mock, patch

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.movie_data import MovieData, ReviewData
from rag.vector_database import VectorDatabase
from scrapers.base_scraper import ScrapingConfig


@pytest.fixture
def sample_movie_data() -> MovieData:
    """Create sample movie data for testing."""
    reviews = [
        ReviewData(
            content="This movie is absolutely fantastic! Great acting and plot.",
            author="John Doe",
            rating=9.0,
            source="Test Source",
            review_type="user",
        ),
        ReviewData(
            content="Not impressed. The story was boring and predictable.",
            author="Jane Smith",
            rating=3.0,
            source="Test Source",
            review_type="critic",
        ),
        ReviewData(
            content="Decent movie with good visuals but weak character development.",
            author="Bob Wilson",
            rating=6.5,
            source="Another Source",
            review_type="user",
        ),
    ]

    return MovieData(
        title="Test Movie",
        year=2023,
        director="Test Director",
        cast=["Actor One", "Actor Two"],
        genre="Action, Drama",
        plot_summary="A test movie about testing movies.",
        ratings={"test_source": 7.5, "another_source": 6.0},
        reviews=reviews,
    )


@pytest.fixture
def sample_review_data() -> ReviewData:
    """Create sample review data for testing."""
    return ReviewData(
        content="This is a test review with good insights about the movie.",
        author="Test Reviewer",
        rating=8.0,
        source="Test Source",
        review_type="user",
        metadata={"test_key": "test_value"},
    )


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Create a temporary database path for testing."""
    temp_dir = tempfile.mkdtemp()
    db_path = os.path.join(temp_dir, "test_db")

    yield db_path

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def mock_vector_db(temp_db_path: str) -> Generator[Mock, None, None]:
    """Mock vector database for testing."""
    with patch("rag.vector_database.VectorDatabase") as mock_db:
        mock_instance = Mock()
        mock_instance.add_movie_data.return_value = None
        mock_instance.query.return_value = []
        mock_instance.get_movies_list.return_value = []
        mock_instance.get_stats.return_value = {"total_documents": 0}
        mock_db.return_value = mock_instance

        yield mock_instance


@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API calls."""
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
        with patch("langchain.chat_models.ChatOpenAI") as mock_openai:
            mock_instance = Mock()
            mock_instance.return_value.content = "Test response from AI"
            mock_openai.return_value = mock_instance
            yield mock_instance


@pytest.fixture
def scraping_config() -> ScrapingConfig:
    """Create test scraping configuration."""
    return ScrapingConfig(
        delay=0.1, timeout=5, max_retries=1, max_reviews=5  # Fast for testing
    )


@pytest.fixture
def mock_html_response() -> str:
    """Mock HTML response for scraper testing."""
    return """
    <html>
        <head><title>Test Movie</title></head>
        <body>
            <h1>Test Movie (2023)</h1>
            <div class="review">
                <p>This is a test review.</p>
                <span class="author">Test Author</span>
                <span class="rating">8/10</span>
            </div>
        </body>
    </html>
    """


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment variables."""
    test_env = {
        "OPENAI_API_KEY": "test-key",
        "CHROMA_DB_PATH": "./test_data/chroma_db",
        "SCRAPING_DELAY": "0.1",
        "MAX_RETRIES": "1",
        "TIMEOUT": "5",
        "DEBUG": "True",
    }

    with patch.dict(os.environ, test_env):
        yield


class MockAsyncSession:
    """Mock aiohttp session for testing."""

    def __init__(self, response_text: str = "", status_code: int = 200):
        self.response_text = response_text
        self.status_code = status_code

    async def get(self, url: str):
        return MockResponse(self.response_text, self.status_code)

    async def close(self):
        pass


class MockResponse:
    """Mock HTTP response for testing."""

    def __init__(self, text: str, status: int = 200):
        self._text = text
        self.status = status

    async def text(self):
        return self._text

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
