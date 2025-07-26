"""Tests for base scraper functionality."""

import asyncio
import os
import sys
from unittest.mock import AsyncMock, Mock, patch

import pytest
from bs4 import BeautifulSoup

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.movie_data import MovieData, ReviewData, ScrapingResult
from scrapers.base_scraper import (
    BaseScraper,
    ScrapingConfig,
    clean_text,
    extract_rating,
)


class TestScrapingConfig:
    """Test ScrapingConfig class."""

    def test_scraping_config_defaults(self):
        """Test default configuration values."""
        config = ScrapingConfig()

        assert config.delay == 1.0
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.max_reviews == 30
        assert "Mozilla" in config.user_agent

    def test_scraping_config_custom(self):
        """Test custom configuration values."""
        config = ScrapingConfig(delay=2.0, timeout=60, max_retries=5, max_reviews=50)

        assert config.delay == 2.0
        assert config.timeout == 60
        assert config.max_retries == 5
        assert config.max_reviews == 50


class MockScraper(BaseScraper):
    """Mock scraper for testing base functionality."""

    async def search_movie(self, title: str, year=None):
        return f"http://test.com/movie/{title.replace(' ', '-')}"

    async def scrape_movie_data(self, movie_url: str):
        return MovieData(title="Test Movie")

    async def scrape_reviews(self, movie_url: str, max_reviews: int = 30):
        return [
            ReviewData(content="Test review", source="test"),
            ReviewData(content="Another review", source="test"),
        ]


class TestBaseScraper:
    """Test BaseScraper base class."""

    @pytest.fixture
    def mock_scraper(self, scraping_config):
        """Create a mock scraper instance."""
        return MockScraper(scraping_config)

    @pytest.mark.asyncio
    async def test_scraper_context_manager(self, mock_scraper):
        """Test scraper as async context manager."""
        async with mock_scraper as scraper:
            assert scraper.session is not None

        # Session should be closed after context exit
        assert mock_scraper.session is None or mock_scraper.session.closed

    @pytest.mark.asyncio
    async def test_fetch_page_success(self, mock_scraper):
        """Test successful page fetching."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.text.return_value = "<html>Test content</html>"
            mock_get.return_value.__aenter__.return_value = mock_response

            async with mock_scraper:
                html = await mock_scraper.fetch_page("http://test.com")

            assert html == "<html>Test content</html>"

    @pytest.mark.asyncio
    async def test_fetch_page_failure(self, mock_scraper):
        """Test page fetching with HTTP error."""
        with patch("aiohttp.ClientSession.get") as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 404
            mock_get.return_value.__aenter__.return_value = mock_response

            async with mock_scraper:
                html = await mock_scraper.fetch_page("http://test.com/notfound")

            assert html is None

    def test_parse_html(self, mock_scraper):
        """Test HTML parsing."""
        html = "<html><body><h1>Test</h1></body></html>"
        soup = mock_scraper.parse_html(html)

        assert isinstance(soup, BeautifulSoup)
        assert soup.find("h1").text == "Test"

    @pytest.mark.asyncio
    async def test_rate_limit(self, mock_scraper):
        """Test rate limiting functionality."""
        import time

        start_time = time.time()
        await mock_scraper.rate_limit()
        end_time = time.time()

        # Should wait at least the configured delay
        assert end_time - start_time >= mock_scraper.config.delay

    @pytest.mark.asyncio
    async def test_scrape_movie_success(self, mock_scraper):
        """Test successful movie scraping."""
        async with mock_scraper:
            result = await mock_scraper.scrape_movie("Test Movie")

        assert isinstance(result, ScrapingResult)
        assert result.success is True
        assert result.movie_data is not None
        assert result.movie_data.title == "Test Movie"
        assert result.reviews_count == 2

    @pytest.mark.asyncio
    async def test_scrape_movie_not_found(self, mock_scraper):
        """Test movie scraping when movie not found."""
        # Override search_movie to return None
        mock_scraper.search_movie = AsyncMock(return_value=None)

        async with mock_scraper:
            result = await mock_scraper.scrape_movie("Nonexistent Movie")

        assert isinstance(result, ScrapingResult)
        assert result.success is False
        assert result.error_message == "Movie not found"


class TestUtilityFunctions:
    """Test utility functions."""

    def test_clean_text_basic(self):
        """Test basic text cleaning."""
        dirty_text = "  This  is   a\n\r  test  "
        clean = clean_text(dirty_text)

        assert clean == "This is a test"

    def test_clean_text_empty(self):
        """Test cleaning empty or None text."""
        assert clean_text("") == ""
        assert clean_text(None) == ""
        assert clean_text("   ") == ""

    def test_clean_text_newlines(self):
        """Test cleaning text with newlines."""
        text_with_newlines = "Line 1\nLine 2\r\nLine 3"
        clean = clean_text(text_with_newlines)

        assert clean == "Line 1 Line 2 Line 3"

    def test_extract_rating_ten_scale(self):
        """Test rating extraction for 10-point scale."""
        assert extract_rating("8.5/10") == 8.5
        assert extract_rating("Rating: 7/10") == 7.0
        assert extract_rating("Score 9.2 / 10") == 9.2

    def test_extract_rating_five_scale(self):
        """Test rating extraction for 5-point scale."""
        assert extract_rating("4/5 stars") == 8.0  # Converted to 10-point
        assert extract_rating("3.5/5") == 7.0

    def test_extract_rating_percentage(self):
        """Test rating extraction for percentage."""
        assert extract_rating("85%") == 8.5
        assert extract_rating("Score: 92%") == 9.2

    def test_extract_rating_hundred_scale(self):
        """Test rating extraction for 100-point scale."""
        assert extract_rating("75/100") == 7.5
        assert extract_rating("88 out of 100") == 8.8

    def test_extract_rating_no_match(self):
        """Test rating extraction with no valid pattern."""
        assert extract_rating("Great movie!") is None
        assert extract_rating("No numbers here") is None
        assert extract_rating("") is None

    def test_extract_rating_edge_cases(self):
        """Test rating extraction edge cases."""
        # Very high numbers should be normalized
        assert extract_rating("95") == 9.5  # Assumes /10 if > 10

        # Multiple patterns - should match first
        assert extract_rating("8/10 and 4/5") == 8.0
