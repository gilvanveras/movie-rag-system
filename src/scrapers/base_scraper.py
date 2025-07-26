"""Base scraper class and common utilities."""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options
from selenium.webdriver.edge.service import Service
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tenacity import retry, stop_after_attempt, wait_exponential

from models.movie_data import MovieData, ReviewData, ScrapingResult

logger = logging.getLogger(__name__)


@dataclass
class ScrapingConfig:
    """Configuration for web scraping."""

    delay: float = 1.0
    timeout: int = 30
    max_retries: int = 3
    user_agent: str = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    max_reviews: int = 30


class BaseScraper(ABC):
    """Base class for all movie review scrapers."""

    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.session = None
        self.driver = None

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self.session and not self.session.closed:
            # Schedule cleanup if we're in an event loop
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self.cleanup())
            except RuntimeError:
                pass  # No event loop running

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup()

    async def setup(self) -> None:
        """Setup the scraper (sessions, drivers, etc.)."""
        if not self.session:
            headers = {
                "User-Agent": self.config.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers=headers,
            )

    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self.session:
            await self.session.close()
            self.session = None
        if self.driver:
            self.driver.quit()
            self.driver = None

    def get_selenium_driver(self) -> webdriver.Edge:
        """Get a configured Edge WebDriver."""
        if not self.driver:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"--user-agent={self.config.user_agent}")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)

            # Try to find the driver in the project's drivers directory
            import os
            from pathlib import Path

            driver_path = None

            # Check project drivers directory first
            project_driver = (
                Path(__file__).parent.parent.parent / "drivers" / "msedgedriver.exe"
            )
            if project_driver.exists():
                driver_path = str(project_driver)

            # Create the driver with or without explicit path
            if driver_path:
                from selenium.webdriver.edge.service import Service

                service = Service(executable_path=driver_path)
                self.driver = webdriver.Edge(service=service, options=options)
            else:
                # Fall back to system PATH
                self.driver = webdriver.Edge(options=options)

            self.driver.execute_script(
                "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
            )

        return self.driver

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch a web page with retry logic."""
        # Ensure session is initialized
        if not self.session:
            await self.setup()

        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    logger.warning(f"HTTP {response.status} for URL: {url}")
                    return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            raise

    def parse_html(self, html: str) -> BeautifulSoup:
        """Parse HTML content with BeautifulSoup."""
        return BeautifulSoup(html, "html.parser")

    async def rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        await asyncio.sleep(self.config.delay)

    @abstractmethod
    async def search_movie(
        self, title: str, year: Optional[int] = None
    ) -> Optional[str]:
        """Search for a movie and return the URL."""
        pass

    @abstractmethod
    async def scrape_movie_data(self, movie_url: str) -> MovieData:
        """Scrape basic movie information."""
        pass

    @abstractmethod
    async def scrape_reviews(
        self, movie_url: str, max_reviews: int = 30
    ) -> List[ReviewData]:
        """Scrape reviews for a movie."""
        pass

    async def scrape_movie(
        self, title: str, year: Optional[int] = None
    ) -> ScrapingResult:
        """Complete movie scraping process."""
        start_time = time.time()
        source = self.__class__.__name__.replace("Scraper", "").lower()

        try:
            # Search for movie
            movie_url = await self.search_movie(title, year)
            if not movie_url:
                return ScrapingResult(
                    source=source, success=False, error_message="Movie not found"
                )

            # Scrape movie data
            movie_data = await self.scrape_movie_data(movie_url)

            # Scrape reviews
            reviews = await self.scrape_reviews(movie_url, self.config.max_reviews)
            movie_data.reviews = reviews

            processing_time = time.time() - start_time

            return ScrapingResult(
                source=source,
                success=True,
                movie_data=movie_data,
                reviews_count=len(reviews),
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error scraping {title} from {source}: {e}")
            return ScrapingResult(
                source=source,
                success=False,
                error_message=str(e),
                processing_time=time.time() - start_time,
            )


def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""

    # Remove extra whitespace
    text = " ".join(text.split())

    # Remove common artifacts
    text = text.replace("\n", " ").replace("\r", " ")
    text = text.replace("  ", " ").strip()

    return text


def extract_rating(text: str) -> Optional[float]:
    """Extract numeric rating from text."""
    import re

    # Common rating patterns
    patterns = [
        r"(\d+(?:\.\d+)?)\s*/\s*10",  # X/10
        r"(\d+(?:\.\d+)?)\s*/\s*5",  # X/5
        r"(\d+(?:\.\d+)?)\s*/\s*100",  # X/100
        r"(\d+(?:\.\d+)?)%",  # X%
        r"(\d+(?:\.\d+)?)",  # Just number
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            rating = float(match.group(1))

            # Normalize to 0-10 scale
            if "/10" in text:
                return rating
            elif "/5" in text:
                return rating * 2
            elif "/100" in text or "%" in text:
                return rating / 10
            else:
                # Assume 0-10 if unclear
                return rating if rating <= 10 else rating / 10

    return None
