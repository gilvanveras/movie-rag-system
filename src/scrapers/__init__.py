"""Scrapers package initialization."""

from .base_scraper import BaseScraper, ScrapingConfig
from .imdb_scraper import IMDBScraper
from .metacritic_scraper import MetacriticScraper
from .rotten_tomatoes_scraper import RottenTomatoesScraper
from .scraper_manager import ScraperManager

__all__ = [
    "BaseScraper",
    "ScrapingConfig",
    "RottenTomatoesScraper",
    "IMDBScraper",
    "MetacriticScraper",
    "ScraperManager",
]
