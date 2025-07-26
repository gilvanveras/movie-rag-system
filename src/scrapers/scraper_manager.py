"""Scraper manager for coordinating multiple review sources."""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Type

from models.movie_data import MovieData, ReviewData, ScrapingResult
from scrapers.base_scraper import BaseScraper, ScrapingConfig
from scrapers.imdb_scraper import IMDBScraper
from scrapers.metacritic_scraper import MetacriticScraper
from scrapers.rotten_tomatoes_scraper import RottenTomatoesScraper

logger = logging.getLogger(__name__)


class ScraperManager:
    """Manages multiple scrapers for different movie review sources."""

    SCRAPERS: Dict[str, Type[BaseScraper]] = {
        "Rotten Tomatoes": RottenTomatoesScraper,
        "IMDB": IMDBScraper,  # Re-enabled with improved URL handling
        "Metacritic": MetacriticScraper,
    }

    def __init__(self, config: Optional[ScrapingConfig] = None):
        self.config = config or ScrapingConfig()
        self.scrapers: Dict[str, BaseScraper] = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.setup_scrapers()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.cleanup_scrapers()

    async def setup_scrapers(self) -> None:
        """Initialize all scrapers."""
        for source_name, scraper_class in self.SCRAPERS.items():
            scraper = scraper_class(self.config)
            await scraper.setup()
            self.scrapers[source_name] = scraper

    async def cleanup_scrapers(self) -> None:
        """Cleanup all scrapers."""
        cleanup_tasks = []
        for scraper in self.scrapers.values():
            cleanup_tasks.append(scraper.cleanup())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        self.scrapers.clear()

    async def scrape_movie_from_sources(
        self,
        title: str,
        sources: List[str],
        year: Optional[int] = None,
        max_reviews: int = 30,
    ) -> Dict[str, ScrapingResult]:
        """Scrape movie data from multiple sources concurrently."""

        # Filter valid sources
        valid_sources = [source for source in sources if source in self.SCRAPERS]

        if not valid_sources:
            logger.warning(f"No valid sources found from: {sources}")
            return {}

        # Update config with max_reviews
        temp_config = ScrapingConfig(
            delay=self.config.delay,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            user_agent=self.config.user_agent,
            max_reviews=max_reviews,
        )

        # Create scraping tasks
        tasks = []
        for source in valid_sources:
            if source in self.scrapers:
                # Update scraper config
                self.scrapers[source].config = temp_config
                task = self._scrape_single_source(source, title, year)
                tasks.append(task)

        # Execute all scraping tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        scraping_results = {}
        for i, result in enumerate(results):
            source = valid_sources[i]

            if isinstance(result, Exception):
                logger.error(f"Error scraping {source}: {result}")
                scraping_results[source] = ScrapingResult(
                    source=source, success=False, error_message=str(result)
                )
            else:
                scraping_results[source] = result

        return scraping_results

    async def _scrape_single_source(
        self, source: str, title: str, year: Optional[int] = None
    ) -> ScrapingResult:
        """Scrape from a single source."""
        try:
            scraper = self.scrapers[source]
            result = await scraper.scrape_movie(title, year)

            logger.info(
                f"Scraped {source}: "
                f"{'Success' if result.success else 'Failed'}, "
                f"{result.reviews_count} reviews, "
                f"{result.processing_time:.2f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Unexpected error scraping {source}: {e}")
            return ScrapingResult(source=source, success=False, error_message=str(e))

    async def combine_movie_data(
        self, scraping_results: Dict[str, ScrapingResult]
    ) -> Optional[MovieData]:
        """Combine movie data from multiple sources into a single MovieData object."""

        successful_results = {
            source: result
            for source, result in scraping_results.items()
            if result.success and result.movie_data
        }

        if not successful_results:
            logger.warning("No successful scraping results to combine")
            return None

        # Use the first successful result as base
        base_source, base_result = next(iter(successful_results.items()))
        combined_data = base_result.movie_data

        logger.info(f"Using {base_source} as base for movie data")

        # Merge data from other sources
        for source, result in successful_results.items():
            if source == base_source:
                continue

            source_data = result.movie_data

            # Merge basic info (prefer non-empty values)
            if not combined_data.year and source_data.year:
                combined_data.year = source_data.year

            if not combined_data.director and source_data.director:
                combined_data.director = source_data.director

            if not combined_data.genre and source_data.genre:
                combined_data.genre = source_data.genre

            if not combined_data.plot_summary and source_data.plot_summary:
                combined_data.plot_summary = source_data.plot_summary

            # Merge cast (combine and deduplicate)
            if source_data.cast:
                combined_cast = set(combined_data.cast + source_data.cast)
                combined_data.cast = list(combined_cast)

            # Merge ratings
            combined_data.ratings.update(source_data.ratings)

            # Add all reviews
            combined_data.reviews.extend(source_data.reviews)

        # Sort reviews by date (newest first) if dates are available
        combined_data.reviews.sort(key=lambda r: r.date or datetime.min, reverse=True)

        logger.info(
            f"Combined movie data: {len(combined_data.reviews)} total reviews "
            f"from {len(successful_results)} sources"
        )

        return combined_data

    async def scrape_movie(
        self,
        title: str,
        sources: List[str] = None,
        year: Optional[int] = None,
        max_reviews: int = 30,
    ) -> Optional[MovieData]:
        """Complete movie scraping workflow."""

        if sources is None:
            sources = list(self.SCRAPERS.keys())

        logger.info(f"Starting scrape for '{title}' from sources: {sources}")

        # Scrape from all sources
        scraping_results = await self.scrape_movie_from_sources(
            title, sources, year, max_reviews
        )

        # Combine results
        combined_data = await self.combine_movie_data(scraping_results)

        if combined_data:
            logger.info(f"Successfully scraped movie: {combined_data.title}")
            logger.info(f"Total reviews collected: {len(combined_data.reviews)}")

            # Log source breakdown
            source_counts = {}
            for review in combined_data.reviews:
                source_counts[review.source] = source_counts.get(review.source, 0) + 1

            for source, count in source_counts.items():
                logger.info(f"  {source}: {count} reviews")

        return combined_data

    def get_available_sources(self) -> List[str]:
        """Get list of available scraper sources."""
        return list(self.SCRAPERS.keys())

    async def test_scrapers(self) -> Dict[str, bool]:
        """Test all scrapers with a simple movie."""
        test_movie = "The Matrix"
        test_results = {}

        for source in self.SCRAPERS.keys():
            try:
                scraper = self.scrapers[source]
                url = await scraper.search_movie(test_movie)
                test_results[source] = url is not None

            except Exception as e:
                logger.error(f"Test failed for {source}: {e}")
                test_results[source] = False

        return test_results
