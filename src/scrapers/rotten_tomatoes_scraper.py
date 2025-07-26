"""Rotten Tomatoes scraper implementation."""

import json
import logging
import re
from typing import List, Optional
from urllib.parse import quote, urljoin

from models.movie_data import MovieData, ReviewData
from scrapers.base_scraper import BaseScraper, clean_text, extract_rating

logger = logging.getLogger(__name__)


class RottenTomatoesScraper(BaseScraper):
    """Scraper for Rotten Tomatoes movie reviews."""

    BASE_URL = "https://www.rottentomatoes.com"
    SEARCH_URL = "https://www.rottentomatoes.com/search"

    async def search_movie(
        self, title: str, year: Optional[int] = None
    ) -> Optional[str]:
        """Search for a movie on Rotten Tomatoes."""
        search_query = f"{title}"
        if year:
            search_query += f" {year}"

        search_url = f"{self.SEARCH_URL}?search={quote(search_query)}"

        try:
            html = await self.fetch_page(search_url)
            if not html:
                return None

            soup = self.parse_html(html)

            # Look for movie results with more specific selectors
            movie_selectors = [
                'a[href*="/m/"][data-qa="thumbnail-link"]',
                'a[href*="/m/"]',
                'search-page-result[type="movie"] a',
            ]

            for selector in movie_selectors:
                links = soup.select(selector)
                for link in links:
                    href = link.get("href")
                    if href and "/m/" in href:
                        # Validate this is the right movie by checking title match
                        full_url = urljoin(self.BASE_URL, href)
                        if await self._validate_movie_match(full_url, title, year):
                            return full_url

            return None

        except Exception as e:
            logger.error(f"Error searching for {title}: {e}")
            return None

    async def _validate_movie_match(
        self, movie_url: str, expected_title: str, expected_year: Optional[int] = None
    ) -> bool:
        """Validate that the found movie matches the search criteria."""
        try:
            movie_data = await self.scrape_movie_data(movie_url)

            # Normalize titles for comparison
            expected_normalized = expected_title.lower().strip()
            actual_normalized = movie_data.title.lower().strip()

            # Remove common articles and punctuation for better matching
            for article in ["the ", "a ", "an "]:
                if expected_normalized.startswith(article):
                    expected_normalized = expected_normalized[len(article) :]
                if actual_normalized.startswith(article):
                    actual_normalized = actual_normalized[len(article) :]

            # Check title similarity (flexible matching)
            title_match = (
                expected_normalized in actual_normalized
                or actual_normalized in expected_normalized
                or
                # Check if they share significant words
                len(set(expected_normalized.split()) & set(actual_normalized.split()))
                >= 2
            )

            # Check year if provided (allow ±2 year difference for flexibility)
            year_match = True
            if expected_year and movie_data.year:
                year_match = abs(movie_data.year - expected_year) <= 2

            logger.info(
                f"Validation for {movie_url}: title_match={title_match}, year_match={year_match}"
            )
            logger.info(
                f"Expected: '{expected_title}' ({expected_year}), Found: '{movie_data.title}' ({movie_data.year})"
            )

            return title_match and year_match

        except Exception as e:
            logger.warning(f"Error validating movie match for {movie_url}: {e}")
            # If validation fails, be conservative and reject
            return False

    async def scrape_movie_data(self, movie_url: str) -> MovieData:
        """Scrape basic movie information from Rotten Tomatoes."""
        html = await self.fetch_page(movie_url)
        if not html:
            raise ValueError("Could not fetch movie page")

        soup = self.parse_html(html)

        # Extract title
        title_elem = soup.find("h1", {"data-qa": "score-panel-movie-title"})
        if not title_elem:
            title_elem = soup.find("h1")
        title = clean_text(title_elem.text) if title_elem else "Unknown"

        # Extract year
        year = None
        year_elem = soup.find("span", {"data-qa": "movie-info-item"})
        if year_elem:
            year_text = year_elem.text
            year_match = re.search(r"(\d{4})", year_text)
            if year_match:
                year = int(year_match.group(1))

        # Extract director
        director = None
        director_elem = soup.find("a", {"data-qa": "movie-info-director"})
        if director_elem:
            director = clean_text(director_elem.text)

        # Extract genre
        genre = None
        genre_elem = soup.find("span", {"data-qa": "movie-info-item-genre"})
        if genre_elem:
            genre = clean_text(genre_elem.text)

        # Extract plot summary
        plot_summary = None
        synopsis_elem = soup.find("div", {"data-qa": "movie-info-synopsis"})
        if synopsis_elem:
            plot_summary = clean_text(synopsis_elem.text)

        # Extract ratings
        ratings = {}

        # Tomatometer score
        tomatometer_elem = soup.find("score-board")
        if tomatometer_elem:
            tomatometer = tomatometer_elem.get("tomatometerscore")
            if tomatometer:
                ratings["tomatometer"] = float(tomatometer) / 10

        # Audience score
        audience_elem = soup.find("score-board")
        if audience_elem:
            audience = audience_elem.get("audiencescore")
            if audience:
                ratings["audience"] = float(audience) / 10

        return MovieData(
            title=title,
            year=year,
            director=director,
            genre=genre,
            plot_summary=plot_summary,
            ratings=ratings,
        )

    async def scrape_reviews(
        self, movie_url: str, max_reviews: int = 30
    ) -> List[ReviewData]:
        """Scrape reviews from Rotten Tomatoes."""
        reviews = []

        # Get critic reviews
        critic_reviews = await self._scrape_critic_reviews(movie_url, max_reviews // 2)
        reviews.extend(critic_reviews)

        # Get audience reviews
        audience_reviews = await self._scrape_audience_reviews(
            movie_url, max_reviews // 2
        )
        reviews.extend(audience_reviews)

        return reviews[:max_reviews]

    async def _scrape_critic_reviews(
        self, movie_url: str, max_reviews: int
    ) -> List[ReviewData]:
        """Scrape critic reviews."""
        reviews = []

        # The #critics-reviews fragment can't be fetched directly via HTTP
        # We need to fetch the main movie page and look for the critics section
        base_url = movie_url.rstrip("/")

        try:
            html = await self.fetch_page(base_url)
            if not html:
                return reviews

            soup = self.parse_html(html)

            # Look for the critics reviews section on the main page
            # Try multiple selectors as the site structure may vary
            review_selectors = [
                '[data-qa="review-row"]',
                "div.review-row",
                'div[class*="review-row"]',
                "div.review_table_row",
                '[data-testid="critics-review"]',
                ".critics-reviews .review-row",
            ]

            review_elements = []
            for selector in review_selectors:
                elements = soup.select(selector)
                if elements:
                    review_elements = elements
                    break

            for elem in review_elements[:max_reviews]:
                try:
                    # Extract review content - try multiple selectors
                    content = None
                    content_selectors = [
                        '[data-qa="review-text"]',
                        ".the_review",
                        '[class*="review-text"]',
                        ".review-content",
                        "p",  # Last resort
                    ]

                    for selector in content_selectors:
                        content_elem = elem.select_one(selector)
                        if content_elem:
                            content = clean_text(content_elem.text)
                            if len(content) > 20:  # Ensure it's substantial content
                                break

                    if not content or len(content) < 20:
                        continue

                    # Extract author
                    author = None
                    author_selectors = [
                        '[data-qa="review-critic-name"]',
                        ".display-name",
                        '[class*="critic-name"]',
                        'a[href*="/critics/"]',
                        ".author",
                    ]

                    for selector in author_selectors:
                        author_elem = elem.select_one(selector)
                        if author_elem:
                            author = clean_text(author_elem.text)
                            break

                    # Extract rating (fresh/rotten)
                    rating = None
                    rating_elem = elem.select_one(
                        '[class*="icon"], .review-icon, [data-qa="review-icon"]'
                    )
                    if rating_elem:
                        classes = rating_elem.get("class", [])
                        class_text = " ".join(classes).lower()
                        if "fresh" in class_text:
                            rating = 8.0  # Fresh = positive
                        elif "rotten" in class_text:
                            rating = 3.0  # Rotten = negative

                    # Extract source publication
                    source_publication = None
                    pub_selectors = [
                        '[data-qa="review-publication"]',
                        ".subtle",
                        '[class*="publication"]',
                        ".source",
                    ]

                    for selector in pub_selectors:
                        source_elem = elem.select_one(selector)
                        if source_elem:
                            source_publication = clean_text(source_elem.text)
                            break

                    review = ReviewData(
                        content=content,
                        author=author or "Anonymous Critic",
                        rating=rating,
                        source="Rotten Tomatoes",
                        review_type="critic",
                        metadata={
                            "publication": source_publication,
                            "review_type": "critic",
                            "url": base_url,
                        },
                    )

                    reviews.append(review)

                except Exception as e:
                    logger.warning(f"Error parsing critic review: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping critic reviews from {base_url}: {e}")

        return reviews

    async def _scrape_audience_reviews(
        self, movie_url: str, max_reviews: int
    ) -> List[ReviewData]:
        """Scrape audience reviews."""
        reviews = []

        # Construct audience reviews URL
        audience_url = f"{movie_url}/reviews?type=user"

        try:
            html = await self.fetch_page(audience_url)
            if not html:
                return reviews

            soup = self.parse_html(html)

            # Find audience review elements
            review_elements = soup.find_all(
                "div", class_=re.compile(r"audience-review")
            )

            for elem in review_elements[:max_reviews]:
                try:
                    # Extract review content
                    content_elem = elem.find("p", class_=re.compile(r"pre-wrap"))
                    if not content_elem:
                        continue

                    content = clean_text(content_elem.text)

                    # Extract author
                    author_elem = elem.find("span", class_=re.compile(r"display-name"))
                    author = clean_text(author_elem.text) if author_elem else None

                    # Extract star rating
                    rating = None
                    rating_elem = elem.find("span", class_=re.compile(r"star-display"))
                    if rating_elem:
                        stars = len(
                            rating_elem.find_all("span", class_=re.compile(r"filled"))
                        )
                        rating = stars * 2.0  # Convert 5-star to 10-point scale

                    review = ReviewData(
                        content=content,
                        author=author,
                        rating=rating,
                        source="Rotten Tomatoes",
                        review_type="user",
                        metadata={"review_type": "audience"},
                    )

                    reviews.append(review)

                except Exception as e:
                    logger.warning(f"Error parsing audience review: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping audience reviews: {e}")

        return reviews
