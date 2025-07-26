"""IMDB scraper implementation."""

import json
import logging
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote, urljoin

from models.movie_data import MovieData, ReviewData
from scrapers.base_scraper import BaseScraper, clean_text, extract_rating

logger = logging.getLogger(__name__)


class IMDBScraper(BaseScraper):
    """Scraper for IMDB movie reviews."""

    BASE_URL = "https://www.imdb.com"
    SEARCH_URL = "https://www.imdb.com/find"

    async def search_movie(
        self, title: str, year: Optional[int] = None
    ) -> Optional[str]:
        """Search for a movie on IMDB."""
        search_query = f"{title}"
        if year:
            search_query += f" {year}"

        search_url = f"{self.SEARCH_URL}?q={quote(search_query)}&s=tt&ttype=ft"

        try:
            html = await self.fetch_page(search_url)
            if not html:
                return None

            soup = self.parse_html(html)

            # Look for movie results with improved selectors
            result_selectors = [
                "td.result_text",
                '[data-testid="find-result-section-title"] li',
                ".findResult",
            ]

            for selector in result_selectors:
                result_elements = soup.select(selector)
                for elem in result_elements:
                    link = elem.select_one('a[href*="/title/"]')
                    if link and link.get("href"):
                        href = link.get("href")
                        if "/title/tt" in href:
                            full_url = urljoin(self.BASE_URL, href)
                            # Validate this is the right movie
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
        """Scrape basic movie information from IMDB."""
        html = await self.fetch_page(movie_url)
        if not html:
            raise ValueError("Could not fetch movie page")

        soup = self.parse_html(html)

        # Extract title
        title_elem = soup.find("h1", {"data-testid": "hero__pageTitle"})
        if not title_elem:
            title_elem = soup.find("h1")
        title = clean_text(title_elem.text) if title_elem else "Unknown"

        # Extract year
        year = None
        year_elem = soup.find("span", class_="sc-52284603-2")
        if year_elem:
            year_text = year_elem.text
            year_match = re.search(r"(\d{4})", year_text)
            if year_match:
                year = int(year_match.group(1))

        # Extract director
        director = None
        director_elem = soup.find(
            "a", {"class": "ipc-metadata-list-item__list-content-item"}
        )
        if director_elem:
            director = clean_text(director_elem.text)

        # Extract cast
        cast = []
        cast_section = soup.find("section", {"data-testid": "title-cast"})
        if cast_section:
            cast_links = cast_section.find_all("a", class_="sc-bfec09a1-1")
            for link in cast_links[:5]:  # Top 5 cast members
                cast.append(clean_text(link.text))

        # Extract genre
        genre = None
        genre_section = soup.find("div", {"data-testid": "genres"})
        if genre_section:
            genre_links = genre_section.find_all("a")
            if genre_links:
                genres = [clean_text(link.text) for link in genre_links]
                genre = ", ".join(genres)

        # Extract plot summary
        plot_summary = None
        plot_elem = soup.find("span", {"data-testid": "plot-summary"})
        if plot_elem:
            plot_summary = clean_text(plot_elem.text)

        # Extract rating
        ratings = {}
        rating_elem = soup.find("span", class_="sc-7ab21ed2-1")
        if rating_elem:
            rating_text = rating_elem.text
            rating_match = re.search(r"(\d+\.?\d*)", rating_text)
            if rating_match:
                ratings["imdb"] = float(rating_match.group(1))

        return MovieData(
            title=title,
            year=year,
            director=director,
            cast=cast,
            genre=genre,
            plot_summary=plot_summary,
            ratings=ratings,
        )

    async def scrape_reviews(
        self, movie_url: str, max_reviews: int = 30
    ) -> List[ReviewData]:
        """Scrape reviews from IMDB."""
        reviews = []

        # Extract movie ID from URL - handle both formats
        movie_id_match = re.search(r"/title/(tt\d+)", movie_url)
        if not movie_id_match:
            return reviews

        movie_id = movie_id_match.group(1)
        # Use the exact format from the user's URL with proper reviews path
        # Try both international and localized versions
        reviews_urls = [
            f"{self.BASE_URL}/title/{movie_id}/reviews/?ref_=tt_ov_ururv",
            f"{self.BASE_URL}/pt/title/{movie_id}/reviews/?ref_=tt_ov_ururv",
        ]

        html = None
        successful_url = None

        # Try each URL until one works
        for reviews_url in reviews_urls:
            try:
                html = await self.fetch_page(reviews_url)
                if html:
                    successful_url = reviews_url
                    break
            except Exception as e:
                logger.warning(f"Failed to fetch {reviews_url}: {e}")
                continue

        if not html:
            return reviews

        try:
            soup = self.parse_html(html)

            # Find review containers - try multiple selectors
            review_selectors = [
                "div.review-container",
                'div[data-testid="review-container"]',
                "div.lister-item",
                'div[class*="review"]',
            ]

            review_containers = []
            for selector in review_selectors:
                containers = soup.select(selector)
                if containers:
                    review_containers = containers
                    break

            for container in review_containers[:max_reviews]:
                try:
                    # Extract review content - try multiple selectors
                    content = None
                    content_selectors = [
                        "div.text.show-more__control",
                        "div.text",
                        '[data-testid="review-summary"]',
                        ".content",
                    ]

                    for selector in content_selectors:
                        content_elem = container.select_one(selector)
                        if content_elem:
                            content = clean_text(content_elem.text)
                            break

                    if not content:
                        continue

                    # Extract author
                    author = None
                    author_selectors = [
                        ".display-name-link a",
                        '[data-testid="author-name"]',
                        ".author a",
                    ]

                    for selector in author_selectors:
                        author_elem = container.select_one(selector)
                        if author_elem:
                            author = clean_text(author_elem.text)
                            break

                    # Extract rating
                    rating = None
                    rating_selectors = [
                        ".rating-other-user-rating span",
                        '[data-testid="review-rating"]',
                        ".ipl-ratings-bar span",
                    ]

                    for selector in rating_selectors:
                        rating_elem = container.select_one(selector)
                        if rating_elem:
                            rating_text = rating_elem.text
                            rating_match = re.search(r"(\d+)", rating_text)
                            if rating_match:
                                rating = float(rating_match.group(1))
                                break

                    # Extract date
                    date = None
                    date_selectors = [
                        ".review-date",
                        '[data-testid="review-date"]',
                        ".date",
                    ]

                    for selector in date_selectors:
                        date_elem = container.select_one(selector)
                        if date_elem:
                            date = clean_text(date_elem.text)
                            break

                    review = ReviewData(
                        content=content,
                        author=author or "Anonymous User",
                        rating=rating,
                        source="IMDB",
                        review_type="user",
                        metadata={
                            "date": date,
                            "review_type": "user",
                            "url": successful_url,
                        },
                    )

                    reviews.append(review)

                except Exception as e:
                    logger.warning(f"Error parsing IMDB review: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping IMDB reviews from {successful_url}: {e}")

        return reviews

    async def _load_more_reviews(
        self, base_url: str, existing_reviews: List[ReviewData], needed: int
    ) -> None:
        """Load additional reviews if available."""
        try:
            # IMDB uses pagination, try to get next page
            current_count = len(existing_reviews)
            next_page_url = f"{base_url}?start={current_count}"

            html = await self.fetch_page(next_page_url)
            if not html:
                return

            soup = self.parse_html(html)
            review_containers = soup.find_all("div", class_="review-container")

            for container in review_containers[:needed]:
                try:
                    # Similar parsing logic as above
                    content_elem = container.find("div", class_="text")
                    if not content_elem:
                        continue

                    content = clean_text(content_elem.text)

                    author_elem = container.find("span", class_="display-name-link")
                    author = clean_text(author_elem.text) if author_elem else None

                    rating = None
                    rating_elem = container.find(
                        "span", class_="rating-other-user-rating"
                    )
                    if rating_elem:
                        rating_span = rating_elem.find("span")
                        if rating_span:
                            rating_text = rating_span.text
                            rating_match = re.search(r"(\d+)", rating_text)
                            if rating_match:
                                rating = float(rating_match.group(1))

                    review = ReviewData(
                        content=content,
                        author=author,
                        rating=rating,
                        source="IMDB",
                        url=next_page_url,
                        review_type="user",
                        metadata={"review_type": "user", "page": 2},
                    )

                    existing_reviews.append(review)

                except Exception as e:
                    logger.warning(f"Error parsing additional IMDB review: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error loading more IMDB reviews: {e}")
