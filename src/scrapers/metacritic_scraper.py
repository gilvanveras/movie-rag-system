"""Metacritic scraper implementation."""

import json
import logging
import re
from datetime import datetime
from typing import List, Optional
from urllib.parse import quote, urljoin

from models.movie_data import MovieData, ReviewData
from scrapers.base_scraper import BaseScraper, clean_text, extract_rating

logger = logging.getLogger(__name__)


class MetacriticScraper(BaseScraper):
    """Scraper for Metacritic movie reviews."""

    BASE_URL = "https://www.metacritic.com"
    SEARCH_URL = "https://www.metacritic.com/search/movie/"

    async def search_movie(
        self, title: str, year: Optional[int] = None
    ) -> Optional[str]:
        """Search for a movie on Metacritic."""
        search_query = title.replace(" ", "-").lower()
        search_url = f"{self.SEARCH_URL}{quote(search_query)}/results"

        try:
            html = await self.fetch_page(search_url)
            if not html:
                # Try direct URL construction
                movie_slug = title.replace(" ", "-").lower().replace(":", "")
                direct_url = f"{self.BASE_URL}/movie/{movie_slug}"
                test_html = await self.fetch_page(direct_url)
                if test_html and "404" not in test_html:
                    return direct_url
                return None

            soup = self.parse_html(html)

            # Look for movie results
            result_elements = soup.find_all("div", class_="result_wrap")

            for elem in result_elements:
                link = elem.find("a")
                if link and link.get("href"):
                    href = link.get("href")
                    if "/movie/" in href:
                        return urljoin(self.BASE_URL, href)

            return None

        except Exception as e:
            logger.error(f"Error searching for {title}: {e}")
            return None

    async def scrape_movie_data(self, movie_url: str) -> MovieData:
        """Scrape basic movie information from Metacritic."""
        html = await self.fetch_page(movie_url)
        if not html:
            raise ValueError("Could not fetch movie page")

        soup = self.parse_html(html)

        # Extract title
        title_elem = soup.find("h1", class_="product_page_title")
        if not title_elem:
            title_elem = soup.find("h1")
        title = clean_text(title_elem.text) if title_elem else "Unknown"

        # Extract year
        year = None
        year_elem = soup.find("span", class_="release_year")
        if year_elem:
            year_text = year_elem.text
            year_match = re.search(r"(\d{4})", year_text)
            if year_match:
                year = int(year_match.group(1))

        # Extract director
        director = None
        director_elem = soup.find("span", string=re.compile(r"Director:"))
        if director_elem:
            director_parent = director_elem.parent
            director_link = director_parent.find("a")
            if director_link:
                director = clean_text(director_link.text)

        # Extract cast
        cast = []
        cast_section = soup.find("div", class_="summary_cast")
        if cast_section:
            cast_links = cast_section.find_all("a")
            for link in cast_links:
                cast.append(clean_text(link.text))

        # Extract genre
        genre = None
        genre_elem = soup.find("span", string=re.compile(r"Genre:"))
        if genre_elem:
            genre_parent = genre_elem.parent
            genre_text = genre_parent.text.replace("Genre(s):", "").strip()
            genre = clean_text(genre_text)

        # Extract plot summary
        plot_summary = None
        plot_elem = soup.find("div", class_="summary_deck")
        if plot_elem:
            plot_summary = clean_text(plot_elem.text)

        # Extract ratings
        ratings = {}

        # Metascore
        metascore_elem = soup.find("div", class_="metascore_w")
        if metascore_elem:
            metascore_text = metascore_elem.text.strip()
            if metascore_text.isdigit():
                ratings["metascore"] = float(metascore_text) / 10

        # User score
        user_score_elem = soup.find("div", class_="user_score")
        if user_score_elem:
            score_elem = user_score_elem.find("div", class_="metascore_w")
            if score_elem:
                score_text = score_elem.text.strip()
                try:
                    ratings["user_score"] = float(score_text)
                except ValueError:
                    pass

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
        """Scrape reviews from Metacritic."""
        reviews = []

        # Get critic reviews
        critic_reviews = await self._scrape_critic_reviews(movie_url, max_reviews // 2)
        reviews.extend(critic_reviews)

        # Get user reviews
        user_reviews = await self._scrape_user_reviews(movie_url, max_reviews // 2)
        reviews.extend(user_reviews)

        return reviews[:max_reviews]

    async def _scrape_critic_reviews(
        self, movie_url: str, max_reviews: int
    ) -> List[ReviewData]:
        """Scrape professional critic reviews."""
        reviews = []

        # Construct critic reviews URL
        critic_url = f"{movie_url}/critic-reviews"

        try:
            html = await self.fetch_page(critic_url)
            if not html:
                return reviews

            soup = self.parse_html(html)

            # Find critic review elements
            review_elements = soup.find_all("div", class_="review_section")

            for elem in review_elements[:max_reviews]:
                try:
                    # Extract review content
                    content_elem = elem.find("div", class_="review_body")
                    if not content_elem:
                        continue

                    content = clean_text(content_elem.text)

                    # Extract author and publication
                    author_elem = elem.find("div", class_="review_critic")
                    author = None
                    publication = None

                    if author_elem:
                        critic_link = author_elem.find("a")
                        if critic_link:
                            author = clean_text(critic_link.text)

                        source_elem = author_elem.find("em")
                        if source_elem:
                            publication = clean_text(source_elem.text)

                    # Extract score
                    rating = None
                    score_elem = elem.find("div", class_="review_grade")
                    if score_elem:
                        score_text = score_elem.text.strip()
                        rating = extract_rating(score_text)

                    # Extract date
                    date = None
                    date_elem = elem.find("div", class_="review_date")
                    if date_elem:
                        date_text = date_elem.text.strip()
                        try:
                            date = datetime.strptime(date_text, "%b %d, %Y")
                        except:
                            pass

                    review = ReviewData(
                        content=content,
                        author=author,
                        rating=rating,
                        source="Metacritic",
                        url=critic_url,
                        date=date,
                        review_type="critic",
                        metadata={"publication": publication, "review_type": "critic"},
                    )

                    reviews.append(review)

                except Exception as e:
                    logger.warning(f"Error parsing Metacritic critic review: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping Metacritic critic reviews: {e}")

        return reviews

    async def _scrape_user_reviews(
        self, movie_url: str, max_reviews: int
    ) -> List[ReviewData]:
        """Scrape user reviews."""
        reviews = []

        # Construct user reviews URL
        user_url = f"{movie_url}/user-reviews"

        try:
            html = await self.fetch_page(user_url)
            if not html:
                return reviews

            soup = self.parse_html(html)

            # Find user review elements
            review_elements = soup.find_all("div", class_="review_section")

            for elem in review_elements[:max_reviews]:
                try:
                    # Extract review content
                    content_elem = elem.find("div", class_="review_body")
                    if not content_elem:
                        continue

                    content = clean_text(content_elem.text)

                    # Extract author
                    author_elem = elem.find("div", class_="review_username")
                    author = None
                    if author_elem:
                        author_link = author_elem.find("a")
                        if author_link:
                            author = clean_text(author_link.text)

                    # Extract score
                    rating = None
                    score_elem = elem.find("div", class_="review_grade")
                    if score_elem:
                        score_text = score_elem.text.strip()
                        try:
                            rating = float(score_text)
                        except ValueError:
                            pass

                    # Extract date
                    date = None
                    date_elem = elem.find("div", class_="review_date")
                    if date_elem:
                        date_text = date_elem.text.strip()
                        try:
                            date = datetime.strptime(date_text, "%b %d, %Y")
                        except:
                            pass

                    # Extract helpful votes
                    helpful_votes = None
                    helpful_elem = elem.find("span", class_="helpful_summary")
                    if helpful_elem:
                        helpful_text = helpful_elem.text
                        helpful_match = re.search(r"(\d+) of (\d+)", helpful_text)
                        if helpful_match:
                            helpful_votes = int(helpful_match.group(1))

                    review = ReviewData(
                        content=content,
                        author=author,
                        rating=rating,
                        source="Metacritic",
                        url=user_url,
                        date=date,
                        review_type="user",
                        helpful_votes=helpful_votes,
                        metadata={"review_type": "user"},
                    )

                    reviews.append(review)

                except Exception as e:
                    logger.warning(f"Error parsing Metacritic user review: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error scraping Metacritic user reviews: {e}")

        return reviews
