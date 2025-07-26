"""Quick test of The Matrix scraping."""

import asyncio

from scrapers.rotten_tomatoes_scraper import RottenTomatoesScraper


async def quick_test():
    """Quick test of Rotten Tomatoes scraping."""
    print("Testing Rotten Tomatoes for The Matrix...")

    scraper = RottenTomatoesScraper()

    try:
        # Test direct URL
        rt_url = "https://www.rottentomatoes.com/m/matrix"
        print(f"Testing URL: {rt_url}")

        # Test movie data
        movie_data = await scraper.scrape_movie_data(rt_url)
        print(f"Movie: {movie_data.title} ({movie_data.year})")
        print(f"Director: {movie_data.director}")

        # Test reviews
        reviews = await scraper.scrape_reviews(rt_url, max_reviews=3)
        print(f"Found {len(reviews)} reviews")

        for i, review in enumerate(reviews, 1):
            print(f"Review {i}: {review.content[:50]}...")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        await scraper.cleanup()


if __name__ == "__main__":
    asyncio.run(quick_test())
