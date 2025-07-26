"""Command-line interface for the Movie RAG System."""

import argparse
import asyncio
import logging
import sys
from typing import List, Optional

from models.movie_data import MovieData
from rag.movie_rag_system import MovieRAGSystem


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("movie_rag.log"),
        ],
    )


async def add_movie_command(
    rag_system: MovieRAGSystem,
    title: str,
    sources: List[str],
    max_reviews: int,
    year: Optional[int] = None,
) -> None:
    """Add a movie to the RAG system."""
    print(f"\n🎬 Adding movie: {title}")
    print(f"Sources: {', '.join(sources)}")
    print(f"Max reviews per source: {max_reviews}")

    try:
        # Collect movie data
        print("\n🔍 Collecting movie data...")
        movie_data = await rag_system.collect_movie_data(
            movie_title=title, sources=sources, max_reviews=max_reviews, year=year
        )

        if not movie_data:
            print("❌ Movie not found or no data available!")
            return

        # Add to database
        print("📝 Adding to database...")
        rag_system.add_movie_data(movie_data)

        # Show summary
        print(f"\n✅ Successfully added '{movie_data.title}'")
        print(f"   Year: {movie_data.year}")
        print(f"   Director: {movie_data.director}")
        print(f"   Reviews collected: {len(movie_data.reviews)}")

        # Show source breakdown
        source_counts = {}
        for review in movie_data.reviews:
            source_counts[review.source] = source_counts.get(review.source, 0) + 1

        print("   Review breakdown:")
        for source, count in source_counts.items():
            print(f"     {source}: {count} reviews")

    except Exception as e:
        print(f"❌ Error adding movie: {e}")
        logging.error(f"Error adding movie {title}: {e}")


def query_command(
    rag_system: MovieRAGSystem, question: str, movie_title: Optional[str] = None
) -> None:
    """Query the RAG system."""
    print(f"\n❓ Question: {question}")
    if movie_title:
        print(f"🎬 Context: {movie_title}")

    try:
        response = rag_system.query(question, movie_title)
        print(f"\n💬 Answer:\n{response}")

    except Exception as e:
        print(f"❌ Error processing query: {e}")
        logging.error(f"Error processing query '{question}': {e}")


def list_movies_command(rag_system: MovieRAGSystem) -> None:
    """List all movies in the database."""
    try:
        movies = rag_system.get_available_movies()

        if not movies:
            print("📁 No movies found in the database.")
            print("Use the 'add' command to add movies first.")
            return

        print(f"\n📚 Movies in database ({len(movies)}):")
        for i, movie in enumerate(movies, 1):
            print(f"  {i}. {movie}")

    except Exception as e:
        print(f"❌ Error listing movies: {e}")


def stats_command(rag_system: MovieRAGSystem) -> None:
    """Show database statistics."""
    try:
        stats = rag_system.get_database_stats()

        print("\n📊 Database Statistics:")
        print(f"  Total documents: {stats.get('total_documents', 0)}")
        print(f"  Movies: {stats.get('movies_count', 0)}")
        print(f"  Reviews: {stats.get('reviews_count', 0)}")

        source_breakdown = stats.get("source_breakdown", {})
        if source_breakdown:
            print("  Reviews by source:")
            for source, count in source_breakdown.items():
                print(f"    {source}: {count}")

    except Exception as e:
        print(f"❌ Error getting stats: {e}")


def sentiment_command(rag_system: MovieRAGSystem, movie_title: str) -> None:
    """Analyze sentiment for a movie."""
    try:
        sentiment = rag_system.get_sentiment_analysis(movie_title)

        if "error" in sentiment:
            print(f"❌ Error: {sentiment['error']}")
            return

        print(f"\n😊 Sentiment Analysis for '{movie_title}':")
        print(f"  Total reviews: {sentiment['total_reviews']}")
        print(f"  Rated reviews: {sentiment['rated_reviews']}")

        if sentiment["rated_reviews"] > 0:
            print(f"  Average rating: {sentiment['average_rating']}/10")

        percentages = sentiment["sentiment_percentages"]
        print("  Sentiment distribution:")
        print(f"    Positive: {percentages['positive']}%")
        print(f"    Neutral: {percentages['neutral']}%")
        print(f"    Negative: {percentages['negative']}%")

    except Exception as e:
        print(f"❌ Error analyzing sentiment: {e}")


def main() -> None:
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="Movie RAG System - Analyze movie reviews with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a movie
  python cli.py add "The Matrix" --sources "IMDB" "Rotten Tomatoes" --max-reviews 50

  # Query about a movie
  python cli.py query "What do critics think about the acting?" --movie "The Matrix"

  # List all movies
  python cli.py list

  # Show database stats
  python cli.py stats

  # Analyze sentiment
  python cli.py sentiment "The Matrix"
        """,
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add movie command
    add_parser = subparsers.add_parser("add", help="Add a movie to the database")
    add_parser.add_argument("title", help="Movie title")
    add_parser.add_argument(
        "--sources",
        nargs="+",
        choices=["Rotten Tomatoes", "IMDB", "Metacritic"],
        default=["Rotten Tomatoes", "IMDB", "Metacritic"],
        help="Review sources to scrape from",
    )
    add_parser.add_argument(
        "--max-reviews",
        type=int,
        default=30,
        help="Maximum reviews per source (default: 30)",
    )
    add_parser.add_argument("--year", type=int, help="Movie release year")

    # Query command
    query_parser = subparsers.add_parser("query", help="Ask a question about movies")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--movie", help="Specific movie to ask about")

    # List command
    subparsers.add_parser("list", help="List all movies in the database")

    # Stats command
    subparsers.add_parser("stats", help="Show database statistics")

    # Sentiment command
    sentiment_parser = subparsers.add_parser(
        "sentiment", help="Analyze sentiment for a movie"
    )
    sentiment_parser.add_argument("movie", help="Movie title to analyze")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging(args.verbose)

    # Initialize RAG system
    print("🚀 Initializing Movie RAG System...")
    rag_system = MovieRAGSystem()

    try:
        # Execute command
        if args.command == "add":
            asyncio.run(
                add_movie_command(
                    rag_system, args.title, args.sources, args.max_reviews, args.year
                )
            )
        elif args.command == "query":
            query_command(rag_system, args.question, args.movie)
        elif args.command == "list":
            list_movies_command(rag_system)
        elif args.command == "stats":
            stats_command(rag_system)
        elif args.command == "sentiment":
            sentiment_command(rag_system, args.movie)

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        logging.error(f"Unexpected error in CLI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
