#!/usr/bin/env python3
"""Add sample movie data to test Streamlit functionality."""

import os
import sys

from dotenv import load_dotenv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
load_dotenv()


def add_sample_movies():
    """Add sample movie data to the database for testing."""
    print("🎬 Adding Sample Movie Data")
    print("=" * 40)

    try:
        from models.movie_data import MovieData, ReviewData
        from rag.movie_rag_system import MovieRAGSystem

        rag_system = MovieRAGSystem()

        # Sample movie 1: The Matrix (popular movie)
        matrix_reviews = [
            ReviewData(
                content="The Matrix is a groundbreaking sci-fi masterpiece. The special effects were revolutionary for 1999, and Keanu Reeves delivers an excellent performance. The philosophical themes about reality and choice are thought-provoking.",
                rating=9.2,
                source="IMDB",
                author="MovieFan2023",
                review_type="user",
            ),
            ReviewData(
                content="A visually stunning achievement in cinema. The Wachowskis created something truly special here. The action sequences are incredible and the story is compelling throughout.",
                rating=9.0,
                source="Rotten Tomatoes",
                author="CriticReviewer",
                review_type="critic",
            ),
            ReviewData(
                content="While the effects are impressive, the plot can be confusing at times. Still an entertaining watch with great action scenes.",
                rating=7.5,
                source="IMDB",
                author="CasualViewer",
                review_type="user",
            ),
            ReviewData(
                content="The Matrix redefined action cinema. Groundbreaking visual effects, stellar performances, and a story that challenges perception of reality.",
                rating=9.5,
                source="Metacritic",
                author="FilmCritic",
                review_type="critic",
            ),
            ReviewData(
                content="Amazing movie! The bullet-time effects blew my mind. Great story about questioning reality and fighting for freedom.",
                rating=8.8,
                source="IMDB",
                author="ActionFan",
                review_type="user",
            ),
        ]

        matrix_movie = MovieData(
            title="The Matrix",
            year=1999,
            director="The Wachowskis",
            genre="Action, Sci-Fi",
            reviews=matrix_reviews,
            ratings={"imdb": 8.7, "rotten_tomatoes": 88, "metacritic": 73},
        )

        # Sample movie 2: Inception
        inception_reviews = [
            ReviewData(
                content="Christopher Nolan creates a mind-bending masterpiece. The layered dream sequences are brilliantly executed, and Leonardo DiCaprio gives a powerful performance.",
                rating=9.0,
                source="IMDB",
                author="DreamWatcher",
                review_type="user",
            ),
            ReviewData(
                content="Inception is a complex and ambitious film that rewards multiple viewings. The visual effects are stunning and the concept is original.",
                rating=8.5,
                source="Rotten Tomatoes",
                author="FilmAnalyst",
                review_type="critic",
            ),
            ReviewData(
                content="Confusing plot but amazing visuals. Had to watch it twice to understand everything. Worth it for the spectacle alone.",
                rating=7.0,
                source="IMDB",
                author="ConfusedViewer",
                review_type="user",
            ),
            ReviewData(
                content="A tour de force of filmmaking. Nolan constructs a labyrinthine plot that never loses its emotional core. Technical excellence throughout.",
                rating=9.2,
                source="Metacritic",
                author="CinePhile",
                review_type="critic",
            ),
        ]

        inception_movie = MovieData(
            title="Inception",
            year=2010,
            director="Christopher Nolan",
            genre="Action, Sci-Fi, Thriller",
            reviews=inception_reviews,
            ratings={"imdb": 8.8, "rotten_tomatoes": 87, "metacritic": 74},
        )

        # Add movies to database
        print("Adding 'The Matrix' to database...")
        rag_system.add_movie_data(matrix_movie)
        print("✅ The Matrix added!")

        print("Adding 'Inception' to database...")
        rag_system.add_movie_data(inception_movie)
        print("✅ Inception added!")

        # Verify database
        stats = rag_system.get_database_stats()
        movies = rag_system.get_available_movies()

        print(f"\n📊 Database updated:")
        print(f"   Total movies: {stats.get('movies_count', 0)}")
        print(f"   Total reviews: {stats.get('reviews_count', 0)}")
        print(f"   Available movies: {movies}")

        print(f"\n🎉 SUCCESS! You can now:")
        print(f"1. Search for 'The Matrix' in Streamlit")
        print(f"2. Search for 'Inception' in Streamlit")
        print(
            f"3. Ask questions like: 'What do people think about the special effects?'"
        )

        return True

    except Exception as e:
        print(f"❌ Error adding sample data: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_sample_query():
    """Test querying the sample data."""
    print(f"\n🔍 Testing Sample Query")
    print("=" * 30)

    try:
        from rag.movie_rag_system import MovieRAGSystem

        rag_system = MovieRAGSystem()

        # Test query
        question = "What do people think about The Matrix special effects?"
        print(f"Question: {question}")

        answer = rag_system.query(question)
        print(f"Answer: {answer[:200]}...")

        print(f"\n✅ Querying works! Streamlit Q&A should work too.")
        return True

    except Exception as e:
        print(f"❌ Query test failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Setting up sample data for Streamlit testing")
    print("=" * 60)

    success1 = add_sample_movies()
    success2 = test_sample_query()

    if success1 and success2:
        print(f"\n🎉 READY FOR STREAMLIT!")
        print(f"Run: streamlit run src/app.py")
        print(f"Then search for 'The Matrix' or 'Inception'")
    else:
        print(f"\n❌ Setup failed. Check errors above.")
