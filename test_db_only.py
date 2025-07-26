"""Test just the database part without LLM."""

import sys

sys.path.insert(0, "src")

try:
    print("Testing vector database only...")
    from rag.vector_database import VectorDatabase

    db = VectorDatabase("./data/chroma_db")
    print("✅ Database connected")

    movies = db.get_movies_list()
    print(f"Movies in DB: {movies}")

    if movies:
        movie = movies[0]
        print(f"Testing query for: {movie}")

        results = db.query(
            query_text="tell me about this movie", movie_title=movie, n_results=3
        )

        print(f"Found {len(results)} results:")
        for i, result in enumerate(results, 1):
            score = result["relevance_score"]
            content = result["content"][:100]
            print(f"  {i}. Score: {score:.3f} - {content}...")

except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
