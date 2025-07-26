import sys

sys.path.insert(0, "src")

try:
    print("1. Importing MovieRAGSystem...")
    from rag.movie_rag_system import MovieRAGSystem

    print("✅ Import successful")

    print("2. Creating RAG system...")
    rag = MovieRAGSystem()
    print("✅ RAG system created")

    print("3. Getting available movies...")
    movies = rag.get_available_movies()
    print(f"✅ Available movies: {movies}")

    if movies:
        print("4. Testing simple query...")
        answer = rag.query(
            "Tell me about Inception", movie_title="Inception", similarity_threshold=0.1
        )
        if len(answer) > 50:
            print(f"✅ Query successful: {answer[:100]}...")
        else:
            print(f"⚠️ Short answer: {answer}")
    else:
        print("❌ No movies found in database")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
