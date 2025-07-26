import sys

sys.path.insert(0, "src")
from rag.movie_rag_system import MovieRAGSystem

print("Testing query system...")
rag = MovieRAGSystem()
movies = rag.get_available_movies()
print(f"Movies: {movies}")

print("Testing query with threshold 0.1...")
answer = rag.query(
    "What do people think about Inception?",
    movie_title="Inception",
    similarity_threshold=0.1,
)
print(f"Answer: {answer}")
