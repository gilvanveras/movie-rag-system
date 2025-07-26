"""Vector database manager using ChromaDB."""

import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from models.movie_data import MovieData, ReviewData

logger = logging.getLogger(__name__)


class VectorDatabase:
    """Vector database for storing and retrieving movie review embeddings."""

    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = db_path
        self.client = None
        self.collection = None

        # Ensure database directory exists
        os.makedirs(db_path, exist_ok=True)

        # Initialize ChromaDB client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection."""
        try:
            # Check for OpenAI embeddings first, then use default
            openai_api_key = os.getenv("OPENAI_API_KEY")

            if openai_api_key:
                logger.info("Using OpenAI embeddings for vector database")
                embedding_function = embedding_functions.OpenAIEmbeddingFunction(
                    api_key=openai_api_key, model_name="text-embedding-ada-002"
                )
            else:
                logger.info(
                    "Using default sentence transformer embeddings for vector database"
                )
                # Fallback to default sentence transformer
                embedding_function = embedding_functions.DefaultEmbeddingFunction()

            # Initialize client
            self.client = chromadb.PersistentClient(
                path=self.db_path, settings=Settings(anonymized_telemetry=False)
            )

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name="movie_reviews",
                embedding_function=embedding_function,
                metadata={"description": "Movie reviews and metadata for RAG system"},
            )

            logger.info(f"Initialized vector database at {self.db_path}")
            logger.info(f"Collection has {self.collection.count()} documents")

        except Exception as e:
            logger.error(f"Failed to initialize vector database: {e}")
            raise

    def add_movie_data(self, movie_data: MovieData) -> None:
        """Add movie data to the vector database."""
        try:
            documents = []
            metadatas = []
            ids = []

            # Add movie overview document
            overview_text = self._create_movie_overview(movie_data)
            documents.append(overview_text)

            # Safely handle genre - convert list to string if needed
            genre_str = movie_data.genre or ""
            if isinstance(genre_str, list):
                genre_str = ", ".join(genre_str)

            metadatas.append(
                {
                    "movie_title": movie_data.title,
                    "year": movie_data.year or 0,
                    "director": movie_data.director or "",
                    "genre": genre_str,
                    "type": "overview",
                    "source": "combined",
                    "date_added": datetime.now().isoformat(),
                }
            )
            ids.append(f"movie_overview_{movie_data.title}_{uuid.uuid4().hex[:8]}")

            # Add individual reviews
            for review in movie_data.reviews:
                if len(review.content.strip()) < 20:  # Skip very short reviews
                    continue

                documents.append(review.content)
                metadatas.append(
                    {
                        "movie_title": movie_data.title,
                        "year": movie_data.year or 0,
                        "author": review.author or "",
                        "source": review.source,
                        "rating": review.rating or 0.0,
                        "review_type": review.review_type,
                        "type": "review",
                        "date_added": datetime.now().isoformat(),
                        "review_date": review.date.isoformat() if review.date else "",
                        "helpful_votes": review.helpful_votes or 0,
                    }
                )
                ids.append(f"review_{movie_data.title}_{uuid.uuid4().hex[:8]}")

            # Add to collection
            if documents:
                self.collection.add(documents=documents, metadatas=metadatas, ids=ids)

                logger.info(
                    f"Added {len(documents)} documents for '{movie_data.title}'"
                )

        except Exception as e:
            logger.error(f"Error adding movie data to vector database: {e}")
            raise

    def _create_movie_overview(self, movie_data: MovieData) -> str:
        """Create a comprehensive overview text for the movie."""
        overview_parts = [f"Movie: {movie_data.title}"]

        if movie_data.year:
            overview_parts.append(f"Year: {movie_data.year}")

        if movie_data.director:
            overview_parts.append(f"Director: {movie_data.director}")

        if movie_data.genre:
            overview_parts.append(f"Genre: {movie_data.genre}")

        if movie_data.cast:
            overview_parts.append(f"Cast: {', '.join(movie_data.cast[:5])}")

        if movie_data.plot_summary:
            overview_parts.append(f"Plot: {movie_data.plot_summary}")

        if movie_data.ratings:
            ratings_text = []
            for source, rating in movie_data.ratings.items():
                ratings_text.append(f"{source}: {rating}")
            overview_parts.append(f"Ratings: {', '.join(ratings_text)}")

        # Add review summary
        if movie_data.reviews:
            review_count = len(movie_data.reviews)
            sources = set(review.source for review in movie_data.reviews)
            overview_parts.append(
                f"Reviews: {review_count} reviews from {', '.join(sources)}"
            )

        return " | ".join(overview_parts)

    def query(
        self,
        query_text: str,
        movie_title: Optional[str] = None,
        n_results: int = 5,
        include_overview: bool = True,
    ) -> List[Dict[str, Any]]:
        """Query the vector database for relevant documents."""
        try:
            # Build where clause for filtering
            where_clause = {}

            if movie_title:
                where_clause["movie_title"] = {"$eq": movie_title}

            # Debug: Log the query
            print(f"DEBUG: Querying with text: '{query_text}', movie: {movie_title}")
            logger.info(
                f"Querying with text: '{query_text}', movie: {movie_title}, where: {where_clause}"
            )

            # Query the collection
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"],
            )

            # Debug: Log raw results
            num_results = (
                len(results.get("documents", [[]])[0])
                if results.get("documents")
                else 0
            )
            print(f"DEBUG: Raw query returned {num_results} results")
            logger.info(f"Raw query returned {num_results} results")

            # Format results
            formatted_results = []

            if results["documents"] and results["documents"][0]:
                for i, doc in enumerate(results["documents"][0]):
                    metadata = (
                        results["metadatas"][0][i] if results["metadatas"] else {}
                    )
                    distance = (
                        results["distances"][0][i] if results["distances"] else 0.0
                    )
                    relevance_score = 1.0 - distance

                    logger.debug(
                        f"Result {i+1}: distance={distance:.4f}, relevance={relevance_score:.4f}"
                    )

                    formatted_results.append(
                        {
                            "content": doc,
                            "metadata": metadata,
                            "relevance_score": relevance_score,  # Use the calculated relevance_score
                            "distance": distance,
                        }
                    )

            # Sort by relevance
            formatted_results.sort(key=lambda x: x["relevance_score"], reverse=True)

            logger.debug(f"Query returned {len(formatted_results)} results")

            return formatted_results

        except Exception as e:
            logger.error(f"Error querying vector database: {e}")
            return []

    def get_movie_reviews(self, movie_title: str) -> List[Dict[str, Any]]:
        """Get all reviews for a specific movie."""
        try:
            results = self.collection.get(
                where={
                    "$and": [
                        {"movie_title": {"$eq": movie_title}},
                        {"type": {"$eq": "review"}},
                    ]
                },
                include=["documents", "metadatas"],
            )

            formatted_results = []

            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    metadata = results["metadatas"][i] if results["metadatas"] else {}

                    formatted_results.append({"content": doc, "metadata": metadata})

            return formatted_results

        except Exception as e:
            logger.error(f"Error getting movie reviews: {e}")
            return []

    def get_movies_list(self) -> List[str]:
        """Get list of all movies in the database."""
        try:
            results = self.collection.get(
                where={"type": {"$eq": "overview"}}, include=["metadatas"]
            )

            movies = []
            if results["metadatas"]:
                for metadata in results["metadatas"]:
                    movie_title = metadata.get("movie_title")
                    if movie_title and movie_title not in movies:
                        movies.append(movie_title)

            return sorted(movies)

        except Exception as e:
            logger.error(f"Error getting movies list: {e}")
            return []

    def delete_movie(self, movie_title: str) -> bool:
        """Delete all data for a specific movie."""
        try:
            # Get all documents for the movie
            results = self.collection.get(where={"movie_title": {"$eq": movie_title}})

            if results and "ids" in results and results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    f"Deleted {len(results['ids'])} documents for '{movie_title}'"
                )
                return True
            else:
                logger.info(f"No documents found for movie '{movie_title}'")
                return False

        except Exception as e:
            logger.error(f"Error deleting movie data: {e}")
            return False

    def clear_database(self) -> bool:
        """Clear all data from the database."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection("movie_reviews")
            self._initialize_client()
            logger.info("Cleared vector database")
            return True

        except Exception as e:
            logger.error(f"Error clearing database: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            total_docs = self.collection.count()

            # Get overview docs count (number of movies)
            overview_results = self.collection.get(
                where={"type": {"$eq": "overview"}}, include=["metadatas"]
            )
            movies_count = (
                len(overview_results["metadatas"])
                if overview_results["metadatas"]
                else 0
            )

            # Get review docs count
            review_results = self.collection.get(
                where={"type": {"$eq": "review"}}, include=["metadatas"]
            )
            reviews_count = (
                len(review_results["metadatas"]) if review_results["metadatas"] else 0
            )

            # Get source breakdown
            source_counts = {}
            if review_results["metadatas"]:
                for metadata in review_results["metadatas"]:
                    source = metadata.get("source", "unknown")
                    source_counts[source] = source_counts.get(source, 0) + 1

            return {
                "total_documents": total_docs,
                "movies_count": movies_count,
                "reviews_count": reviews_count,
                "source_breakdown": source_counts,
                "database_path": self.db_path,
            }

        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}
