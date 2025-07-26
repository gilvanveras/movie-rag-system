"""Tests for RAG system functionality."""

import os
import sys
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.movie_data import MovieData, ReviewData
from rag.movie_rag_system import MovieRAGSystem
from rag.vector_database import VectorDatabase


class TestVectorDatabase:
    """Test VectorDatabase functionality."""

    @pytest.fixture
    def mock_chroma_client(self):
        """Mock ChromaDB client."""
        with patch("chromadb.PersistentClient") as mock_client:
            mock_collection = Mock()
            mock_collection.count.return_value = 0
            mock_collection.add.return_value = None
            mock_collection.query.return_value = {
                "documents": [["Test document"]],
                "metadatas": [[{"movie_title": "Test Movie"}]],
                "distances": [[0.1]],
            }
            mock_collection.get.return_value = {
                "documents": ["Test document"],
                "metadatas": [{"movie_title": "Test Movie"}],
            }

            mock_client.return_value.get_or_create_collection.return_value = (
                mock_collection
            )
            yield mock_client, mock_collection

    def test_vector_database_initialization(self, temp_db_path, mock_chroma_client):
        """Test vector database initialization."""
        mock_client, mock_collection = mock_chroma_client

        db = VectorDatabase(temp_db_path)

        assert db.db_path == temp_db_path
        assert db.client is not None
        assert db.collection is not None

    def test_add_movie_data(self, temp_db_path, mock_chroma_client, sample_movie_data):
        """Test adding movie data to vector database."""
        mock_client, mock_collection = mock_chroma_client

        db = VectorDatabase(temp_db_path)
        db.add_movie_data(sample_movie_data)

        # Should call add on the collection
        mock_collection.add.assert_called_once()

        # Check that correct number of documents were added
        call_args = mock_collection.add.call_args
        documents = call_args[1]["documents"]

        # Should have movie overview + reviews
        expected_docs = 1 + len(sample_movie_data.reviews)  # overview + reviews
        assert len(documents) == expected_docs

    def test_query_vector_database(self, temp_db_path, mock_chroma_client):
        """Test querying the vector database."""
        mock_client, mock_collection = mock_chroma_client

        db = VectorDatabase(temp_db_path)
        results = db.query("test query", movie_title="Test Movie")

        # Should call query on collection
        mock_collection.query.assert_called_once()

        # Should return formatted results
        assert len(results) == 1
        assert results[0]["content"] == "Test document"
        assert results[0]["metadata"]["movie_title"] == "Test Movie"
        assert "relevance_score" in results[0]

    def test_get_movies_list(self, temp_db_path, mock_chroma_client):
        """Test getting list of movies."""
        mock_client, mock_collection = mock_chroma_client

        # Mock response for overview documents
        mock_collection.get.return_value = {
            "metadatas": [{"movie_title": "Movie 1"}, {"movie_title": "Movie 2"}]
        }

        db = VectorDatabase(temp_db_path)
        movies = db.get_movies_list()

        assert len(movies) == 2
        assert "Movie 1" in movies
        assert "Movie 2" in movies

    def test_delete_movie(self, temp_db_path, mock_chroma_client):
        """Test deleting a movie from database."""
        mock_client, mock_collection = mock_chroma_client

        # Mock response with document IDs
        mock_collection.get.return_value = {"ids": ["doc1", "doc2", "doc3"]}

        db = VectorDatabase(temp_db_path)
        result = db.delete_movie("Test Movie")

        assert result is True
        mock_collection.delete.assert_called_once_with(ids=["doc1", "doc2", "doc3"])


class TestMovieRAGSystem:
    """Test MovieRAGSystem functionality."""

    @pytest.fixture
    def mock_rag_system(self, temp_db_path):
        """Create a mock RAG system."""
        with patch("rag.movie_rag_system.VectorDatabase") as mock_vdb:
            with patch("rag.movie_rag_system.ChatOpenAI") as mock_llm:
                mock_vdb_instance = Mock()
                mock_llm_instance = Mock()
                mock_llm_instance.return_value.content = "Test AI response"

                mock_vdb.return_value = mock_vdb_instance
                mock_llm.return_value = mock_llm_instance

                rag_system = MovieRAGSystem(temp_db_path)
                yield rag_system, mock_vdb_instance, mock_llm_instance

    def test_rag_system_initialization(self, temp_db_path):
        """Test RAG system initialization."""
        with patch("rag.movie_rag_system.VectorDatabase"):
            rag_system = MovieRAGSystem(temp_db_path)

            assert rag_system.db_path == temp_db_path
            assert rag_system.vector_db is not None

    @pytest.mark.asyncio
    async def test_collect_movie_data(self, mock_rag_system, sample_movie_data):
        """Test collecting movie data through scraping."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        with patch("rag.movie_rag_system.ScraperManager") as mock_scraper_mgr:
            mock_manager = AsyncMock()
            mock_manager.scrape_movie.return_value = sample_movie_data
            mock_scraper_mgr.return_value.__aenter__.return_value = mock_manager

            result = await rag_system.collect_movie_data(
                "Test Movie", sources=["Test Source"]
            )

            assert result == sample_movie_data
            mock_manager.scrape_movie.assert_called_once()

    def test_add_movie_data(self, mock_rag_system, sample_movie_data):
        """Test adding movie data to RAG system."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        # Mock existing movies check
        mock_vdb.get_movies_list.return_value = []

        rag_system.add_movie_data(sample_movie_data)

        mock_vdb.add_movie_data.assert_called_once_with(sample_movie_data)

    def test_add_movie_data_existing(self, mock_rag_system, sample_movie_data):
        """Test adding movie data when movie already exists."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        # Mock existing movie
        mock_vdb.get_movies_list.return_value = ["Test Movie"]
        mock_vdb.delete_movie.return_value = True

        rag_system.add_movie_data(sample_movie_data)

        mock_vdb.delete_movie.assert_called_once_with("Test Movie")
        mock_vdb.add_movie_data.assert_called_once_with(sample_movie_data)

    def test_query_with_llm(self, mock_rag_system):
        """Test querying RAG system with LLM."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        # Mock vector database response
        mock_vdb.query.return_value = [
            {
                "content": "Test content",
                "metadata": {"source": "test"},
                "relevance_score": 0.8,
            }
        ]

        response = rag_system.query("What do you think about this movie?")

        assert isinstance(response, str)
        mock_vdb.query.assert_called_once()

    def test_query_no_results(self, mock_rag_system):
        """Test querying when no relevant documents found."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        # Mock empty response
        mock_vdb.query.return_value = []

        response = rag_system.query("Unknown question")

        assert "couldn't find" in response.lower()

    def test_query_low_similarity(self, mock_rag_system):
        """Test querying with low similarity scores."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        # Mock response with low similarity
        mock_vdb.query.return_value = [
            {
                "content": "Test content",
                "metadata": {"source": "test"},
                "relevance_score": 0.3,  # Below threshold
            }
        ]

        response = rag_system.query("Test question", similarity_threshold=0.7)

        assert "couldn't find" in response.lower()

    def test_get_sentiment_analysis(self, mock_rag_system):
        """Test sentiment analysis functionality."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        # Mock reviews data
        mock_reviews = [
            {"content": "Great movie!", "metadata": {"rating": 9.0, "source": "test"}},
            {"content": "Terrible film", "metadata": {"rating": 2.0, "source": "test"}},
            {"content": "Average movie", "metadata": {"rating": 6.0, "source": "test"}},
        ]
        mock_vdb.get_movie_reviews.return_value = mock_reviews

        result = rag_system.get_sentiment_analysis("Test Movie")

        assert "total_reviews" in result
        assert "sentiment_distribution" in result
        assert "sentiment_percentages" in result
        assert result["total_reviews"] == 3

    def test_get_sentiment_analysis_no_reviews(self, mock_rag_system):
        """Test sentiment analysis with no reviews."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        mock_vdb.get_movie_reviews.return_value = []

        result = rag_system.get_sentiment_analysis("Unknown Movie")

        assert "error" in result

    def test_get_database_stats(self, mock_rag_system):
        """Test getting database statistics."""
        rag_system, mock_vdb, mock_llm = mock_rag_system

        mock_stats = {"total_documents": 100, "movies_count": 10, "reviews_count": 90}
        mock_vdb.get_stats.return_value = mock_stats

        stats = rag_system.get_database_stats()

        assert stats == mock_stats
        mock_vdb.get_stats.assert_called_once()

    def test_fallback_without_llm(self, temp_db_path):
        """Test RAG system functionality without LLM."""
        with patch("rag.movie_rag_system.VectorDatabase") as mock_vdb:
            with patch.dict(os.environ, {}, clear=True):  # No OpenAI key
                rag_system = MovieRAGSystem(temp_db_path)

                # Should initialize without LLM
                assert rag_system.llm is None

                # Mock vector database response
                mock_vdb_instance = Mock()
                mock_vdb_instance.query.return_value = [
                    {
                        "content": "Test review content",
                        "metadata": {"source": "test", "author": "reviewer"},
                        "relevance_score": 0.8,
                    }
                ]
                rag_system.vector_db = mock_vdb_instance

                # Should still work with fallback response
                response = rag_system.query("Test question")
                assert isinstance(response, str)
                assert "test review content" in response.lower()
