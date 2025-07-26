"""Tests for movie data models."""

import os
import sys
from datetime import datetime

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.movie_data import AnalysisResult, MovieData, MovieSearchQuery, ReviewData


class TestReviewData:
    """Test ReviewData model."""

    def test_review_data_creation(self):
        """Test creating a ReviewData instance."""
        review = ReviewData(
            content="Great movie!", author="John Doe", rating=8.5, source="Test Source"
        )

        assert review.content == "Great movie!"
        assert review.author == "John Doe"
        assert review.rating == 8.5
        assert review.source == "Test Source"
        assert review.review_type == "user"  # default value

    def test_review_data_with_metadata(self):
        """Test ReviewData with metadata."""
        metadata = {"helpful_votes": 10, "verified_purchase": True}
        review = ReviewData(content="Good movie", source="Test", metadata=metadata)

        assert review.metadata == metadata

    def test_review_data_minimal(self):
        """Test ReviewData with minimal required fields."""
        review = ReviewData(content="Test review", source="Test")

        assert review.content == "Test review"
        assert review.source == "Test"
        assert review.author is None
        assert review.rating is None


class TestMovieData:
    """Test MovieData model."""

    def test_movie_data_creation(self, sample_movie_data):
        """Test creating a MovieData instance."""
        movie = sample_movie_data

        assert movie.title == "Test Movie"
        assert movie.year == 2023
        assert movie.director == "Test Director"
        assert len(movie.cast) == 2
        assert movie.genre == "Action, Drama"
        assert len(movie.reviews) == 3

    def test_add_review(self, sample_movie_data, sample_review_data):
        """Test adding a review to movie data."""
        movie = sample_movie_data
        initial_count = len(movie.reviews)

        movie.add_review(sample_review_data)

        assert len(movie.reviews) == initial_count + 1
        assert movie.reviews[-1] == sample_review_data

    def test_get_reviews_by_source(self, sample_movie_data):
        """Test filtering reviews by source."""
        movie = sample_movie_data

        test_source_reviews = movie.get_reviews_by_source("Test Source")
        another_source_reviews = movie.get_reviews_by_source("Another Source")

        assert len(test_source_reviews) == 2
        assert len(another_source_reviews) == 1
        assert all(review.source == "Test Source" for review in test_source_reviews)

    def test_get_average_rating(self, sample_movie_data):
        """Test calculating average rating."""
        movie = sample_movie_data

        avg_rating = movie.get_average_rating()
        expected_avg = (7.5 + 6.0) / 2

        assert avg_rating == expected_avg

    def test_get_average_rating_no_ratings(self):
        """Test average rating with no ratings."""
        movie = MovieData(title="No Ratings Movie")

        assert movie.get_average_rating() is None


class TestMovieSearchQuery:
    """Test MovieSearchQuery model."""

    def test_movie_search_query_creation(self):
        """Test creating a MovieSearchQuery."""
        query = MovieSearchQuery(
            title="The Matrix", year=1999, sources=["imdb", "rotten_tomatoes"]
        )

        assert query.title == "The Matrix"
        assert query.year == 1999
        assert query.sources == ["imdb", "rotten_tomatoes"]

    def test_movie_search_query_defaults(self):
        """Test MovieSearchQuery with default values."""
        query = MovieSearchQuery(title="Test Movie")

        assert query.title == "Test Movie"
        assert query.year is None
        assert query.sources == ["rotten_tomatoes", "imdb", "metacritic"]
        assert query.max_reviews == 30

    def test_movie_search_query_validation(self):
        """Test MovieSearchQuery validation."""
        # Test empty title
        with pytest.raises(ValueError):
            MovieSearchQuery(title="")

        # Test invalid year
        with pytest.raises(ValueError):
            MovieSearchQuery(title="Test", year=1800)

        # Test invalid max_reviews
        with pytest.raises(ValueError):
            MovieSearchQuery(title="Test", max_reviews=0)


class TestAnalysisResult:
    """Test AnalysisResult model."""

    def test_analysis_result_creation(self):
        """Test creating an AnalysisResult."""
        result = AnalysisResult(
            movie_title="Test Movie",
            summary="This is a test analysis",
            sentiment={"positive": 0.7, "negative": 0.2, "neutral": 0.1},
            themes=["action", "drama"],
            confidence_score=0.85,
        )

        assert result.movie_title == "Test Movie"
        assert result.summary == "This is a test analysis"
        assert result.sentiment["positive"] == 0.7
        assert result.themes == ["action", "drama"]
        assert result.confidence_score == 0.85

    def test_analysis_result_defaults(self):
        """Test AnalysisResult with default values."""
        result = AnalysisResult(movie_title="Test Movie", summary="Test summary")

        assert result.sentiment == {}
        assert result.themes == []
        assert result.pros_cons == {}
        assert result.key_insights == []
        assert result.confidence_score == 0.0
        assert result.processing_time == 0.0
