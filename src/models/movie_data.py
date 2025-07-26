"""Data models for the movie RAG system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


@dataclass
class ReviewData:
    """Data model for a single movie review."""

    content: str
    author: Optional[str] = None
    rating: Optional[float] = None
    source: str = ""
    url: Optional[str] = None
    date: Optional[datetime] = None
    review_type: str = "user"  # "user" or "critic"
    helpful_votes: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MovieData:
    """Data model for movie information and reviews."""

    title: str
    year: Optional[int] = None
    director: Optional[str] = None
    cast: List[str] = field(default_factory=list)
    genre: Optional[str] = None
    plot_summary: Optional[str] = None
    ratings: Dict[str, float] = field(default_factory=dict)  # source -> rating
    reviews: List[ReviewData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_review(self, review: ReviewData) -> None:
        """Add a review to the movie data."""
        self.reviews.append(review)

    def get_reviews_by_source(self, source: str) -> List[ReviewData]:
        """Get all reviews from a specific source."""
        return [review for review in self.reviews if review.source == source]

    def get_average_rating(self) -> Optional[float]:
        """Calculate the average rating across all sources."""
        if not self.ratings:
            return None
        return sum(self.ratings.values()) / len(self.ratings)


class MovieSearchQuery(BaseModel):
    """Query model for movie search."""

    title: str = Field(..., min_length=1, description="Movie title to search for")
    year: Optional[int] = Field(None, ge=1900, le=2030, description="Movie year")
    sources: List[str] = Field(
        default=["rotten_tomatoes", "imdb", "metacritic"],
        description="Sources to search",
    )
    max_reviews: int = Field(
        default=30, ge=1, le=100, description="Maximum reviews per source"
    )


class AnalysisRequest(BaseModel):
    """Request model for movie analysis."""

    movie_data: MovieData
    analysis_type: str = Field(
        default="comprehensive", description="Type of analysis to perform"
    )
    include_sentiment: bool = Field(
        default=True, description="Include sentiment analysis"
    )
    include_themes: bool = Field(default=True, description="Include theme extraction")


class AnalysisResult(BaseModel):
    """Result model for movie analysis."""

    movie_title: str
    summary: str
    sentiment: Dict[str, float] = Field(default_factory=dict)
    themes: List[str] = Field(default_factory=list)
    pros_cons: Dict[str, List[str]] = Field(default_factory=dict)
    key_insights: List[str] = Field(default_factory=list)
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    processing_time: float = Field(
        default=0.0, description="Processing time in seconds"
    )


@dataclass
class ScrapingResult:
    """Result from scraping a movie source."""

    source: str
    success: bool
    movie_data: Optional[MovieData] = None
    error_message: Optional[str] = None
    reviews_count: int = 0
    processing_time: float = 0.0


class RAGQuery(BaseModel):
    """Query model for RAG system."""

    question: str = Field(..., min_length=1, description="Question about the movie")
    movie_title: Optional[str] = Field(None, description="Specific movie context")
    max_results: int = Field(
        default=5, ge=1, le=20, description="Maximum results to retrieve"
    )
    similarity_threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold"
    )


class RAGResponse(BaseModel):
    """Response model from RAG system."""

    answer: str
    sources: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    retrieved_documents: List[Dict[str, Any]] = Field(default_factory=list)
