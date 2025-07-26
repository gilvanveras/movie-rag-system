"""Main RAG system for movie reviews."""

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from models.movie_data import MovieData, RAGQuery, RAGResponse
from rag.vector_database import VectorDatabase
from scrapers.scraper_manager import ScraperManager, ScrapingConfig

logger = logging.getLogger(__name__)


class MovieRAGSystem:
    """Complete RAG system for movie review analysis and querying."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or os.getenv("CHROMA_DB_PATH", "./data/chroma_db")
        self.vector_db = VectorDatabase(self.db_path)
        self.llm = None
        self.llm_type = None
        self.model_name = None
        self._initialize_llm_with_config()
        self.scraper_manager = None

    def _initialize_llm_with_config(self):
        """Initialize the language model and store configuration."""
        try:
            # Try Groq API first, then fallback to OpenAI
            groq_api_key = os.getenv("GROQ_API_KEY")
            groq_model = os.getenv("GROQ_MODEL", "meta-llama/llama-3.1-70b-versatile")
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

            if groq_api_key:
                logger.info(f"Using Groq API for LLM with model: {groq_model}")
                self.llm = Groq(api_key=groq_api_key)
                self.llm_type = "groq"
                self.model_name = groq_model
            elif openai_api_key:
                logger.info(f"Using OpenAI API for LLM with model: {openai_model}")
                self.llm = ChatOpenAI(
                    temperature=0.1,
                    model_name=openai_model,
                    openai_api_key=openai_api_key,
                    max_tokens=1000,
                )
                self.llm_type = "openai"
                self.model_name = openai_model
            else:
                logger.warning(
                    "No Groq or OpenAI API key found. Some features may not work."
                )
                self.llm = None
                self.llm_type = None
                self.model_name = None

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
            self.llm_type = None
            self.model_name = None

    async def collect_movie_data(
        self,
        movie_title: str,
        sources: List[str] = None,
        max_reviews: int = 30,
        year: Optional[int] = None,
    ) -> Optional[MovieData]:
        """Collect movie data from specified sources."""

        if sources is None:
            sources = ["Rotten Tomatoes", "IMDB", "Metacritic"]

        # Initialize scraper manager
        scraping_config = ScrapingConfig(
            delay=float(os.getenv("SCRAPING_DELAY", 1.0)),
            timeout=int(os.getenv("TIMEOUT", 30)),
            max_retries=int(os.getenv("MAX_RETRIES", 3)),
            max_reviews=max_reviews,
        )

        async with ScraperManager(scraping_config) as scraper_manager:
            movie_data = await scraper_manager.scrape_movie(
                title=movie_title, sources=sources, year=year, max_reviews=max_reviews
            )

        return movie_data

    def add_movie_data(self, movie_data: MovieData) -> None:
        """Add movie data to the vector database."""
        try:
            # Check if movie already exists
            existing_movies = self.vector_db.get_movies_list()

            if movie_data.title in existing_movies:
                logger.info(
                    f"Movie '{movie_data.title}' already exists. Updating data."
                )
                self.vector_db.delete_movie(movie_data.title)

            # Add new data
            self.vector_db.add_movie_data(movie_data)
            logger.info(f"Successfully added '{movie_data.title}' to database")

        except Exception as e:
            logger.error(f"Error adding movie data: {e}")
            raise

    def query(
        self,
        question: str,
        movie_title: Optional[str] = None,
        max_results: int = 5,
        similarity_threshold: float = 0.1,
    ) -> str:
        """Query the RAG system with a question about movies."""

        try:
            # Retrieve relevant documents
            relevant_docs = self.vector_db.query(
                query_text=question, movie_title=movie_title, n_results=max_results
            )

            # Filter by similarity threshold
            print(
                f"DEBUG: Filtering {len(relevant_docs)} docs with threshold {similarity_threshold}"
            )
            for i, doc in enumerate(relevant_docs):
                print(f"DEBUG: Doc {i+1} score: {doc['relevance_score']:.4f}")

            filtered_docs = [
                doc
                for doc in relevant_docs
                if doc["relevance_score"] >= similarity_threshold
            ]

            print(f"DEBUG: After filtering: {len(filtered_docs)} docs remain")

            if not filtered_docs:
                return self._generate_no_results_response(question, movie_title)

            # Generate response using LLM
            response = self._generate_llm_response(question, filtered_docs, movie_title)

            return response

        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

    def _generate_llm_response(
        self,
        question: str,
        relevant_docs: List[Dict[str, Any]],
        movie_title: Optional[str] = None,
    ) -> str:
        """Generate a response using the LLM with retrieved context."""

        if not self.llm:
            return self._generate_fallback_response(
                question, relevant_docs, movie_title
            )

        try:
            # Prepare context from retrieved documents
            context_parts = []

            for i, doc in enumerate(relevant_docs[:5]):  # Limit to top 5 results
                metadata = doc["metadata"]
                content = doc["content"]

                if metadata.get("type") == "overview":
                    context_parts.append(f"Movie Overview: {content}")
                else:
                    author = metadata.get("author", "Anonymous")
                    source = metadata.get("source", "Unknown")
                    rating = metadata.get("rating", "N/A")

                    context_parts.append(
                        f"Review by {author} from {source} (Rating: {rating}): {content[:500]}..."
                    )

            context = "\n\n".join(context_parts)

            # Create system prompt
            system_prompt = """
You are a knowledgeable movie expert assistant powered by advanced AI. You help users understand movies based on reviews and information from multiple sources including Rotten Tomatoes, IMDB, and Metacritic.

Guidelines:
- Provide comprehensive, balanced answers based on the provided context
- Cite specific sources when mentioning opinions or ratings
- Distinguish between critic and audience opinions when relevant
- If asked about specific aspects (acting, plot, cinematography), focus on those areas
- Be objective and present different perspectives when they exist
- If the context doesn't contain enough information, say so honestly
- Use your expertise to provide insightful analysis while staying grounded in the provided data

Always base your response on the provided context and avoid making up information.
"""

            movie_context = f" about the movie '{movie_title}'" if movie_title else ""

            user_prompt = f"""
Based on the following movie reviews and information, please answer this question{movie_context}:

Question: {question}

Context:
{context}

Please provide a comprehensive answer based on the available information.
"""

            # Check if using Groq or OpenAI
            if self.llm_type == "groq":
                # Use Groq API with configured model
                completion = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.1,
                    max_tokens=1000,
                    top_p=1,
                    stream=False,
                    stop=None,
                )

                return completion.choices[0].message.content.strip()

            elif self.llm_type == "openai":
                # Use OpenAI/LangChain
                system_message = SystemMessage(content=system_prompt)
                human_message = HumanMessage(content=user_prompt)
                response = self.llm([system_message, human_message])
                return response.content.strip()
            else:
                # Fallback to basic response
                return self._generate_fallback_response(
                    question, relevant_docs, movie_title
                )

        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(
                question, relevant_docs, movie_title
            )

    def _generate_fallback_response(
        self,
        question: str,
        relevant_docs: List[Dict[str, Any]],
        movie_title: Optional[str] = None,
    ) -> str:
        """Generate a fallback response without LLM."""

        if not relevant_docs:
            return f"I couldn't find specific information to answer your question{f' about {movie_title}' if movie_title else ''}."

        # Simple concatenation of top results
        response_parts = []

        if movie_title:
            response_parts.append(
                f"Based on available reviews and information about '{movie_title}':"
            )
        else:
            response_parts.append("Based on available movie reviews and information:")

        for doc in relevant_docs[:3]:  # Show top 3 results
            metadata = doc["metadata"]
            content = doc["content"]

            if metadata.get("type") == "overview":
                response_parts.append(f"\nMovie Information: {content}")
            else:
                source = metadata.get("source", "Unknown source")
                author = metadata.get("author", "Anonymous reviewer")

                response_parts.append(f"\nFrom {source} ({author}): {content[:300]}...")

        return "\n".join(response_parts)

    def _generate_no_results_response(
        self, question: str, movie_title: Optional[str] = None
    ) -> str:
        """Generate response when no relevant documents are found."""

        if movie_title:
            return (
                f"I don't have enough information about '{movie_title}' to answer your question. "
                f"You might want to add this movie to the database first by searching for it."
            )
        else:
            return (
                "I couldn't find relevant information to answer your question. "
                "Please make sure you've added some movies to the database first, "
                "or try rephrasing your question."
            )

    def get_movie_summary(self, movie_title: str) -> Optional[str]:
        """Get a comprehensive summary of a movie."""

        try:
            # Get all movie data
            relevant_docs = self.vector_db.query(
                query_text=f"summary overview analysis {movie_title}",
                movie_title=movie_title,
                n_results=10,
            )

            if not relevant_docs:
                return None

            # Generate summary
            summary_question = f"Provide a comprehensive summary and analysis of {movie_title} based on the available reviews and information."

            return self._generate_llm_response(
                summary_question, relevant_docs, movie_title
            )

        except Exception as e:
            logger.error(f"Error generating movie summary: {e}")
            return None

    def get_sentiment_analysis(self, movie_title: str) -> Dict[str, Any]:
        """Get sentiment analysis for a movie's reviews."""

        try:
            reviews = self.vector_db.get_movie_reviews(movie_title)

            if not reviews:
                return {"error": "No reviews found for this movie"}

            # Simple sentiment analysis based on ratings
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            total_rating = 0
            rated_reviews = 0

            for review in reviews:
                metadata = review["metadata"]
                rating = metadata.get("rating", 0)

                if rating > 0:
                    total_rating += rating
                    rated_reviews += 1

                    if rating >= 7:
                        positive_count += 1
                    elif rating <= 4:
                        negative_count += 1
                    else:
                        neutral_count += 1
                else:
                    neutral_count += 1

            total_reviews = len(reviews)
            avg_rating = total_rating / rated_reviews if rated_reviews > 0 else 0

            return {
                "total_reviews": total_reviews,
                "rated_reviews": rated_reviews,
                "average_rating": round(avg_rating, 2),
                "sentiment_distribution": {
                    "positive": positive_count,
                    "neutral": neutral_count,
                    "negative": negative_count,
                },
                "sentiment_percentages": {
                    "positive": round(positive_count / total_reviews * 100, 1),
                    "neutral": round(neutral_count / total_reviews * 100, 1),
                    "negative": round(negative_count / total_reviews * 100, 1),
                },
            }

        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"error": str(e)}

    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the database."""
        return self.vector_db.get_stats()

    def get_available_movies(self) -> List[str]:
        """Get list of movies in the database."""
        return self.vector_db.get_movies_list()

    def delete_movie(self, movie_title: str) -> bool:
        """Delete a movie from the database."""
        return self.vector_db.delete_movie(movie_title)

    def clear_database(self) -> bool:
        """Clear the entire database."""
        return self.vector_db.clear_database()
