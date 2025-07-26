"""Main application entry point using Streamlit."""

import asyncio
import os
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

from crews.movie_analysis_crew import MovieAnalysisCrew
from models.movie_data import MovieData, ReviewData
from rag.movie_rag_system import MovieRAGSystem

# Load environment variables
load_dotenv()


# Initialize the RAG system
@st.cache_resource
def get_rag_system() -> MovieRAGSystem:
    """Initialize and cache the RAG system."""
    return MovieRAGSystem()


@st.cache_resource
def get_analysis_crew() -> MovieAnalysisCrew:
    """Initialize and cache the analysis crew."""
    return MovieAnalysisCrew()


def main() -> None:
    """Main Streamlit application."""
    st.set_page_config(page_title="Movie RAG System", page_icon="🎬", layout="wide")

    st.title("🎬 Movie Review Analysis System")
    st.markdown("Get comprehensive insights about movies from multiple review sources!")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        # Movie search input
        movie_title = st.text_input(
            "Movie Title",
            placeholder="Enter movie title...",
            help="Enter the exact movie title for best results",
        )

        # Source selection
        st.subheader("Review Sources")
        sources = {
            "Rotten Tomatoes": st.checkbox("Rotten Tomatoes", value=True),
            "IMDB": st.checkbox("IMDB", value=True),
            "Metacritic": st.checkbox("Metacritic", value=True),
        }

        # Advanced options
        with st.expander("Advanced Options"):
            max_reviews = st.slider("Max Reviews per Source", 10, 100, 30)
            analysis_depth = st.selectbox(
                "Analysis Depth", ["Quick", "Standard", "Comprehensive"], index=1
            )

        # Show existing movies in database
        with st.expander("🎬 Movies in Database"):
            rag_system = get_rag_system()
            available_movies = rag_system.get_available_movies()
            stats = rag_system.get_database_stats()

            if available_movies:
                st.write(f"**{stats.get('movies_count', 0)} movies available:**")
                for movie in available_movies:
                    if st.button(f"📊 Analyze '{movie}'", key=f"analyze_{movie}"):
                        st.session_state["selected_movie"] = movie
                        st.rerun()
            else:
                st.info("No movies in database yet. Try adding one above!")

    # Main content area
    # Check if a movie was selected from the database
    if "selected_movie" in st.session_state:
        movie_title = st.session_state["selected_movie"]
        del st.session_state["selected_movie"]  # Clear the selection
        analyze_existing_movie(movie_title)
    elif movie_title:
        if st.button("🔍 Analyze Movie", type="primary"):
            analyze_movie(movie_title, sources, max_reviews, analysis_depth)
    else:
        st.info("👆 Enter a movie title in the sidebar to get started!")

        # Show example usage
        st.subheader("How it works:")
        cols = st.columns(3)

        with cols[0]:
            st.markdown(
                """
            **1. Data Collection**
            - Scrapes reviews from multiple sources
            - Collects both critic and audience reviews
            - Handles rate limiting and errors gracefully
            """
            )

        with cols[1]:
            st.markdown(
                """
            **2. AI Analysis**
            - Uses CrewAI for intelligent processing
            - Sentiment analysis and topic extraction
            - Identifies key themes and opinions
            """
            )

        with cols[2]:
            st.markdown(
                """
            **3. Smart Retrieval**
            - Vector database for similarity search
            - Context-aware question answering
            - Comprehensive movie insights
            """
            )


def analyze_existing_movie(movie_title: str) -> None:
    """Analyze a movie that's already in the database."""
    rag_system = get_rag_system()
    analysis_crew = get_analysis_crew()

    try:
        # Check if movie exists in database
        available_movies = rag_system.get_available_movies()
        if movie_title not in available_movies:
            st.error(f"Movie '{movie_title}' not found in database!")
            return

        # Get movie data from database (we'll need to reconstruct this)
        # For now, create a basic MovieData object for analysis
        st.success(f"✅ Analyzing '{movie_title}' from database...")

        # Query the database for this movie's reviews
        query_result = rag_system.query(f"Tell me about {movie_title}")

        # Create a simple analysis result for existing movies
        st.header(f"📊 Analysis: {movie_title}")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("💬 Database Query")
            st.markdown(query_result)

        with col2:
            st.subheader("🔍 Ask Questions")
            question = st.text_input(
                f"Ask about {movie_title}:", key=f"question_{movie_title}"
            )
            if question:
                if st.button("Get Answer", key=f"answer_{movie_title}"):
                    answer = rag_system.query(f"{movie_title}: {question}")
                    st.markdown("**Answer:**")
                    st.markdown(answer)

    except Exception as e:
        st.error(f"❌ Error analyzing '{movie_title}': {str(e)}")


def analyze_movie(
    movie_title: str, sources: Dict[str, bool], max_reviews: int, analysis_depth: str
) -> None:
    """Analyze a movie using the RAG system."""
    rag_system = get_rag_system()
    analysis_crew = get_analysis_crew()

    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # Step 1: Collect movie data
        status_text.text("🔍 Searching for movie data...")
        progress_bar.progress(20)

        selected_sources = [source for source, selected in sources.items() if selected]
        movie_data = asyncio.run(
            rag_system.collect_movie_data(movie_title, selected_sources, max_reviews)
        )

        if not movie_data:
            st.error("❌ Movie not found or no reviews available!")
            return

        # Step 2: Process and store in vector database
        status_text.text("📝 Processing reviews...")
        progress_bar.progress(50)

        rag_system.add_movie_data(movie_data)

        # Step 3: AI Analysis
        status_text.text("🤖 Analyzing with AI...")
        progress_bar.progress(80)

        analysis_result = analysis_crew.analyze_movie(movie_data, analysis_depth)

        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")

        # Display results
        display_results(movie_data, analysis_result, rag_system)

    except Exception as e:
        st.error(f"❌ Error during analysis: {str(e)}")
        st.exception(e)


def display_results(
    movie_data: MovieData, analysis_result: Dict[str, Any], rag_system: MovieRAGSystem
) -> None:
    """Display the analysis results."""

    # Movie overview
    st.header(f"📊 Analysis: {movie_data.title}")

    # Basic movie info
    cols = st.columns(4)
    with cols[0]:
        st.metric("Year", movie_data.year or "N/A")
    with cols[1]:
        st.metric("Total Reviews", len(movie_data.reviews))
    with cols[2]:
        if movie_data.ratings:
            avg_rating = sum(movie_data.ratings.values()) / len(movie_data.ratings)
            st.metric("Avg Rating", f"{avg_rating:.1f}")
    with cols[3]:
        if movie_data.genre:
            st.metric("Genre", movie_data.genre)

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📈 Analysis Summary",
            "💬 Reviews",
            "🔍 Ask Questions",
            "📊 Detailed Insights",
        ]
    )

    with tab1:
        display_analysis_summary(analysis_result)

    with tab2:
        display_reviews(movie_data.reviews)

    with tab3:
        display_qa_interface(rag_system, movie_data.title)

    with tab4:
        display_detailed_insights(analysis_result)


def display_analysis_summary(analysis_result: Dict[str, Any]) -> None:
    """Display the AI analysis summary."""
    if "summary" in analysis_result:
        st.markdown("### 🎯 Key Insights")
        st.markdown(analysis_result["summary"])

    if "sentiment" in analysis_result:
        st.markdown("### 😊 Overall Sentiment")
        sentiment = analysis_result["sentiment"]

        cols = st.columns(3)
        with cols[0]:
            st.metric("Positive", f"{sentiment.get('positive', 0):.1%}")
        with cols[1]:
            st.metric("Neutral", f"{sentiment.get('neutral', 0):.1%}")
        with cols[2]:
            st.metric("Negative", f"{sentiment.get('negative', 0):.1%}")


def display_reviews(reviews: List[ReviewData]) -> None:
    """Display individual reviews."""
    st.markdown("### 💬 Recent Reviews")

    # Group reviews by source
    sources = {}
    for review in reviews:
        if review.source not in sources:
            sources[review.source] = []
        sources[review.source].append(review)

    for source, source_reviews in sources.items():
        with st.expander(f"{source} ({len(source_reviews)} reviews)"):
            for review in source_reviews[:5]:  # Show first 5 reviews
                cols = st.columns([3, 1])
                with cols[0]:
                    st.markdown(f"**{review.author or 'Anonymous'}**")
                    st.markdown(
                        review.content[:300] + "..."
                        if len(review.content) > 300
                        else review.content
                    )
                with cols[1]:
                    if review.rating:
                        st.metric("Rating", f"{review.rating}")


def display_qa_interface(rag_system: MovieRAGSystem, movie_title: str) -> None:
    """Display the question-answering interface."""
    st.markdown("### ❓ Ask About This Movie")

    question = st.text_input(
        "Your Question",
        placeholder="e.g., What do critics think about the acting?",
        key="qa_input",
    )

    if question and st.button("Get Answer"):
        with st.spinner("🔍 Searching for relevant information..."):
            answer = rag_system.query(f"{movie_title}: {question}")
            st.markdown("**Answer:**")
            st.markdown(answer)


def display_detailed_insights(analysis_result: Dict[str, Any]) -> None:
    """Display detailed analysis insights."""
    if "themes" in analysis_result:
        st.markdown("### 🎭 Main Themes")
        for theme in analysis_result["themes"][:5]:
            st.markdown(f"- {theme}")

    if "pros_cons" in analysis_result:
        cols = st.columns(2)
        with cols[0]:
            st.markdown("### ✅ Commonly Praised")
            for pro in analysis_result["pros_cons"].get("pros", [])[:5]:
                st.markdown(f"- {pro}")

        with cols[1]:
            st.markdown("### ❌ Common Criticisms")
            for con in analysis_result["pros_cons"].get("cons", [])[:5]:
                st.markdown(f"- {con}")


if __name__ == "__main__":
    main()
