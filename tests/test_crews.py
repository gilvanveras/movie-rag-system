"""Tests for CrewAI movie analysis functionality."""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from crews.movie_analysis_crew import MovieAnalysisCrew
from models.movie_data import MovieData, ReviewData


class TestMovieAnalysisCrew:
    """Test MovieAnalysisCrew functionality."""

    @pytest.fixture
    def mock_crew_with_llm(self):
        """Mock CrewAI components with LLM."""
        with patch("crews.movie_analysis_crew.ChatOpenAI") as mock_llm:
            with patch("crews.movie_analysis_crew.Agent") as mock_agent:
                with patch("crews.movie_analysis_crew.Crew") as mock_crew:
                    mock_llm_instance = Mock()
                    mock_agent_instance = Mock()
                    mock_crew_instance = Mock()

                    mock_llm.return_value = mock_llm_instance
                    mock_agent.return_value = mock_agent_instance
                    mock_crew.return_value = mock_crew_instance

                    crew = MovieAnalysisCrew()
                    yield crew, mock_crew_instance

    @pytest.fixture
    def mock_crew_no_llm(self):
        """Mock CrewAI without LLM (fallback mode)."""
        with patch.dict("os.environ", {}, clear=True):  # No OpenAI key
            crew = MovieAnalysisCrew()
            yield crew

    def test_crew_initialization_with_llm(self, mock_openai_api):
        """Test crew initialization with LLM."""
        with patch("crews.movie_analysis_crew.Agent") as mock_agent:
            # Mock Agent instances
            mock_agent_instance = Mock()
            mock_agent.return_value = mock_agent_instance

            crew = MovieAnalysisCrew()

            assert crew.llm is not None
            assert len(crew.agents) > 0

            # Verify all expected agents are created
            expected_agents = ["review_analyst", "sentiment_analyst", "summarizer"]
            for agent_name in expected_agents:
                assert agent_name in crew.agents

    def test_crew_initialization_no_llm(self, mock_crew_no_llm):
        """Test crew initialization without LLM."""
        crew = mock_crew_no_llm

        assert crew.llm is None
        assert crew.agents == {}

    def test_analyze_movie_with_crew(self, mock_crew_with_llm, sample_movie_data):
        """Test movie analysis with CrewAI."""
        crew, mock_crew_instance = mock_crew_with_llm

        # Mock crew execution result
        mock_result = Mock()
        mock_result.raw = "Test analysis summary from CrewAI execution"

        with patch("crews.movie_analysis_crew.Crew") as mock_crew_class:
            mock_crew_class.return_value = mock_crew_instance
            mock_crew_instance.kickoff.return_value = mock_result

            result = crew.analyze_movie(sample_movie_data, "Standard")

            assert isinstance(result, dict)
            assert "summary" in result
            assert "sentiment" in result
            assert "themes" in result
            assert "pros_cons" in result
            assert result["analysis_method"] == "CrewAI"

            # Verify crew was created with tasks
            mock_crew_class.assert_called_once()
            call_args = mock_crew_class.call_args
            assert "agents" in call_args.kwargs
            assert "tasks" in call_args.kwargs
            assert len(call_args.kwargs["tasks"]) > 0

            mock_crew_instance.kickoff.assert_called_once()

    def test_analyze_movie_fallback(self, mock_crew_no_llm, sample_movie_data):
        """Test movie analysis fallback without CrewAI."""
        crew = mock_crew_no_llm

        result = crew.analyze_movie(sample_movie_data, "Standard")

        assert isinstance(result, dict)
        assert "summary" in result
        assert "sentiment" in result
        assert "themes" in result
        assert "pros_cons" in result
        assert result["analysis_method"] == "Fallback"

    def test_extract_sentiment_from_reviews(self, sample_movie_data):
        """Test sentiment extraction from reviews."""
        crew = MovieAnalysisCrew()

        sentiment = crew._extract_sentiment_from_reviews(sample_movie_data.reviews)

        assert isinstance(sentiment, dict)
        assert "positive" in sentiment
        assert "neutral" in sentiment
        assert "negative" in sentiment

        # Values should sum to 1.0
        total = sentiment["positive"] + sentiment["neutral"] + sentiment["negative"]
        assert abs(total - 1.0) < 0.01

    def test_extract_sentiment_positive_reviews(self):
        """Test sentiment extraction with positive reviews."""
        reviews = [
            ReviewData(
                content="Excellent movie! Amazing acting and fantastic plot.",
                rating=9.0,
                source="test",
            ),
            ReviewData(
                content="Great film with brilliant performances.",
                rating=8.5,
                source="test",
            ),
        ]

        crew = MovieAnalysisCrew()
        sentiment = crew._extract_sentiment_from_reviews(reviews)

        # Should be mostly positive
        assert sentiment["positive"] > sentiment["negative"]
        assert sentiment["positive"] > sentiment["neutral"]

    def test_extract_sentiment_negative_reviews(self):
        """Test sentiment extraction with negative reviews."""
        reviews = [
            ReviewData(
                content="Terrible movie with awful acting and boring plot.",
                rating=2.0,
                source="test",
            ),
            ReviewData(
                content="Horrible film, worst thing I've seen.",
                rating=1.0,
                source="test",
            ),
        ]

        crew = MovieAnalysisCrew()
        sentiment = crew._extract_sentiment_from_reviews(reviews)

        # Should be mostly negative
        assert sentiment["negative"] > sentiment["positive"]

    def test_extract_sentiment_empty_reviews(self):
        """Test sentiment extraction with no reviews."""
        crew = MovieAnalysisCrew()
        sentiment = crew._extract_sentiment_from_reviews([])

        assert sentiment == {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    def test_create_analysis_tasks(self, sample_movie_data):
        """Test creation and validation of analysis tasks."""
        with patch("crews.movie_analysis_crew.Agent") as mock_agent:
            # Mock Agent instances
            mock_agent_instance = Mock()
            mock_agent.return_value = mock_agent_instance

            # Create crew with mocked agents
            crew = MovieAnalysisCrew()
            crew.agents = {
                "review_analyst": mock_agent_instance,
                "sentiment_analyst": mock_agent_instance,
                "summarizer": mock_agent_instance,
            }

            # Prepare review content
            all_reviews = "\n\n".join(
                [review.content for review in sample_movie_data.reviews]
            )
            critic_reviews = "\n\n".join(
                [
                    review.content
                    for review in sample_movie_data.reviews
                    if review.review_type == "critic"
                ]
            )
            user_reviews = "\n\n".join(
                [
                    review.content
                    for review in sample_movie_data.reviews
                    if review.review_type == "user"
                ]
            )

            # Test task creation
            tasks = crew._create_analysis_tasks(
                sample_movie_data, all_reviews, critic_reviews, user_reviews, "Standard"
            )

            # Validate tasks
            assert isinstance(tasks, list)
            assert len(tasks) == 3  # Should create 3 tasks

            for task in tasks:
                # Each task should have required attributes
                assert hasattr(task, "description")
                assert hasattr(task, "expected_output")
                assert hasattr(task, "agent")

                # Validate content
                assert task.description is not None
                assert len(task.description) > 0
                assert task.expected_output is not None
                assert len(task.expected_output) > 0
                assert task.agent is not None

    def test_task_content_validation(self, sample_movie_data):
        """Test that tasks contain appropriate content and movie references."""
        with patch("crews.movie_analysis_crew.Agent") as mock_agent:
            mock_agent_instance = Mock()
            mock_agent.return_value = mock_agent_instance

            crew = MovieAnalysisCrew()
            crew.agents = {
                "review_analyst": mock_agent_instance,
                "sentiment_analyst": mock_agent_instance,
                "summarizer": mock_agent_instance,
            }

            all_reviews = "Great movie with excellent acting."
            critic_reviews = "Professional review: Outstanding film."
            user_reviews = "User review: Really enjoyed it!"

            tasks = crew._create_analysis_tasks(
                sample_movie_data, all_reviews, critic_reviews, user_reviews, "Standard"
            )

            # Check that movie title is referenced in task descriptions
            for task in tasks:
                assert sample_movie_data.title in task.description

            # Check for specific task types
            task_descriptions = [task.description for task in tasks]
            combined_descriptions = " ".join(task_descriptions)

            # Should contain analysis-related keywords
            assert (
                "analyze" in combined_descriptions.lower()
                or "analysis" in combined_descriptions.lower()
            )
            assert "sentiment" in combined_descriptions.lower()
            assert (
                "summary" in combined_descriptions.lower()
                or "summarize" in combined_descriptions.lower()
            )

    def test_extract_themes_from_reviews(self, sample_movie_data):
        """Test theme extraction from reviews."""
        crew = MovieAnalysisCrew()

        themes = crew._extract_themes_from_reviews(sample_movie_data.reviews)

        assert isinstance(themes, list)
        assert len(themes) <= 5  # Should return max 5 themes

        # Should identify relevant themes
        all_content = " ".join(
            [review.content for review in sample_movie_data.reviews]
        ).lower()
        if "acting" in all_content:
            assert "Acting" in themes
        if "plot" in all_content or "story" in all_content:
            assert "Plot" in themes

    def test_extract_themes_specific_content(self):
        """Test theme extraction with specific content."""
        reviews = [
            ReviewData(
                content="The acting was superb and the cinematography was stunning.",
                source="test",
            ),
            ReviewData(
                content="Great direction and excellent music score throughout.",
                source="test",
            ),
            ReviewData(
                content="The visual effects were amazing and the camera work outstanding.",
                source="test",
            ),
        ]

        crew = MovieAnalysisCrew()
        themes = crew._extract_themes_from_reviews(reviews)

        # Should identify specific themes mentioned
        assert "Acting" in themes
        assert "Cinematography" in themes
        assert "Direction" in themes
        assert "Music/Sound" in themes

    def test_extract_pros_cons_from_reviews(self, sample_movie_data):
        """Test pros/cons extraction."""
        crew = MovieAnalysisCrew()

        pros_cons = crew._extract_pros_cons_from_reviews(sample_movie_data.reviews)

        assert isinstance(pros_cons, dict)
        assert "pros" in pros_cons
        assert "cons" in pros_cons
        assert isinstance(pros_cons["pros"], list)
        assert isinstance(pros_cons["cons"], list)

    def test_extract_pros_cons_positive_content(self):
        """Test pros/cons extraction with positive content."""
        reviews = [
            ReviewData(
                content="Great acting and excellent performance by the cast.",
                source="test",
            ),
            ReviewData(
                content="Very entertaining movie with stunning visuals.", source="test"
            ),
        ]

        crew = MovieAnalysisCrew()
        pros_cons = crew._extract_pros_cons_from_reviews(reviews)

        # Should identify positive aspects
        assert len(pros_cons["pros"]) > 0
        assert any("acting" in pro.lower() for pro in pros_cons["pros"])

    def test_extract_pros_cons_negative_content(self):
        """Test pros/cons extraction with negative content."""
        reviews = [
            ReviewData(
                content="Bad acting and poor performance throughout.", source="test"
            ),
            ReviewData(
                content="Very boring movie with terrible direction.", source="test"
            ),
        ]

        crew = MovieAnalysisCrew()
        pros_cons = crew._extract_pros_cons_from_reviews(reviews)

        # Should identify negative aspects
        assert len(pros_cons["cons"]) > 0
        assert any("acting" in con.lower() for con in pros_cons["cons"])

    def test_different_analysis_depths(self, mock_crew_with_llm, sample_movie_data):
        """Test different analysis depth options."""
        crew, mock_crew_instance = mock_crew_with_llm

        mock_crew_instance.kickoff.return_value = Mock(raw="Test analysis")

        # Test different depths
        for depth in ["Quick", "Standard", "Comprehensive"]:
            result = crew.analyze_movie(sample_movie_data, depth)
            assert isinstance(result, dict)
            assert "summary" in result

    def test_crew_error_handling(self, mock_crew_with_llm, sample_movie_data):
        """Test error handling in crew analysis."""
        crew, mock_crew_instance = mock_crew_with_llm

        # Mock crew execution error
        mock_crew_instance.kickoff.side_effect = Exception("CrewAI error")

        result = crew.analyze_movie(sample_movie_data, "Standard")

        # Should fall back to basic analysis
        assert isinstance(result, dict)
        assert result["analysis_method"] == "Fallback"
