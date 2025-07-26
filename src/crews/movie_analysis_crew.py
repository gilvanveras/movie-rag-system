"""CrewAI-powered movie analysis crew."""

import logging
import os
from typing import Any, Dict, List

from crewai import Agent, Crew, Task
from crewai.tools import tool
from langchain_openai import ChatOpenAI

from models.movie_data import AnalysisResult, MovieData

logger = logging.getLogger(__name__)


class MovieAnalysisCrew:
    """CrewAI crew for comprehensive movie analysis."""

    def __init__(self):
        self.llm = self._initialize_llm()
        self.agents = self._create_agents()

    def _initialize_llm(self):
        """Initialize the language model for CrewAI."""
        try:
            # CrewAI works best with OpenAI API
            # For Groq-only setups, we'll use the fallback analysis which works great
            openai_api_key = os.getenv("OPENAI_API_KEY")
            openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            groq_api_key = os.getenv("GROQ_API_KEY")

            if openai_api_key:
                logger.info(f"CrewAI using OpenAI API with model: {openai_model}")
                return ChatOpenAI(
                    temperature=0.1,
                    model_name=openai_model,
                    openai_api_key=openai_api_key,
                )
            else:
                if groq_api_key:
                    logger.info(
                        "Groq API detected but CrewAI integration is complex. Using enhanced fallback analysis instead."
                    )
                else:
                    logger.info("No API keys found for CrewAI.")
                logger.info(
                    "The system will use the comprehensive fallback analysis which provides excellent results."
                )
                return None

        except Exception as e:
            logger.error(f"Failed to initialize LLM for CrewAI: {e}")
            return None

    def _create_agents(self) -> Dict[str, Agent]:
        """Create specialized agents for movie analysis."""

        if not self.llm:
            return {}

        agents = {}

        # Review Analyst Agent
        agents["review_analyst"] = Agent(
            role="Movie Review Analyst",
            goal="Analyze movie reviews to extract key themes, sentiments, and insights",
            backstory="""You are an expert movie critic and data analyst with years of experience 
            in analyzing film reviews from various sources. You excel at identifying patterns in 
            audience and critic opinions, extracting meaningful themes, and understanding the 
            overall reception of movies.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # Sentiment Analyst Agent
        agents["sentiment_analyst"] = Agent(
            role="Sentiment Analysis Specialist",
            goal="Perform detailed sentiment analysis on movie reviews and ratings",
            backstory="""You are a specialist in natural language processing and sentiment analysis 
            with expertise in understanding emotional nuances in text. You can accurately gauge 
            public opinion, identify positive and negative aspects, and quantify overall sentiment.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        # Movie Summarizer Agent
        agents["summarizer"] = Agent(
            role="Movie Summary Specialist",
            goal="Create comprehensive and engaging summaries of movie analysis",
            backstory="""You are a professional entertainment journalist and editor who excels 
            at creating clear, engaging, and comprehensive summaries. You can distill complex 
            analysis into accessible insights that help readers understand a movie's reception.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm,
        )

        return agents

    @tool
    def extract_review_themes(reviews_text: str) -> str:
        """Extract main themes from movie reviews."""
        # This would normally use more sophisticated NLP
        # For now, return a simple analysis
        return (
            f"Analyzed themes from {len(reviews_text.split())} words of review content"
        )

    @tool
    def calculate_sentiment_scores(reviews_text: str) -> str:
        """Calculate sentiment scores for reviews."""
        # Simple sentiment calculation
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "fantastic",
            "brilliant",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "horrible",
            "disappointing",
            "poor",
        ]

        text_lower = reviews_text.lower()
        positive_count = sum(text_lower.count(word) for word in positive_words)
        negative_count = sum(text_lower.count(word) for word in negative_words)

        return f"Positive indicators: {positive_count}, Negative indicators: {negative_count}"

    def analyze_movie(
        self, movie_data: MovieData, analysis_depth: str = "Standard"
    ) -> Dict[str, Any]:
        """Perform comprehensive movie analysis using CrewAI."""

        if not self.agents:
            return self._fallback_analysis(movie_data)

        try:
            # Prepare review content
            all_reviews = "\n\n".join([review.content for review in movie_data.reviews])
            critic_reviews = "\n\n".join(
                [
                    review.content
                    for review in movie_data.reviews
                    if review.review_type == "critic"
                ]
            )
            user_reviews = "\n\n".join(
                [
                    review.content
                    for review in movie_data.reviews
                    if review.review_type == "user"
                ]
            )

            # Create tasks based on analysis depth
            tasks = self._create_analysis_tasks(
                movie_data, all_reviews, critic_reviews, user_reviews, analysis_depth
            )

            # Create crew with tasks
            crew = Crew(agents=list(self.agents.values()), tasks=tasks, verbose=True)

            # Execute the crew
            results = crew.kickoff()

            # Process results
            return self._process_crew_results(results, movie_data)

        except Exception as e:
            logger.error(f"Error in CrewAI analysis: {e}")
            return self._fallback_analysis(movie_data)

    def _create_analysis_tasks(
        self,
        movie_data: MovieData,
        all_reviews: str,
        critic_reviews: str,
        user_reviews: str,
        analysis_depth: str,
    ) -> List[Task]:
        """Create analysis tasks for the crew."""

        tasks = []

        # Review Analysis Task
        review_task = Task(
            description=f"""
            Analyze the following movie reviews for '{movie_data.title}':
            
            ALL REVIEWS:
            {all_reviews[:2000]}...
            
            Extract:
            1. Main themes and topics discussed
            2. Most praised aspects
            3. Most criticized aspects
            4. Overall tone and reception
            5. Key insights about the movie's reception
            
            Provide a structured analysis focusing on patterns and insights.
            """,
            expected_output="""A structured analysis report containing:
            - Main themes and topics (3-5 key themes)
            - Most praised aspects (top 3)
            - Most criticized aspects (top 3)
            - Overall tone and reception summary
            - Key insights about the movie's reception""",
            agent=self.agents["review_analyst"],
        )
        tasks.append(review_task)

        # Sentiment Analysis Task
        sentiment_task = Task(
            description=f"""
            Perform detailed sentiment analysis on reviews for '{movie_data.title}':
            
            CRITIC REVIEWS:
            {critic_reviews[:1500]}...
            
            USER REVIEWS:
            {user_reviews[:1500]}...
            
            Analyze:
            1. Overall sentiment distribution (positive/neutral/negative)
            2. Differences between critic and audience sentiment
            3. Specific emotional responses and reactions
            4. Rating trends and patterns
            
            Provide quantitative and qualitative sentiment insights.
            """,
            expected_output="""A detailed sentiment analysis report containing:
            - Overall sentiment distribution with percentages
            - Comparison between critic and audience sentiment
            - Specific emotional responses identified
            - Rating trends and patterns analysis
            - Quantitative metrics and qualitative insights""",
            agent=self.agents["sentiment_analyst"],
        )
        tasks.append(sentiment_task)

        # Summary Task
        summary_task = Task(
            description=f"""
            Create a comprehensive summary of the analysis for '{movie_data.title}':
            
            Movie Details:
            - Title: {movie_data.title}
            - Year: {movie_data.year}
            - Director: {movie_data.director}
            - Genre: {movie_data.genre}
            - Ratings: {movie_data.ratings}
            - Total Reviews: {len(movie_data.reviews)}
            
            Based on the review analysis and sentiment analysis, create:
            1. Executive summary of the movie's reception
            2. Key strengths and weaknesses identified by reviewers
            3. Target audience and appeal
            4. Overall recommendation and context
            
            Make it engaging and informative for general audiences.
            """,
            expected_output="""A comprehensive summary report containing:
            - Executive summary of movie reception (2-3 paragraphs)
            - Key strengths identified by reviewers (3-5 points)
            - Key weaknesses identified by reviewers (3-5 points)
            - Target audience and appeal analysis
            - Overall recommendation with context
            - Engaging and accessible language for general audiences""",
            agent=self.agents["summarizer"],
        )
        tasks.append(summary_task)

        return tasks

    def _process_crew_results(
        self, results: Any, movie_data: MovieData
    ) -> Dict[str, Any]:
        """Process the results from CrewAI execution."""

        try:
            # Extract results (format depends on CrewAI version)
            if hasattr(results, "raw"):
                summary = results.raw
            else:
                summary = str(results)

            # Extract sentiment information
            sentiment = self._extract_sentiment_from_reviews(movie_data.reviews)

            # Extract themes
            themes = self._extract_themes_from_reviews(movie_data.reviews)

            # Extract pros and cons
            pros_cons = self._extract_pros_cons_from_reviews(movie_data.reviews)

            return {
                "summary": summary,
                "sentiment": sentiment,
                "themes": themes,
                "pros_cons": pros_cons,
                "total_reviews": len(movie_data.reviews),
                "sources": list(set(review.source for review in movie_data.reviews)),
                "analysis_method": "CrewAI",
            }

        except Exception as e:
            logger.error(f"Error processing crew results: {e}")
            return self._fallback_analysis(movie_data)

    def _fallback_analysis(self, movie_data: MovieData) -> Dict[str, Any]:
        """Enhanced fallback analysis when CrewAI is not available."""

        logger.info("Using enhanced fallback analysis method")

        # Simple sentiment analysis
        sentiment = self._extract_sentiment_from_reviews(movie_data.reviews)

        # Extract basic themes
        themes = self._extract_themes_from_reviews(movie_data.reviews)

        # Extract pros and cons
        pros_cons = self._extract_pros_cons_from_reviews(movie_data.reviews)

        # Create comprehensive summary
        total_reviews = len(movie_data.reviews)
        sources = list(set(review.source for review in movie_data.reviews))
        avg_rating = movie_data.get_average_rating()

        # Enhanced summary generation
        critic_reviews = [r for r in movie_data.reviews if r.review_type == "critic"]
        user_reviews = [r for r in movie_data.reviews if r.review_type == "user"]

        # Calculate more detailed sentiment
        positive_pct = sentiment.get("positive", 0) * 100
        negative_pct = sentiment.get("negative", 0) * 100

        # Determine overall reception
        if positive_pct > 70:
            reception = "overwhelmingly positive"
        elif positive_pct > 55:
            reception = "generally positive"
        elif positive_pct > 45:
            reception = "mixed"
        elif positive_pct > 30:
            reception = "generally negative"
        else:
            reception = "overwhelmingly negative"

        # Generate comprehensive summary
        summary_parts = []

        # Introduction
        summary_parts.append(
            f"Analysis of '{movie_data.title}' based on {total_reviews} reviews from {', '.join(sources)}."
        )

        # Rating information
        if avg_rating:
            summary_parts.append(
                f"The film has an average rating of {avg_rating:.1f}/10 across platforms."
            )

        # Overall reception
        summary_parts.append(
            f"The movie has received {reception} reception from viewers, with {positive_pct:.0f}% positive sentiment and {negative_pct:.0f}% negative sentiment."
        )

        # Source breakdown
        if critic_reviews and user_reviews:
            critic_sentiment = self._extract_sentiment_from_reviews(critic_reviews)
            user_sentiment = self._extract_sentiment_from_reviews(user_reviews)

            critic_pos = critic_sentiment.get("positive", 0) * 100
            user_pos = user_sentiment.get("positive", 0) * 100

            if abs(critic_pos - user_pos) > 15:
                if critic_pos > user_pos:
                    summary_parts.append(
                        f"Critics are more favorable ({critic_pos:.0f}% positive) compared to general audiences ({user_pos:.0f}% positive)."
                    )
                else:
                    summary_parts.append(
                        f"General audiences are more favorable ({user_pos:.0f}% positive) compared to critics ({critic_pos:.0f}% positive)."
                    )
            else:
                summary_parts.append(
                    f"Critics and audiences are generally aligned in their opinions."
                )

        # Themes and aspects
        if themes:
            top_themes = themes[:3]
            summary_parts.append(
                f"The most discussed aspects include {', '.join(top_themes[:-1])}, and {top_themes[-1]}."
            )

        # Strengths and weaknesses
        pros = pros_cons.get("pros", [])
        cons = pros_cons.get("cons", [])

        if pros:
            summary_parts.append(
                f"Viewers particularly praise the film for {', '.join(pros[:2])}."
            )

        if cons:
            summary_parts.append(f"Common criticisms focus on {', '.join(cons[:2])}.")

        # Conclusion
        if positive_pct > 60:
            summary_parts.append(
                "Overall, the film appears to be well-received and worth watching for most audiences."
            )
        elif positive_pct > 40:
            summary_parts.append(
                "The film has divided opinions and may appeal to specific audiences."
            )
        else:
            summary_parts.append(
                "The film has faced significant criticism and may not appeal to general audiences."
            )

        summary = " ".join(summary_parts)

        return {
            "summary": summary,
            "sentiment": sentiment,
            "themes": themes,
            "pros_cons": pros_cons,
            "total_reviews": total_reviews,
            "sources": sources,
            "analysis_method": "Enhanced Fallback",
            "critic_count": len(critic_reviews),
            "user_count": len(user_reviews),
            "average_rating": avg_rating,
        }

    def _extract_sentiment_from_reviews(self, reviews: List) -> Dict[str, float]:
        """Extract sentiment from reviews using simple heuristics."""

        if not reviews:
            return {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

        positive_count = 0
        negative_count = 0
        neutral_count = 0

        positive_words = [
            "excellent",
            "amazing",
            "fantastic",
            "brilliant",
            "outstanding",
            "superb",
            "wonderful",
            "great",
            "good",
            "impressive",
            "remarkable",
            "magnificent",
        ]

        negative_words = [
            "terrible",
            "awful",
            "horrible",
            "disappointing",
            "poor",
            "bad",
            "worst",
            "pathetic",
            "boring",
            "dull",
            "weak",
            "failed",
        ]

        for review in reviews:
            content_lower = review.content.lower()

            positive_score = sum(content_lower.count(word) for word in positive_words)
            negative_score = sum(content_lower.count(word) for word in negative_words)

            # Also consider ratings if available
            if review.rating:
                if review.rating >= 7:
                    positive_score += 2
                elif review.rating <= 4:
                    negative_score += 2

            if positive_score > negative_score:
                positive_count += 1
            elif negative_score > positive_score:
                negative_count += 1
            else:
                neutral_count += 1

        total = len(reviews)
        return {
            "positive": positive_count / total,
            "neutral": neutral_count / total,
            "negative": negative_count / total,
        }

    def _extract_themes_from_reviews(self, reviews: List) -> List[str]:
        """Extract common themes from reviews."""

        theme_keywords = {
            "Acting": [
                "acting",
                "performance",
                "actor",
                "actress",
                "cast",
                "character",
            ],
            "Plot": ["plot", "story", "storyline", "narrative", "script"],
            "Direction": ["direction", "director", "directed", "directing"],
            "Cinematography": [
                "cinematography",
                "visuals",
                "camera",
                "shots",
                "photography",
            ],
            "Music/Sound": ["music", "soundtrack", "score", "sound", "audio"],
            "Special Effects": ["effects", "cgi", "visual effects", "special effects"],
            "Pacing": ["pacing", "pace", "slow", "fast", "rushed", "dragged"],
            "Entertainment": [
                "entertaining",
                "fun",
                "enjoyable",
                "engaging",
                "thrilling",
            ],
        }

        theme_scores = {}
        all_content = " ".join([review.content.lower() for review in reviews])

        for theme, keywords in theme_keywords.items():
            score = sum(all_content.count(keyword) for keyword in keywords)
            if score > 0:
                theme_scores[theme] = score

        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        return [theme for theme, score in sorted_themes[:5]]

    def _extract_pros_cons_from_reviews(self, reviews: List) -> Dict[str, List[str]]:
        """Extract commonly mentioned pros and cons."""

        pros_keywords = {
            "Great acting": ["great acting", "excellent performance", "brilliant cast"],
            "Engaging plot": [
                "interesting story",
                "compelling plot",
                "engaging narrative",
            ],
            "Beautiful visuals": [
                "stunning visuals",
                "beautiful cinematography",
                "gorgeous shots",
            ],
            "Good direction": [
                "well directed",
                "excellent direction",
                "masterful directing",
            ],
            "Entertaining": ["very entertaining", "highly enjoyable", "lots of fun"],
        }

        cons_keywords = {
            "Poor acting": ["bad acting", "poor performance", "terrible cast"],
            "Weak plot": ["boring story", "confusing plot", "weak narrative"],
            "Bad visuals": ["poor visuals", "bad cinematography", "ugly shots"],
            "Poor direction": ["badly directed", "poor direction", "weak directing"],
            "Boring": ["very boring", "not entertaining", "dull movie"],
        }

        all_content = " ".join([review.content.lower() for review in reviews])

        pros = []
        for pro, keywords in pros_keywords.items():
            if any(keyword in all_content for keyword in keywords):
                pros.append(pro)

        cons = []
        for con, keywords in cons_keywords.items():
            if any(keyword in all_content for keyword in keywords):
                cons.append(con)

        return {"pros": pros[:5], "cons": cons[:5]}
