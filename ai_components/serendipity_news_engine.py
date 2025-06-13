# ai_components/serendipity_news_engine.py
"""
Enhanced Serendipitous Newsletter Engine for Scenario 3:
Provide newsletters with content user will be interested in but never knew they would be
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation
import networkx as nx
import requests
from bs4 import BeautifulSoup
import feedparser
from sqlalchemy.orm import Session
import google.generativeai as genai

logger = logging.getLogger(__name__)

class ContentSource(Enum):
    """Different content sources for newsletter"""
    NEWS_API = "news_api"
    RSS_FEEDS = "rss_feeds"
    ACADEMIC_PAPERS = "academic_papers"
    TECH_BLOGS = "tech_blogs"
    SOCIAL_MEDIA = "social_media"
    PODCASTS = "podcasts"
    VIDEOS = "videos"

class InterestType(Enum):
    """Types of interests for discovery"""
    EXPLICIT = "explicit"      # Direct interests
    IMPLICIT = "implicit"      # Derived from behavior
    ADJACENT = "adjacent"      # Related to current interests
    CONTRARIAN = "contrarian"  # Opposite viewpoints
    SERENDIPITOUS = "serendipitous"  # Unexpected discoveries

@dataclass
class ContentItem:
    """Represents a piece of content"""
    title: str
    url: str
    content: str
    source: str
    published_date: datetime
    topics: List[str]
    sentiment: float
    quality_score: float
    novelty_score: float
    engagement_potential: float

@dataclass
class UserInterestGraph:
    """User's interest graph for recommendation"""
    user_id: str
    explicit_interests: Dict[str, float]  # topic -> strength
    implicit_interests: Dict[str, float]  # derived interests
    interest_evolution: List[Dict[str, Any]]  # temporal changes
    curiosity_profile: Dict[str, float]  # openness to new topics
    exploration_zones: List[str]  # areas for discovery
    last_updated: datetime

@dataclass
class NewsletterItem:
    """Single item in newsletter"""
    content_item: ContentItem
    interest_type: InterestType
    relevance_score: float
    surprise_factor: float
    explanation: str  # Why this was selected
    reading_time: int  # Estimated minutes

@dataclass
class Newsletter:
    """Complete newsletter"""
    user_id: str
    items: List[NewsletterItem]
    theme: str
    generation_date: datetime
    personalization_score: float
    diversity_score: float
    surprise_score: float

class ContentAggregator:
    """Aggregates content from multiple sources"""
    
    def __init__(self):
        self.sources = {
            ContentSource.RSS_FEEDS: [
                "https://feeds.feedburner.com/oreilly/radar/radar",
                "https://feeds.feedburner.com/TechCrunch",
                "https://rss.cnn.com/rss/edition.rss",
                "https://feeds.bbci.co.uk/news/rss.xml",
                "https://www.reddit.com/r/technology/.rss",
                "https://arxiv.org/rss/cs.AI",
                "https://medium.com/feed/@towardsdatascience"
            ],
            ContentSource.NEWS_API: "news_api_key_here",
            ContentSource.ACADEMIC_PAPERS: [
                "https://arxiv.org/",
                "https://scholar.google.com/"
            ]
        }
        
    async def fetch_recent_content(self, hours_back: int = 24) -> List[ContentItem]:
        """Fetch recent content from all sources"""
        all_content = []
        
        # Fetch from RSS feeds
        rss_content = await self._fetch_rss_content(hours_back)
        all_content.extend(rss_content)
        
        # Fetch from News API
        news_content = await self._fetch_news_api_content(hours_back)
        all_content.extend(news_content)
        
        # Remove duplicates and filter quality
        filtered_content = self._filter_and_deduplicate(all_content)
        
        return filtered_content
    
    async def _fetch_rss_content(self, hours_back: int) -> List[ContentItem]:
        """Fetch content from RSS feeds"""
        content_items = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        
        for feed_url in self.sources[ContentSource.RSS_FEEDS]:
            try:
                feed = feedparser.parse(feed_url)
                
                for entry in feed.entries:
                    # Parse publication date
                    pub_date = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        pub_date = datetime(*entry.published_parsed[:6])
                    
                    # Skip old content
                    if pub_date < cutoff_time:
                        continue
                    
                    # Extract content
                    content = getattr(entry, 'summary', '') or getattr(entry, 'description', '')
                    
                    if content and len(content) > 100:  # Minimum content length
                        item = ContentItem(
                            title=entry.title,
                            url=entry.link,
                            content=content,
                            source=feed_url,
                            published_date=pub_date,
                            topics=[],  # Will be populated later
                            sentiment=0.0,  # Will be calculated
                            quality_score=0.0,  # Will be calculated
                            novelty_score=0.0,
                            engagement_potential=0.0
                        )
                        content_items.append(item)
                        
            except Exception as e:
                logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
        
        return content_items
    
    async def _fetch_news_api_content(self, hours_back: int) -> List[ContentItem]:
        """Fetch content from News API"""
        # Placeholder for News API integration
        # In production, this would use actual News API
        return []
    
    def _filter_and_deduplicate(self, content_items: List[ContentItem]) -> List[ContentItem]:
        """Filter low-quality content and remove duplicates"""
        # Remove duplicates based on title similarity
        unique_items = []
        seen_titles = set()
        
        for item in content_items:
            title_lower = item.title.lower()
            
            # Simple duplicate detection
            is_duplicate = any(
                self._calculate_title_similarity(title_lower, seen_title) > 0.8 
                for seen_title in seen_titles
            )
            
            if not is_duplicate and len(item.content) > 200:  # Quality filter
                unique_items.append(item)
                seen_titles.add(title_lower)
        
        return unique_items
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles"""
        # Simple word overlap similarity
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class SurpriseCalculator:
    """Calculates surprise factor for content recommendations"""
    
    def __init__(self):
        self.surprise_factors = {
            'topic_distance': 0.4,      # How far from known interests
            'source_novelty': 0.3,      # New source or publication
            'perspective_shift': 0.2,   # Different viewpoint
            'temporal_relevance': 0.1   # Timely but unexpected
        }
    
    def calculate_surprise_score(
        self, 
        content_item: ContentItem, 
        user_interests: Dict[str, float],
        user_history: List[str]
    ) -> float:
        """Calculate how surprising this content would be for the user"""
        
        # Topic distance surprise
        topic_surprise = self._calculate_topic_distance_surprise(content_item.topics, user_interests)
        
        # Source novelty surprise
        source_surprise = self._calculate_source_novelty(content_item.source, user_history)
        
        # Perspective shift surprise
        perspective_surprise = self._calculate_perspective_surprise(content_item, user_interests)
        
        # Temporal relevance surprise
        temporal_surprise = self._calculate_temporal_surprise(content_item)
        
        # Weighted combination
        total_surprise = (
            topic_surprise * self.surprise_factors['topic_distance'] +
            source_surprise * self.surprise_factors['source_novelty'] +
            perspective_surprise * self.surprise_factors['perspective_shift'] +
            temporal_surprise * self.surprise_factors['temporal_relevance']
        )
        
        return min(1.0, total_surprise)
    
    def _calculate_topic_distance_surprise(self, content_topics: List[str], user_interests: Dict[str, float]) -> float:
        """Calculate surprise based on topic distance from user interests"""
        if not content_topics or not user_interests:
            return 0.5
        
        # Calculate minimum distance to any user interest
        min_distance = float('inf')
        
        for topic in content_topics:
            for user_topic, strength in user_interests.items():
                # Simple word-based distance (in production, use embeddings)
                distance = self._calculate_topic_distance(topic, user_topic)
                weighted_distance = distance / (strength + 0.1)  # Closer to stronger interests
                min_distance = min(min_distance, weighted_distance)
        
        # Convert distance to surprise (farther = more surprising)
        surprise = min(1.0, min_distance / 2.0) if min_distance != float('inf') else 0.5
        return surprise
    
    def _calculate_topic_distance(self, topic1: str, topic2: str) -> float:
        """Calculate distance between two topics"""
        # Simple word overlap distance
        words1 = set(topic1.lower().split())
        words2 = set(topic2.lower().split())
        
        if not words1 or not words2:
            return 1.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        similarity = len(intersection) / len(union)
        return 1.0 - similarity
    
    def _calculate_source_novelty(self, source: str, user_history: List[str]) -> float:
        """Calculate surprise based on source novelty"""
        # Check if user has seen this source before
        source_count = user_history.count(source)
        
        if source_count == 0:
            return 1.0  # Completely new source
        elif source_count < 3:
            return 0.7  # Somewhat new
        elif source_count < 10:
            return 0.3  # Familiar
        else:
            return 0.1  # Very familiar
    
    def _calculate_perspective_surprise(self, content_item: ContentItem, user_interests: Dict[str, float]) -> float:
        """Calculate surprise based on perspective shift"""
        # This would analyze sentiment and stance in production
        # For now, use a simple heuristic
        
        # Check if content challenges common viewpoints
        contrarian_indicators = ['however', 'but', 'contrary', 'challenge', 'myth', 'wrong']
        content_lower = content_item.content.lower()
        
        contrarian_score = sum(1 for indicator in contrarian_indicators if indicator in content_lower)
        return min(1.0, contrarian_score / 5.0)  # Normalize to max 5 indicators
    
    def _calculate_temporal_surprise(self, content_item: ContentItem) -> float:
        """Calculate surprise based on temporal relevance"""
        # Recent content is less surprising, but very timely content can be surprising
        hours_ago = (datetime.now() - content_item.published_date).total_seconds() / 3600
        
        if hours_ago < 1:
            return 0.8  # Very recent, potentially surprising
        elif hours_ago < 6:
            return 0.4  # Recent
        elif hours_ago < 24:
            return 0.2  # Today's news
        else:
            return 0.1  # Old news

class SerendipityNewsEngine:
    """
    Main engine for generating serendipitous newsletters
    
    Features:
    - Multi-source content aggregation
    - Interest graph construction and evolution
    - Surprise factor calculation
    - Exploration vs exploitation balance
    - Personalized newsletter generation
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.content_aggregator = ContentAggregator()
        self.surprise_calculator = SurpriseCalculator()
        
        # Initialize ML components
        self.topic_model = LatentDirichletAllocation(n_components=20, random_state=42)
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Exploration parameters
        self.exploration_rate = 0.3  # 30% exploration, 70% exploitation
        self.max_newsletter_items = 8
        self.diversity_threshold = 0.7
        
        # Content quality thresholds
        self.min_quality_score = 0.6
        self.min_novelty_score = 0.4
        
    async def generate_personalized_newsletter(self, user_id: str) -> Newsletter:
        """Generate a personalized newsletter with serendipitous content"""
        try:
            # Get user interest profile
            user_interests = await self._get_user_interest_graph(user_id)
            
            # Fetch recent content
            recent_content = await self.content_aggregator.fetch_recent_content(hours_back=48)
            
            # Analyze and score content
            analyzed_content = await self._analyze_content_batch(recent_content)
            
            # Generate interest recommendations
            newsletter_items = await self._select_newsletter_items(
                analyzed_content, user_interests, user_id
            )
            
            # Create newsletter
            newsletter = Newsletter(
                user_id=user_id,
                items=newsletter_items,
                theme=self._generate_newsletter_theme(newsletter_items),
                generation_date=datetime.now(),
                personalization_score=self._calculate_personalization_score(newsletter_items),
                diversity_score=self._calculate_diversity_score(newsletter_items),
                surprise_score=self._calculate_newsletter_surprise_score(newsletter_items)
            )
            
            # Store newsletter and update user interests
            await self._store_newsletter(newsletter)
            await self._update_user_interests_from_newsletter(user_id, newsletter)
            
            return newsletter
            
        except Exception as e:
            logger.error(f"Failed to generate newsletter for user {user_id}: {e}")
            raise
    
    async def _get_user_interest_graph(self, user_id: str) -> UserInterestGraph:
        """Get or create user interest graph"""
        try:
            # Try to load existing profile
            profile = await self._load_user_interest_graph(user_id)
            
            if profile:
                return profile
            
            # Create new profile if none exists
            return await self._create_initial_interest_graph(user_id)
            
        except Exception as e:
            logger.error(f"Failed to get user interest graph: {e}")
            return self._default_interest_graph(user_id)
    
    async def _create_initial_interest_graph(self, user_id: str) -> UserInterestGraph:
        """Create initial interest graph from user's existing data"""
        # Get user's notes and interactions
        user_notes = await self._get_user_notes(user_id)
        user_interactions = await self._get_user_interactions(user_id)
        
        # Extract interests from notes
        explicit_interests = await self._extract_interests_from_notes(user_notes)
        
        # Extract implicit interests from behavior
        implicit_interests = await self._extract_implicit_interests(user_interactions)
        
        # Determine exploration zones
        exploration_zones = await self._identify_exploration_zones(explicit_interests, implicit_interests)
        
        return UserInterestGraph(
            user_id=user_id,
            explicit_interests=explicit_interests,
            implicit_interests=implicit_interests,
            interest_evolution=[],
            curiosity_profile=self._calculate_curiosity_profile(user_interactions),
            exploration_zones=exploration_zones,
            last_updated=datetime.now()
        )
    
    def _default_interest_graph(self, user_id: str) -> UserInterestGraph:
        """Create default interest graph for new users"""
        return UserInterestGraph(
            user_id=user_id,
            explicit_interests={"technology": 0.5, "science": 0.3, "business": 0.2},
            implicit_interests={},
            interest_evolution=[],
            curiosity_profile={"openness": 0.7, "exploration_tendency": 0.5},
            exploration_zones=["philosophy", "art", "psychology"],
            last_updated=datetime.now()
        )
    
    async def _analyze_content_batch(self, content_items: List[ContentItem]) -> List[ContentItem]:
        """Analyze a batch of content items for topics, quality, and novelty"""
        if not content_items:
            return []
        
        # Extract topics for all content
        all_content_text = [item.content for item in content_items]
        
        # Fit topic model
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_content_text)
            topic_distributions = self.topic_model.fit_transform(tfidf_matrix)
            
            # Get feature names for topic interpretation
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Analyze each content item
            for i, item in enumerate(content_items):
                # Extract topics
                item.topics = self._extract_item_topics(topic_distributions[i], feature_names)
                
                # Calculate quality score
                item.quality_score = self._calculate_content_quality(item)
                
                # Calculate novelty score
                item.novelty_score = self._calculate_content_novelty(item, content_items)
                
                # Calculate engagement potential
                item.engagement_potential = self._calculate_engagement_potential(item)
                
        except Exception as e:
            logger.error(f"Failed to analyze content batch: {e}")
            # Return content with default scores
            for item in content_items:
                item.topics = []
                item.quality_score = 0.5
                item.novelty_score = 0.5
                item.engagement_potential = 0.5
        
        return content_items
    
    def _extract_item_topics(self, topic_distribution: np.ndarray, feature_names: np.ndarray) -> List[str]:
        """Extract main topics for a content item"""
        # Get top topics from distribution
        top_topic_indices = topic_distribution.argsort()[-3:][::-1]  # Top 3 topics
        
        topics = []
        for topic_idx in top_topic_indices:
            if topic_distribution[topic_idx] > 0.1:  # Minimum threshold
                # Get top words for this topic (simplified)
                topic_words = feature_names[:10]  # Simplified - would use proper topic modeling
                topic_name = " ".join(topic_words[:2])  # Use first 2 words as topic name
                topics.append(topic_name)
        
        return topics
    
    def _calculate_content_quality(self, item: ContentItem) -> float:
        """Calculate content quality score"""
        score = 0.0
        
        # Length factor (not too short, not too long)
        word_count = len(item.content.split())
        if 100 <= word_count <= 1000:
            score += 0.3
        elif word_count > 50:
            score += 0.1
        
        # Title quality
        if len(item.title.split()) >= 5:  # Descriptive title
            score += 0.2
        
        # Content structure
        if '. ' in item.content and len(item.content.split('. ')) >= 3:  # Multiple sentences
            score += 0.2
        
        # Readability (simple heuristic)
        avg_word_length = np.mean([len(word) for word in item.content.split()])
        if 4 <= avg_word_length <= 6:  # Reasonable word complexity
            score += 0.2
        
        # URL quality (has proper domain)
        if item.url and '.' in item.url and len(item.url) > 10:
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_content_novelty(self, item: ContentItem, all_items: List[ContentItem]) -> float:
        """Calculate how novel this content is compared to others"""
        if len(all_items) <= 1:
            return 1.0
        
        # Calculate similarity to other items
        similarities = []
        for other_item in all_items:
            if other_item != item:
                similarity = self._calculate_content_similarity(item, other_item)
                similarities.append(similarity)
        
        # Novelty is inverse of maximum similarity
        max_similarity = max(similarities) if similarities else 0
        novelty = 1.0 - max_similarity
        
        return novelty
    
    def _calculate_content_similarity(self, item1: ContentItem, item2: ContentItem) -> float:
        """Calculate similarity between two content items"""
        # Simple approach: word overlap in titles and content
        
        # Title similarity
        title1_words = set(item1.title.lower().split())
        title2_words = set(item2.title.lower().split())
        title_similarity = len(title1_words.intersection(title2_words)) / len(title1_words.union(title2_words)) if title1_words.union(title2_words) else 0
        
        # Content similarity (first 200 words)
        content1_words = set(item1.content.lower().split()[:200])
        content2_words = set(item2.content.lower().split()[:200])
        content_similarity = len(content1_words.intersection(content2_words)) / len(content1_words.union(content2_words)) if content1_words.union(content2_words) else 0
        
        # Combined similarity
        return (title_similarity * 0.4 + content_similarity * 0.6)
    
    def _calculate_engagement_potential(self, item: ContentItem) -> float:
        """Calculate potential for user engagement"""
        score = 0.0
        
        # Recency factor
        hours_ago = (datetime.now() - item.published_date).total_seconds() / 3600
        if hours_ago < 24:
            score += 0.3
        elif hours_ago < 48:
            score += 0.2
        else:
            score += 0.1
        
        # Title engagement indicators
        engaging_words = ['new', 'breakthrough', 'discover', 'reveal', 'secret', 'surprising', 'innovative']
        title_lower = item.title.lower()
        engagement_word_count = sum(1 for word in engaging_words if word in title_lower)
        score += min(0.3, engagement_word_count * 0.1)
        
        # Content engagement indicators
        question_count = item.content.count('?')
        score += min(0.2, question_count * 0.05)
        
        # Numbers and data (people like concrete information)
        number_count = len([word for word in item.content.split() if any(char.isdigit() for char in word)])
        score += min(0.2, number_count * 0.01)
        
        return min(1.0, score)
    
    async def _select_newsletter_items(
        self, 
        analyzed_content: List[ContentItem], 
        user_interests: UserInterestGraph, 
        user_id: str
    ) -> List[NewsletterItem]:
        """Select items for newsletter using exploration vs exploitation"""
        
        # Filter content by quality
        quality_content = [item for item in analyzed_content if item.quality_score >= self.min_quality_score]
        
        if not quality_content:
            logger.warning(f"No quality content found for user {user_id}")
            return []
        
        # Get user history for surprise calculation
        user_history = await self._get_user_content_history(user_id)
        
        # Score all content items
        scored_items = []
        for item in quality_content:
            # Calculate relevance to known interests
            relevance_score = self._calculate_relevance_score(item, user_interests)
            
            # Calculate surprise factor
            surprise_factor = self.surprise_calculator.calculate_surprise_score(
                item, user_interests.explicit_interests, user_history
            )
            
            # Determine interest type
            interest_type = self._classify_interest_type(item, user_interests, surprise_factor)
            
            # Generate explanation
            explanation = self._generate_item_explanation(item, interest_type, relevance_score, surprise_factor)
            
            # Estimate reading time
            reading_time = max(1, len(item.content.split()) // 200)  # ~200 words per minute
            
            newsletter_item = NewsletterItem(
                content_item=item,
                interest_type=interest_type,
                relevance_score=relevance_score,
                surprise_factor=surprise_factor,
                explanation=explanation,
                reading_time=reading_time
            )
            
            scored_items.append(newsletter_item)
        
        # Select diverse set of items
        selected_items = self._select_diverse_items(scored_items, user_interests)
        
        return selected_items[:self.max_newsletter_items]
    
    def _calculate_relevance_score(self, item: ContentItem, user_interests: UserInterestGraph) -> float:
        """Calculate relevance to user's known interests"""
        if not item.topics:
            return 0.2  # Default low relevance
        
        relevance_scores = []
        
        # Check against explicit interests
        for topic in item.topics:
            for interest, strength in user_interests.explicit_interests.items():
                topic_similarity = 1.0 - self.surprise_calculator._calculate_topic_distance(topic, interest)
                relevance_scores.append(topic_similarity * strength)
        
        # Check against implicit interests
        for topic in item.topics:
            for interest, strength in user_interests.implicit_interests.items():
                topic_similarity = 1.0 - self.surprise_calculator._calculate_topic_distance(topic, interest)
                relevance_scores.append(topic_similarity * strength * 0.7)  # Lower weight for implicit
        
        return max(relevance_scores) if relevance_scores else 0.2
    
    def _classify_interest_type(self, item: ContentItem, user_interests: UserInterestGraph, surprise_factor: float) -> InterestType:
        """Classify what type of interest this item represents"""
        relevance = self._calculate_relevance_score(item, user_interests)
        
        if relevance > 0.7:
            return InterestType.EXPLICIT
        elif relevance > 0.4:
            return InterestType.IMPLICIT
        elif surprise_factor > 0.7:
            return InterestType.SERENDIPITOUS
        elif any(topic in item.topics for topic in user_interests.exploration_zones):
            return InterestType.ADJACENT
        else:
            return InterestType.CONTRARIAN
    
    def _generate_item_explanation(self, item: ContentItem, interest_type: InterestType, relevance: float, surprise: float) -> str:
        """Generate explanation for why this item was selected"""
        explanations = {
            InterestType.EXPLICIT: f"This aligns with your known interests (relevance: {relevance:.1f})",
            InterestType.IMPLICIT: f"Based on your reading patterns, you might find this interesting",
            InterestType.ADJACENT: f"This explores an area related to your interests",
            InterestType.CONTRARIAN: f"A different perspective on topics you've engaged with",
            InterestType.SERENDIPITOUS: f"Something unexpected that might spark new interests (surprise: {surprise:.1f})"
        }
        
        base_explanation = explanations.get(interest_type, "Selected for discovery")
        
        # Add topic context if available
        if item.topics:
            topic_context = f" - focuses on {', '.join(item.topics[:2])}"
            return base_explanation + topic_context
        
        return base_explanation
    
    def _select_diverse_items(self, scored_items: List[NewsletterItem], user_interests: UserInterestGraph) -> List[NewsletterItem]:
        """Select diverse set of items balancing exploration and exploitation"""
        if not scored_items:
            return []
        
        selected = []
        
        # Sort by combined score (relevance + surprise + quality)
        def combined_score(item):
            return (
                item.relevance_score * 0.4 +
                item.surprise_factor * 0.3 +
                item.content_item.quality_score * 0.2 +
                item.content_item.engagement_potential * 0.1
            )
        
        scored_items.sort(key=combined_score, reverse=True)
        
        # Select items ensuring diversity
        exploitation_count = int(self.max_newsletter_items * (1 - self.exploration_rate))
        exploration_count = self.max_newsletter_items - exploitation_count
        
        # Select high-relevance items (exploitation)
        exploitation_items = [item for item in scored_items if item.relevance_score > 0.5]
        selected.extend(exploitation_items[:exploitation_count])
        
        # Select surprising/novel items (exploration)
        exploration_items = [item for item in scored_items 
                           if item not in selected and 
                           (item.surprise_factor > 0.5 or item.interest_type == InterestType.SERENDIPITOUS)]
        selected.extend(exploration_items[:exploration_count])
        
        # Fill remaining slots with best available
        remaining_slots = self.max_newsletter_items - len(selected)
        remaining_items = [item for item in scored_items if item not in selected]
        selected.extend(remaining_items[:remaining_slots])
        
        return selected
    
    def _generate_newsletter_theme(self, items: List[NewsletterItem]) -> str:
        """Generate a theme for the newsletter based on selected items"""
        if not items:
            return "Your Personalized Discovery"
        
        # Count interest types
        type_counts = {}
        for item in items:
            type_counts[item.interest_type] = type_counts.get(item.interest_type, 0) + 1
        
        dominant_type = max(type_counts, key=type_counts.get)
        
        themes = {
            InterestType.EXPLICIT: "Deep Dive into Your Interests",
            InterestType.IMPLICIT: "Following Your Curiosity",
            InterestType.ADJACENT: "Expanding Your Horizons",
            InterestType.CONTRARIAN: "Different Perspectives",
            InterestType.SERENDIPITOUS: "Unexpected Discoveries"
        }
        
        return themes.get(dominant_type, "Your Personalized Discovery")
    
    def _calculate_personalization_score(self, items: List[NewsletterItem]) -> float:
        """Calculate how personalized the newsletter is"""
        if not items:
            return 0.0
        
        relevance_scores = [item.relevance_score for item in items]
        return np.mean(relevance_scores)
    
    def _calculate_diversity_score(self, items: List[NewsletterItem]) -> float:
        """Calculate diversity of content in newsletter"""
        if not items:
            return 0.0
        
        # Count unique topics
        all_topics = []
        for item in items:
            all_topics.extend(item.content_item.topics)
        
        unique_topics = len(set(all_topics))
        total_topics = len(all_topics)
        
        return unique_topics / max(1, total_topics)
    
    def _calculate_newsletter_surprise_score(self, items: List[NewsletterItem]) -> float:
        """Calculate overall surprise score for newsletter"""
        if not items:
            return 0.0
        
        surprise_scores = [item.surprise_factor for item in items]
        return np.mean(surprise_scores)
    
    # Database and utility methods (implement based on your schema)
    async def _load_user_interest_graph(self, user_id: str) -> Optional[UserInterestGraph]:
        """Load user interest graph from database"""
        # Implementation depends on your database schema
        return None
    
    async def _get_user_notes(self, user_id: str) -> List[str]:
        """Get user's notes for interest extraction"""
        # Implementation depends on your database schema
        return []
    
    async def _get_user_interactions(self, user_id: str) -> List[Dict]:
        """Get user's interaction history"""
        # Implementation depends on your database schema
        return []
    
    async def _get_user_content_history(self, user_id: str) -> List[str]:
        """Get user's content consumption history"""
        # Implementation depends on your database schema
        return []
    
    async def _extract_interests_from_notes(self, notes: List[str]) -> Dict[str, float]:
        """Extract explicit interests from user notes"""
        if not notes:
            return {}
        
        # Simple TF-IDF approach
        try:
            vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(notes)
            feature_names = vectorizer.get_feature_names_out()
            
            # Calculate average importance across all notes
            importance_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            interests = {}
            for feature, score in zip(feature_names, importance_scores):
                if score > 0.1:  # Threshold for significance
                    interests[feature] = float(score)
            
            return interests
            
        except Exception as e:
            logger.error(f"Failed to extract interests from notes: {e}")
            return {}
    
    async def _extract_implicit_interests(self, interactions: List[Dict]) -> Dict[str, float]:
        """Extract implicit interests from user behavior"""
        # Analyze reading time, clicks, etc.
        # Implementation depends on your interaction data structure
        return {}
    
    async def _identify_exploration_zones(self, explicit: Dict[str, float], implicit: Dict[str, float]) -> List[str]:
        """Identify areas for exploration based on current interests"""
        # Define domain adjacencies
        domain_adjacencies = {
            "technology": ["science", "business", "philosophy"],
            "science": ["technology", "philosophy", "art"],
            "business": ["technology", "psychology", "economics"],
            "art": ["science", "philosophy", "psychology"],
            "psychology": ["science", "business", "philosophy"],
            "philosophy": ["science", "art", "psychology"]
        }
        
        exploration_zones = set()
        
        # Add adjacent domains to current interests
        for interest in list(explicit.keys()) + list(implicit.keys()):
            for domain, adjacencies in domain_adjacencies.items():
                if domain.lower() in interest.lower():
                    exploration_zones.update(adjacencies)
        
        # Remove current interests
        current_domains = set()
        for interest in list(explicit.keys()) + list(implicit.keys()):
            for domain in domain_adjacencies.keys():
                if domain.lower() in interest.lower():
                    current_domains.add(domain)
        
        exploration_zones -= current_domains
        
        return list(exploration_zones)[:5]  # Limit to 5 exploration zones
    
    def _calculate_curiosity_profile(self, interactions: List[Dict]) -> Dict[str, float]:
        """Calculate user's curiosity and exploration tendency"""
        # Simple heuristic based on interaction diversity
        # Implementation depends on your interaction data structure
        return {"openness": 0.7, "exploration_tendency": 0.5}
    
    async def _store_newsletter(self, newsletter: Newsletter) -> None:
        """Store newsletter in database"""
        # Implementation depends on your database schema
        pass
    
    async def _update_user_interests_from_newsletter(self, user_id: str, newsletter: Newsletter) -> None:
        """Update user interests based on newsletter interaction"""
        # This would track which items user clicked/read and update interest graph
        # Implementation depends on your tracking system
        pass