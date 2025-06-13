# ai_components/user_behavior_analyzer.py
"""
Advanced user behavior analysis for personalized AI assistant
Analyzes user interaction patterns to improve personalization
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy.orm import Session
from pydantic import BaseModel
import pandas as pd

logger = logging.getLogger(__name__)

@dataclass
class InteractionEvent:
    """Represents a single user interaction event"""
    user_id: str
    timestamp: datetime
    event_type: str  # 'question', 'document_upload', 'feedback', 'reading_time'
    content: str
    metadata: Dict[str, Any]
    reading_time: Optional[float] = None
    satisfaction_score: Optional[float] = None

@dataclass
class UserProfile:
    """Comprehensive user profile with behavioral patterns"""
    user_id: str
    interest_clusters: List[str]
    communication_style: Dict[str, float]
    preferred_content_types: Dict[str, float]
    optimal_interaction_times: List[int]  # Hours of day
    attention_span: float  # Average reading time
    curiosity_score: float  # Tendency to explore new topics
    last_updated: datetime

class UserBehaviorAnalyzer:
    """
    Advanced user behavior analysis system for personalization
    
    Features:
    - Interaction pattern analysis
    - Interest clustering and evolution tracking
    - Communication style profiling
    - Optimal timing detection
    - Attention span measurement
    """
    
    def __init__(self, db_session: Session):
        self.db_session = db_session
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.interest_clusters = 8  # Number of interest clusters
        self.min_interactions = 5   # Minimum interactions for profiling
        
    async def track_interaction(self, interaction: InteractionEvent) -> None:
        """Track a new user interaction event"""
        try:
            # Store interaction in database
            await self._store_interaction(interaction)
            
            # Update user profile if enough data
            if await self._get_interaction_count(interaction.user_id) >= self.min_interactions:
                await self.update_user_profile(interaction.user_id)
                
        except Exception as e:
            logger.error(f"Failed to track interaction: {e}")
    
    async def analyze_interaction_patterns(self, user_id: str) -> Dict[str, Any]:
        """Analyze comprehensive interaction patterns for a user"""
        try:
            interactions = await self._get_user_interactions(user_id)
            if len(interactions) < self.min_interactions:
                return {"status": "insufficient_data", "interactions_needed": self.min_interactions - len(interactions)}
            
            # Temporal analysis
            temporal_patterns = self._analyze_temporal_patterns(interactions)
            
            # Content analysis
            content_patterns = await self._analyze_content_patterns(interactions)
            
            # Behavioral metrics
            behavioral_metrics = self._calculate_behavioral_metrics(interactions)
            
            return {
                "temporal_patterns": temporal_patterns,
                "content_patterns": content_patterns,
                "behavioral_metrics": behavioral_metrics,
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze patterns for user {user_id}: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_temporal_patterns(self, interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Analyze when users are most active and engaged"""
        hours = [interaction.timestamp.hour for interaction in interactions]
        days = [interaction.timestamp.weekday() for interaction in interactions]
        
        # Find peak activity hours
        hour_counter = Counter(hours)
        peak_hours = [hour for hour, count in hour_counter.most_common(3)]
        
        # Calculate session patterns
        sessions = self._identify_sessions(interactions)
        avg_session_length = np.mean([session['duration'] for session in sessions]) if sessions else 0
        
        return {
            "peak_hours": peak_hours,
            "day_distribution": dict(Counter(days)),
            "average_session_length": avg_session_length,
            "total_sessions": len(sessions),
            "activity_score": len(interactions) / max(1, (datetime.now() - interactions[0].timestamp).days)
        }
    
    async def _analyze_content_patterns(self, interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Analyze content preferences and interest evolution"""
        # Extract content for analysis
        content_texts = [interaction.content for interaction in interactions if interaction.content]
        
        if not content_texts:
            return {"status": "no_content"}
        
        # TF-IDF analysis for topics
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(content_texts)
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Get most important terms
            importance_scores = np.array(tfidf_matrix.sum(axis=0)).flatten()
            top_indices = importance_scores.argsort()[-20:][::-1]
            top_topics = [(feature_names[i], importance_scores[i]) for i in top_indices]
            
            # Cluster interests
            if len(content_texts) >= self.interest_clusters:
                kmeans = KMeans(n_clusters=self.interest_clusters, random_state=42)
                clusters = kmeans.fit_predict(tfidf_matrix)
                cluster_distribution = dict(Counter(clusters))
            else:
                cluster_distribution = {}
            
            return {
                "top_topics": top_topics,
                "cluster_distribution": cluster_distribution,
                "content_diversity": len(set(content_texts)) / len(content_texts),
                "vocabulary_richness": len(feature_names)
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {"status": "analysis_failed"}
    
    def _calculate_behavioral_metrics(self, interactions: List[InteractionEvent]) -> Dict[str, Any]:
        """Calculate key behavioral metrics"""
        reading_times = [i.reading_time for i in interactions if i.reading_time]
        satisfaction_scores = [i.satisfaction_score for i in interactions if i.satisfaction_score]
        
        # Communication style analysis
        question_interactions = [i for i in interactions if i.event_type == 'question']
        avg_question_length = np.mean([len(i.content.split()) for i in question_interactions]) if question_interactions else 0
        
        # Exploration behavior
        unique_content_types = len(set(i.event_type for i in interactions))
        exploration_score = unique_content_types / len(set(['question', 'document_upload', 'feedback']))
        
        return {
            "average_reading_time": np.mean(reading_times) if reading_times else 0,
            "average_satisfaction": np.mean(satisfaction_scores) if satisfaction_scores else 0,
            "average_question_length": avg_question_length,
            "exploration_score": exploration_score,
            "interaction_frequency": len(interactions),
            "engagement_consistency": self._calculate_consistency_score(interactions)
        }
    
    def _identify_sessions(self, interactions: List[InteractionEvent], session_gap_minutes: int = 30) -> List[Dict]:
        """Identify user sessions based on temporal gaps"""
        if not interactions:
            return []
        
        sorted_interactions = sorted(interactions, key=lambda x: x.timestamp)
        sessions = []
        current_session_start = sorted_interactions[0].timestamp
        current_session_interactions = [sorted_interactions[0]]
        
        for i in range(1, len(sorted_interactions)):
            current = sorted_interactions[i]
            previous = sorted_interactions[i-1]
            
            gap = (current.timestamp - previous.timestamp).total_seconds() / 60
            
            if gap <= session_gap_minutes:
                current_session_interactions.append(current)
            else:
                # End current session
                session_end = previous.timestamp
                sessions.append({
                    "start": current_session_start,
                    "end": session_end,
                    "duration": (session_end - current_session_start).total_seconds() / 60,
                    "interaction_count": len(current_session_interactions)
                })
                
                # Start new session
                current_session_start = current.timestamp
                current_session_interactions = [current]
        
        # Add the last session
        if current_session_interactions:
            session_end = current_session_interactions[-1].timestamp
            sessions.append({
                "start": current_session_start,
                "end": session_end,
                "duration": (session_end - current_session_start).total_seconds() / 60,
                "interaction_count": len(current_session_interactions)
            })
        
        return sessions
    
    def _calculate_consistency_score(self, interactions: List[InteractionEvent]) -> float:
        """Calculate how consistently a user engages over time"""
        if len(interactions) < 2:
            return 0.0
        
        # Calculate daily interaction counts
        daily_counts = defaultdict(int)
        for interaction in interactions:
            date_key = interaction.timestamp.date()
            daily_counts[date_key] += 1
        
        if len(daily_counts) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower = more consistent)
        counts = list(daily_counts.values())
        mean_count = np.mean(counts)
        std_count = np.std(counts)
        
        if mean_count == 0:
            return 0.0
        
        cv = std_count / mean_count
        # Convert to consistency score (higher = more consistent)
        consistency_score = max(0.0, 1.0 - cv)
        
        return consistency_score
    
    async def update_user_profile(self, user_id: str) -> UserProfile:
        """Update comprehensive user profile based on latest interactions"""
        try:
            analysis = await self.analyze_interaction_patterns(user_id)
            
            if analysis.get("status") == "insufficient_data":
                raise ValueError("Insufficient data for profile update")
            
            # Extract profile components
            temporal = analysis.get("temporal_patterns", {})
            content = analysis.get("content_patterns", {})
            behavioral = analysis.get("behavioral_metrics", {})
            
            # Build user profile
            profile = UserProfile(
                user_id=user_id,
                interest_clusters=self._extract_interest_clusters(content),
                communication_style=self._extract_communication_style(behavioral),
                preferred_content_types=self._extract_content_preferences(analysis),
                optimal_interaction_times=temporal.get("peak_hours", []),
                attention_span=behavioral.get("average_reading_time", 0),
                curiosity_score=behavioral.get("exploration_score", 0),
                last_updated=datetime.now()
            )
            
            # Store profile
            await self._store_user_profile(profile)
            
            return profile
            
        except Exception as e:
            logger.error(f"Failed to update profile for user {user_id}: {e}")
            raise
    
    def _extract_interest_clusters(self, content_analysis: Dict) -> List[str]:
        """Extract main interest clusters from content analysis"""
        top_topics = content_analysis.get("top_topics", [])
        return [topic[0] for topic in top_topics[:5]]  # Top 5 topics
    
    def _extract_communication_style(self, behavioral_metrics: Dict) -> Dict[str, float]:
        """Extract communication style characteristics"""
        return {
            "verbosity": min(1.0, behavioral_metrics.get("average_question_length", 0) / 20),  # Normalized
            "engagement": behavioral_metrics.get("engagement_consistency", 0),
            "exploration": behavioral_metrics.get("exploration_score", 0)
        }
    
    def _extract_content_preferences(self, analysis: Dict) -> Dict[str, float]:
        """Extract content type preferences"""
        # This would be enhanced based on actual content types in your system
        return {
            "text": 0.8,
            "documents": 0.6,
            "audio": 0.4,
            "visual": 0.3
        }
    
    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve current user profile"""
        try:
            # Implementation would retrieve from database
            # This is a placeholder for the database query
            return await self._get_user_profile_from_db(user_id)
        except Exception as e:
            logger.error(f"Failed to get profile for user {user_id}: {e}")
            return None
    
    async def predict_optimal_interaction_time(self, user_id: str) -> List[int]:
        """Predict optimal hours for user interaction"""
        profile = await self.get_user_profile(user_id)
        if profile and profile.optimal_interaction_times:
            return profile.optimal_interaction_times
        
        # Default to common active hours if no profile
        return [9, 14, 20]  # 9 AM, 2 PM, 8 PM
    
    # Database interaction methods (to be implemented based on your actual DB schema)
    async def _store_interaction(self, interaction: InteractionEvent) -> None:
        """Store interaction in database"""
        # Implementation depends on your database schema
        pass
    
    async def _get_user_interactions(self, user_id: str, limit: int = 1000) -> List[InteractionEvent]:
        """Retrieve user interactions from database"""
        # Implementation depends on your database schema
        return []
    
    async def _get_interaction_count(self, user_id: str) -> int:
        """Get total interaction count for user"""
        # Implementation depends on your database schema
        return 0
    
    async def _store_user_profile(self, profile: UserProfile) -> None:
        """Store user profile in database"""
        # Implementation depends on your database schema
        pass
    
    async def _get_user_profile_from_db(self, user_id: str) -> Optional[UserProfile]:
        """Retrieve user profile from database"""
        # Implementation depends on your database schema
        return None