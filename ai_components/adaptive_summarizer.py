# ai_components/adaptive_summarizer.py
"""
Enhanced Adaptive Summarization Engine for Scenario 2:
Input notes and generate summaries based on user's thinking style
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
import google.generativeai as genai

logger = logging.getLogger(__name__)

class SummaryLevel(Enum):
    """Different levels of summary detail"""
    EXECUTIVE = "executive"  # 30-second read
    DETAILED = "detailed"   # 2-3 minute read
    COMPREHENSIVE = "comprehensive"  # Full analysis

class WritingStyle(Enum):
    """Different writing style categories"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    TECHNICAL = "technical"
    CONVERSATIONAL = "conversational"
    ACADEMIC = "academic"
    PRACTICAL = "practical"

@dataclass
class UserWritingProfile:
    """User's writing style profile"""
    user_id: str
    preferred_length: str  # "concise", "moderate", "detailed"
    sentence_complexity: float  # 0-1, simple to complex
    vocabulary_level: str  # "basic", "intermediate", "advanced"
    tone: str  # "formal", "casual", "technical"
    structure_preference: str  # "bullet_points", "paragraphs", "mixed"
    emphasis_style: str  # "bold", "italics", "headers", "mixed"
    example_preference: bool  # Likes concrete examples
    question_style: bool  # Includes rhetorical questions
    metaphor_usage: float  # 0-1, frequency of metaphors/analogies
    action_orientation: float  # 0-1, focus on actionable items
    last_updated: datetime

@dataclass
class SummaryOutput:
    """Structured summary output"""
    executive_summary: str
    detailed_summary: str
    key_insights: List[str]
    action_items: List[str]
    questions_raised: List[str]
    related_concepts: List[str]
    confidence_score: float
    style_match_score: float

class WritingStyleAnalyzer:
    """Analyzes and learns user's writing style from samples"""
    
    def __init__(self):
        self.style_features = {
            'sentence_length': [],
            'word_complexity': [],
            'punctuation_usage': {},
            'vocabulary_richness': 0,
            'structure_patterns': [],
            'tone_indicators': []
        }
        
    def analyze_writing_samples(self, writing_samples: List[str]) -> UserWritingProfile:
        """Analyze user's writing samples to extract style profile"""
        if not writing_samples:
            return self._default_profile()
        
        combined_text = " ".join(writing_samples)
        
        # Analyze sentence structure
        sentences = re.split(r'[.!?]+', combined_text)
        sentence_lengths = [len(sentence.split()) for sentence in sentences if sentence.strip()]
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 10
        
        # Determine complexity level
        complexity = self._calculate_sentence_complexity(avg_sentence_length)
        
        # Analyze vocabulary
        vocab_level = self._analyze_vocabulary_level(combined_text)
        
        # Determine tone
        tone = self._analyze_tone(combined_text)
        
        # Analyze structure preferences
        structure_pref = self._analyze_structure_preference(writing_samples)
        
        # Calculate other metrics
        metaphor_usage = self._calculate_metaphor_usage(combined_text)
        action_orientation = self._calculate_action_orientation(combined_text)
        
        return UserWritingProfile(
            user_id="",  # Will be set by caller
            preferred_length=self._determine_length_preference(writing_samples),
            sentence_complexity=complexity,
            vocabulary_level=vocab_level,
            tone=tone,
            structure_preference=structure_pref,
            emphasis_style=self._analyze_emphasis_style(writing_samples),
            example_preference=self._has_example_preference(combined_text),
            question_style=self._uses_questions(combined_text),
            metaphor_usage=metaphor_usage,
            action_orientation=action_orientation,
            last_updated=datetime.now()
        )
    
    def _default_profile(self) -> UserWritingProfile:
        """Return default writing profile"""
        return UserWritingProfile(
            user_id="",
            preferred_length="moderate",
            sentence_complexity=0.5,
            vocabulary_level="intermediate",
            tone="casual",
            structure_preference="mixed",
            emphasis_style="mixed",
            example_preference=True,
            question_style=False,
            metaphor_usage=0.3,
            action_orientation=0.6,
            last_updated=datetime.now()
        )
    
    def _calculate_sentence_complexity(self, avg_length: float) -> float:
        """Calculate sentence complexity score"""
        # Simple heuristic: longer sentences = more complex
        if avg_length < 8:
            return 0.2  # Simple
        elif avg_length < 15:
            return 0.5  # Moderate
        elif avg_length < 25:
            return 0.8  # Complex
        else:
            return 1.0  # Very complex
    
    def _analyze_vocabulary_level(self, text: str) -> str:
        """Analyze vocabulary sophistication"""
        words = text.lower().split()
        
        # Simple heuristic based on word length and complexity
        avg_word_length = np.mean([len(word) for word in words])
        
        # Count complex words (>6 letters)
        complex_words = sum(1 for word in words if len(word) > 6)
        complexity_ratio = complex_words / len(words) if words else 0
        
        if avg_word_length < 4.5 and complexity_ratio < 0.15:
            return "basic"
        elif avg_word_length < 5.5 and complexity_ratio < 0.25:
            return "intermediate"
        else:
            return "advanced"
    
    def _analyze_tone(self, text: str) -> str:
        """Analyze writing tone"""
        text_lower = text.lower()
        
        # Simple keyword-based tone detection
        formal_indicators = ['furthermore', 'therefore', 'consequently', 'moreover', 'nevertheless']
        casual_indicators = ['really', 'pretty', 'quite', 'sort of', 'kind of', 'actually']
        technical_indicators = ['implement', 'analyze', 'optimize', 'configure', 'process']
        
        formal_score = sum(1 for indicator in formal_indicators if indicator in text_lower)
        casual_score = sum(1 for indicator in casual_indicators if indicator in text_lower)
        technical_score = sum(1 for indicator in technical_indicators if indicator in text_lower)
        
        if technical_score > max(formal_score, casual_score):
            return "technical"
        elif formal_score > casual_score:
            return "formal"
        else:
            return "casual"
    
    def _analyze_structure_preference(self, samples: List[str]) -> str:
        """Analyze preferred text structure"""
        bullet_count = 0
        paragraph_count = 0
        
        for sample in samples:
            # Count bullet points or list items
            bullet_count += len(re.findall(r'[•\-\*]\s+', sample))
            bullet_count += len(re.findall(r'\n\d+\.\s+', sample))
            
            # Count paragraph breaks
            paragraph_count += len(re.findall(r'\n\s*\n', sample))
        
        if bullet_count > paragraph_count * 1.5:
            return "bullet_points"
        elif paragraph_count > bullet_count * 1.5:
            return "paragraphs"
        else:
            return "mixed"
    
    def _determine_length_preference(self, samples: List[str]) -> str:
        """Determine preferred content length"""
        if not samples:
            return "moderate"
        
        avg_length = np.mean([len(sample.split()) for sample in samples])
        
        if avg_length < 50:
            return "concise"
        elif avg_length < 200:
            return "moderate"
        else:
            return "detailed"
    
    def _analyze_emphasis_style(self, samples: List[str]) -> str:
        """Analyze preferred emphasis style"""
        combined = " ".join(samples)
        
        bold_count = len(re.findall(r'\*\*[^*]+\*\*', combined))
        italic_count = len(re.findall(r'\*[^*]+\*', combined))
        header_count = len(re.findall(r'^#+\s+', combined, re.MULTILINE))
        
        if header_count > max(bold_count, italic_count):
            return "headers"
        elif bold_count > italic_count:
            return "bold"
        elif italic_count > 0:
            return "italics"
        else:
            return "mixed"
    
    def _has_example_preference(self, text: str) -> bool:
        """Check if user prefers concrete examples"""
        example_indicators = ['for example', 'such as', 'like', 'instance', 'e.g.', 'i.e.']
        return any(indicator in text.lower() for indicator in example_indicators)
    
    def _uses_questions(self, text: str) -> bool:
        """Check if user uses rhetorical questions"""
        return '?' in text and len(re.findall(r'\?', text)) > 1
    
    def _calculate_metaphor_usage(self, text: str) -> float:
        """Calculate frequency of metaphors and analogies"""
        metaphor_indicators = ['like', 'as if', 'similar to', 'reminds me of', 'metaphorically']
        count = sum(text.lower().count(indicator) for indicator in metaphor_indicators)
        words = len(text.split())
        return min(1.0, count / max(1, words / 100))  # Normalize per 100 words
    
    def _calculate_action_orientation(self, text: str) -> float:
        """Calculate focus on actionable items"""
        action_words = ['should', 'must', 'need to', 'will', 'plan', 'implement', 'execute', 'action']
        count = sum(text.lower().count(word) for word in action_words)
        words = len(text.split())
        return min(1.0, count / max(1, words / 50))  # Normalize per 50 words

class AdaptiveSummarizer:
    """
    Main adaptive summarization engine that learns and adapts to user's style
    
    Features:
    - Multi-level summarization (executive, detailed, comprehensive)
    - Style adaptation based on user writing samples
    - Key insight extraction
    - Action item identification
    - Question generation for deeper thinking
    """
    
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.style_analyzer = WritingStyleAnalyzer()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize summarization models
        self.summarization_pipeline = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )
        
        # Template library for different styles
        self.style_templates = {
            WritingStyle.ANALYTICAL: {
                "intro": "Analysis reveals that",
                "structure": "paragraphs",
                "emphasis": "**",
                "conclusion": "In conclusion"
            },
            WritingStyle.CREATIVE: {
                "intro": "Imagine",
                "structure": "mixed",
                "emphasis": "*",
                "conclusion": "The story continues"
            },
            WritingStyle.TECHNICAL: {
                "intro": "Technical analysis indicates",
                "structure": "bullet_points",
                "emphasis": "###",
                "conclusion": "Implementation summary"
            },
            WritingStyle.CONVERSATIONAL: {
                "intro": "Here's what's interesting",
                "structure": "paragraphs",
                "emphasis": "**",
                "conclusion": "What do you think?"
            }
        }
    
    async def learn_user_style(self, user_id: str, writing_samples: List[str]) -> UserWritingProfile:
        """Learn and store user's writing style from samples"""
        try:
            profile = self.style_analyzer.analyze_writing_samples(writing_samples)
            profile.user_id = user_id
            
            # Store profile in database
            await self._store_user_profile(profile)
            
            logger.info(f"Learned writing style for user {user_id}")
            return profile
            
        except Exception as e:
            logger.error(f"Failed to learn user style: {e}")
            return self.style_analyzer._default_profile()
    
    async def generate_adaptive_summary(
        self, 
        content: str, 
        user_id: str, 
        summary_level: SummaryLevel = SummaryLevel.DETAILED,
        additional_context: Optional[str] = None
    ) -> SummaryOutput:
        """Generate summary adapted to user's writing style"""
        try:
            # Get user's style profile
            user_profile = await self._get_user_profile(user_id)
            if not user_profile:
                user_profile = self.style_analyzer._default_profile()
                user_profile.user_id = user_id
            
            # Analyze content structure
            content_analysis = await self._analyze_content_structure(content)
            
            # Generate summaries at different levels
            executive_summary = await self._generate_executive_summary(content, user_profile)
            detailed_summary = await self._generate_detailed_summary(content, user_profile, content_analysis)
            
            # Extract key components
            key_insights = await self._extract_key_insights(content, user_profile)
            action_items = await self._extract_action_items(content, user_profile)
            questions = await self._generate_questions(content, user_profile)
            related_concepts = await self._extract_related_concepts(content)
            
            # Calculate confidence scores
            confidence_score = self._calculate_confidence_score(content, content_analysis)
            style_match_score = self._calculate_style_match_score(user_profile)
            
            return SummaryOutput(
                executive_summary=executive_summary,
                detailed_summary=detailed_summary,
                key_insights=key_insights,
                action_items=action_items,
                questions_raised=questions,
                related_concepts=related_concepts,
                confidence_score=confidence_score,
                style_match_score=style_match_score
            )
            
        except Exception as e:
            logger.error(f"Failed to generate adaptive summary: {e}")
            raise
    
    async def _analyze_content_structure(self, content: str) -> Dict[str, Any]:
        """Analyze the structure and characteristics of input content"""
        # Split into chunks for analysis
        chunks = self.text_splitter.split_text(content)
        
        # Basic metrics
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        paragraph_count = len(content.split('\n\n'))
        
        # Content type detection
        content_type = self._detect_content_type(content)
        
        # Key topic extraction
        topics = await self._extract_main_topics(content)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "chunk_count": len(chunks),
            "content_type": content_type,
            "main_topics": topics,
            "complexity_score": self._calculate_content_complexity(content)
        }
    
    def _detect_content_type(self, content: str) -> str:
        """Detect the type of content (academic, technical, creative, etc.)"""
        content_lower = content.lower()
        
        # Define indicators for different content types
        academic_indicators = ['research', 'study', 'analysis', 'methodology', 'conclusion', 'abstract']
        technical_indicators = ['implementation', 'algorithm', 'system', 'configuration', 'architecture']
        creative_indicators = ['story', 'imagine', 'creative', 'artistic', 'inspiration']
        business_indicators = ['strategy', 'market', 'revenue', 'customer', 'growth', 'roi']
        personal_indicators = ['i think', 'my experience', 'personally', 'reflection']
        
        # Count indicators
        scores = {
            'academic': sum(1 for indicator in academic_indicators if indicator in content_lower),
            'technical': sum(1 for indicator in technical_indicators if indicator in content_lower),
            'creative': sum(1 for indicator in creative_indicators if indicator in content_lower),
            'business': sum(1 for indicator in business_indicators if indicator in content_lower),
            'personal': sum(1 for indicator in personal_indicators if indicator in content_lower)
        }
        
        # Return type with highest score, default to 'general'
        max_type = max(scores, key=scores.get) if max(scores.values()) > 0 else 'general'
        return max_type
    
    async def _extract_main_topics(self, content: str) -> List[str]:
        """Extract main topics from content using TF-IDF"""
        try:
            vectorizer = TfidfVectorizer(
                max_features=20,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get top topics
            topic_scores = list(zip(feature_names, scores))
            topic_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [topic for topic, score in topic_scores[:10] if score > 0]
            
        except Exception as e:
            logger.error(f"Failed to extract topics: {e}")
            return []
    
    def _calculate_content_complexity(self, content: str) -> float:
        """Calculate content complexity score"""
        sentences = re.split(r'[.!?]+', content)
        
        if not sentences:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()])
        
        # Vocabulary diversity
        words = content.lower().split()
        unique_words = len(set(words))
        vocab_diversity = unique_words / len(words) if words else 0
        
        # Complex word ratio
        complex_words = sum(1 for word in words if len(word) > 6)
        complex_ratio = complex_words / len(words) if words else 0
        
        # Normalize and combine scores
        length_complexity = min(1.0, avg_sentence_length / 25)  # Normalize to 25 words
        vocab_complexity = min(1.0, vocab_diversity * 2)  # Boost diversity score
        word_complexity = min(1.0, complex_ratio * 3)  # Boost complex words
        
        return (length_complexity + vocab_complexity + word_complexity) / 3
    
    async def _generate_executive_summary(self, content: str, user_profile: UserWritingProfile) -> str:
        """Generate 30-second executive summary"""
        try:
            # Use extractive summarization for very short summary
            max_length = 50 if user_profile.preferred_length == "concise" else 80
            
            # Use LLM for better quality (placeholder for actual implementation)
            prompt = self._build_summary_prompt(content, user_profile, "executive")
            
            # For now, use simple extractive method
            sentences = re.split(r'[.!?]+', content)
            if len(sentences) <= 2:
                return content[:200] + "..." if len(content) > 200 else content
            
            # Select most important sentences
            important_sentences = sentences[:2]  # Simple heuristic
            
            summary = ". ".join(important_sentences).strip()
            
            # Apply user style
            summary = self._apply_user_style(summary, user_profile, "executive")
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate executive summary: {e}")
            return "Unable to generate executive summary."
    
    async def _generate_detailed_summary(self, content: str, user_profile: UserWritingProfile, content_analysis: Dict) -> str:
        """Generate 2-3 minute detailed summary"""
        try:
            # Determine target length based on user preference
            target_length = {
                "concise": 150,
                "moderate": 300,
                "detailed": 500
            }.get(user_profile.preferred_length, 300)
            
            # Split content into sections
            chunks = self.text_splitter.split_text(content)
            
            # Summarize each chunk
            chunk_summaries = []
            for chunk in chunks[:5]:  # Limit to 5 chunks
                chunk_summary = await self._summarize_chunk(chunk, user_profile)
                chunk_summaries.append(chunk_summary)
            
            # Combine and refine
            combined_summary = " ".join(chunk_summaries)
            
            # Apply user style
            styled_summary = self._apply_user_style(combined_summary, user_profile, "detailed")
            
            # Ensure target length
            if len(styled_summary.split()) > target_length:
                styled_summary = self._truncate_to_length(styled_summary, target_length)
            
            return styled_summary
            
        except Exception as e:
            logger.error(f"Failed to generate detailed summary: {e}")
            return "Unable to generate detailed summary."
    
    async def _summarize_chunk(self, chunk: str, user_profile: UserWritingProfile) -> str:
        """Summarize a single chunk of text"""
        try:
            # Simple extractive summarization for now
            sentences = re.split(r'[.!?]+', chunk)
            
            if len(sentences) <= 1:
                return chunk
            
            # Select key sentences (simple heuristic)
            key_sentences = sentences[:max(1, len(sentences) // 3)]
            return ". ".join(key_sentences).strip()
            
        except Exception as e:
            logger.error(f"Failed to summarize chunk: {e}")
            return chunk[:200] + "..." if len(chunk) > 200 else chunk
    
    def _build_summary_prompt(self, content: str, user_profile: UserWritingProfile, summary_type: str) -> str:
        """Build prompt for LLM summarization"""
        style_description = f"""
        User preferences:
        - Length: {user_profile.preferred_length}
        - Tone: {user_profile.tone}
        - Structure: {user_profile.structure_preference}
        - Complexity: {user_profile.sentence_complexity}
        - Examples: {"Include examples" if user_profile.example_preference else "Avoid examples"}
        - Action focus: {"High" if user_profile.action_orientation > 0.7 else "Medium" if user_profile.action_orientation > 0.4 else "Low"}
        """
        
        if summary_type == "executive":
            return f"""Create a very brief executive summary (30-50 words) of the following content.
            
            {style_description}
            
            Content: {content[:1000]}"""
        else:
            return f"""Create a detailed summary (200-400 words) of the following content.
            
            {style_description}
            
            Content: {content}"""
    
    def _apply_user_style(self, summary: str, user_profile: UserWritingProfile, summary_type: str) -> str:
        """Apply user's writing style to the summary"""
        styled_summary = summary
        
        # Apply structure preference
        if user_profile.structure_preference == "bullet_points" and summary_type == "detailed":
            styled_summary = self._convert_to_bullet_points(styled_summary)
        
        # Apply emphasis style
        if user_profile.emphasis_style == "bold":
            styled_summary = self._add_bold_emphasis(styled_summary)
        elif user_profile.emphasis_style == "headers" and summary_type == "detailed":
            styled_summary = self._add_headers(styled_summary)
        
        # Adjust sentence complexity
        if user_profile.sentence_complexity < 0.3:
            styled_summary = self._simplify_sentences(styled_summary)
        elif user_profile.sentence_complexity > 0.7:
            styled_summary = self._complexify_sentences(styled_summary)
        
        # Add questions if user prefers them
        if user_profile.question_style and summary_type == "detailed":
            styled_summary = self._add_rhetorical_questions(styled_summary)
        
        return styled_summary
    
    def _convert_to_bullet_points(self, text: str) -> str:
        """Convert text to bullet point format"""
        sentences = re.split(r'[.!?]+', text)
        bullet_points = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                bullet_points.append(f"• {sentence}")
        
        return "\n".join(bullet_points)
    
    def _add_bold_emphasis(self, text: str) -> str:
        """Add bold emphasis to key terms"""
        # Simple heuristic: bold important words
        important_words = ['key', 'important', 'significant', 'critical', 'main', 'primary']
        
        for word in important_words:
            text = re.sub(f'\\b{word}\\b', f'**{word}**', text, flags=re.IGNORECASE)
        
        return text
    
    def _add_headers(self, text: str) -> str:
        """Add section headers to text"""
        # Simple approach: add a header at the beginning
        return f"## Summary\n\n{text}"
    
    def _simplify_sentences(self, text: str) -> str:
        """Simplify sentence structure"""
        # Simple approach: split long sentences
        sentences = re.split(r'[.!?]+', text)
        simplified = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence.split()) > 20:  # Long sentence
                # Split on commas or conjunctions
                parts = re.split(r',\s*(?:and|but|or)\s*', sentence)
                simplified.extend(parts)
            else:
                simplified.append(sentence)
        
        return ". ".join(simplified).strip()
    
    def _complexify_sentences(self, text: str) -> str:
        """Make sentences more complex (placeholder)"""
        # For now, just return original text
        # In production, this would use more sophisticated NLP
        return text
    
    def _add_rhetorical_questions(self, text: str) -> str:
        """Add rhetorical questions to engage reader"""
        # Simple approach: add a question at the end
        questions = [
            "What does this mean for the future?",
            "How might this change your approach?",
            "What are the implications?",
            "Where do we go from here?"
        ]
        
        question = np.random.choice(questions)
        return f"{text}\n\n{question}"
    
    def _truncate_to_length(self, text: str, target_words: int) -> str:
        """Truncate text to target word count"""
        words = text.split()
        if len(words) <= target_words:
            return text
        
        truncated = " ".join(words[:target_words])
        return truncated + "..."
    
    async def _extract_key_insights(self, content: str, user_profile: UserWritingProfile) -> List[str]:
        """Extract key insights from content"""
        insights = []
        
        # Simple approach: look for conclusion indicators
        insight_patterns = [
            r'(?:key|main|important)\s+(?:insight|finding|point|takeaway)(?:s)?[:\s]+([^.!?]+)',
            r'(?:in conclusion|to summarize|therefore)[:\s]+([^.!?]+)',
            r'(?:this shows|this indicates|this suggests)[:\s]+([^.!?]+)'
        ]
        
        for pattern in insight_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            insights.extend(matches)
        
        # If no explicit insights found, extract from important sentences
        if not insights:
            sentences = re.split(r'[.!?]+', content)
            # Simple heuristic: sentences with important keywords
            important_keywords = ['important', 'significant', 'key', 'critical', 'main', 'crucial']
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in important_keywords):
                    insights.append(sentence.strip())
        
        return insights[:5]  # Return top 5 insights
    
    async def _extract_action_items(self, content: str, user_profile: UserWritingProfile) -> List[str]:
        """Extract actionable items from content"""
        action_items = []
        
        # Look for action-oriented language
        action_patterns = [
            r'(?:should|must|need to|will|plan to)\s+([^.!?]+)',
            r'(?:action|step|task|todo)[:\s]+([^.!?]+)',
            r'(?:implement|execute|complete|finish)\s+([^.!?]+)'
        ]
        
        for pattern in action_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            action_items.extend(matches)
        
        # Clean and format action items
        cleaned_items = []
        for item in action_items:
            item = item.strip()
            if len(item) > 10 and len(item) < 100:  # Reasonable length
                if not item.startswith(('the', 'a', 'an')):
                    item = item.capitalize()
                cleaned_items.append(item)
        
        return cleaned_items[:5]  # Return top 5 action items
    
    async def _generate_questions(self, content: str, user_profile: UserWritingProfile) -> List[str]:
        """Generate thought-provoking questions based on content"""
        questions = []
        
        # Extract main topics for question generation
        topics = await self._extract_main_topics(content)
        
        # Question templates
        question_templates = [
            "What are the implications of {topic}?",
            "How might {topic} evolve in the future?",
            "What challenges does {topic} present?",
            "How does {topic} relate to other concepts?",
            "What assumptions about {topic} should be questioned?"
        ]
        
        for topic in topics[:3]:  # Use top 3 topics
            template = np.random.choice(question_templates)
            question = template.format(topic=topic)
            questions.append(question)
        
        return questions
    
    async def _extract_related_concepts(self, content: str) -> List[str]:
        """Extract concepts related to the main content"""
        # Use TF-IDF to find important terms
        try:
            vectorizer = TfidfVectorizer(
                max_features=15,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            tfidf_matrix = vectorizer.fit_transform([content])
            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]
            
            # Get concepts with significant scores
            concept_scores = list(zip(feature_names, scores))
            concept_scores.sort(key=lambda x: x[1], reverse=True)
            
            return [concept for concept, score in concept_scores[:10] if score > 0.1]
            
        except Exception as e:
            logger.error(f"Failed to extract related concepts: {e}")
            return []
    
    def _calculate_confidence_score(self, content: str, content_analysis: Dict) -> float:
        """Calculate confidence score for the summary"""
        # Base confidence on content characteristics
        word_count = content_analysis.get('word_count', 0)
        complexity = content_analysis.get('complexity_score', 0)
        
        # Higher confidence for longer, well-structured content
        length_score = min(1.0, word_count / 500)  # Normalize to 500 words
        structure_score = min(1.0, content_analysis.get('paragraph_count', 1) / 5)  # Normalize to 5 paragraphs
        
        return (length_score + structure_score + complexity) / 3
    
    def _calculate_style_match_score(self, user_profile: UserWritingProfile) -> float:
        """Calculate how well the summary matches user's style"""
        # Simple heuristic based on profile completeness
        profile_completeness = 0.8  # Assume good match for now
        
        # In production, this would compare generated summary against user's style
        return profile_completeness
    
    # Database interaction methods (implement based on your schema)
    async def _store_user_profile(self, profile: UserWritingProfile) -> None:
        """Store user writing profile in database"""
        # Implementation depends on your database schema
        pass
    
    async def _get_user_profile(self, user_id: str) -> Optional[UserWritingProfile]:
        """Retrieve user writing profile from database"""
        # Implementation depends on your database schema
        return None