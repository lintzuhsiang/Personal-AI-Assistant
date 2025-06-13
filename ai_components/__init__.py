"""
Enhanced AI Components Package

This package contains advanced ML components for the Personal AI Assistant:
- User behavior analysis and personalization
- Creative discovery and insight generation
- Adaptive summarization with style learning
- Serendipitous content recommendation
"""

from .user_behavior_analyzer import (
    UserBehaviorAnalyzer,
    InteractionEvent,
    UserProfile
)

from .creative_discovery_engine import (
    CreativeDiscoveryEngine,
    ConceptGraph,
    AnalogyDetector,
    ConceptConnection,
    InsightGeneration
)

from .adaptive_summarizer import (
    AdaptiveSummarizer,
    WritingStyleAnalyzer,
    SummaryLevel,
    WritingStyle,
    UserWritingProfile,
    SummaryOutput
)

from .serendipity_news_engine import (
    SerendipityNewsEngine,
    ContentAggregator,
    SurpriseCalculator,
    ContentSource,
    InterestType,
    ContentItem,
    UserInterestGraph,
    NewsletterItem,
    Newsletter
)

__version__ = "1.0.0"
__author__ = "Personal AI Assistant Team"

# Package-level configuration
DEFAULT_CONFIG = {
    "behavior_analysis": {
        "min_interactions": 5,
        "session_gap_minutes": 30
    },
    "creative_discovery": {
        "max_insights": 5,
        "cross_domain_threshold": 0.7
    },
    "adaptive_summarization": {
        "min_quality_score": 0.6,
        "max_content_length": 10000
    },
    "serendipity_news": {
        "exploration_rate": 0.3,
        "max_newsletter_items": 8,
        "content_fetch_hours": 48
    }
}

def get_config(component: str = None):
    """Get configuration for specific component or all components"""
    if component:
        return DEFAULT_CONFIG.get(component, {})
    return DEFAULT_CONFIG

def validate_dependencies():
    """Validate that all required dependencies are installed"""
    required_packages = [
        'scikit-learn',
        'transformers',
        'torch',
        'networkx',
        'feedparser',
        'beautifulsoup4',
        'numpy',
        'pandas'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        raise ImportError(
            f"Missing required packages: {', '.join(missing_packages)}. "
            f"Please install them using: pip install {' '.join(missing_packages)}"
        )
    
    return True

# Initialize package
try:
    validate_dependencies()
    print("✅ AI Components package loaded successfully")
except ImportError as e:
    print(f"⚠️  Warning: {e}")

__all__ = [
    # User Behavior Analysis
    'UserBehaviorAnalyzer',
    'InteractionEvent',
    'UserProfile',
    
    # Creative Discovery
    'CreativeDiscoveryEngine',
    'ConceptGraph',
    'AnalogyDetector',
    'ConceptConnection',
    'InsightGeneration',
    
    # Adaptive Summarization
    'AdaptiveSummarizer',
    'WritingStyleAnalyzer',
    'SummaryLevel',
    'WritingStyle',
    'UserWritingProfile',
    'SummaryOutput',
    
    # Serendipitous Newsletter
    'SerendipityNewsEngine',
    'ContentAggregator',
    'SurpriseCalculator',
    'ContentSource',
    'InterestType',
    'ContentItem',
    'UserInterestGraph',
    'NewsletterItem',
    'Newsletter',
    
    # Utilities
    'get_config',
    'validate_dependencies'
]