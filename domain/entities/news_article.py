"""
News Article Entity - Domain Model

This module defines the NewsArticle entity following Domain Driven Design principles.
The NewsArticle represents the core business concept of a news article with all its
attributes and behaviors.

Classes:
    NewsArticle: Main entity representing a news article
    ArticleType: Enumeration of article types
    SentimentType: Enumeration of sentiment types
    BiasClassification: Enumeration of bias classifications
    FactCheckClassification: Enumeration of fact-check classifications

Author: Claude AI Assistant
Date: 2025-01-18
Version: 1.0
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum


class ArticleType(Enum):
    """Enumeration of article types."""
    FINANCIAL = "financial"
    CRYPTO = "crypto"
    TECHNOLOGY = "technology"
    GENERAL = "general"
    POLITICS = "politics"
    SPORTS = "sports"
    ENTERTAINMENT = "entertainment"


class SentimentType(Enum):
    """Enumeration of sentiment types."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    BULLISH = "bullish"
    BEARISH = "bearish"


class BiasClassification(Enum):
    """Enumeration of bias classifications."""
    LEFT = "left"
    RIGHT = "right"
    CENTER = "center"
    UNKNOWN = "unknown"


class FactCheckClassification(Enum):
    """Enumeration of fact-check classifications."""
    VERIFIED = "verified"
    LIKELY_ACCURATE = "likely_accurate"
    UNVERIFIED = "unverified"
    QUESTIONABLE = "questionable"
    UNRELIABLE = "unreliable"


@dataclass
class NewsArticle:
    """
    News Article Entity representing a single news article.
    
    This entity contains all the information about a news article including
    its content, metadata, and classification results.
    """
    
    # Core attributes
    url: str
    title: str
    source: str
    content: Optional[str] = None
    summary: Optional[str] = None
    published_date: Optional[datetime] = None
    
    # Classification results
    sentiment: Optional[SentimentType] = None
    confidence: Optional[float] = None
    article_type: Optional[ArticleType] = None
    bias_classification: Optional[BiasClassification] = None
    credibility_score: Optional[float] = None
    
    # Financial analysis
    financial_relevance: bool = False
    crypto_relevance: bool = False
    companies_mentioned: List[str] = field(default_factory=list)
    crypto_mentions: List[str] = field(default_factory=list)
    
    # Fact-checking
    fact_check_score: Optional[float] = None
    fact_check_classification: Optional[FactCheckClassification] = None
    fact_checks_found: int = 0
    
    # Technical metadata
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    analyzed_date: Optional[datetime] = None
    extraction_success: bool = True
    
    # Additional metadata
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.published_date is None:
            self.published_date = datetime.now()
        if self.analyzed_date is None:
            self.analyzed_date = datetime.now()
    
    def is_financial_news(self) -> bool:
        """
        Check if this article is financial news.
        
        Returns:
            bool: True if this is financial news
        """
        return (self.financial_relevance or 
                self.article_type == ArticleType.FINANCIAL or
                self.article_type == ArticleType.CRYPTO)
    
    def is_crypto_news(self) -> bool:
        """
        Check if this article is crypto news.
        
        Returns:
            bool: True if this is crypto news
        """
        return (self.crypto_relevance or 
                self.article_type == ArticleType.CRYPTO or
                len(self.crypto_mentions) > 0)
    
    def is_high_confidence(self) -> bool:
        """
        Check if this article has high confidence score.
        
        Returns:
            bool: True if this article has high confidence (>= 0.7)
        """
        return self.confidence is not None and self.confidence >= 0.7
    
    def get_quality_score(self) -> float:
        """
        Calculate overall quality score for this article.
        
        Returns:
            float: Quality score between 0 and 1
        """
        score = 0.0
        factors = 0
        
        # Confidence factor
        if self.confidence is not None:
            score += self.confidence
            factors += 1
        
        # Credibility factor
        if self.credibility_score is not None:
            score += self.credibility_score / 100.0
            factors += 1
        
        # Fact-check factor
        if self.fact_check_score is not None:
            score += self.fact_check_score / 100.0
            factors += 1
        
        # Content quality factor
        if self.content and len(self.content.strip()) > 100:
            score += 0.8
            factors += 1
        elif self.content and len(self.content.strip()) > 50:
            score += 0.6
            factors += 1
        else:
            score += 0.3
            factors += 1
        
        # Extraction success factor
        if self.extraction_success:
            score += 0.9
            factors += 1
        else:
            score += 0.2
            factors += 1
        
        return score / factors if factors > 0 else 0.5
    
    def get_sentiment_label(self) -> str:
        """
        Get human-readable sentiment label.
        
        Returns:
            str: Sentiment label
        """
        if self.sentiment:
            return self.sentiment.value
        return "neutral"
    
    def get_bias_label(self) -> str:
        """
        Get human-readable bias label.
        
        Returns:
            str: Bias label
        """
        if self.bias_classification:
            return self.bias_classification.value
        return "unknown"
    
    def get_type_label(self) -> str:
        """
        Get human-readable article type label.
        
        Returns:
            str: Article type label
        """
        if self.article_type:
            return self.article_type.value
        return "general"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert article to dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation
        """
        return {
            'url': self.url,
            'title': self.title,
            'source': self.source,
            'content': self.content,
            'summary': self.summary,
            'published_date': self.published_date.isoformat() if self.published_date else None,
            'sentiment': self.sentiment.value if self.sentiment else None,
            'confidence': self.confidence,
            'article_type': self.article_type.value if self.article_type else None,
            'bias_classification': self.bias_classification.value if self.bias_classification else None,
            'credibility_score': self.credibility_score,
            'financial_relevance': self.financial_relevance,
            'crypto_relevance': self.crypto_relevance,
            'companies_mentioned': self.companies_mentioned,
            'crypto_mentions': self.crypto_mentions,
            'fact_check_score': self.fact_check_score,
            'fact_check_classification': self.fact_check_classification.value if self.fact_check_classification else None,
            'fact_checks_found': self.fact_checks_found,
            'processing_time': self.processing_time,
            'model_version': self.model_version,
            'analyzed_date': self.analyzed_date.isoformat() if self.analyzed_date else None,
            'extraction_success': self.extraction_success,
            'tags': self.tags,
            'metadata': self.metadata,
            'quality_score': self.get_quality_score()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """
        Create article from dictionary.
        
        Args:
            data (Dict[str, Any]): Dictionary data
            
        Returns:
            NewsArticle: Created article instance
        """
        # Parse dates
        published_date = None
        if data.get('published_date'):
            try:
                published_date = datetime.fromisoformat(data['published_date'])
            except:
                published_date = None
        
        analyzed_date = None
        if data.get('analyzed_date'):
            try:
                analyzed_date = datetime.fromisoformat(data['analyzed_date'])
            except:
                analyzed_date = None
        
        # Parse enums
        sentiment = None
        if data.get('sentiment'):
            try:
                sentiment = SentimentType(data['sentiment'])
            except:
                sentiment = None
        
        article_type = None
        if data.get('article_type'):
            try:
                article_type = ArticleType(data['article_type'])
            except:
                article_type = None
        
        bias_classification = None
        if data.get('bias_classification'):
            try:
                bias_classification = BiasClassification(data['bias_classification'])
            except:
                bias_classification = None
        
        fact_check_classification = None
        if data.get('fact_check_classification'):
            try:
                fact_check_classification = FactCheckClassification(data['fact_check_classification'])
            except:
                fact_check_classification = None
        
        return cls(
            url=data['url'],
            title=data['title'],
            source=data['source'],
            content=data.get('content'),
            summary=data.get('summary'),
            published_date=published_date,
            sentiment=sentiment,
            confidence=data.get('confidence'),
            article_type=article_type,
            bias_classification=bias_classification,
            credibility_score=data.get('credibility_score'),
            financial_relevance=data.get('financial_relevance', False),
            crypto_relevance=data.get('crypto_relevance', False),
            companies_mentioned=data.get('companies_mentioned', []),
            crypto_mentions=data.get('crypto_mentions', []),
            fact_check_score=data.get('fact_check_score'),
            fact_check_classification=fact_check_classification,
            fact_checks_found=data.get('fact_checks_found', 0),
            processing_time=data.get('processing_time'),
            model_version=data.get('model_version'),
            analyzed_date=analyzed_date,
            extraction_success=data.get('extraction_success', True),
            tags=data.get('tags', []),
            metadata=data.get('metadata', {})
        )
    
    def __str__(self) -> str:
        """String representation of the article."""
        return f"NewsArticle(title='{self.title[:50]}...', source='{self.source}', type='{self.get_type_label()}')"
    
    def __repr__(self) -> str:
        """Detailed string representation of the article."""
        return f"NewsArticle(url='{self.url}', title='{self.title}', source='{self.source}', sentiment='{self.get_sentiment_label()}')" 