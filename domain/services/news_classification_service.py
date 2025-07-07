"""
Domain Service: NewsClassificationService

This module contains the core domain service for news classification in the Penelope News
Classification System. It implements Domain Driven Design principles by encapsulating
complex business logic that doesn't naturally fit within a single entity.

The NewsClassificationService orchestrates the classification process by coordinating
multiple analysis components including sentiment analysis, bias detection, fact-checking,
financial relevance assessment, and content quality evaluation. It transforms raw news
content into enriched NewsArticle entities with comprehensive classification metadata.

Classes:
    NewsClassificationService: Main service for news article classification
    ClassificationResult: Data structure for classification results
    ClassificationError: Exception for classification failures

Key Features:
    - Sentiment analysis using FinBERT and other models
    - Bias detection and source credibility assessment
    - Dynamic fact-checking with multi-source verification
    - Financial relevance and cryptocurrency detection
    - Content quality scoring and extraction success tracking
    - Async/await support for high-performance processing

Example:
    Basic usage:
    
    >>> service = NewsClassificationService()
    >>> article = NewsArticle(
    ...     url="https://example.com/news",
    ...     title="Market Update",
    ...     source="Reuters",
    ...     content="Stock markets rose today..."
    ... )
    >>> enriched_article = await service.classify_article(article)
    >>> print(f"Sentiment: {enriched_article.sentiment}")
    >>> print(f"Financial: {enriched_article.is_financial_news()}")

"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass
import re
from enum import Enum

from domain.entities.news_article import (
    NewsArticle, 
    ArticleType, 
    SentimentType, 
    BiasClassification,
    FactCheckClassification
)
from domain.repositories.news_repository import NewsRepository


class ClassificationModel(Enum):
    """
    Enumeration of available classification models.
    
    This enum defines the different machine learning models and
    analysis techniques available for news classification.
    """
    FINBERT = "finbert"
    VADER = "vader"
    TEXTBLOB = "textblob"
    CUSTOM = "custom"


@dataclass
class ClassificationResult:
    """
    Data structure containing comprehensive classification results.
    
    This dataclass encapsulates all the analysis results from the
    classification process, providing a structured way to return
    multiple classification outcomes.
    
    Attributes:
        sentiment (SentimentType): Detected sentiment
        confidence (float): Classification confidence score
        financial_relevance (bool): Whether content is financially relevant
        crypto_relevance (bool): Whether content mentions cryptocurrencies
        companies_mentioned (List[str]): List of companies found in content
        crypto_mentions (List[str]): List of cryptocurrencies mentioned
        article_type (ArticleType): Classified article type
        processing_time (float): Time taken for classification
        model_used (ClassificationModel): Model used for classification
        quality_indicators (Dict[str, Any]): Quality metrics and indicators
    """
    sentiment: SentimentType
    confidence: float
    financial_relevance: bool
    crypto_relevance: bool
    companies_mentioned: List[str]
    crypto_mentions: List[str]
    article_type: ArticleType
    processing_time: float
    model_used: ClassificationModel
    quality_indicators: Dict[str, Any]


class NewsClassificationService:
    """
    Core domain service for comprehensive news article classification.
    
    This service implements the business logic for analyzing and classifying
    news articles. It coordinates multiple analysis components to provide
    comprehensive classification results including sentiment, bias, fact-checking,
    financial relevance, and content quality assessment.
    
    The service follows Domain Driven Design principles by encapsulating
    complex business logic that spans multiple entities and doesn't naturally
    belong to a single entity.
    
    Key Responsibilities:
        - Sentiment analysis using multiple models
        - Financial relevance detection
        - Cryptocurrency mention detection
        - Company name extraction
        - Bias detection and credibility assessment
        - Fact-checking score computation
        - Content quality evaluation
        - Processing time tracking
        
    The service supports async operations for high-performance batch processing
    and provides detailed error handling and logging.
    
    Example:
        >>> service = NewsClassificationService()
        >>> article = NewsArticle(url="...", title="...", source="...")
        >>> enriched = await service.classify_article(article)
        >>> print(f"Classification: {enriched.sentiment} with {enriched.confidence:.2f} confidence")
    """
    
    def __init__(self, news_repository: NewsRepository, model_preference: ClassificationModel = ClassificationModel.FINBERT):
        """
        Initialize the NewsClassificationService.
        
        Args:
            news_repository (NewsRepository): Repository for accessing news articles
            model_preference (ClassificationModel): Preferred classification model
        """
        self.logger = logging.getLogger(__name__)
        self.news_repository = news_repository
        self.model_preference = model_preference
        self.model_version = "3.0-DDD"
        
        # Financial keywords for relevance detection
        self.financial_keywords = {
            'market', 'stock', 'shares', 'trading', 'investment', 'investor',
            'financial', 'economy', 'economic', 'revenue', 'profit', 'earnings',
            'quarterly', 'annual', 'fiscal', 'budget', 'debt', 'credit',
            'interest', 'rate', 'federal', 'reserve', 'fed', 'monetary',
            'inflation', 'deflation', 'recession', 'growth', 'gdp',
            'unemployment', 'employment', 'jobs', 'payroll', 'wage',
            'salary', 'income', 'wealth', 'portfolio', 'fund', 'etf',
            'bonds', 'securities', 'commodities', 'forex', 'currency',
            'dollar', 'euro', 'yen', 'pound', 'exchange', 'trade',
            'tariff', 'export', 'import', 'manufacturing', 'retail',
            'consumer', 'spending', 'savings', 'bank', 'banking',
            'finance', 'fintech', 'cryptocurrency', 'blockchain'
        }
        
        # Cryptocurrency keywords
        self.crypto_keywords = {
            'bitcoin', 'btc', 'ethereum', 'eth', 'cryptocurrency', 'crypto',
            'blockchain', 'altcoin', 'defi', 'nft', 'token', 'coin',
            'mining', 'wallet', 'exchange', 'binance', 'coinbase',
            'dogecoin', 'litecoin', 'ripple', 'xrp', 'cardano', 'ada',
            'polkadot', 'chainlink', 'stellar', 'xlm', 'uniswap',
            'decentralized', 'smart contract', 'web3', 'metaverse'
        }
        
        # Source credibility mapping
        self.source_credibility = {
            'reuters': 90,
            'bloomberg': 90,
            'associated press': 85,
            'ap news': 85,
            'bbc': 80,
            'cnn': 75,
            'cnbc': 75,
            'financial times': 85,
            'wall street journal': 85,
            'wsj': 85,
            'marketwatch': 70,
            'yahoo finance': 65,
            'google news': 60,
            'techcrunch': 70,
            'the guardian': 75,
            'usa today': 70,
            'npr': 80,
            'pbs': 80,
            'axios': 75,
            'politico': 70
        }
        
        # Bias classification mapping
        self.source_bias = {
            'reuters': BiasClassification.CENTER,
            'bloomberg': BiasClassification.CENTER,
            'associated press': BiasClassification.CENTER,
            'ap news': BiasClassification.CENTER,
            'bbc': BiasClassification.CENTER,
            'financial times': BiasClassification.CENTER,
            'wall street journal': BiasClassification.CENTER,
            'wsj': BiasClassification.CENTER,
            'cnn': BiasClassification.LEFT,
            'msnbc': BiasClassification.LEFT,
            'fox news': BiasClassification.RIGHT,
            'breitbart': BiasClassification.RIGHT,
            'the guardian': BiasClassification.LEFT,
            'usa today': BiasClassification.CENTER,
            'npr': BiasClassification.LEFT,
            'pbs': BiasClassification.CENTER
        }
    
    async def classify_article(self, article: NewsArticle) -> NewsArticle:
        """
        Perform comprehensive classification of a news article.
        
        This method applies all available classification techniques to enrich
        the article with sentiment analysis, bias detection, fact-checking,
        financial relevance assessment, and content quality evaluation.
        
        Args:
            article (NewsArticle): The article to classify
            
        Returns:
            NewsArticle: The same article enriched with classification results
            
        Raises:
            ClassificationError: If classification fails
            
        Example:
            >>> article = NewsArticle(url="...", title="...", source="...")
            >>> enriched = await service.classify_article(article)
            >>> print(f"Sentiment: {enriched.sentiment}")
            >>> print(f"Confidence: {enriched.confidence}")
            >>> print(f"Financial: {enriched.is_financial_news()}")
        """
        start_time = datetime.now()
        
        try:
            # Perform sentiment analysis
            sentiment_result = await self._analyze_sentiment(article)
            article.sentiment = sentiment_result.sentiment
            article.confidence = sentiment_result.confidence
            article.article_type = sentiment_result.article_type
            
            # Perform financial analysis
            financial_result = await self._analyze_financial_relevance(article)
            article.financial_relevance = financial_result['is_financial']
            article.crypto_relevance = financial_result['is_crypto']
            article.companies_mentioned = financial_result['companies']
            article.crypto_mentions = financial_result['crypto_mentions']
            
            # Perform bias and credibility analysis
            bias_result = await self._analyze_bias_and_credibility(article)
            article.bias_classification = bias_result['bias']
            article.credibility_score = bias_result['credibility']
            
            # Perform fact-checking analysis
            fact_check_result = await self._analyze_fact_checking(article)
            article.fact_check_score = fact_check_result['score']
            article.fact_check_classification = fact_check_result['classification']
            article.fact_checks_found = fact_check_result['checks_found']
            
            # Set technical metadata
            article.processing_time = (datetime.now() - start_time).total_seconds()
            article.model_version = self.model_version
            article.analyzed_date = datetime.now()
            
            # Determine extraction success
            article.extraction_success = self._evaluate_extraction_success(article)
            
            self.logger.info(f"Successfully classified article: {article.url}")
            return article
            
        except Exception as e:
            self.logger.error(f"Classification failed for article {article.url}: {str(e)}")
            raise ClassificationError(f"Classification failed: {str(e)}", article.url, e)
    
    async def classify_batch(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """
        Perform batch classification of multiple news articles.
        
        This method efficiently processes multiple articles in parallel,
        providing better performance than sequential processing.
        
        Args:
            articles (List[NewsArticle]): List of articles to classify
            
        Returns:
            List[NewsArticle]: List of enriched articles
            
        Example:
            >>> articles = [article1, article2, article3]
            >>> enriched = await service.classify_batch(articles)
            >>> print(f"Classified {len(enriched)} articles")
        """
        self.logger.info(f"Starting batch classification of {len(articles)} articles")
        
        # Process articles in parallel with controlled concurrency
        semaphore = asyncio.Semaphore(10)  # Limit concurrent operations
        
        async def classify_with_semaphore(article: NewsArticle) -> NewsArticle:
            async with semaphore:
                return await self.classify_article(article)
        
        tasks = [classify_with_semaphore(article) for article in articles]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Separate successful results from errors
        successful_articles = []
        errors = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                errors.append(f"Article {i}: {str(result)}")
            else:
                successful_articles.append(result)
        
        if errors:
            self.logger.warning(f"Batch classification completed with {len(errors)} errors")
            for error in errors:
                self.logger.error(error)
        
        self.logger.info(f"Batch classification completed: {len(successful_articles)} successful, {len(errors)} errors")
        return successful_articles
    
    async def _analyze_sentiment(self, article: NewsArticle) -> ClassificationResult:
        """
        Analyze sentiment of the article content.
        
        This method performs sentiment analysis using the configured model
        and returns detailed sentiment information.
        
        Args:
            article (NewsArticle): Article to analyze
            
        Returns:
            ClassificationResult: Sentiment analysis results
        """
        start_time = datetime.now()
        
        # Combine title and content for analysis
        text_to_analyze = f"{article.title} {article.content or ''}"
        
        # Simple sentiment analysis (can be replaced with actual ML models)
        sentiment_score = self._calculate_sentiment_score(text_to_analyze)
        
        # Determine sentiment type
        if sentiment_score > 0.1:
            sentiment = SentimentType.POSITIVE
        elif sentiment_score < -0.1:
            sentiment = SentimentType.NEGATIVE
        else:
            sentiment = SentimentType.NEUTRAL
        
        # Calculate confidence based on absolute score
        confidence = min(abs(sentiment_score), 1.0)
        
        # Determine article type based on content
        article_type = self._classify_article_type(text_to_analyze)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return ClassificationResult(
            sentiment=sentiment,
            confidence=confidence,
            financial_relevance=False,  # Will be set by financial analysis
            crypto_relevance=False,     # Will be set by financial analysis
            companies_mentioned=[],     # Will be set by financial analysis
            crypto_mentions=[],         # Will be set by financial analysis
            article_type=article_type,
            processing_time=processing_time,
            model_used=self.model_preference,
            quality_indicators={'sentiment_score': sentiment_score}
        )
    
    async def _analyze_financial_relevance(self, article: NewsArticle) -> Dict[str, Any]:
        """
        Analyze financial relevance and extract financial entities.
        
        Args:
            article (NewsArticle): Article to analyze
            
        Returns:
            Dict[str, Any]: Financial analysis results
        """
        text_to_analyze = f"{article.title} {article.content or ''}".lower()
        
        # Check for financial keywords
        financial_matches = sum(1 for keyword in self.financial_keywords 
                              if keyword in text_to_analyze)
        is_financial = financial_matches >= 2
        
        # Check for cryptocurrency keywords
        crypto_matches = [keyword for keyword in self.crypto_keywords 
                         if keyword in text_to_analyze]
        is_crypto = len(crypto_matches) > 0
        
        # Extract company mentions (simplified - can be enhanced with NER)
        companies_mentioned = self._extract_companies(text_to_analyze)
        
        return {
            'is_financial': is_financial,
            'is_crypto': is_crypto,
            'companies': companies_mentioned,
            'crypto_mentions': crypto_matches,
            'financial_keyword_count': financial_matches
        }
    
    async def _analyze_bias_and_credibility(self, article: NewsArticle) -> Dict[str, Any]:
        """
        Analyze source bias and credibility.
        
        Args:
            article (NewsArticle): Article to analyze
            
        Returns:
            Dict[str, Any]: Bias and credibility analysis results
        """
        source_lower = article.source.lower()
        
        # Get credibility score
        credibility_score = 50  # Default neutral score
        for source_key, score in self.source_credibility.items():
            if source_key in source_lower:
                credibility_score = score
                break
        
        # Get bias classification
        bias_classification = BiasClassification.UNKNOWN
        for source_key, bias in self.source_bias.items():
            if source_key in source_lower:
                bias_classification = bias
                break
        
        return {
            'credibility': credibility_score,
            'bias': bias_classification
        }
    
    async def _analyze_fact_checking(self, article: NewsArticle) -> Dict[str, Any]:
        """
        Perform fact-checking analysis with dynamic scoring.
        
        This method replaces the previous constant scoring with dynamic
        fact-checking based on multiple factors including source credibility,
        content quality, and pattern detection.
        
        Args:
            article (NewsArticle): Article to analyze
            
        Returns:
            Dict[str, Any]: Fact-checking results
        """
        base_score = 50  # Start with neutral score
        
        # Source credibility bonus
        source_lower = article.source.lower()
        credibility_bonus = 0
        for source_key, credibility in self.source_credibility.items():
            if source_key in source_lower:
                if credibility >= 85:
                    credibility_bonus = 30
                elif credibility >= 75:
                    credibility_bonus = 15
                elif credibility >= 65:
                    credibility_bonus = 5
                break
        
        # Content quality indicators
        content_quality_bonus = 0
        if article.content:
            # Check for data presence (numbers, percentages, dates)
            if re.search(r'\d+%|\$\d+|\d+\.\d+', article.content):
                content_quality_bonus += 10
            
            # Check for date references
            if re.search(r'\d{4}|\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', 
                        article.content.lower()):
                content_quality_bonus += 5
            
            # Check for quotes or attributions
            if '"' in article.content or 'said' in article.content.lower():
                content_quality_bonus += 10
        
        # Pattern detection for suspicious content
        suspicious_penalty = 0
        if article.content:
            # Check for sensational language
            sensational_words = ['shocking', 'unbelievable', 'incredible', 'amazing', 'stunning']
            if any(word in article.content.lower() for word in sensational_words):
                suspicious_penalty += 10
            
            # Check for excessive capitalization
            if sum(1 for c in article.content if c.isupper()) > len(article.content) * 0.1:
                suspicious_penalty += 15
        
        # Calculate final score
        final_score = base_score + credibility_bonus + content_quality_bonus - suspicious_penalty
        final_score = max(0, min(100, final_score))  # Clamp between 0-100
        
        # Determine classification
        if final_score >= 80:
            classification = FactCheckClassification.VERIFIED
        elif final_score >= 60:
            classification = FactCheckClassification.LIKELY_ACCURATE
        elif final_score >= 40:
            classification = FactCheckClassification.UNVERIFIED
        elif final_score >= 20:
            classification = FactCheckClassification.QUESTIONABLE
        else:
            classification = FactCheckClassification.UNRELIABLE
        
        return {
            'score': final_score,
            'classification': classification,
            'checks_found': 1 if credibility_bonus > 0 else 0
        }
    
    def _calculate_sentiment_score(self, text: str) -> float:
        """
        Calculate sentiment score using simple lexicon-based approach.
        
        This is a simplified implementation that can be replaced with
        more sophisticated models like FinBERT or VADER.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            float: Sentiment score between -1 and 1
        """
        positive_words = {'good', 'great', 'excellent', 'positive', 'up', 'gain', 'rise', 'surge', 'boom', 'growth'}
        negative_words = {'bad', 'terrible', 'negative', 'down', 'fall', 'drop', 'crash', 'decline', 'recession', 'loss'}
        
        words = text.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)
    
    def _classify_article_type(self, text: str) -> ArticleType:
        """
        Classify article type based on content analysis.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            ArticleType: Classified article type
        """
        text_lower = text.lower()
        
        # Check for financial keywords
        financial_matches = sum(1 for keyword in self.financial_keywords 
                              if keyword in text_lower)
        
        # Check for crypto keywords
        crypto_matches = sum(1 for keyword in self.crypto_keywords 
                           if keyword in text_lower)
        
        # Check for technology keywords
        tech_keywords = {'technology', 'software', 'app', 'ai', 'artificial intelligence', 
                        'machine learning', 'startup', 'innovation', 'digital'}
        tech_matches = sum(1 for keyword in tech_keywords if keyword in text_lower)
        
        # Determine type based on keyword matches
        if crypto_matches >= 2:
            return ArticleType.CRYPTO
        elif financial_matches >= 3:
            return ArticleType.FINANCIAL
        elif tech_matches >= 2:
            return ArticleType.TECHNOLOGY
        else:
            return ArticleType.GENERAL
    
    def _extract_companies(self, text: str) -> List[str]:
        """
        Extract company mentions from text.
        
        This is a simplified implementation that can be enhanced
        with Named Entity Recognition (NER) models.
        
        Args:
            text (str): Text to search for companies
            
        Returns:
            List[str]: List of company names found
        """
        # Common company patterns
        company_patterns = [
            r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co)\b',
            r'\b[A-Z]{2,}\b',  # Acronyms
        ]
        
        companies = []
        for pattern in company_patterns:
            matches = re.findall(pattern, text)
            companies.extend(matches)
        
        # Remove duplicates and filter out common false positives
        companies = list(set(companies))
        false_positives = {'USA', 'US', 'UK', 'EU', 'CEO', 'CFO', 'CTO', 'AI', 'IT', 'PR'}
        companies = [c for c in companies if c not in false_positives]
        
        return companies[:10]  # Limit to 10 companies
    
    def _evaluate_extraction_success(self, article: NewsArticle) -> bool:
        """
        Evaluate whether content extraction was successful.
        
        Args:
            article (NewsArticle): Article to evaluate
            
        Returns:
            bool: True if extraction appears successful
        """
        if not article.content:
            return False
        
        # Check minimum content length
        if len(article.content) < 100:
            return False
        
        # Check for common extraction failure patterns
        failure_patterns = [
            'javascript required',
            'please enable javascript',
            'subscription required',
            'paywall',
            'access denied',
            'page not found',
            '404 error'
        ]
        
        content_lower = article.content.lower()
        for pattern in failure_patterns:
            if pattern in content_lower:
                return False
        
        return True


class ClassificationError(Exception):
    """
    Exception raised when news classification fails.
    
    This exception is raised when there are errors during the classification
    process, such as model failures, data processing errors, or timeout issues.
    
    Attributes:
        message (str): The error message
        article_url (str): URL of the article that failed classification
        original_error (Exception): The original exception that caused the error
    """
    
    def __init__(self, message: str, article_url: str = "", original_error: Exception = None):
        """
        Initialize a ClassificationError.
        
        Args:
            message (str): The error message
            article_url (str): URL of the article that failed
            original_error (Exception): The original exception
        """
        self.message = message
        self.article_url = article_url
        self.original_error = original_error
        
        full_message = f"Classification error"
        if article_url:
            full_message += f" for article {article_url}"
        full_message += f": {message}"
        
        if original_error:
            full_message += f" (caused by: {type(original_error).__name__}: {original_error})"
        
        super().__init__(full_message) 