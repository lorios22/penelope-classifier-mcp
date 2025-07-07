"""
Application DTO: NewsClassificationRequest

This module contains the Data Transfer Object for news classification requests in the
Penelope News Classification System. It implements Domain Driven Design principles by
providing a clean interface for request data validation and serialization.

The NewsClassificationRequest DTO encapsulates all parameters needed to request news
classification operations, including processing options, filtering criteria, and
output format preferences. It provides validation to ensure data integrity and
proper error handling for invalid requests.

Classes:
    ProcessingOptions: Configuration options for classification processing
    FilterCriteria: Criteria for filtering news articles
    OutputFormat: Enumeration of supported output formats
    NewsClassificationRequest: Main DTO for classification requests

Example:
    Creating a classification request:
    
    >>> request = NewsClassificationRequest(
    ...     source_urls=["https://example.com/rss"],
    ...     max_articles=50,
    ...     processing_options=ProcessingOptions(
    ...         enable_fact_checking=True,
    ...         enable_sentiment_analysis=True
    ...     )
    ... )
    >>> request.is_valid()
    True

Author: Claude AI Assistant
Date: 2025-07-04
Version: 3.0 (DDD Architecture)
"""

from enum import Enum
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from domain.entities.news_article import ArticleType, SentimentType


class OutputFormat(Enum):
    """
    Enumeration of supported output formats for classification results.
    
    This enum defines the different formats in which classification results
    can be returned to the client application.
    """
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    EXCEL = "excel"


class ProcessingPriority(Enum):
    """
    Enumeration of processing priority levels.
    
    This enum defines the priority levels for processing requests,
    allowing for queue management and resource allocation.
    """
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class ProcessingOptions:
    """
    Configuration options for news classification processing.
    
    This dataclass encapsulates various processing options that control
    how news articles are classified and what analysis techniques are applied.
    
    Attributes:
        enable_sentiment_analysis (bool): Whether to perform sentiment analysis
        enable_fact_checking (bool): Whether to perform fact-checking
        enable_bias_detection (bool): Whether to detect source bias
        enable_financial_analysis (bool): Whether to analyze financial relevance
        enable_crypto_detection (bool): Whether to detect crypto mentions
        enable_company_extraction (bool): Whether to extract company names
        parallel_processing (bool): Whether to enable parallel processing
        max_concurrent_requests (int): Maximum concurrent classification requests
        timeout_seconds (int): Timeout for individual article processing
        retry_attempts (int): Number of retry attempts for failed articles
        
    Example:
        >>> options = ProcessingOptions(
        ...     enable_sentiment_analysis=True,
        ...     enable_fact_checking=True,
        ...     parallel_processing=True,
        ...     max_concurrent_requests=10
        ... )
        >>> options.is_advanced_processing_enabled()
        True
    """
    enable_sentiment_analysis: bool = True
    enable_fact_checking: bool = True
    enable_bias_detection: bool = True
    enable_financial_analysis: bool = True
    enable_crypto_detection: bool = True
    enable_company_extraction: bool = True
    parallel_processing: bool = True
    max_concurrent_requests: int = 10
    timeout_seconds: int = 30
    retry_attempts: int = 3
    
    def is_advanced_processing_enabled(self) -> bool:
        """
        Check if advanced processing features are enabled.
        
        Returns:
            bool: True if any advanced processing option is enabled
        """
        return (
            self.enable_sentiment_analysis or
            self.enable_fact_checking or
            self.enable_bias_detection or
            self.enable_financial_analysis or
            self.enable_crypto_detection or
            self.enable_company_extraction
        )
    
    def get_enabled_features(self) -> List[str]:
        """
        Get a list of enabled processing features.
        
        Returns:
            List[str]: List of enabled feature names
        """
        features = []
        if self.enable_sentiment_analysis:
            features.append("sentiment_analysis")
        if self.enable_fact_checking:
            features.append("fact_checking")
        if self.enable_bias_detection:
            features.append("bias_detection")
        if self.enable_financial_analysis:
            features.append("financial_analysis")
        if self.enable_crypto_detection:
            features.append("crypto_detection")
        if self.enable_company_extraction:
            features.append("company_extraction")
        return features


@dataclass
class FilterCriteria:
    """
    Criteria for filtering news articles during processing.
    
    This dataclass defines various filters that can be applied to news articles
    before or after classification to focus on specific types of content.
    
    Attributes:
        article_types (Optional[List[ArticleType]]): Filter by article types
        sentiment_types (Optional[List[SentimentType]]): Filter by sentiment types
        min_confidence (Optional[float]): Minimum confidence score filter
        max_confidence (Optional[float]): Maximum confidence score filter
        min_credibility (Optional[float]): Minimum credibility score filter
        sources_include (Optional[List[str]]): Include only these sources
        sources_exclude (Optional[List[str]]): Exclude these sources
        keywords_include (Optional[List[str]]): Include articles with these keywords
        keywords_exclude (Optional[List[str]]): Exclude articles with these keywords
        language_filter (Optional[str]): Filter by language (e.g., 'en', 'es')
        date_from (Optional[datetime]): Filter articles from this date
        date_to (Optional[datetime]): Filter articles up to this date
        
    Example:
        >>> criteria = FilterCriteria(
        ...     article_types=[ArticleType.FINANCIAL, ArticleType.CRYPTO],
        ...     min_confidence=0.7,
        ...     sources_include=["Reuters", "Bloomberg"]
        ... )
        >>> criteria.has_filters()
        True
    """
    article_types: Optional[List[ArticleType]] = None
    sentiment_types: Optional[List[SentimentType]] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    min_credibility: Optional[float] = None
    sources_include: Optional[List[str]] = None
    sources_exclude: Optional[List[str]] = None
    keywords_include: Optional[List[str]] = None
    keywords_exclude: Optional[List[str]] = None
    language_filter: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    
    def has_filters(self) -> bool:
        """
        Check if any filters are configured.
        
        Returns:
            bool: True if any filter criteria are set
        """
        return any([
            self.article_types,
            self.sentiment_types,
            self.min_confidence is not None,
            self.max_confidence is not None,
            self.min_credibility is not None,
            self.sources_include,
            self.sources_exclude,
            self.keywords_include,
            self.keywords_exclude,
            self.language_filter,
            self.date_from,
            self.date_to
        ])
    
    def validate(self) -> List[str]:
        """
        Validate the filter criteria and return any validation errors.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
        """
        errors = []
        
        # Validate confidence ranges
        if self.min_confidence is not None and not 0 <= self.min_confidence <= 1:
            errors.append("min_confidence must be between 0 and 1")
        
        if self.max_confidence is not None and not 0 <= self.max_confidence <= 1:
            errors.append("max_confidence must be between 0 and 1")
        
        if (self.min_confidence is not None and self.max_confidence is not None 
            and self.min_confidence > self.max_confidence):
            errors.append("min_confidence cannot be greater than max_confidence")
        
        # Validate credibility ranges
        if self.min_credibility is not None and not 0 <= self.min_credibility <= 100:
            errors.append("min_credibility must be between 0 and 100")
        
        # Validate date ranges
        if (self.date_from is not None and self.date_to is not None 
            and self.date_from > self.date_to):
            errors.append("date_from cannot be after date_to")
        
        return errors


@dataclass
class NewsClassificationRequest:
    """
    Main DTO for news classification requests.
    
    This Data Transfer Object encapsulates all parameters needed to request
    news classification operations. It provides validation, serialization,
    and convenient methods for handling classification requests.
    
    The request DTO serves as the interface between the application layer
    and external clients, ensuring data integrity and proper error handling.
    
    Attributes:
        source_urls (List[str]): URLs of RSS feeds or news sources to process
        max_articles (int): Maximum number of articles to process
        processing_options (ProcessingOptions): Configuration for processing
        filter_criteria (FilterCriteria): Criteria for filtering articles
        output_format (OutputFormat): Desired output format for results
        processing_priority (ProcessingPriority): Priority level for processing
        request_id (Optional[str]): Unique identifier for the request
        client_id (Optional[str]): Identifier of the requesting client
        callback_url (Optional[str]): URL for result delivery (async processing)
        metadata (Optional[Dict[str, Any]]): Additional metadata for the request
        
    Example:
        >>> request = NewsClassificationRequest(
        ...     source_urls=["https://feeds.reuters.com/reuters/businessNews"],
        ...     max_articles=100,
        ...     processing_options=ProcessingOptions(
        ...         enable_fact_checking=True,
        ...         parallel_processing=True
        ...     ),
        ...     filter_criteria=FilterCriteria(
        ...         article_types=[ArticleType.FINANCIAL],
        ...         min_confidence=0.6
        ...     )
        ... )
        >>> errors = request.validate()
        >>> if not errors:
        ...     print("Request is valid")
        
    Business Rules:
        - At least one source URL must be provided
        - max_articles must be between 1 and 1000
        - source_urls must be valid URLs
        - All filter criteria must be valid
        - Request ID should be unique if provided
    """
    
    source_urls: List[str]
    max_articles: int = 100
    processing_options: ProcessingOptions = None
    filter_criteria: FilterCriteria = None
    output_format: OutputFormat = OutputFormat.JSON
    processing_priority: ProcessingPriority = ProcessingPriority.NORMAL
    request_id: Optional[str] = None
    client_id: Optional[str] = None
    callback_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Post-initialization processing for the request.
        
        This method is called after object creation to set default values
        and perform initial validation.
        """
        if self.processing_options is None:
            self.processing_options = ProcessingOptions()
        
        if self.filter_criteria is None:
            self.filter_criteria = FilterCriteria()
        
        if self.metadata is None:
            self.metadata = {}
        
        # Add creation timestamp
        self.metadata['created_at'] = datetime.now().isoformat()
    
    def validate(self) -> List[str]:
        """
        Validate the classification request and return any validation errors.
        
        This method performs comprehensive validation of all request parameters
        and returns a list of error messages if any validation fails.
        
        Returns:
            List[str]: List of validation error messages (empty if valid)
            
        Example:
            >>> request = NewsClassificationRequest(source_urls=[])
            >>> errors = request.validate()
            >>> if errors:
            ...     print(f"Validation errors: {errors}")
        """
        errors = []
        
        # Validate source URLs
        if not self.source_urls:
            errors.append("At least one source URL must be provided")
        else:
            for i, url in enumerate(self.source_urls):
                if not url or not isinstance(url, str):
                    errors.append(f"Source URL {i+1} is invalid")
                elif not url.startswith(('http://', 'https://')):
                    errors.append(f"Source URL {i+1} must start with http:// or https://")
        
        # Validate max_articles
        if not isinstance(self.max_articles, int) or self.max_articles < 1:
            errors.append("max_articles must be a positive integer")
        elif self.max_articles > 1000:
            errors.append("max_articles cannot exceed 1000")
        
        # Validate processing options
        if not isinstance(self.processing_options.timeout_seconds, int) or self.processing_options.timeout_seconds < 1:
            errors.append("timeout_seconds must be a positive integer")
        
        if not isinstance(self.processing_options.retry_attempts, int) or self.processing_options.retry_attempts < 0:
            errors.append("retry_attempts must be a non-negative integer")
        
        if not isinstance(self.processing_options.max_concurrent_requests, int) or self.processing_options.max_concurrent_requests < 1:
            errors.append("max_concurrent_requests must be a positive integer")
        
        # Validate filter criteria
        if self.filter_criteria:
            errors.extend(self.filter_criteria.validate())
        
        # Validate callback URL if provided
        if self.callback_url and not self.callback_url.startswith(('http://', 'https://')):
            errors.append("callback_url must be a valid HTTP(S) URL")
        
        return errors
    
    def is_valid(self) -> bool:
        """
        Check if the request is valid.
        
        Returns:
            bool: True if the request passes all validation rules
        """
        return len(self.validate()) == 0
    
    def is_async_request(self) -> bool:
        """
        Check if this is an asynchronous processing request.
        
        Returns:
            bool: True if a callback URL is provided
        """
        return self.callback_url is not None
    
    def is_high_priority(self) -> bool:
        """
        Check if this is a high-priority request.
        
        Returns:
            bool: True if priority is HIGH or URGENT
        """
        return self.processing_priority in [ProcessingPriority.HIGH, ProcessingPriority.URGENT]
    
    def get_estimated_processing_time(self) -> int:
        """
        Estimate the processing time in seconds based on request parameters.
        
        Returns:
            int: Estimated processing time in seconds
        """
        base_time_per_article = 2  # seconds
        
        # Adjust for processing options
        if self.processing_options.is_advanced_processing_enabled():
            base_time_per_article += 1
        
        # Adjust for parallel processing
        if self.processing_options.parallel_processing:
            concurrent_factor = min(self.processing_options.max_concurrent_requests, 10)
            base_time_per_article = base_time_per_article / concurrent_factor
        
        return int(self.max_articles * base_time_per_article)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the request to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the request
        """
        return {
            'source_urls': self.source_urls,
            'max_articles': self.max_articles,
            'processing_options': {
                'enable_sentiment_analysis': self.processing_options.enable_sentiment_analysis,
                'enable_fact_checking': self.processing_options.enable_fact_checking,
                'enable_bias_detection': self.processing_options.enable_bias_detection,
                'enable_financial_analysis': self.processing_options.enable_financial_analysis,
                'enable_crypto_detection': self.processing_options.enable_crypto_detection,
                'enable_company_extraction': self.processing_options.enable_company_extraction,
                'parallel_processing': self.processing_options.parallel_processing,
                'max_concurrent_requests': self.processing_options.max_concurrent_requests,
                'timeout_seconds': self.processing_options.timeout_seconds,
                'retry_attempts': self.processing_options.retry_attempts,
            },
            'filter_criteria': {
                'article_types': [t.value for t in self.filter_criteria.article_types] if self.filter_criteria.article_types else None,
                'sentiment_types': [s.value for s in self.filter_criteria.sentiment_types] if self.filter_criteria.sentiment_types else None,
                'min_confidence': self.filter_criteria.min_confidence,
                'max_confidence': self.filter_criteria.max_confidence,
                'min_credibility': self.filter_criteria.min_credibility,
                'sources_include': self.filter_criteria.sources_include,
                'sources_exclude': self.filter_criteria.sources_exclude,
                'keywords_include': self.filter_criteria.keywords_include,
                'keywords_exclude': self.filter_criteria.keywords_exclude,
                'language_filter': self.filter_criteria.language_filter,
                'date_from': self.filter_criteria.date_from.isoformat() if self.filter_criteria.date_from else None,
                'date_to': self.filter_criteria.date_to.isoformat() if self.filter_criteria.date_to else None,
            },
            'output_format': self.output_format.value,
            'processing_priority': self.processing_priority.value,
            'request_id': self.request_id,
            'client_id': self.client_id,
            'callback_url': self.callback_url,
            'metadata': self.metadata,
            'estimated_processing_time': self.get_estimated_processing_time(),
            'is_async': self.is_async_request(),
            'is_high_priority': self.is_high_priority(),
        }
    
    def __str__(self) -> str:
        """
        String representation of the request.
        
        Returns:
            str: Human-readable string representation
        """
        return (f"NewsClassificationRequest(sources={len(self.source_urls)}, "
                f"max_articles={self.max_articles}, "
                f"priority={self.processing_priority.value}, "
                f"format={self.output_format.value})")
    
    def __repr__(self) -> str:
        """
        Developer representation of the request.
        
        Returns:
            str: Detailed string representation for debugging
        """
        return (f"NewsClassificationRequest(source_urls={self.source_urls}, "
                f"max_articles={self.max_articles}, "
                f"processing_options={self.processing_options}, "
                f"filter_criteria={self.filter_criteria})") 