"""
Application DTO: NewsClassificationResponse

This module contains the Data Transfer Object for news classification responses in the
Penelope News Classification System. It implements Domain Driven Design principles by
providing a structured interface for returning classification results to clients.

The NewsClassificationResponse DTO encapsulates all data returned from classification
operations, including processing statistics, error information, and classified articles.
It provides comprehensive metrics and validation to ensure data integrity and proper
error handling for client applications.

Classes:
    ProcessingStatistics: Metrics about the classification process
    ErrorInformation: Details about processing errors
    NewsClassificationResponse: Main DTO for classification responses

Example:
    Creating a classification response:
    
    >>> response = NewsClassificationResponse(
    ...     success=True,
    ...     articles_processed=100,
    ...     processing_statistics=ProcessingStatistics(
    ...         total_processing_time=158.9,
    ...         articles_per_minute=37.8
    ...     )
    ... )
    >>> print(f"Success rate: {response.get_success_rate():.1%}")

Author: Claude AI Assistant
Date: 2025-07-04
Version: 3.0 (DDD Architecture)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

from domain.entities.news_article import NewsArticle


class ProcessingStatus(Enum):
    """
    Enumeration of processing status values.
    
    This enum defines the possible states of a classification operation.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


@dataclass
class ProcessingStatistics:
    """
    Comprehensive statistics about the classification processing.
    
    This dataclass contains detailed metrics about the classification operation,
    including performance indicators, success rates, and processing efficiency.
    
    Attributes:
        total_processing_time (float): Total time taken for processing (seconds)
        articles_per_minute (float): Processing rate (articles per minute)
        extraction_success_rate (float): Percentage of successful content extractions
        classification_success_rate (float): Percentage of successful classifications
        average_confidence (float): Average confidence score of classifications
        average_credibility (float): Average credibility score of sources
        average_fact_check_score (float): Average fact-checking score
        financial_articles_percentage (float): Percentage of financial articles
        crypto_articles_percentage (float): Percentage of crypto articles
        high_confidence_percentage (float): Percentage of high-confidence articles
        verified_articles_percentage (float): Percentage of verified articles
        error_rate (float): Overall error rate during processing
        
    Example:
        >>> stats = ProcessingStatistics(
        ...     total_processing_time=158.9,
        ...     articles_per_minute=37.8,
        ...     extraction_success_rate=87.5,
        ...     classification_success_rate=96.2
        ... )
        >>> print(f"Processed at {stats.articles_per_minute:.1f} articles/minute")
    """
    total_processing_time: float
    articles_per_minute: float
    extraction_success_rate: float
    classification_success_rate: float
    average_confidence: float
    average_credibility: float
    average_fact_check_score: float
    financial_articles_percentage: float
    crypto_articles_percentage: float
    high_confidence_percentage: float
    verified_articles_percentage: float
    error_rate: float
    
    def is_high_performance(self) -> bool:
        """
        Check if processing achieved high performance metrics.
        
        Returns:
            bool: True if processing was high-performance
        """
        return (
            self.articles_per_minute >= 30 and
            self.extraction_success_rate >= 85 and
            self.classification_success_rate >= 95 and
            self.error_rate <= 5
        )
    
    def get_quality_score(self) -> float:
        """
        Calculate an overall quality score for the processing.
        
        Returns:
            float: Quality score between 0 and 100
        """
        quality_factors = [
            self.extraction_success_rate,
            self.classification_success_rate,
            self.average_confidence * 100,
            self.average_credibility,
            self.average_fact_check_score,
            (100 - self.error_rate)
        ]
        return sum(quality_factors) / len(quality_factors)


@dataclass
class ErrorInformation:
    """
    Information about errors encountered during processing.
    
    This dataclass contains details about errors that occurred during
    classification operations, providing debugging information and
    error recovery suggestions.
    
    Attributes:
        error_count (int): Total number of errors encountered
        error_types (Dict[str, int]): Count of errors by type
        failed_urls (List[str]): URLs of articles that failed processing
        error_messages (List[str]): List of error messages
        recoverable_errors (int): Number of errors that could be retried
        critical_errors (int): Number of critical errors that stopped processing
        
    Example:
        >>> error_info = ErrorInformation(
        ...     error_count=3,
        ...     error_types={"timeout": 2, "extraction_failed": 1},
        ...     failed_urls=["https://example.com/article1"],
        ...     error_messages=["Timeout after 30 seconds"]
        ... )
        >>> print(f"Error rate: {error_info.get_error_rate(100):.1%}")
    """
    error_count: int
    error_types: Dict[str, int]
    failed_urls: List[str]
    error_messages: List[str]
    recoverable_errors: int = 0
    critical_errors: int = 0
    
    def get_error_rate(self, total_articles: int) -> float:
        """
        Calculate the error rate as a percentage.
        
        Args:
            total_articles (int): Total number of articles processed
            
        Returns:
            float: Error rate as a percentage (0-100)
        """
        if total_articles == 0:
            return 0.0
        return (self.error_count / total_articles) * 100
    
    def has_critical_errors(self) -> bool:
        """
        Check if any critical errors occurred.
        
        Returns:
            bool: True if critical errors were encountered
        """
        return self.critical_errors > 0
    
    def get_most_common_error_type(self) -> Optional[str]:
        """
        Get the most common error type.
        
        Returns:
            Optional[str]: Most common error type, or None if no errors
        """
        if not self.error_types:
            return None
        return max(self.error_types.items(), key=lambda x: x[1])[0]


@dataclass
class NewsClassificationResponse:
    """
    Main DTO for news classification responses.
    
    This Data Transfer Object encapsulates all data returned from news
    classification operations. It provides comprehensive information about
    the processing results, including success metrics, error details,
    and the classified articles themselves.
    
    The response DTO serves as the interface between the application layer
    and client applications, ensuring consistent data format and complete
    information delivery.
    
    Attributes:
        success (bool): Whether the operation completed successfully
        status (ProcessingStatus): Current status of the processing operation
        request_id (Optional[str]): Identifier of the original request
        articles_processed (int): Number of articles that were processed
        articles_classified (int): Number of articles successfully classified
        classified_articles (List[NewsArticle]): List of classified articles
        processing_statistics (ProcessingStatistics): Detailed processing metrics
        error_information (Optional[ErrorInformation]): Error details if any
        started_at (datetime): When processing started
        completed_at (Optional[datetime]): When processing completed
        metadata (Dict[str, Any]): Additional metadata about the response
        
    Example:
        >>> response = NewsClassificationResponse(
        ...     success=True,
        ...     status=ProcessingStatus.COMPLETED,
        ...     articles_processed=100,
        ...     articles_classified=96,
        ...     classified_articles=classified_articles_list,
        ...     processing_statistics=processing_stats
        ... )
        >>> print(f"Success rate: {response.get_success_rate():.1%}")
        >>> print(f"Quality score: {response.get_quality_score():.1f}")
        
    Business Rules:
        - articles_classified cannot exceed articles_processed
        - completed_at must be after started_at if both are set
        - success should be False if error_information contains critical errors
        - processing_statistics must be provided for completed operations
    """
    
    success: bool
    status: ProcessingStatus
    articles_processed: int
    articles_classified: int
    classified_articles: List[NewsArticle]
    processing_statistics: ProcessingStatistics
    request_id: Optional[str] = None
    error_information: Optional[ErrorInformation] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """
        Post-initialization processing for the response.
        
        This method is called after object creation to set default values
        and perform validation.
        """
        if self.metadata is None:
            self.metadata = {}
        
        if self.started_at is None:
            self.started_at = datetime.now()
        
        # Add response creation timestamp
        self.metadata['response_created_at'] = datetime.now().isoformat()
        
        # Validate business rules
        if self.articles_classified > self.articles_processed:
            raise ValueError("articles_classified cannot exceed articles_processed")
        
        if (self.completed_at is not None and self.started_at is not None 
            and self.completed_at < self.started_at):
            raise ValueError("completed_at cannot be before started_at")
    
    def get_success_rate(self) -> float:
        """
        Calculate the success rate of the classification operation.
        
        Returns:
            float: Success rate as a percentage (0-100)
        """
        if self.articles_processed == 0:
            return 0.0
        return (self.articles_classified / self.articles_processed) * 100
    
    def get_processing_duration(self) -> Optional[float]:
        """
        Get the total processing duration in seconds.
        
        Returns:
            Optional[float]: Processing duration in seconds, or None if not completed
        """
        if self.completed_at is None or self.started_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()
    
    def get_quality_score(self) -> float:
        """
        Calculate an overall quality score for the response.
        
        Returns:
            float: Quality score between 0 and 100
        """
        if not self.processing_statistics:
            return 0.0
        return self.processing_statistics.get_quality_score()
    
    def is_complete(self) -> bool:
        """
        Check if the processing operation is complete.
        
        Returns:
            bool: True if the operation is complete
        """
        return self.status == ProcessingStatus.COMPLETED
    
    def has_errors(self) -> bool:
        """
        Check if any errors occurred during processing.
        
        Returns:
            bool: True if errors were encountered
        """
        return (self.error_information is not None and 
                self.error_information.error_count > 0)
    
    def has_critical_errors(self) -> bool:
        """
        Check if any critical errors occurred.
        
        Returns:
            bool: True if critical errors were encountered
        """
        return (self.error_information is not None and 
                self.error_information.has_critical_errors())
    
    def get_financial_articles(self) -> List[NewsArticle]:
        """
        Get all articles classified as financial news.
        
        Returns:
            List[NewsArticle]: List of financial articles
        """
        return [article for article in self.classified_articles 
                if article.is_financial_news()]
    
    def get_high_confidence_articles(self) -> List[NewsArticle]:
        """
        Get all articles with high confidence classification.
        
        Returns:
            List[NewsArticle]: List of high-confidence articles
        """
        return [article for article in self.classified_articles 
                if article.is_high_confidence()]
    
    def get_credible_articles(self) -> List[NewsArticle]:
        """
        Get all articles from credible sources.
        
        Returns:
            List[NewsArticle]: List of articles from credible sources
        """
        return [article for article in self.classified_articles 
                if article.is_credible_source()]
    
    def get_articles_by_sentiment(self, sentiment: str) -> List[NewsArticle]:
        """
        Get articles filtered by sentiment type.
        
        Args:
            sentiment (str): Sentiment type to filter by
            
        Returns:
            List[NewsArticle]: List of articles with specified sentiment
        """
        return [article for article in self.classified_articles 
                if article.sentiment and article.sentiment.value == sentiment]
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get a summary of key statistics for the response.
        
        Returns:
            Dict[str, Any]: Summary statistics dictionary
        """
        return {
            'success': self.success,
            'status': self.status.value,
            'articles_processed': self.articles_processed,
            'articles_classified': self.articles_classified,
            'success_rate': self.get_success_rate(),
            'processing_duration': self.get_processing_duration(),
            'quality_score': self.get_quality_score(),
            'has_errors': self.has_errors(),
            'has_critical_errors': self.has_critical_errors(),
            'financial_articles': len(self.get_financial_articles()),
            'high_confidence_articles': len(self.get_high_confidence_articles()),
            'credible_articles': len(self.get_credible_articles()),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'processing_statistics': {
                'articles_per_minute': self.processing_statistics.articles_per_minute,
                'extraction_success_rate': self.processing_statistics.extraction_success_rate,
                'average_confidence': self.processing_statistics.average_confidence,
                'average_credibility': self.processing_statistics.average_credibility,
                'is_high_performance': self.processing_statistics.is_high_performance()
            } if self.processing_statistics else None
        }
    
    def to_dict(self, include_articles: bool = False) -> Dict[str, Any]:
        """
        Convert the response to a dictionary for serialization.
        
        Args:
            include_articles (bool): Whether to include full article data
            
        Returns:
            Dict[str, Any]: Dictionary representation of the response
        """
        result = {
            'success': self.success,
            'status': self.status.value,
            'request_id': self.request_id,
            'articles_processed': self.articles_processed,
            'articles_classified': self.articles_classified,
            'success_rate': self.get_success_rate(),
            'processing_duration': self.get_processing_duration(),
            'quality_score': self.get_quality_score(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'metadata': self.metadata,
            'processing_statistics': {
                'total_processing_time': self.processing_statistics.total_processing_time,
                'articles_per_minute': self.processing_statistics.articles_per_minute,
                'extraction_success_rate': self.processing_statistics.extraction_success_rate,
                'classification_success_rate': self.processing_statistics.classification_success_rate,
                'average_confidence': self.processing_statistics.average_confidence,
                'average_credibility': self.processing_statistics.average_credibility,
                'average_fact_check_score': self.processing_statistics.average_fact_check_score,
                'financial_articles_percentage': self.processing_statistics.financial_articles_percentage,
                'crypto_articles_percentage': self.processing_statistics.crypto_articles_percentage,
                'high_confidence_percentage': self.processing_statistics.high_confidence_percentage,
                'verified_articles_percentage': self.processing_statistics.verified_articles_percentage,
                'error_rate': self.processing_statistics.error_rate,
                'is_high_performance': self.processing_statistics.is_high_performance(),
                'quality_score': self.processing_statistics.get_quality_score()
            } if self.processing_statistics else None,
            'error_information': {
                'error_count': self.error_information.error_count,
                'error_types': self.error_information.error_types,
                'failed_urls': self.error_information.failed_urls,
                'error_messages': self.error_information.error_messages,
                'recoverable_errors': self.error_information.recoverable_errors,
                'critical_errors': self.error_information.critical_errors,
                'error_rate': self.error_information.get_error_rate(self.articles_processed),
                'most_common_error_type': self.error_information.get_most_common_error_type()
            } if self.error_information else None
        }
        
        if include_articles:
            result['classified_articles'] = [
                article.to_dict() for article in self.classified_articles
            ]
        else:
            result['classified_articles_count'] = len(self.classified_articles)
            result['sample_articles'] = [
                article.to_dict() for article in self.classified_articles[:3]
            ]
        
        return result
    
    def __str__(self) -> str:
        """
        String representation of the response.
        
        Returns:
            str: Human-readable string representation
        """
        return (f"NewsClassificationResponse(success={self.success}, "
                f"status={self.status.value}, "
                f"processed={self.articles_processed}, "
                f"classified={self.articles_classified}, "
                f"success_rate={self.get_success_rate():.1f}%)")
    
    def __repr__(self) -> str:
        """
        Developer representation of the response.
        
        Returns:
            str: Detailed string representation for debugging
        """
        return (f"NewsClassificationResponse(success={self.success}, "
                f"status={self.status}, "
                f"articles_processed={self.articles_processed}, "
                f"articles_classified={self.articles_classified}, "
                f"request_id={self.request_id})")
    
    @classmethod
    def create_success_response(
        cls,
        articles_processed: int,
        classified_articles: List[NewsArticle],
        processing_statistics: ProcessingStatistics,
        request_id: Optional[str] = None
    ) -> 'NewsClassificationResponse':
        """
        Create a successful classification response.
        
        Args:
            articles_processed (int): Number of articles processed
            classified_articles (List[NewsArticle]): Classified articles
            processing_statistics (ProcessingStatistics): Processing metrics
            request_id (Optional[str]): Request identifier
            
        Returns:
            NewsClassificationResponse: Success response
        """
        return cls(
            success=True,
            status=ProcessingStatus.COMPLETED,
            articles_processed=articles_processed,
            articles_classified=len(classified_articles),
            classified_articles=classified_articles,
            processing_statistics=processing_statistics,
            request_id=request_id,
            completed_at=datetime.now()
        )
    
    @classmethod
    def create_error_response(
        cls,
        error_message: str,
        request_id: str,
        articles_processed: int = 0,
        started_at: Optional[datetime] = None
    ) -> 'NewsClassificationResponse':
        """
        Create an error response for failed classification operations.
        
        Args:
            error_message (str): Error message describing the failure
            request_id (str): Request identifier
            articles_processed (int): Number of articles processed before failure
            started_at (Optional[datetime]): When the process started
            
        Returns:
            NewsClassificationResponse: Error response
        """
        now = datetime.now()
        start_time = started_at if started_at else now
        
        # Create empty processing statistics for error case
        processing_statistics = ProcessingStatistics(
            total_processing_time=0.0,
            articles_per_minute=0.0,
            extraction_success_rate=0.0,
            classification_success_rate=0.0,
            average_confidence=0.0,
            average_credibility=0.0,
            average_fact_check_score=0.0,
            financial_articles_percentage=0.0,
            crypto_articles_percentage=0.0,
            high_confidence_percentage=0.0,
            verified_articles_percentage=0.0,
            error_rate=100.0
        )
        
        return cls(
            success=False,
            status=ProcessingStatus.FAILED,
            request_id=request_id,
            articles_processed=articles_processed,
            articles_classified=0,
            classified_articles=[],
            processing_statistics=processing_statistics,
            error_information=ErrorInformation(
                error_count=1,
                error_types={'system_error': 1},
                failed_urls=[],
                error_messages=[error_message],
                recoverable_errors=0,
                critical_errors=1
            ),
            started_at=start_time,
            completed_at=now if now > start_time else start_time
        ) 