"""
Domain Repository Interface: NewsRepository

This module defines the abstract repository interface for news articles in the Penelope News
Classification System. It follows Domain Driven Design principles by providing a clean
abstraction layer between the domain layer and infrastructure implementations.

The repository pattern allows the domain layer to remain independent of specific storage
implementations (file system, database, API, etc.) while providing a consistent interface
for data access operations.

Classes:
    NewsRepository: Abstract base class defining the repository interface

Example:
    Implementing a concrete repository:
    
    >>> class FileNewsRepository(NewsRepository):
    ...     async def save(self, article: NewsArticle) -> bool:
    ...         # Implementation specific to file storage
    ...         pass
    
    >>> repository = FileNewsRepository()
    >>> success = await repository.save(article)

Author: Claude AI Assistant
Date: 2025-07-04
Version: 3.0 (DDD Architecture)
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from datetime import datetime

from ..entities.news_article import NewsArticle


class NewsRepository(ABC):
    """
    Abstract repository interface for news article persistence operations.
    
    This abstract base class defines the contract that all concrete repository
    implementations must follow. It provides methods for common CRUD operations
    and querying capabilities for news articles.
    
    The repository pattern ensures that the domain layer remains independent
    of specific storage implementations, allowing for easy testing and flexibility
    in choosing different persistence strategies.
    
    Methods:
        save: Store a single news article
        save_batch: Store multiple news articles efficiently
        find_by_url: Retrieve an article by its URL
        find_by_source: Retrieve articles from a specific source
        find_by_date_range: Retrieve articles within a date range
        find_financial_articles: Retrieve articles classified as financial
        find_high_confidence_articles: Retrieve articles with high confidence scores
        get_statistics: Get aggregated statistics about stored articles
        delete_by_url: Remove an article by its URL
        clear_all: Remove all stored articles
        
    Example:
        >>> class ConcreteRepository(NewsRepository):
        ...     async def save(self, article: NewsArticle) -> bool:
        ...         # Implementation specific logic
        ...         return True
        
        >>> repo = ConcreteRepository()
        >>> article = NewsArticle(url="...", title="...", source="...")
        >>> success = await repo.save(article)
    """
    
    @abstractmethod
    async def save(self, article: NewsArticle) -> bool:
        """
        Save a single news article to the repository.
        
        This method persists a news article entity to the underlying storage
        system. The implementation should handle any necessary data validation,
        serialization, and error handling.
        
        Args:
            article (NewsArticle): The news article entity to save
            
        Returns:
            bool: True if the article was successfully saved, False otherwise
            
        Raises:
            RepositoryError: If there's an error during the save operation
            
        Example:
            >>> article = NewsArticle(url="...", title="...", source="...")
            >>> success = await repository.save(article)
            >>> assert success is True
        """
        pass
    
    @abstractmethod
    async def save_batch(self, articles: List[NewsArticle]) -> Dict[str, bool]:
        """
        Save multiple news articles efficiently in a batch operation.
        
        This method provides an optimized way to persist multiple articles
        at once, which is more efficient than individual save operations.
        The implementation should handle partial failures gracefully.
        
        Args:
            articles (List[NewsArticle]): List of news articles to save
            
        Returns:
            Dict[str, bool]: Mapping of article URLs to success status
            
        Raises:
            RepositoryError: If there's a critical error during batch save
            
        Example:
            >>> articles = [article1, article2, article3]
            >>> results = await repository.save_batch(articles)
            >>> success_count = sum(results.values())
        """
        pass
    
    @abstractmethod
    async def find_by_url(self, url: str) -> Optional[NewsArticle]:
        """
        Retrieve a news article by its URL.
        
        This method searches for and returns a news article with the specified
        URL. URLs are treated as unique identifiers for articles.
        
        Args:
            url (str): The URL of the article to retrieve
            
        Returns:
            Optional[NewsArticle]: The article if found, None otherwise
            
        Example:
            >>> url = "https://example.com/news/article-123"
            >>> article = await repository.find_by_url(url)
            >>> if article:
            ...     print(f"Found: {article.title}")
        """
        pass
    
    @abstractmethod
    async def find_by_source(self, source: str, limit: Optional[int] = None) -> List[NewsArticle]:
        """
        Retrieve news articles from a specific source.
        
        This method searches for and returns articles from a given news source.
        It supports pagination through the limit parameter.
        
        Args:
            source (str): The name of the news source to filter by
            limit (Optional[int]): Maximum number of articles to return
            
        Returns:
            List[NewsArticle]: List of articles from the specified source
            
        Example:
            >>> articles = await repository.find_by_source("Reuters", limit=10)
            >>> print(f"Found {len(articles)} articles from Reuters")
        """
        pass
    
    @abstractmethod
    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[NewsArticle]:
        """
        Retrieve news articles within a specified date range.
        
        This method searches for articles that were published between the
        specified start and end dates (inclusive).
        
        Args:
            start_date (datetime): The earliest publication date to include
            end_date (datetime): The latest publication date to include
            
        Returns:
            List[NewsArticle]: List of articles within the date range
            
        Example:
            >>> from datetime import datetime, timedelta
            >>> end_date = datetime.now()
            >>> start_date = end_date - timedelta(days=7)
            >>> articles = await repository.find_by_date_range(start_date, end_date)
            >>> print(f"Found {len(articles)} articles from last week")
        """
        pass
    
    @abstractmethod
    async def find_financial_articles(self, limit: Optional[int] = None) -> List[NewsArticle]:
        """
        Retrieve news articles classified as financial.
        
        This method searches for articles that have been identified as
        financial news through classification algorithms. It includes
        articles with financial relevance, crypto relevance, or classified
        as financial type.
        
        Args:
            limit (Optional[int]): Maximum number of articles to return
            
        Returns:
            List[NewsArticle]: List of financial news articles
            
        Example:
            >>> financial_articles = await repository.find_financial_articles(limit=20)
            >>> for article in financial_articles:
            ...     print(f"Financial: {article.title}")
        """
        pass
    
    @abstractmethod
    async def find_high_confidence_articles(self, min_confidence: float = 0.7) -> List[NewsArticle]:
        """
        Retrieve news articles with high classification confidence.
        
        This method searches for articles where the machine learning
        classification confidence score meets or exceeds the specified
        minimum threshold.
        
        Args:
            min_confidence (float): Minimum confidence score (0.0-1.0)
            
        Returns:
            List[NewsArticle]: List of high-confidence articles
            
        Example:
            >>> high_conf_articles = await repository.find_high_confidence_articles(0.8)
            >>> print(f"Found {len(high_conf_articles)} high-confidence articles")
        """
        pass
    
    @abstractmethod
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get aggregated statistics about stored news articles.
        
        This method provides comprehensive statistics about the articles
        stored in the repository, including counts, averages, and distributions
        of various attributes.
        
        Returns:
            Dict[str, Any]: Dictionary containing various statistics including:
                - total_articles: Total number of articles
                - by_source: Count of articles per source
                - by_type: Count of articles per type
                - by_sentiment: Count of articles per sentiment
                - average_confidence: Average confidence score
                - average_credibility: Average credibility score
                - financial_percentage: Percentage of financial articles
                - date_range: Earliest and latest publication dates
                
        Example:
            >>> stats = await repository.get_statistics()
            >>> print(f"Total articles: {stats['total_articles']}")
            >>> print(f"Financial articles: {stats['financial_percentage']:.1f}%")
        """
        pass
    
    @abstractmethod
    async def delete_by_url(self, url: str) -> bool:
        """
        Delete a news article by its URL.
        
        This method removes an article from the repository using its URL
        as the unique identifier.
        
        Args:
            url (str): The URL of the article to delete
            
        Returns:
            bool: True if the article was successfully deleted, False if not found
            
        Example:
            >>> url = "https://example.com/news/old-article"
            >>> deleted = await repository.delete_by_url(url)
            >>> if deleted:
            ...     print("Article deleted successfully")
        """
        pass
    
    @abstractmethod
    async def clear_all(self) -> bool:
        """
        Remove all news articles from the repository.
        
        This method clears all stored articles from the repository.
        Use with caution as this operation cannot be undone.
        
        Returns:
            bool: True if all articles were successfully cleared
            
        Raises:
            RepositoryError: If the clear operation fails
            
        Example:
            >>> cleared = await repository.clear_all()
            >>> if cleared:
            ...     print("All articles cleared")
        """
        pass
    
    # Optional helper methods that concrete implementations may override
    
    async def count_total(self) -> int:
        """
        Get the total number of articles in the repository.
        
        This is a convenience method that can be overridden by concrete
        implementations for more efficient counting.
        
        Returns:
            int: Total number of articles
        """
        stats = await self.get_statistics()
        return stats.get('total_articles', 0)
    
    async def exists(self, url: str) -> bool:
        """
        Check if an article with the given URL exists in the repository.
        
        This is a convenience method that can be overridden by concrete
        implementations for more efficient existence checks.
        
        Args:
            url (str): The URL to check for existence
            
        Returns:
            bool: True if an article with the URL exists, False otherwise
        """
        article = await self.find_by_url(url)
        return article is not None
    
    async def find_by_confidence_range(self, min_confidence: float, max_confidence: float) -> List[NewsArticle]:
        """
        Find articles within a specific confidence range.
        
        This helper method can be overridden by concrete implementations
        for more efficient range queries.
        
        Args:
            min_confidence (float): Minimum confidence score (0.0-1.0)
            max_confidence (float): Maximum confidence score (0.0-1.0)
            
        Returns:
            List[NewsArticle]: Articles within the confidence range
        """
        # Default implementation using get_statistics and filtering
        # Concrete implementations should override for efficiency
        raise NotImplementedError("Concrete implementation should override this method")


class RepositoryError(Exception):
    """
    Exception raised when repository operations fail.
    
    This exception is raised when there are errors during repository
    operations such as save failures, connection issues, or data corruption.
    
    Attributes:
        message (str): The error message describing what went wrong
        operation (str): The repository operation that failed
        original_error (Exception): The original exception that caused the error
    """
    
    def __init__(self, message: str, operation: str = "", original_error: Exception = None):
        """
        Initialize a RepositoryError.
        
        Args:
            message (str): The error message
            operation (str): The operation that failed
            original_error (Exception): The original exception
        """
        self.message = message
        self.operation = operation
        self.original_error = original_error
        
        full_message = f"Repository error"
        if operation:
            full_message += f" during {operation}"
        full_message += f": {message}"
        
        if original_error:
            full_message += f" (caused by: {type(original_error).__name__}: {original_error})"
        
        super().__init__(full_message) 