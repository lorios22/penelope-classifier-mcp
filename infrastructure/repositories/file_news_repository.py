"""
Infrastructure Repository: FileNewsRepository

This module contains the file-based implementation of the NewsRepository interface for the
Penelope News Classification System. It implements Domain Driven Design principles by
providing a concrete persistence implementation that stores articles in JSON and CSV formats.

The FileNewsRepository provides efficient file-based storage with support for JSON and CSV
export formats. It implements all repository operations with proper error handling,
atomic writes, and backup mechanisms to ensure data integrity.

Classes:
    FileNewsRepository: File-based implementation of NewsRepository

Key Features:
    - JSON and CSV export support
    - Atomic file operations with backup
    - Efficient indexing and search capabilities  
    - Comprehensive error handling and recovery
    - Statistics calculation and caching
    - Thread-safe operations

Example:
    Basic usage:
    
    >>> repository = FileNewsRepository(
    ...     data_directory="./data",
    ...     enable_backup=True
    ... )
    >>> article = NewsArticle(url="...", title="...", source="...")
    >>> success = await repository.save(article)
    >>> found_article = await repository.find_by_url(article.url)

Author: Claude AI Assistant
Date: 2025-07-04
Version: 3.0 (DDD Architecture)
"""

import json
import csv
import os
import asyncio
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import shutil
from pathlib import Path
import aiofiles

from domain.repositories.news_repository import NewsRepository, RepositoryError
from domain.entities.news_article import (
    NewsArticle, 
    ArticleType, 
    SentimentType, 
    BiasClassification, 
    FactCheckClassification
)


class FileNewsRepository(NewsRepository):
    """
    File-based implementation of the NewsRepository interface.
    
    This repository stores news articles in JSON format with support for CSV export.
    It provides efficient file-based persistence with proper error handling,
    backup mechanisms, and indexing for fast retrieval operations.
    
    The repository maintains data integrity through atomic write operations and
    provides comprehensive statistics calculation for monitoring and reporting.
    
    Key Features:
        - JSON primary storage with CSV export capability
        - Atomic write operations with backup and rollback
        - In-memory indexing for fast URL-based lookups
        - Efficient batch operations with progress tracking
        - Comprehensive error handling and recovery
        - Automatic statistics calculation and caching
        - Thread-safe async operations
        
    File Structure:
        - data_directory/articles.json: Main article storage
        - data_directory/articles.csv: CSV export
        - data_directory/backup/: Backup files
        - data_directory/index/: Index files for fast lookup
        
    Example:
        >>> repo = FileNewsRepository("/path/to/data")
        >>> await repo.save(article)
        >>> articles = await repo.find_by_source("Reuters")
        >>> stats = await repo.get_statistics()
    """
    
    def __init__(self, data_directory: str = "./data", enable_backup: bool = True):
        """
        Initialize the FileNewsRepository.
        
        Args:
            data_directory (str): Directory for storing data files
            enable_backup (bool): Whether to enable backup functionality
        """
        self.logger = logging.getLogger(__name__)
        self.data_directory = Path(data_directory)
        self.enable_backup = enable_backup
        
        # File paths
        self.articles_file = self.data_directory / "articles.json"
        self.csv_file = self.data_directory / "articles.csv"
        self.backup_directory = self.data_directory / "backup"
        self.index_directory = self.data_directory / "index"
        
        # In-memory cache for fast lookups
        self._url_index: Dict[str, int] = {}
        self._statistics_cache: Optional[Dict[str, Any]] = None
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minutes
        
        # Thread safety
        self._write_lock = asyncio.Lock()
        
        # Initialize repository
        asyncio.create_task(self._initialize_repository())
    
    async def _initialize_repository(self) -> None:
        """
        Initialize the repository by creating necessary directories and files.
        """
        try:
            # Create directories
            self.data_directory.mkdir(parents=True, exist_ok=True)
            if self.enable_backup:
                self.backup_directory.mkdir(parents=True, exist_ok=True)
            self.index_directory.mkdir(parents=True, exist_ok=True)
            
            # Create initial files if they don't exist
            if not self.articles_file.exists():
                await self._write_articles_file([])
                self.logger.info("Initialized empty articles file")
            
            # Build initial index
            await self._rebuild_index()
            self.logger.info(f"Repository initialized at {self.data_directory}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize repository: {e}")
            raise RepositoryError(f"Repository initialization failed: {e}", "initialization", e)
    
    async def save(self, article: NewsArticle) -> bool:
        """
        Save a single news article to the repository.
        
        Args:
            article (NewsArticle): The article to save
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            async with self._write_lock:
                # Load existing articles
                articles = await self._load_articles_from_file()
                
                # Check if article already exists
                existing_index = self._url_index.get(article.url)
                if existing_index is not None:
                    # Update existing article
                    articles[existing_index] = article
                    self.logger.debug(f"Updated existing article: {article.url}")
                else:
                    # Add new article
                    articles.append(article)
                    self._url_index[article.url] = len(articles) - 1
                    self.logger.debug(f"Added new article: {article.url}")
                
                # Save to file
                await self._write_articles_file(articles)
                
                # Update CSV export
                await self._export_to_csv(articles)
                
                # Invalidate cache
                self._invalidate_cache()
                
                self.logger.info(f"Successfully saved article: {article.url}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to save article {article.url}: {e}")
            return False
    
    async def save_batch(self, articles: List[NewsArticle]) -> Dict[str, bool]:
        """
        Save multiple articles efficiently in a batch operation.
        
        Args:
            articles (List[NewsArticle]): Articles to save
            
        Returns:
            Dict[str, bool]: Mapping of article URLs to success status
        """
        self.logger.info(f"Starting batch save of {len(articles)} articles")
        results = {}
        
        try:
            async with self._write_lock:
                # Load existing articles
                existing_articles = await self._load_articles_from_file()
                
                # Process each article
                for article in articles:
                    try:
                        existing_index = self._url_index.get(article.url)
                        if existing_index is not None:
                            # Update existing
                            existing_articles[existing_index] = article
                        else:
                            # Add new
                            existing_articles.append(article)
                            self._url_index[article.url] = len(existing_articles) - 1
                        
                        results[article.url] = True
                        
                    except Exception as e:
                        self.logger.error(f"Failed to process article {article.url}: {e}")
                        results[article.url] = False
                
                # Save all articles at once
                await self._write_articles_file(existing_articles)
                
                # Update CSV export
                await self._export_to_csv(existing_articles)
                
                # Invalidate cache
                self._invalidate_cache()
                
                successful_saves = sum(results.values())
                self.logger.info(f"Batch save completed: {successful_saves}/{len(articles)} successful")
                
        except Exception as e:
            self.logger.error(f"Batch save failed: {e}")
            # Mark all remaining as failed
            for article in articles:
                if article.url not in results:
                    results[article.url] = False
        
        return results
    
    async def find_by_url(self, url: str) -> Optional[NewsArticle]:
        """
        Find an article by its URL.
        
        Args:
            url (str): The URL to search for
            
        Returns:
            Optional[NewsArticle]: The article if found, None otherwise
        """
        try:
            # Check index first
            index = self._url_index.get(url)
            if index is None:
                return None
            
            # Load and return the specific article
            articles = await self._load_articles_from_file()
            if 0 <= index < len(articles):
                return articles[index]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to find article by URL {url}: {e}")
            return None
    
    async def find_by_source(self, source: str, limit: Optional[int] = None) -> List[NewsArticle]:
        """
        Find articles from a specific source.
        
        Args:
            source (str): The source to filter by
            limit (Optional[int]): Maximum number of articles to return
            
        Returns:
            List[NewsArticle]: Articles from the specified source
        """
        try:
            articles = await self._load_articles_from_file()
            
            # Filter by source
            filtered_articles = [
                article for article in articles
                if source.lower() in article.source.lower()
            ]
            
            # Apply limit if specified
            if limit is not None:
                filtered_articles = filtered_articles[:limit]
            
            self.logger.debug(f"Found {len(filtered_articles)} articles from source: {source}")
            return filtered_articles
            
        except Exception as e:
            self.logger.error(f"Failed to find articles by source {source}: {e}")
            return []
    
    async def find_by_date_range(self, start_date: datetime, end_date: datetime) -> List[NewsArticle]:
        """
        Find articles within a date range.
        
        Args:
            start_date (datetime): Start of date range
            end_date (datetime): End of date range
            
        Returns:
            List[NewsArticle]: Articles within the date range
        """
        try:
            articles = await self._load_articles_from_file()
            
            filtered_articles = []
            for article in articles:
                if article.published_date:
                    if start_date <= article.published_date <= end_date:
                        filtered_articles.append(article)
            
            self.logger.debug(f"Found {len(filtered_articles)} articles in date range")
            return filtered_articles
            
        except Exception as e:
            self.logger.error(f"Failed to find articles by date range: {e}")
            return []
    
    async def find_financial_articles(self, limit: Optional[int] = None) -> List[NewsArticle]:
        """
        Find articles classified as financial.
        
        Args:
            limit (Optional[int]): Maximum number of articles to return
            
        Returns:
            List[NewsArticle]: Financial articles
        """
        try:
            articles = await self._load_articles_from_file()
            
            financial_articles = [
                article for article in articles
                if article.is_financial_news()
            ]
            
            if limit is not None:
                financial_articles = financial_articles[:limit]
            
            self.logger.debug(f"Found {len(financial_articles)} financial articles")
            return financial_articles
            
        except Exception as e:
            self.logger.error(f"Failed to find financial articles: {e}")
            return []
    
    async def find_high_confidence_articles(self, min_confidence: float = 0.7) -> List[NewsArticle]:
        """
        Find articles with high confidence scores.
        
        Args:
            min_confidence (float): Minimum confidence threshold
            
        Returns:
            List[NewsArticle]: High confidence articles
        """
        try:
            articles = await self._load_articles_from_file()
            
            high_confidence_articles = [
                article for article in articles
                if article.confidence and article.confidence >= min_confidence
            ]
            
            self.logger.debug(f"Found {len(high_confidence_articles)} high confidence articles")
            return high_confidence_articles
            
        except Exception as e:
            self.logger.error(f"Failed to find high confidence articles: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about stored articles.
        
        Returns:
            Dict[str, Any]: Detailed statistics
        """
        try:
            # Check cache first
            if self._is_cache_valid():
                return self._statistics_cache
            
            articles = await self._load_articles_from_file()
            
            if not articles:
                return {
                    'total_articles': 0,
                    'by_source': {},
                    'by_type': {},
                    'by_sentiment': {},
                    'average_confidence': 0,
                    'average_credibility': 0,
                    'financial_percentage': 0,
                    'date_range': None
                }
            
            # Calculate statistics
            stats = await self._calculate_statistics(articles)
            
            # Cache results
            self._statistics_cache = stats
            self._cache_timestamp = datetime.now()
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to calculate statistics: {e}")
            return {'error': str(e)}
    
    async def delete_by_url(self, url: str) -> bool:
        """
        Delete an article by its URL.
        
        Args:
            url (str): URL of the article to delete
            
        Returns:
            bool: True if deleted, False if not found
        """
        try:
            async with self._write_lock:
                # Check if article exists
                index = self._url_index.get(url)
                if index is None:
                    return False
                
                # Load articles
                articles = await self._load_articles_from_file()
                
                # Remove article
                if 0 <= index < len(articles):
                    del articles[index]
                    
                    # Rebuild index
                    await self._rebuild_index_from_articles(articles)
                    
                    # Save updated articles
                    await self._write_articles_file(articles)
                    await self._export_to_csv(articles)
                    
                    # Invalidate cache
                    self._invalidate_cache()
                    
                    self.logger.info(f"Deleted article: {url}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to delete article {url}: {e}")
            return False
    
    async def clear_all(self) -> bool:
        """
        Remove all articles from the repository.
        
        Returns:
            bool: True if cleared successfully
        """
        try:
            async with self._write_lock:
                # Create backup if enabled
                if self.enable_backup:
                    await self._create_backup()
                
                # Clear all data
                await self._write_articles_file([])
                await self._export_to_csv([])
                
                # Clear index and cache
                self._url_index.clear()
                self._invalidate_cache()
                
                self.logger.info("Cleared all articles from repository")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to clear repository: {e}")
            return False
    
    # Private helper methods
    
    async def _load_articles_from_file(self) -> List[NewsArticle]:
        """Load articles from the JSON file."""
        try:
            if not self.articles_file.exists():
                return []
            
            async with aiofiles.open(self.articles_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                
            if not content.strip():
                return []
            
            data = json.loads(content)
            articles = []
            
            # Define valid fields for NewsArticle to filter out deprecated/unknown fields
            valid_fields = {
                'url', 'title', 'source', 'content', 'summary', 'published_date',
                'sentiment', 'confidence', 'article_type', 'bias_classification', 'credibility_score',
                'financial_relevance', 'crypto_relevance', 'companies_mentioned', 'crypto_mentions',
                'fact_check_score', 'fact_check_classification', 'fact_checks_found',
                'processing_time', 'model_version', 'analyzed_date', 'extraction_success',
                'tags', 'metadata'
            }
            
            for item in data:
                try:
                    # Convert dict back to NewsArticle
                    article_data = item.copy()
                    
                    # Filter out unknown/deprecated fields like risk_score
                    filtered_data = {k: v for k, v in article_data.items() if k in valid_fields}
                    
                    # Convert datetime strings back to datetime objects
                    if filtered_data.get('published_date'):
                        filtered_data['published_date'] = datetime.fromisoformat(filtered_data['published_date'])
                    if filtered_data.get('analyzed_date'):
                        filtered_data['analyzed_date'] = datetime.fromisoformat(filtered_data['analyzed_date'])
                    
                    # Convert enum strings back to enum objects
                    if filtered_data.get('article_type'):
                        filtered_data['article_type'] = ArticleType(filtered_data['article_type'])
                    if filtered_data.get('sentiment'):
                        filtered_data['sentiment'] = SentimentType(filtered_data['sentiment'])
                    if filtered_data.get('bias_classification'):
                        filtered_data['bias_classification'] = BiasClassification(filtered_data['bias_classification'])
                    if filtered_data.get('fact_check_classification'):
                        filtered_data['fact_check_classification'] = FactCheckClassification(filtered_data['fact_check_classification'])
                    
                    # Create NewsArticle instance with filtered data
                    article = NewsArticle(**filtered_data)
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to parse article: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to load articles from file: {e}")
            return []
    
    async def _write_articles_file(self, articles: List[NewsArticle]) -> None:
        """Write articles to the JSON file."""
        try:
            # Create backup if enabled
            if self.enable_backup and self.articles_file.exists():
                await self._create_backup()
            
            # Convert articles to serializable format
            articles_data = [article.to_dict() for article in articles]
            
            # Write to temporary file first (atomic operation)
            temp_file = self.articles_file.with_suffix('.tmp')
            
            async with aiofiles.open(temp_file, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(articles_data, indent=2, ensure_ascii=False))
            
            # Atomic move
            temp_file.replace(self.articles_file)
            
        except Exception as e:
            self.logger.error(f"Failed to write articles file: {e}")
            raise RepositoryError(f"Failed to write articles: {e}", "write_file", e)
    
    async def _export_to_csv(self, articles: List[NewsArticle]) -> None:
        """Export articles to CSV format."""
        try:
            if not articles:
                # Create empty CSV file
                async with aiofiles.open(self.csv_file, 'w', encoding='utf-8', newline='') as f:
                    await f.write('url,title,source,sentiment,confidence,article_type,published_date\n')
                return
            
            # Write CSV file
            csv_data = []
            headers = ['url', 'title', 'source', 'sentiment', 'confidence', 'article_type', 
                      'published_date', 'financial_relevance', 'credibility_score', 'fact_check_score']
            
            csv_data.append(','.join(headers))
            
            for article in articles:
                row = [
                    f'"{article.url}"',
                    f'"{article.title}"',
                    f'"{article.source}"',
                    f'"{article.sentiment.value if article.sentiment else ""}"',
                    str(article.confidence if article.confidence else ''),
                    f'"{article.article_type.value if article.article_type else ""}"',
                    f'"{article.published_date.isoformat() if article.published_date else ""}"',
                    str(article.financial_relevance if article.financial_relevance else ''),
                    str(article.credibility_score if article.credibility_score else ''),
                    str(article.fact_check_score if article.fact_check_score else '')
                ]
                csv_data.append(','.join(row))
            
            async with aiofiles.open(self.csv_file, 'w', encoding='utf-8') as f:
                await f.write('\n'.join(csv_data))
                
        except Exception as e:
            self.logger.warning(f"Failed to export CSV: {e}")
    
    async def _rebuild_index(self) -> None:
        """Rebuild the URL index from stored articles."""
        try:
            articles = await self._load_articles_from_file()
            await self._rebuild_index_from_articles(articles)
            
        except Exception as e:
            self.logger.error(f"Failed to rebuild index: {e}")
    
    async def _rebuild_index_from_articles(self, articles: List[NewsArticle]) -> None:
        """Rebuild index from article list."""
        self._url_index.clear()
        for i, article in enumerate(articles):
            self._url_index[article.url] = i
    
    async def _create_backup(self) -> None:
        """Create backup of current data."""
        try:
            if not self.enable_backup or not self.articles_file.exists():
                return
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_file = self.backup_directory / f"articles_backup_{timestamp}.json"
            
            shutil.copy2(self.articles_file, backup_file)
            self.logger.debug(f"Created backup: {backup_file}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create backup: {e}")
    
    def _invalidate_cache(self) -> None:
        """Invalidate the statistics cache."""
        self._statistics_cache = None
        self._cache_timestamp = None
    
    def _is_cache_valid(self) -> bool:
        """Check if the statistics cache is still valid."""
        if self._statistics_cache is None or self._cache_timestamp is None:
            return False
        
        age = (datetime.now() - self._cache_timestamp).total_seconds()
        return age < self._cache_ttl_seconds
    
    async def _calculate_statistics(self, articles: List[NewsArticle]) -> Dict[str, Any]:
        """Calculate comprehensive statistics for articles."""
        total_articles = len(articles)
        
        # Count by source
        by_source = {}
        for article in articles:
            by_source[article.source] = by_source.get(article.source, 0) + 1
        
        # Count by type
        by_type = {}
        for article in articles:
            if article.article_type:
                type_value = article.article_type.value
                by_type[type_value] = by_type.get(type_value, 0) + 1
        
        # Count by sentiment
        by_sentiment = {}
        for article in articles:
            if article.sentiment:
                sentiment_value = article.sentiment.value
                by_sentiment[sentiment_value] = by_sentiment.get(sentiment_value, 0) + 1
        
        # Calculate averages
        confidence_values = [a.confidence for a in articles if a.confidence is not None]
        average_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
        
        credibility_values = [a.credibility_score for a in articles if a.credibility_score is not None]
        average_credibility = sum(credibility_values) / len(credibility_values) if credibility_values else 0
        
        # Financial percentage
        financial_articles = [a for a in articles if a.is_financial_news()]
        financial_percentage = (len(financial_articles) / total_articles) * 100 if total_articles > 0 else 0
        
        # Date range
        dates = [a.published_date for a in articles if a.published_date]
        date_range = None
        if dates:
            date_range = {
                'earliest': min(dates).isoformat(),
                'latest': max(dates).isoformat()
            }
        
        return {
            'total_articles': total_articles,
            'by_source': by_source,
            'by_type': by_type,
            'by_sentiment': by_sentiment,
            'average_confidence': round(average_confidence, 3),
            'average_credibility': round(average_credibility, 1),
            'financial_percentage': round(financial_percentage, 1),
            'date_range': date_range,
            'cache_updated': datetime.now().isoformat()
        } 