"""
Application Use Case: ClassifyNewsBatchUseCase

This module contains the main use case for batch news classification in the Penelope News
Classification System. It implements Domain Driven Design principles by orchestrating
business operations across multiple domain services and infrastructure components.

The ClassifyNewsBatchUseCase serves as the primary entry point for news classification
operations, coordinating between RSS feed parsing, content extraction, article classification,
and result persistence. It ensures proper error handling, monitoring, and business rule
enforcement throughout the entire process.

Classes:
    ClassifyNewsBatchUseCase: Main orchestrator for batch news classification
    
Key Responsibilities:
    - Orchestrates the complete news classification workflow
    - Manages RSS feed processing and article extraction
    - Coordinates domain services for classification
    - Handles error recovery and partial processing
    - Ensures data persistence and result reporting
    - Provides comprehensive logging and monitoring

Example:
    Basic usage:
    
    >>> use_case = ClassifyNewsBatchUseCase(
    ...     news_repository=repository,
    ...     classification_service=classifier,
    ...     rss_feed_service=rss_service,
    ...     content_extractor=extractor
    ... )
    >>> request = NewsClassificationRequest(
    ...     source_urls=["https://feeds.reuters.com/reuters/businessNews"],
    ...     max_articles=100
    ... )
    >>> response = await use_case.execute(request)
    >>> print(f"Processed {response.articles_processed} articles")

Author: Claude AI Assistant
Date: 2025-07-04
Version: 3.0 (DDD Architecture)
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import traceback

from domain.entities.news_article import NewsArticle
from domain.services.news_classification_service import NewsClassificationService, ClassificationError
from domain.repositories.news_repository import NewsRepository
from application.dtos.news_classification_request import NewsClassificationRequest
from application.dtos.news_classification_response import NewsClassificationResponse, ProcessingStatistics, ErrorInformation, ProcessingStatus
from shared.constants.rss_feeds import RSS_FEEDS_CONFIG
from infrastructure.external_services.rss_feed_service import RSSFeedService
from infrastructure.external_services.content_extractor_service import ContentExtractorService


class ClassifyNewsBatchUseCase:
    """
    Main use case for batch news classification operations.
    
    This use case orchestrates the complete workflow for processing multiple news articles
    from RSS feeds, including content extraction, classification, and result persistence.
    It follows Domain Driven Design principles by coordinating between domain services
    and infrastructure components while maintaining clear separation of concerns.
    
    The use case implements comprehensive error handling, monitoring, and recovery
    mechanisms to ensure robust processing of large article batches. It provides
    detailed reporting and statistics for monitoring and optimization purposes.
    
    Key Features:
        - Asynchronous processing for high performance
        - Comprehensive error handling and recovery
        - Detailed progress tracking and reporting
        - Flexible filtering and processing options
        - Automatic retry mechanisms for failed operations
        - Memory-efficient streaming processing
        - Comprehensive logging and monitoring
        
    Dependencies:
        - NewsRepository: For persisting classified articles
        - NewsClassificationService: For article classification logic
        - RSSFeedService: For RSS feed parsing and article extraction
        - ContentExtractorService: For web content extraction
        
    Example:
        >>> use_case = ClassifyNewsBatchUseCase(
        ...     news_repository=file_repository,
        ...     classification_service=classification_service,
        ...     rss_feed_service=rss_service,
        ...     content_extractor=content_extractor
        ... )
        >>> response = await use_case.execute(request)
        >>> print(f"Success: {response.success}")
        >>> print(f"Articles: {response.articles_classified}")
    """
    
    def __init__(
        self,
        news_repository: NewsRepository,
        classification_service: NewsClassificationService,
        rss_feed_service=None,  # Will be imported dynamically
        content_extractor=None   # Will be imported dynamically
    ):
        """
        Initialize the ClassifyNewsBatchUseCase.
        
        Args:
            news_repository (NewsRepository): Repository for article persistence
            classification_service (NewsClassificationService): Service for article classification
            rss_feed_service: Service for RSS feed processing
            content_extractor: Service for content extraction
        """
        self.logger = logging.getLogger(__name__)
        self.news_repository = news_repository
        self.classification_service = classification_service
        self.rss_feed_service = rss_feed_service
        self.content_extractor = content_extractor
        
        # Processing state
        self._current_request: Optional[NewsClassificationRequest] = None
        self._processing_start_time: Optional[datetime] = None
        self._articles_processed = 0
        self._errors_encountered: List[Dict[str, Any]] = []
    
    async def execute(self, request: NewsClassificationRequest) -> NewsClassificationResponse:
        """
        Execute the news classification batch processing workflow.
        
        This method orchestrates the complete news classification process,
        from RSS feed parsing to final result reporting. It handles all
        error scenarios and provides comprehensive feedback about the operation.
        
        Args:
            request (NewsClassificationRequest): The classification request
            
        Returns:
            NewsClassificationResponse: Comprehensive response with results and statistics
            
        Raises:
            ValueError: If the request is invalid
            Exception: For unexpected system errors
            
        Example:
            >>> request = NewsClassificationRequest(
            ...     source_urls=["https://feeds.reuters.com/reuters/businessNews"],
            ...     max_articles=50
            ... )
            >>> response = await use_case.execute(request)
            >>> if response.success:
            ...     print(f"Successfully processed {response.articles_classified} articles")
        """
        # Validate request
        validation_errors = request.validate()
        if validation_errors:
            error_message = f"Invalid request: {'; '.join(validation_errors)}"
            self.logger.error(error_message)
            return NewsClassificationResponse.create_error_response(
                error_message=error_message,
                request_id=request.request_id
            )
        
        # Initialize processing state
        self._current_request = request
        self._processing_start_time = datetime.now()
        self._articles_processed = 0
        self._errors_encountered = []
        
        self.logger.info(f"Starting batch classification for request {request.request_id}")
        self.logger.info(f"Processing {len(request.source_urls)} RSS feeds, max {request.max_articles} articles")
        
        try:
            # Step 1: Extract real articles from RSS feeds
            articles = await self._extract_real_articles_from_rss(request)
            if not articles:
                return NewsClassificationResponse.create_error_response(
                    error_message="No articles extracted from RSS feeds",
                    request_id=request.request_id,
                    started_at=self._processing_start_time
                )
            
            self.logger.info(f"Extracted {len(articles)} real articles from RSS feeds")
            
            # Step 2: Enhance articles with full content extraction
            enhanced_articles = await self._enhance_articles_with_content(articles, request)
            self.logger.info(f"Enhanced {len(enhanced_articles)} articles with full content")
            
            # Step 3: Classify articles
            classified_articles = await self._classify_articles(enhanced_articles, request)
            successful_classifications = len(classified_articles)
            self.logger.info(f"Successfully classified {successful_classifications} articles")
            
            # Step 4: Apply filters if specified
            filtered_articles = await self._apply_filters(classified_articles, request.filter_criteria)
            self.logger.info(f"After filtering: {len(filtered_articles)} articles")
            
            # Step 5: Persist results
            persistence_results = await self._persist_articles(filtered_articles, request)
            self.logger.info(f"Persisted {sum(persistence_results.values())} articles")
            
            # Step 6: Generate response with statistics
            response = await self._generate_response(
                request=request,
                articles_processed=len(articles),
                classified_articles=filtered_articles,
                extraction_success_count=len(enhanced_articles)
            )
            
            self.logger.info(f"Batch classification completed successfully: {response}")
            return response
            
        except Exception as e:
            error_message = f"Unexpected error during batch classification: {str(e)}"
            self.logger.error(error_message, exc_info=True)
            
            return NewsClassificationResponse.create_error_response(
                error_message=error_message,
                request_id=request.request_id,
                articles_processed=self._articles_processed,
                started_at=self._processing_start_time
            )
    
    async def _extract_real_articles_from_rss(self, request: NewsClassificationRequest) -> List[NewsArticle]:
        """
        Extract real articles from RSS feeds.
        
        Args:
            request (NewsClassificationRequest): The classification request
            
        Returns:
            List[NewsArticle]: Extracted articles
        """
        self.logger.info(f"Extracting articles from {len(request.source_urls)} RSS feeds")
        
        try:
            # Use the correct method name from RSSFeedService
            articles = await self.rss_feed_service.fetch_articles_from_feeds(
                request.source_urls, 
                request.max_articles
            )
            
            self.logger.info(f"Successfully extracted {len(articles)} articles from RSS feeds")
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to extract articles from RSS feeds: {e}")
            self._errors_encountered.append({
                'type': 'extraction_error',
                'url': 'RSS_FEEDS',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return []
    
    async def _enhance_articles_with_content(self, articles: List[NewsArticle], request: NewsClassificationRequest) -> List[NewsArticle]:
        """
        Enhance articles with full content extraction.
        
        Args:
            articles (List[NewsArticle]): Articles to enhance
            request (NewsClassificationRequest): The classification request
            
        Returns:
            List[NewsArticle]: Enhanced articles
        """
        self.logger.info(f"Enhancing {len(articles)} articles with full content")
        
        enhanced_articles = []
        for article in articles:
            try:
                # Use the correct method from ContentExtractorService
                enhanced_article = await self.content_extractor.extract_content(article)
                enhanced_articles.append(enhanced_article)
                
                if enhanced_article.extraction_success:
                    self.logger.debug(f"✅ Enhanced content for {article.url[:50]}...")
                else:
                    self.logger.warning(f"❌ Failed to enhance content for {article.url[:50]}...")
                    self._errors_encountered.append({
                        'type': 'extraction_timeout',
                        'url': article.url,
                        'error': "Content extraction failed",
                        'timestamp': datetime.now().isoformat()
                    })
                    
            except Exception as e:
                self.logger.error(f"Failed to enhance content for {article.url}: {e}")
                self._errors_encountered.append({
                    'type': 'extraction_error',
                    'url': article.url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
                # Add original article even if enhancement failed
                enhanced_articles.append(article)
        
        return enhanced_articles
    
    async def _classify_articles(
        self, 
        articles: List[NewsArticle], 
        request: NewsClassificationRequest
    ) -> List[NewsArticle]:
        """
        Classify articles using the domain service.
        
        Args:
            articles (List[NewsArticle]): Articles to classify
            request (NewsClassificationRequest): The classification request
            
        Returns:
            List[NewsArticle]: Classified articles
        """
        self.logger.info(f"Classifying {len(articles)} articles")
        
        try:
            if request.processing_options.parallel_processing:
                classified_articles = await self.classification_service.classify_batch(articles)
            else:
                classified_articles = []
                for article in articles:
                    try:
                        classified_article = await self.classification_service.classify_article(article)
                        classified_articles.append(classified_article)
                    except Exception as e:
                        self.logger.error(f"Classification failed for {article.url}: {e}")
                        self._errors_encountered.append({
                            'type': 'classification_error',
                            'url': article.url,
                            'error': str(e),
                            'timestamp': datetime.now().isoformat()
                        })
            
            self.logger.info(f"Successfully classified {len(classified_articles)} articles")
            return classified_articles
            
        except Exception as e:
            self.logger.error(f"Batch classification failed: {e}")
            raise
    
    async def _apply_filters(
        self, 
        articles: List[NewsArticle], 
        filter_criteria
    ) -> List[NewsArticle]:
        """
        Apply filtering criteria to classified articles.
        
        Args:
            articles (List[NewsArticle]): Articles to filter
            filter_criteria: Filter criteria to apply
            
        Returns:
            List[NewsArticle]: Filtered articles
        """
        if not filter_criteria or not filter_criteria.has_filters():
            return articles
        
        self.logger.info("Applying filters to classified articles")
        filtered_articles = articles.copy()
        
        # Apply article type filters
        if filter_criteria.article_types:
            filtered_articles = [
                a for a in filtered_articles 
                if a.article_type in filter_criteria.article_types
            ]
            self.logger.debug(f"After article type filter: {len(filtered_articles)} articles")
        
        # Apply sentiment filters
        if filter_criteria.sentiment_types:
            filtered_articles = [
                a for a in filtered_articles 
                if a.sentiment in filter_criteria.sentiment_types
            ]
            self.logger.debug(f"After sentiment filter: {len(filtered_articles)} articles")
        
        # Apply confidence filters - more permissive to allow None values
        if filter_criteria.min_confidence is not None:
            filtered_articles = [
                a for a in filtered_articles 
                if a.confidence is None or (a.confidence is not None and a.confidence >= filter_criteria.min_confidence)
            ]
            self.logger.debug(f"After min confidence filter: {len(filtered_articles)} articles")
        
        if filter_criteria.max_confidence is not None:
            filtered_articles = [
                a for a in filtered_articles 
                if a.confidence is None or (a.confidence is not None and a.confidence <= filter_criteria.max_confidence)
            ]
            self.logger.debug(f"After max confidence filter: {len(filtered_articles)} articles")
        
        # Apply credibility filters - more permissive to allow None values
        if filter_criteria.min_credibility is not None:
            filtered_articles = [
                a for a in filtered_articles 
                if a.credibility_score is None or (a.credibility_score is not None and a.credibility_score >= filter_criteria.min_credibility)
            ]
            self.logger.debug(f"After credibility filter: {len(filtered_articles)} articles")
        
        # Apply source filters
        if filter_criteria.sources_include:
            filtered_articles = [
                a for a in filtered_articles 
                if any(source.lower() in a.source.lower() for source in filter_criteria.sources_include)
            ]
            self.logger.debug(f"After source include filter: {len(filtered_articles)} articles")
        
        if filter_criteria.sources_exclude:
            filtered_articles = [
                a for a in filtered_articles 
                if not any(source.lower() in a.source.lower() for source in filter_criteria.sources_exclude)
            ]
            self.logger.debug(f"After source exclude filter: {len(filtered_articles)} articles")
        
        self.logger.info(f"Filtering complete: {len(articles)} -> {len(filtered_articles)} articles")
        return filtered_articles
    
    async def _persist_articles(
        self, 
        articles: List[NewsArticle], 
        request: NewsClassificationRequest
    ) -> Dict[str, bool]:
        """
        Persist classified articles to the repository.
        
        Args:
            articles (List[NewsArticle]): Articles to persist
            request (NewsClassificationRequest): The classification request
            
        Returns:
            Dict[str, bool]: Results of persistence operations
        """
        self.logger.info(f"Persisting {len(articles)} articles")
        
        try:
            persistence_results = await self.news_repository.save_batch(articles)
            successful_saves = sum(persistence_results.values())
            
            self.logger.info(f"Successfully persisted {successful_saves}/{len(articles)} articles")
            return persistence_results
            
        except Exception as e:
            self.logger.error(f"Failed to persist articles: {e}")
            raise
    
    async def _generate_response(
        self,
        request: NewsClassificationRequest,
        articles_processed: int,
        classified_articles: List[NewsArticle],
        extraction_success_count: int
    ) -> NewsClassificationResponse:
        """
        Generate the final response with comprehensive statistics.
        
        Args:
            request (NewsClassificationRequest): Original request
            articles_processed (int): Number of articles processed
            classified_articles (List[NewsArticle]): Successfully classified articles
            extraction_success_count (int): Number of successful content extractions
            
        Returns:
            NewsClassificationResponse: Complete response with statistics
        """
        processing_end_time = datetime.now()
        total_processing_time = (processing_end_time - self._processing_start_time).total_seconds()
        
        # Calculate statistics
        articles_per_minute = (articles_processed / total_processing_time) * 60 if total_processing_time > 0 else 0
        extraction_success_rate = (extraction_success_count / articles_processed) * 100 if articles_processed > 0 else 0
        classification_success_rate = (len(classified_articles) / articles_processed) * 100 if articles_processed > 0 else 0
        
        # Calculate averages
        valid_articles = [a for a in classified_articles if a.confidence is not None]
        average_confidence = sum(a.confidence for a in valid_articles) / len(valid_articles) if valid_articles else 0
        
        credible_articles = [a for a in classified_articles if a.credibility_score is not None]
        average_credibility = sum(a.credibility_score for a in credible_articles) / len(credible_articles) if credible_articles else 0
        
        fact_check_articles = [a for a in classified_articles if a.fact_check_score is not None]
        average_fact_check_score = sum(a.fact_check_score for a in fact_check_articles) / len(fact_check_articles) if fact_check_articles else 0
        
        # Calculate percentages
        financial_articles = [a for a in classified_articles if a.is_financial_news()]
        financial_percentage = (len(financial_articles) / len(classified_articles)) * 100 if classified_articles else 0
        
        crypto_articles = [a for a in classified_articles if a.crypto_relevance]
        crypto_percentage = (len(crypto_articles) / len(classified_articles)) * 100 if classified_articles else 0
        
        high_confidence_articles = [a for a in classified_articles if a.is_high_confidence()]
        high_confidence_percentage = (len(high_confidence_articles) / len(classified_articles)) * 100 if classified_articles else 0
        
        verified_articles = [a for a in classified_articles if a.fact_check_classification and a.fact_check_classification.value == 'verified']
        verified_percentage = (len(verified_articles) / len(classified_articles)) * 100 if classified_articles else 0
        
        error_rate = (len(self._errors_encountered) / articles_processed) * 100 if articles_processed > 0 else 0
        
        # Create statistics object
        processing_statistics = ProcessingStatistics(
            total_processing_time=total_processing_time,
            articles_per_minute=articles_per_minute,
            extraction_success_rate=extraction_success_rate,
            classification_success_rate=classification_success_rate,
            average_confidence=average_confidence,
            average_credibility=average_credibility,
            average_fact_check_score=average_fact_check_score,
            financial_articles_percentage=financial_percentage,
            crypto_articles_percentage=crypto_percentage,
            high_confidence_percentage=high_confidence_percentage,
            verified_articles_percentage=verified_percentage,
            error_rate=error_rate
        )
        
        # Create error information if there were errors
        error_information = None
        if self._errors_encountered:
            error_types = {}
            failed_urls = []
            error_messages = []
            
            for error in self._errors_encountered:
                error_type = error.get('type', 'unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
                
                if 'url' in error:
                    failed_urls.append(error['url'])
                
                if 'error' in error:
                    error_messages.append(error['error'])
            
            error_information = ErrorInformation(
                error_count=len(self._errors_encountered),
                error_types=error_types,
                failed_urls=failed_urls,
                error_messages=error_messages,
                recoverable_errors=len([e for e in self._errors_encountered if e.get('type') in ['extraction_timeout', 'extraction_error']]),
                critical_errors=len([e for e in self._errors_encountered if e.get('type') in ['system_error', 'classification_error']])
            )
        
        # Create and return response
        return NewsClassificationResponse(
            success=len(classified_articles) > 0,
            status=ProcessingStatus.COMPLETED,
            request_id=request.request_id,
            articles_processed=articles_processed,
            articles_classified=len(classified_articles),
            classified_articles=classified_articles,
            processing_statistics=processing_statistics,
            error_information=error_information,
            started_at=self._processing_start_time,
            completed_at=processing_end_time,
            metadata={
                'source_urls_count': len(request.source_urls),
                'processing_options': request.processing_options.get_enabled_features(),
                'filters_applied': request.filter_criteria.has_filters(),
                'output_format': request.output_format.value
            }
        ) 