"""
Comprehensive Test Suite: Penelope News Classification System

This module contains comprehensive tests for the Penelope News Classification System,
validating all layers of the Domain Driven Design architecture. The tests cover
unit tests, integration tests, and end-to-end validation of the complete system.

Test Coverage:
    - Domain Layer: Entities, repositories, and services
    - Application Layer: Use cases and DTOs
    - Infrastructure Layer: External services and persistence
    - System Integration: End-to-end workflows

Key Features:
    - Comprehensive test coverage for all components
    - Mock external dependencies for isolated testing
    - Performance and load testing capabilities
    - Error handling and edge case validation
    - Documentation and example generation

Usage:
    python -m pytest test_ddd_system.py -v
    python test_ddd_system.py  # Run directly
"""

import asyncio
import pytest
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any

# Domain layer imports
from domain.entities.news_article import (
    NewsArticle, 
    ArticleType, 
    SentimentType, 
    BiasClassification,
    FactCheckClassification
)
from domain.services.news_classification_service import NewsClassificationService

# Application layer imports
from application.use_cases.classify_news_batch_use_case import ClassifyNewsBatchUseCase
from application.dtos.news_classification_request import (
    NewsClassificationRequest,
    ProcessingOptions,
    FilterCriteria,
    OutputFormat,
    ProcessingPriority
)
from application.dtos.news_classification_response import (
    NewsClassificationResponse,
    ProcessingStatistics,
    ErrorInformation,
    ProcessingStatus
)

# Infrastructure layer imports
from infrastructure.repositories.file_news_repository import FileNewsRepository

# Shared imports
from shared.utils.logger_config import setup_logging


class TestNewsArticleEntity:
    """Test suite for the NewsArticle domain entity."""
    
    def test_news_article_creation(self):
        """Test basic NewsArticle creation and validation."""
        article = NewsArticle(
            url="https://example.com/article1",
            title="Test Article",
            source="Test Source",
            content="Test content about financial markets and trading.",
            published_date=datetime.now()
        )
        
        assert article.url == "https://example.com/article1"
        assert article.title == "Test Article"
        assert article.source == "Test Source"
        assert article.content is not None
        assert article.published_date is not None
        assert article.analyzed_date is None  # Not yet analyzed
    
    def test_financial_news_detection(self):
        """Test financial news detection business logic."""
        # Financial article
        financial_article = NewsArticle(
            url="https://example.com/financial",
            title="Stock Market Report",
            source="Financial Times",
            content="Market analysis shows strong performance in tech stocks.",
            financial_relevance=85.5
        )
        
        assert financial_article.is_financial_news()
        
        # Non-financial article
        non_financial_article = NewsArticle(
            url="https://example.com/sports",
            title="Football Match Results",
            source="Sports News",
            content="Team A defeated Team B in yesterday's match.",
            financial_relevance=15.2
        )
        
        assert not non_financial_article.is_financial_news()
    
    def test_high_confidence_detection(self):
        """Test high confidence detection business logic."""
        # High confidence article
        high_confidence_article = NewsArticle(
            url="https://example.com/reliable",
            title="Breaking News",
            source="Reuters",
            content="Official statement from government spokesperson.",
            confidence=0.85
        )
        
        assert high_confidence_article.is_high_confidence()
        
        # Low confidence article
        low_confidence_article = NewsArticle(
            url="https://example.com/unreliable",
            title="Rumor Report",
            source="Anonymous Blog",
            content="Unconfirmed reports suggest...",
            confidence=0.45
        )
        
        assert not low_confidence_article.is_high_confidence()
    
    def test_credible_source_detection(self):
        """Test credible source detection business logic."""
        # Credible source
        credible_article = NewsArticle(
            url="https://reuters.com/article",
            title="News Article",
            source="Reuters",
            content="Official news content.",
            credibility_score=90.0
        )
        
        assert credible_article.is_credible_source()
        
        # Non-credible source
        non_credible_article = NewsArticle(
            url="https://blog.com/post",
            title="Opinion Piece",
            source="Personal Blog",
            content="Personal opinion on current events.",
            credibility_score=30.0
        )
        
        assert not non_credible_article.is_credible_source()
    
    def test_risk_score_calculation(self):
        """Test risk score calculation business logic."""
        article = NewsArticle(
            url="https://example.com/test",
            title="Test Article",
            source="Test Source",
            content="Test content",
            confidence=0.6,
            credibility_score=70.0,
            bias_score=0.3,
            fact_check_score=60.0
        )
        
        risk_score = article.get_risk_score()
        
        # Risk score should be calculated based on confidence, credibility, bias, and fact-check
        assert isinstance(risk_score, float)
        assert 0.0 <= risk_score <= 100.0
    
    def test_article_serialization(self):
        """Test article serialization to dictionary."""
        article = NewsArticle(
            url="https://example.com/article",
            title="Test Article",
            source="Test Source",
            content="Test content",
            published_date=datetime(2024, 1, 1, 12, 0, 0),
            article_type=ArticleType.FINANCIAL,
            sentiment=SentimentType.POSITIVE,
            confidence=0.85,
            credibility_score=90.0
        )
        
        article_dict = article.to_dict()
        
        assert article_dict['url'] == "https://example.com/article"
        assert article_dict['title'] == "Test Article"
        assert article_dict['source'] == "Test Source"
        assert article_dict['article_type'] == "financial"
        assert article_dict['sentiment'] == "positive"
        assert article_dict['confidence'] == 0.85
        assert article_dict['credibility_score'] == 90.0


class TestNewsClassificationService:
    """Test suite for the NewsClassificationService domain service."""
    
    def setup_method(self):
        """Set up test environment."""
        self.classification_service = NewsClassificationService()
    
    async def test_classify_article(self):
        """Test single article classification."""
        article = NewsArticle(
            url="https://example.com/financial-news",
            title="Stock Market Rises on Positive Economic Data",
            source="Financial Times",
            content="The stock market showed strong gains today following the release of positive economic indicators. Tech stocks led the rally with significant increases across major indices.",
            published_date=datetime.now()
        )
        
        classified_article = await self.classification_service.classify_article(article)
        
        # Verify classification results
        assert classified_article.analyzed_date is not None
        assert classified_article.confidence is not None
        assert classified_article.sentiment is not None
        assert classified_article.article_type is not None
        assert classified_article.credibility_score is not None
        assert classified_article.fact_check_score is not None
        assert classified_article.bias_classification is not None
        
        # Verify financial relevance detection
        assert classified_article.financial_relevance is not None
        assert classified_article.financial_relevance > 50  # Should detect financial content
    
    async def test_classify_batch(self):
        """Test batch article classification."""
        articles = [
            NewsArticle(
                url=f"https://example.com/article-{i}",
                title=f"Test Article {i}",
                source="Test Source",
                content=f"Test content {i} about financial markets and trading.",
                published_date=datetime.now()
            )
            for i in range(5)
        ]
        
        classified_articles = await self.classification_service.classify_batch(articles)
        
        assert len(classified_articles) == 5
        
        for article in classified_articles:
            assert article.analyzed_date is not None
            assert article.confidence is not None
            assert article.sentiment is not None
            assert article.article_type is not None
    
    async def test_sentiment_analysis(self):
        """Test sentiment analysis functionality."""
        # Positive sentiment
        positive_article = NewsArticle(
            url="https://example.com/positive",
            title="Great Economic Growth Reported",
            source="Economic Times",
            content="The economy shows excellent growth with rising employment and strong consumer confidence.",
            published_date=datetime.now()
        )
        
        classified_positive = await self.classification_service.classify_article(positive_article)
        # Note: In a real implementation, this would likely be positive
        assert classified_positive.sentiment is not None
        
        # Negative sentiment
        negative_article = NewsArticle(
            url="https://example.com/negative",
            title="Market Crash Causes Widespread Panic",
            source="Financial News",
            content="Stock markets plummeted today amid fears of economic recession and widespread investor panic.",
            published_date=datetime.now()
        )
        
        classified_negative = await self.classification_service.classify_article(negative_article)
        assert classified_negative.sentiment is not None
    
    async def test_bias_detection(self):
        """Test bias detection functionality."""
        article = NewsArticle(
            url="https://example.com/news",
            title="Political News Article",
            source="News Source",
            content="This is a news article that may contain some bias in its reporting.",
            published_date=datetime.now()
        )
        
        classified_article = await self.classification_service.classify_article(article)
        
        assert classified_article.bias_classification is not None
        assert classified_article.bias_score is not None
        assert 0.0 <= classified_article.bias_score <= 1.0
    
    async def test_fact_checking(self):
        """Test fact-checking functionality."""
        article = NewsArticle(
            url="https://reuters.com/article",
            title="Official Government Statement",
            source="Reuters",
            content="Government spokesperson announced new policy changes in official press conference.",
            published_date=datetime.now()
        )
        
        classified_article = await self.classification_service.classify_article(article)
        
        assert classified_article.fact_check_score is not None
        assert classified_article.fact_check_classification is not None
        assert 0.0 <= classified_article.fact_check_score <= 100.0


class TestNewsClassificationRequest:
    """Test suite for NewsClassificationRequest DTO."""
    
    def test_valid_request_creation(self):
        """Test creating a valid classification request."""
        request = NewsClassificationRequest(
            source_urls=["https://feeds.reuters.com/reuters/businessNews"],
            max_articles=50,
            processing_options=ProcessingOptions(
                enable_sentiment_analysis=True,
                enable_fact_checking=True,
                parallel_processing=True
            ),
            output_format=OutputFormat.JSON,
            processing_priority=ProcessingPriority.NORMAL
        )
        
        assert request.is_valid()
        assert len(request.validate()) == 0
        assert request.source_urls == ["https://feeds.reuters.com/reuters/businessNews"]
        assert request.max_articles == 50
        assert request.processing_options.enable_sentiment_analysis
        assert request.processing_options.enable_fact_checking
    
    def test_invalid_request_validation(self):
        """Test validation of invalid requests."""
        # Empty source URLs
        invalid_request = NewsClassificationRequest(
            source_urls=[],
            max_articles=50
        )
        
        assert not invalid_request.is_valid()
        validation_errors = invalid_request.validate()
        assert len(validation_errors) > 0
        assert "At least one source URL must be provided" in validation_errors
        
        # Invalid max_articles
        invalid_request2 = NewsClassificationRequest(
            source_urls=["https://example.com"],
            max_articles=0
        )
        
        assert not invalid_request2.is_valid()
        validation_errors2 = invalid_request2.validate()
        assert "max_articles must be a positive integer" in validation_errors2
    
    def test_processing_time_estimation(self):
        """Test processing time estimation."""
        request = NewsClassificationRequest(
            source_urls=["https://example.com"],
            max_articles=100,
            processing_options=ProcessingOptions(
                enable_sentiment_analysis=True,
                enable_fact_checking=True,
                parallel_processing=True,
                max_concurrent_requests=10
            )
        )
        
        estimated_time = request.get_estimated_processing_time()
        
        assert isinstance(estimated_time, int)
        assert estimated_time > 0
    
    def test_request_serialization(self):
        """Test request serialization to dictionary."""
        request = NewsClassificationRequest(
            source_urls=["https://example.com"],
            max_articles=50,
            request_id="test-request-123"
        )
        
        request_dict = request.to_dict()
        
        assert request_dict['source_urls'] == ["https://example.com"]
        assert request_dict['max_articles'] == 50
        assert request_dict['request_id'] == "test-request-123"
        assert 'estimated_processing_time' in request_dict
        assert 'is_async' in request_dict
        assert 'is_high_priority' in request_dict


class TestNewsClassificationResponse:
    """Test suite for NewsClassificationResponse DTO."""
    
    def test_success_response_creation(self):
        """Test creating a successful response."""
        articles = [
            NewsArticle(
                url=f"https://example.com/article-{i}",
                title=f"Test Article {i}",
                source="Test Source",
                content=f"Test content {i}",
                confidence=0.8,
                credibility_score=85.0,
                fact_check_score=70.0
            )
            for i in range(10)
        ]
        
        processing_stats = ProcessingStatistics(
            total_processing_time=120.0,
            articles_per_minute=30.0,
            extraction_success_rate=95.0,
            classification_success_rate=100.0,
            average_confidence=0.8,
            average_credibility=85.0,
            average_fact_check_score=70.0,
            financial_articles_percentage=60.0,
            crypto_articles_percentage=10.0,
            high_confidence_percentage=80.0,
            verified_articles_percentage=70.0,
            error_rate=0.0
        )
        
        response = NewsClassificationResponse.create_success_response(
            articles_processed=10,
            classified_articles=articles,
            processing_statistics=processing_stats,
            request_id="test-request-123"
        )
        
        assert response.success
        assert response.status == ProcessingStatus.COMPLETED
        assert response.articles_processed == 10
        assert response.articles_classified == 10
        assert response.get_success_rate() == 100.0
        assert response.request_id == "test-request-123"
        assert response.is_complete()
        assert not response.has_errors()
    
    def test_error_response_creation(self):
        """Test creating an error response."""
        response = NewsClassificationResponse.create_error_response(
            error_message="Test error occurred",
            request_id="test-request-456",
            articles_processed=5
        )
        
        assert not response.success
        assert response.status == ProcessingStatus.FAILED
        assert response.articles_processed == 5
        assert response.articles_classified == 0
        assert response.get_success_rate() == 0.0
        assert response.request_id == "test-request-456"
        assert response.has_errors()
        assert response.has_critical_errors()
    
    def test_response_statistics_calculation(self):
        """Test response statistics calculation."""
        articles = [
            NewsArticle(
                url=f"https://example.com/article-{i}",
                title=f"Financial Article {i}",
                source="Reuters",
                content=f"Financial content {i}",
                confidence=0.9,
                credibility_score=90.0,
                financial_relevance=85.0,
                article_type=ArticleType.FINANCIAL
            )
            for i in range(5)
        ]
        
        processing_stats = ProcessingStatistics(
            total_processing_time=60.0,
            articles_per_minute=60.0,
            extraction_success_rate=100.0,
            classification_success_rate=100.0,
            average_confidence=0.9,
            average_credibility=90.0,
            average_fact_check_score=80.0,
            financial_articles_percentage=100.0,
            crypto_articles_percentage=0.0,
            high_confidence_percentage=100.0,
            verified_articles_percentage=80.0,
            error_rate=0.0
        )
        
        response = NewsClassificationResponse.create_success_response(
            articles_processed=5,
            classified_articles=articles,
            processing_statistics=processing_stats
        )
        
        financial_articles = response.get_financial_articles()
        high_confidence_articles = response.get_high_confidence_articles()
        credible_articles = response.get_credible_articles()
        
        assert len(financial_articles) == 5
        assert len(high_confidence_articles) == 5
        assert len(credible_articles) == 5
        
        quality_score = response.get_quality_score()
        assert isinstance(quality_score, float)
        assert 0.0 <= quality_score <= 100.0
    
    def test_response_serialization(self):
        """Test response serialization to dictionary."""
        articles = [
            NewsArticle(
                url="https://example.com/article-1",
                title="Test Article",
                source="Test Source",
                content="Test content",
                confidence=0.8
            )
        ]
        
        processing_stats = ProcessingStatistics(
            total_processing_time=30.0,
            articles_per_minute=60.0,
            extraction_success_rate=100.0,
            classification_success_rate=100.0,
            average_confidence=0.8,
            average_credibility=80.0,
            average_fact_check_score=70.0,
            financial_articles_percentage=50.0,
            crypto_articles_percentage=0.0,
            high_confidence_percentage=100.0,
            verified_articles_percentage=70.0,
            error_rate=0.0
        )
        
        response = NewsClassificationResponse.create_success_response(
            articles_processed=1,
            classified_articles=articles,
            processing_statistics=processing_stats
        )
        
        response_dict = response.to_dict(include_articles=True)
        
        assert response_dict['success']
        assert response_dict['articles_processed'] == 1
        assert response_dict['articles_classified'] == 1
        assert 'processing_statistics' in response_dict
        assert 'classified_articles' in response_dict
        assert len(response_dict['classified_articles']) == 1


class TestFileNewsRepository:
    """Test suite for FileNewsRepository infrastructure component."""
    
    def setup_method(self):
        """Set up test environment with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.repository = FileNewsRepository(data_directory=self.temp_dir)
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_save_and_find_article(self):
        """Test saving and finding articles."""
        article = NewsArticle(
            url="https://example.com/test-article",
            title="Test Article",
            source="Test Source",
            content="Test content for repository testing.",
            published_date=datetime.now(),
            confidence=0.85,
            credibility_score=80.0
        )
        
        # Save article
        success = await self.repository.save(article)
        assert success
        
        # Find article by URL
        found_article = await self.repository.find_by_url(article.url)
        assert found_article is not None
        assert found_article.url == article.url
        assert found_article.title == article.title
        assert found_article.source == article.source
    
    async def test_save_batch_articles(self):
        """Test batch saving of articles."""
        articles = [
            NewsArticle(
                url=f"https://example.com/article-{i}",
                title=f"Test Article {i}",
                source="Test Source",
                content=f"Test content {i}",
                published_date=datetime.now()
            )
            for i in range(5)
        ]
        
        results = await self.repository.save_batch(articles)
        
        assert len(results) == 5
        assert all(results.values())  # All should be successful
    
    async def test_find_by_source(self):
        """Test finding articles by source."""
        articles = [
            NewsArticle(
                url=f"https://reuters.com/article-{i}",
                title=f"Reuters Article {i}",
                source="Reuters",
                content=f"Reuters content {i}",
                published_date=datetime.now()
            )
            for i in range(3)
        ] + [
            NewsArticle(
                url=f"https://bloomberg.com/article-{i}",
                title=f"Bloomberg Article {i}",
                source="Bloomberg",
                content=f"Bloomberg content {i}",
                published_date=datetime.now()
            )
            for i in range(2)
        ]
        
        await self.repository.save_batch(articles)
        
        reuters_articles = await self.repository.find_by_source("Reuters")
        bloomberg_articles = await self.repository.find_by_source("Bloomberg")
        
        assert len(reuters_articles) == 3
        assert len(bloomberg_articles) == 2
        
        for article in reuters_articles:
            assert "Reuters" in article.source
        
        for article in bloomberg_articles:
            assert "Bloomberg" in article.source
    
    async def test_find_financial_articles(self):
        """Test finding financial articles."""
        articles = [
            NewsArticle(
                url=f"https://example.com/financial-{i}",
                title=f"Financial Article {i}",
                source="Financial Times",
                content=f"Financial content {i}",
                financial_relevance=85.0,
                article_type=ArticleType.FINANCIAL
            )
            for i in range(3)
        ] + [
            NewsArticle(
                url=f"https://example.com/sports-{i}",
                title=f"Sports Article {i}",
                source="Sports News",
                content=f"Sports content {i}",
                financial_relevance=15.0,
                article_type=ArticleType.GENERAL
            )
            for i in range(2)
        ]
        
        await self.repository.save_batch(articles)
        
        financial_articles = await self.repository.find_financial_articles()
        
        assert len(financial_articles) == 3
        for article in financial_articles:
            assert article.is_financial_news()
    
    async def test_repository_statistics(self):
        """Test repository statistics calculation."""
        articles = [
            NewsArticle(
                url=f"https://example.com/article-{i}",
                title=f"Test Article {i}",
                source="Test Source",
                content=f"Test content {i}",
                confidence=0.8,
                credibility_score=85.0,
                financial_relevance=70.0,
                article_type=ArticleType.FINANCIAL,
                sentiment=SentimentType.POSITIVE
            )
            for i in range(10)
        ]
        
        await self.repository.save_batch(articles)
        
        stats = await self.repository.get_statistics()
        
        assert stats['total_articles'] == 10
        assert 'by_source' in stats
        assert 'by_type' in stats
        assert 'by_sentiment' in stats
        assert stats['average_confidence'] == 0.8
        assert stats['average_credibility'] == 85.0
        assert stats['financial_percentage'] == 100.0
    
    async def test_delete_article(self):
        """Test deleting articles."""
        article = NewsArticle(
            url="https://example.com/delete-test",
            title="Delete Test Article",
            source="Test Source",
            content="This article will be deleted.",
            published_date=datetime.now()
        )
        
        # Save article
        await self.repository.save(article)
        
        # Verify it exists
        found_article = await self.repository.find_by_url(article.url)
        assert found_article is not None
        
        # Delete article
        deleted = await self.repository.delete_by_url(article.url)
        assert deleted
        
        # Verify it's gone
        found_article = await self.repository.find_by_url(article.url)
        assert found_article is None


class TestClassifyNewsBatchUseCase:
    """Test suite for ClassifyNewsBatchUseCase application use case."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.repository = FileNewsRepository(data_directory=self.temp_dir)
        self.classification_service = NewsClassificationService()
        
        self.use_case = ClassifyNewsBatchUseCase(
            news_repository=self.repository,
            classification_service=self.classification_service
        )
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_execute_classification(self):
        """Test executing the classification use case."""
        request = NewsClassificationRequest(
            source_urls=["https://example.com/rss"],
            max_articles=10,
            processing_options=ProcessingOptions(
                enable_sentiment_analysis=True,
                enable_fact_checking=True,
                parallel_processing=True
            ),
            request_id="test-batch-123"
        )
        
        response = await self.use_case.execute(request)
        
        assert response.success
        assert response.request_id == "test-batch-123"
        assert response.articles_processed > 0
        assert response.articles_classified > 0
        assert response.is_complete()
        assert response.processing_statistics is not None
    
    async def test_execute_with_invalid_request(self):
        """Test executing with invalid request."""
        invalid_request = NewsClassificationRequest(
            source_urls=[],  # Empty URLs - should be invalid
            max_articles=10
        )
        
        response = await self.use_case.execute(invalid_request)
        
        assert not response.success
        assert response.status == ProcessingStatus.FAILED
        assert response.has_errors()
        assert response.has_critical_errors()
    
    async def test_execute_with_filters(self):
        """Test executing with filtering criteria."""
        request = NewsClassificationRequest(
            source_urls=["https://example.com/rss"],
            max_articles=20,
            filter_criteria=FilterCriteria(
                min_confidence=0.7,
                article_types=[ArticleType.FINANCIAL]
            ),
            request_id="test-filtered-batch"
        )
        
        response = await self.use_case.execute(request)
        
        assert response.success
        assert response.request_id == "test-filtered-batch"
        
        # All returned articles should meet filter criteria
        for article in response.classified_articles:
            if article.confidence is not None:
                assert article.confidence >= 0.7


class TestSystemIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Set up test environment."""
        setup_logging()
        self.temp_dir = tempfile.mkdtemp()
    
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    async def test_end_to_end_classification(self):
        """Test complete end-to-end classification workflow."""
        # Initialize components
        repository = FileNewsRepository(data_directory=self.temp_dir)
        classification_service = NewsClassificationService()
        use_case = ClassifyNewsBatchUseCase(
            news_repository=repository,
            classification_service=classification_service
        )
        
        # Create comprehensive request
        request = NewsClassificationRequest(
            source_urls=[
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.bloomberg.com/markets/news"
            ],
            max_articles=25,
            processing_options=ProcessingOptions(
                enable_sentiment_analysis=True,
                enable_fact_checking=True,
                enable_bias_detection=True,
                enable_financial_analysis=True,
                enable_crypto_detection=True,
                parallel_processing=True,
                max_concurrent_requests=5
            ),
            filter_criteria=FilterCriteria(
                min_confidence=0.5,
                min_credibility=50.0
            ),
            output_format=OutputFormat.JSON,
            processing_priority=ProcessingPriority.HIGH,
            request_id="integration-test-001"
        )
        
        # Execute classification
        response = await use_case.execute(request)
        
        # Verify response
        assert response.success
        assert response.request_id == "integration-test-001"
        assert response.articles_processed > 0
        assert response.articles_classified > 0
        assert response.is_complete()
        
        # Verify processing statistics
        stats = response.processing_statistics
        assert stats is not None
        assert stats.total_processing_time > 0
        assert stats.articles_per_minute > 0
        assert 0 <= stats.extraction_success_rate <= 100
        assert 0 <= stats.classification_success_rate <= 100
        assert 0 <= stats.average_confidence <= 1
        assert 0 <= stats.average_credibility <= 100
        
        # Verify repository persistence
        repo_stats = await repository.get_statistics()
        assert repo_stats['total_articles'] == response.articles_classified
        
        # Verify filtering worked
        for article in response.classified_articles:
            if article.confidence is not None:
                assert article.confidence >= 0.5
            if article.credibility_score is not None:
                assert article.credibility_score >= 50.0
    
    async def test_performance_benchmark(self):
        """Test system performance with larger dataset."""
        # Initialize components
        repository = FileNewsRepository(data_directory=self.temp_dir)
        classification_service = NewsClassificationService()
        use_case = ClassifyNewsBatchUseCase(
            news_repository=repository,
            classification_service=classification_service
        )
        
        # Performance test request
        request = NewsClassificationRequest(
            source_urls=[
                "https://feeds.reuters.com/reuters/businessNews",
                "https://feeds.bloomberg.com/markets/news",
                "https://rss.cnn.com/rss/money_latest.rss"
            ],
            max_articles=100,
            processing_options=ProcessingOptions(
                enable_sentiment_analysis=True,
                enable_fact_checking=True,
                enable_bias_detection=True,
                enable_financial_analysis=True,
                parallel_processing=True,
                max_concurrent_requests=10
            ),
            request_id="performance-test-001"
        )
        
        start_time = datetime.now()
        response = await use_case.execute(request)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Verify performance expectations
        assert response.success
        assert response.articles_processed > 0
        
        # Performance benchmarks
        if response.articles_processed > 0:
            articles_per_second = response.articles_processed / processing_time
            assert articles_per_second > 0.5  # Should process at least 0.5 articles per second
            
            # Quality benchmarks
            stats = response.processing_statistics
            assert stats.classification_success_rate >= 90.0  # At least 90% classification success
            assert stats.average_confidence >= 0.5  # Average confidence at least 50%
    
    async def test_error_handling_and_recovery(self):
        """Test system error handling and recovery mechanisms."""
        # Initialize components
        repository = FileNewsRepository(data_directory=self.temp_dir)
        classification_service = NewsClassificationService()
        use_case = ClassifyNewsBatchUseCase(
            news_repository=repository,
            classification_service=classification_service
        )
        
        # Test with problematic URLs
        request = NewsClassificationRequest(
            source_urls=[
                "https://example.com/valid-rss",
                "https://invalid-url-that-does-not-exist.com/rss",
                "https://example.com/another-valid-rss"
            ],
            max_articles=20,
            processing_options=ProcessingOptions(
                enable_sentiment_analysis=True,
                retry_attempts=2,
                timeout_seconds=5
            ),
            request_id="error-handling-test"
        )
        
        response = await use_case.execute(request)
        
        # Should still succeed with partial results
        assert response.success  # Should succeed with available articles
        assert response.articles_processed > 0
        
        # Should have some error information
        if response.error_information:
            assert response.error_information.error_count > 0
            assert len(response.error_information.error_messages) > 0


async def run_all_tests():
    """Run all tests manually without pytest."""
    print("🧪 Running Penelope News Classification System Tests")
    print("=" * 60)
    
    test_classes = [
        TestNewsArticleEntity,
        TestNewsClassificationService,
        TestNewsClassificationRequest,
        TestNewsClassificationResponse,
        TestFileNewsRepository,
        TestClassifyNewsBatchUseCase,
        TestSystemIntegration
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    
    for test_class in test_classes:
        print(f"\n📋 Running {test_class.__name__}")
        print("-" * 40)
        
        test_instance = test_class()
        test_methods = [method for method in dir(test_instance) if method.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            
            try:
                # Setup if available
                if hasattr(test_instance, 'setup_method'):
                    test_instance.setup_method()
                
                # Run test
                test_method = getattr(test_instance, method_name)
                if asyncio.iscoroutinefunction(test_method):
                    await test_method()
                else:
                    test_method()
                
                print(f"✅ {method_name}")
                passed_tests += 1
                
            except Exception as e:
                print(f"❌ {method_name}: {e}")
                failed_tests += 1
                
            finally:
                # Teardown if available
                if hasattr(test_instance, 'teardown_method'):
                    try:
                        test_instance.teardown_method()
                    except Exception:
                        pass
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
    
    if failed_tests == 0:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {failed_tests} tests failed")
    
    return failed_tests == 0


if __name__ == "__main__":
    """Run tests when executed directly."""
    setup_logging()
    
    success = asyncio.run(run_all_tests())
    
    exit(0 if success else 1) 