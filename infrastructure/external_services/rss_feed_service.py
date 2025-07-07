"""
Infrastructure Service: RSS Feed Service

This service handles fetching articles from RSS feeds.
Following DDD principles, this is part of the infrastructure layer
that implements external service integration.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from domain.entities.news_article import NewsArticle
from shared.constants.rss_feeds import DEFAULT_ARTICLES_PER_FEED, DEFAULT_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)

class RSSFeedService:
    """
    Service for fetching articles from RSS feeds.
    
    This service handles RSS feed parsing and article extraction,
    providing NewsArticle entities to the domain layer.
    """
    
    def __init__(self, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS):
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = {}
        
    async def fetch_articles_from_feeds(
        self, 
        feed_urls: List[str], 
        max_articles: int = 100
    ) -> List[NewsArticle]:
        """
        Fetch articles from multiple RSS feeds.
        
        Args:
            feed_urls: List of RSS feed URLs
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of NewsArticle entities
        """
        logger.info(f"🔄 Fetching articles from {len(feed_urls)} RSS feeds")
        
        all_articles = []
        articles_per_feed = max(DEFAULT_ARTICLES_PER_FEED, max_articles // len(feed_urls))
        
        # Process feeds in parallel
        tasks = []
        for feed_url in feed_urls:
            task = self._fetch_from_single_feed(feed_url, articles_per_feed)
            tasks.append(task)
        
        # Wait for all feeds to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Error fetching from {feed_urls[i]}: {result}")
                continue
            
            if isinstance(result, list):
                all_articles.extend(result)
                
            if len(all_articles) >= max_articles:
                break
        
        # Remove duplicates and limit to max_articles
        unique_articles = self._remove_duplicates(all_articles)
        limited_articles = unique_articles[:max_articles]
        
        logger.info(f"✅ Fetched {len(limited_articles)} unique articles from RSS feeds")
        return limited_articles
    
    async def _fetch_from_single_feed(
        self, 
        feed_url: str, 
        max_articles: int = DEFAULT_ARTICLES_PER_FEED
    ) -> List[NewsArticle]:
        """
        Fetch articles from a single RSS feed.
        
        Args:
            feed_url: RSS feed URL
            max_articles: Maximum articles to fetch from this feed
            
        Returns:
            List of NewsArticle entities
        """
        try:
            # Import here to avoid dependency issues
            import feedparser
            
            # Rate limiting
            await self._apply_rate_limiting(feed_url)
            
            # Parse RSS feed
            logger.debug(f"📡 Parsing RSS feed: {feed_url}")
            feed = feedparser.parse(feed_url)
            
            if feed.bozo:
                logger.warning(f"⚠️ Feed parsing warning for {feed_url}: {feed.bozo_exception}")
            
            articles = []
            entries = feed.entries[:max_articles]
            
            for entry in entries:
                try:
                    article = self._create_article_from_entry(entry, feed_url)
                    if article:
                        articles.append(article)
                except Exception as e:
                    logger.warning(f"❌ Error creating article from entry: {e}")
                    continue
            
            logger.debug(f"✅ Extracted {len(articles)} articles from {feed_url}")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error fetching from RSS feed {feed_url}: {e}")
            return []
    
    def _create_article_from_entry(self, entry: Any, feed_url: str) -> Optional[NewsArticle]:
        """
        Create a NewsArticle entity from RSS entry.
        
        Args:
            entry: RSS feed entry
            feed_url: Source RSS feed URL
            
        Returns:
            NewsArticle entity or None
        """
        try:
            # Extract basic information
            url = getattr(entry, 'link', '')
            title = getattr(entry, 'title', '')
            
            if not url or not title:
                return None
            
            # Extract source from feed or entry
            source = self._extract_source_name(entry, feed_url)
            
            # Extract content
            content = self._extract_content_from_entry(entry)
            
            # Extract published date
            published_date = self._extract_published_date(entry)
            
            # Create and return NewsArticle
            return NewsArticle(
                url=url,
                title=title.strip(),
                source=source,
                content=content,
                published_date=published_date
            )
            
        except Exception as e:
            logger.warning(f"❌ Error creating article from entry: {e}")
            return None
    
    def _extract_source_name(self, entry: Any, feed_url: str) -> str:
        """Extract source name from entry or feed URL."""
        # Try to get source from entry
        if hasattr(entry, 'source') and hasattr(entry.source, 'title'):
            return entry.source.title
        
        # Extract from feed URL
        if 'reuters' in feed_url:
            return 'Reuters'
        elif 'cnn' in feed_url:
            return 'CNN'
        elif 'bbc' in feed_url:
            return 'BBC'
        elif 'bloomberg' in feed_url:
            return 'Bloomberg'
        elif 'wsj' in feed_url:
            return 'Wall Street Journal'
        elif 'techcrunch' in feed_url:
            return 'TechCrunch'
        elif 'coindesk' in feed_url:
            return 'CoinDesk'
        elif 'wired' in feed_url:
            return 'Wired'
        elif 'theverge' in feed_url:
            return 'The Verge'
        elif 'guardian' in feed_url:
            return 'The Guardian'
        elif 'fortune' in feed_url:
            return 'Fortune'
        elif 'economist' in feed_url:
            return 'The Economist'
        elif 'ft.com' in feed_url:
            return 'Financial Times'
        else:
            return 'Unknown Source'
    
    def _extract_content_from_entry(self, entry: Any) -> str:
        """Extract content from RSS entry."""
        # Try different content fields
        content_fields = ['content', 'summary', 'description']
        
        for field in content_fields:
            if hasattr(entry, field):
                content_data = getattr(entry, field)
                
                if isinstance(content_data, list) and len(content_data) > 0:
                    # Handle content list (like feedparser content)
                    return content_data[0].get('value', '')
                elif isinstance(content_data, str):
                    return content_data
        
        return ""
    
    def _extract_published_date(self, entry: Any) -> Optional[datetime]:
        """Extract published date from RSS entry."""
        try:
            import time
            
            # Try different date fields
            date_fields = ['published_parsed', 'updated_parsed']
            
            for field in date_fields:
                if hasattr(entry, field):
                    date_tuple = getattr(entry, field)
                    if date_tuple:
                        return datetime.fromtimestamp(time.mktime(date_tuple))
            
            # Try string date fields
            string_date_fields = ['published', 'updated']
            for field in string_date_fields:
                if hasattr(entry, field):
                    date_string = getattr(entry, field)
                    if date_string:
                        # Try to parse string date
                        try:
                            from dateutil import parser
                            return parser.parse(date_string)
                        except:
                            continue
            
            return None
            
        except Exception:
            return None
    
    def _remove_duplicates(self, articles: List[NewsArticle]) -> List[NewsArticle]:
        """Remove duplicate articles based on URL."""
        seen_urls = set()
        unique_articles = []
        
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        return unique_articles
    
    async def _apply_rate_limiting(self, feed_url: str) -> None:
        """Apply rate limiting to avoid overwhelming RSS feeds."""
        current_time = time.time()
        
        # Check if we need to wait
        if feed_url in self.rate_limiter:
            last_request_time = self.rate_limiter[feed_url]
            time_since_last = current_time - last_request_time
            
            # Wait at least 1 second between requests to the same feed
            if time_since_last < 1.0:
                await asyncio.sleep(1.0 - time_since_last)
        
        # Update last request time
        self.rate_limiter[feed_url] = current_time 