"""
Infrastructure Service: Content Extractor Service

This service handles extracting full article content from news article URLs.
Following DDD principles, this is part of the infrastructure layer
that implements external service integration.
"""

import asyncio
import logging
from typing import Optional
import time
import re

from domain.entities.news_article import NewsArticle

logger = logging.getLogger(__name__)

class ContentExtractorService:
    """
    Service for extracting full content from news article URLs.
    
    This service handles web scraping and content extraction,
    enhancing NewsArticle entities with full content.
    """
    
    def __init__(self, timeout_seconds: int = 30):
        self.timeout_seconds = timeout_seconds
        self.rate_limiter = {}
        
        # Headers for web requests
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
    
    async def extract_content(self, article: NewsArticle) -> NewsArticle:
        """
        Extract full content for a news article.
        
        Args:
            article: NewsArticle entity
            
        Returns:
            NewsArticle with enhanced content
        """
        if not article.url:
            return article
        
        try:
            # Apply rate limiting
            await self._apply_rate_limiting(article.url)
            
            # Extract content
            full_content = await self._fetch_full_content(article.url)
            
            if full_content and len(full_content) > 100:
                article.content = full_content
                article.extraction_success = True
                logger.debug(f"✅ Extracted content for {article.url[:50]}...")
            else:
                article.extraction_success = False
                logger.warning(f"❌ Failed to extract content for {article.url[:50]}...")
            
        except Exception as e:
            logger.error(f"❌ Error extracting content for {article.url}: {e}")
            article.extraction_success = False
        
        return article
    
    async def _fetch_full_content(self, url: str) -> Optional[str]:
        """
        Fetch full content from a URL.
        
        Args:
            url: Article URL
            
        Returns:
            Extracted content or None
        """
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Make request
            response = requests.get(
                url, 
                timeout=self.timeout_seconds,
                headers=self.headers
            )
            
            if response.status_code != 200:
                logger.warning(f"⚠️ HTTP {response.status_code} for {url}")
                return None
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'advertisement']):
                element.decompose()
            
            # Try premium content selectors first
            content = self._extract_premium_content(soup, url)
            
            # Fallback to general selectors
            if not content or len(content) < 200:
                content = self._extract_general_content(soup)
            
            # Final fallback to paragraphs
            if not content or len(content) < 100:
                content = self._extract_paragraph_content(soup)
            
            # Clean and return content
            if content:
                content = self._clean_content(content)
                return content
            
            return None
            
        except Exception as e:
            logger.warning(f"❌ Content extraction failed for {url}: {e}")
            return None
    
    def _extract_premium_content(self, soup, url: str) -> Optional[str]:
        """Extract content using premium site-specific selectors."""
        # Bloomberg
        if 'bloomberg' in url:
            selectors = [
                '[data-module="ArticleBody"]',
                '.paywall',
                '.article-body'
            ]
        
        # CNN
        elif 'cnn' in url:
            selectors = [
                '.ArticleBody-articleBody',
                '.InlineArticleBody',
                '.l-container'
            ]
        
        # Wall Street Journal
        elif 'wsj' in url:
            selectors = [
                '.wsj-snippet-body',
                '.article-content',
                '.snippet-promotion'
            ]
        
        # BBC
        elif 'bbc' in url:
            selectors = [
                '.story-body__inner',
                '[data-component="text-block"]',
                '.story-body'
            ]
        
        # Reuters
        elif 'reuters' in url:
            selectors = [
                '.article-wrap',
                '.StandardArticleBody_body',
                '.article-body'
            ]
        
        # TechCrunch
        elif 'techcrunch' in url:
            selectors = [
                '.article-content',
                '.entry-content',
                '.post-content'
            ]
        
        # CoinDesk
        elif 'coindesk' in url:
            selectors = [
                '.at-body',
                '.entry-content',
                '.article-body'
            ]
        
        # The Verge
        elif 'theverge' in url:
            selectors = [
                '.duet--article--article-body',
                '.c-entry-content',
                '.entry-content'
            ]
        
        # Wired
        elif 'wired' in url:
            selectors = [
                '.article__chunks',
                '.content',
                '.article-body'
            ]
        
        # Guardian
        elif 'guardian' in url:
            selectors = [
                '.article-body-commercial-selector',
                '.content__article-body',
                '.article-body'
            ]
        
        # Fortune
        elif 'fortune' in url:
            selectors = [
                '.article-content',
                '.entry-content',
                '.post-content'
            ]
        
        # Financial Times
        elif 'ft.com' in url:
            selectors = [
                '.article__content-body',
                '.n-content-body',
                '.article-body'
            ]
        
        else:
            return None
        
        # Try each selector
        for selector in selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator=' ', strip=True)
                if len(content) > 200:
                    return content
        
        return None
    
    def _extract_general_content(self, soup) -> Optional[str]:
        """Extract content using general selectors."""
        general_selectors = [
            'article',
            '.article-content',
            '.post-content',
            '.entry-content',
            '.article-body',
            '.story-body',
            'main',
            '.content',
            '.text',
            '.article-text',
            '.post-body',
            '.content-body'
        ]
        
        for selector in general_selectors:
            content_elem = soup.select_one(selector)
            if content_elem:
                content = content_elem.get_text(separator=' ', strip=True)
                if len(content) > 200:
                    return content
        
        return None
    
    def _extract_paragraph_content(self, soup) -> Optional[str]:
        """Extract content from paragraphs as fallback."""
        paragraphs = soup.find_all('p')
        content_parts = []
        
        for p in paragraphs:
            text = p.get_text(strip=True)
            if len(text) > 20:  # Only include substantial paragraphs
                content_parts.append(text)
        
        if content_parts:
            content = ' '.join(content_parts)
            return content
        
        return None
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content."""
        # Remove extra whitespace
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common footer/header patterns
        patterns_to_remove = [
            r'Subscribe to.*?newsletter',
            r'Sign up for.*?updates',
            r'Follow us on.*?social',
            r'Copyright.*?\d{4}',
            r'All rights reserved',
            r'Terms of use',
            r'Privacy policy',
            r'Cookie policy',
            r'Advertisement',
            r'Sponsored content',
        ]
        
        for pattern in patterns_to_remove:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        # Remove excessive punctuation
        content = re.sub(r'[.]{3,}', '...', content)
        
        # Final cleanup
        content = content.strip()
        
        return content
    
    async def _apply_rate_limiting(self, url: str) -> None:
        """Apply rate limiting to avoid overwhelming servers."""
        current_time = time.time()
        
        # Extract domain for rate limiting
        domain = self._extract_domain(url)
        
        # Check if we need to wait
        if domain in self.rate_limiter:
            last_request_time = self.rate_limiter[domain]
            time_since_last = current_time - last_request_time
            
            # Wait at least 2 seconds between requests to the same domain
            if time_since_last < 2.0:
                await asyncio.sleep(2.0 - time_since_last)
        
        # Update last request time
        self.rate_limiter[domain] = current_time
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL for rate limiting."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return url 