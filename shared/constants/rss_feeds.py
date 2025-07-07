"""
RSS feed sources configuration for news extraction.

This module contains the configuration for RSS feeds from various news sources.
The feeds are organized by category and priority for efficient processing.
"""

from typing import List, Dict, Any

# PREMIUM FINANCIAL & CRYPTO NEWS SOURCES
PREMIUM_RSS_FEEDS: List[str] = [
    # Financial News - Premium Sources
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.reuters.com/reuters/businessNews", 
    "https://www.ft.com/rss/feed/companies",
    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    "https://feeds.feedburner.com/bloomberg/most-popular",
    
    # Crypto & Blockchain - Premium Sources  
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://feeds.feedburner.com/bitcoinmagazine/feeds",
    "https://feeds.feedburner.com/CoinTelegraph",
    "https://feeds.feedburner.com/CoinDesk",
    
    # Technology & Business - Premium Sources
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://feeds.feedburner.com/oreilly/radar",
    "https://feeds.feedburner.com/TechCrunch",
    "https://rss.cnn.com/rss/money_latest.rss",
    "https://feeds.feedburner.com/wsj/xml/rss/3_7085.xml",
    
    # International Premium Sources
    "https://www.theguardian.com/business/rss",
    "https://feeds.bbc.co.uk/news/business/rss.xml",
    "https://feeds.bbc.co.uk/news/technology/rss.xml",
    "https://www.economist.com/finance-and-economics/rss.xml",
    "https://feeds.feedburner.com/economist-business-finance"
]

# FINANCIAL NEWS SPECIFIC FEEDS
FINANCIAL_RSS_FEEDS: List[str] = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://www.ft.com/rss/feed/companies",
    "https://feeds.feedburner.com/bloomberg/most-popular",
    "https://www.theguardian.com/business/rss",
    "https://feeds.bbc.co.uk/news/business/rss.xml",
    "https://rss.cnn.com/rss/money_latest.rss",
    "https://feeds.feedburner.com/wsj/xml/rss/3_7085.xml",
    "https://www.economist.com/finance-and-economics/rss.xml"
]

# CRYPTO & BLOCKCHAIN SPECIFIC FEEDS
CRYPTO_RSS_FEEDS: List[str] = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://feeds.feedburner.com/bitcoinmagazine/feeds",
    "https://feeds.feedburner.com/CoinTelegraph",
    "https://feeds.feedburner.com/CoinDesk",
    "https://feeds.feedburner.com/TheCoinTelegraph",
    "https://feeds.feedburner.com/DecryptMedia",
    "https://feeds.feedburner.com/CryptoSlate",
    "https://feeds.feedburner.com/bitcoin-news-updates",
    "https://feeds.feedburner.com/CoinJournal"
]

# TECHNOLOGY & BUSINESS FEEDS
TECH_RSS_FEEDS: List[str] = [
    "https://feeds.arstechnica.com/arstechnica/index",
    "https://feeds.feedburner.com/oreilly/radar",
    "https://feeds.feedburner.com/TechCrunch",
    "https://feeds.bbc.co.uk/news/technology/rss.xml",
    "https://feeds.feedburner.com/VentureBeat",
    "https://feeds.feedburner.com/TheVerge",
    "https://feeds.feedburner.com/wired/index",
    "https://feeds.feedburner.com/fastcompany/headlines",
    "https://feeds.feedburner.com/ZDNet-News",
    "https://feeds.feedburner.com/thenextweb"
]

# ALTERNATIVE WORKING FEEDS (in case premium ones fail)
ALTERNATIVE_RSS_FEEDS: List[str] = [
    "https://feeds.feedburner.com/reuters/businessNews",
    "https://feeds.feedburner.com/reuters/technologyNews",
    "https://feeds.feedburner.com/reuters/marketsNews",
    "https://feeds.feedburner.com/bloomberg/business",
    "https://feeds.feedburner.com/bloomberg/technology",
    "https://feeds.feedburner.com/TheGuardianBusiness",
    "https://feeds.feedburner.com/bbc-news-business",
    "https://feeds.feedburner.com/bbc-news-technology",
    "https://feeds.feedburner.com/MarketWatch-TopStories",
    "https://feeds.feedburner.com/TheEconomist-BusinessAndFinance"
]

# BACKUP FEEDS (if all else fails)
BACKUP_RSS_FEEDS: List[str] = [
    "https://rss.cnn.com/rss/money_latest.rss",
    "https://feeds.yahoo.com/rss/mostviewed",
    "https://feeds.feedburner.com/yahoo/finance",
    "https://feeds.feedburner.com/msnbc/business",
    "https://feeds.feedburner.com/cbcnews/business",
    "https://feeds.feedburner.com/skynews/business",
    "https://feeds.feedburner.com/newsweek/business",
    "https://feeds.feedburner.com/time/business",
    "https://feeds.feedburner.com/fortune/feeds",
    "https://feeds.feedburner.com/forbes/business"
]

# COMBINED FEEDS BY PRIORITY
RSS_FEEDS_CONFIG = PREMIUM_RSS_FEEDS

# Feed metadata for better source identification
FEED_METADATA: Dict[str, Dict[str, Any]] = {
    "bloomberg": {
        "name": "Bloomberg",
        "credibility": 95,
        "bias": "center",
        "specialties": ["finance", "markets", "economics"]
    },
    "bbc": {
        "name": "BBC News",
        "credibility": 95,
        "bias": "center",
        "specialties": ["general", "business", "technology"]
    },
    "marketwatch": {
        "name": "MarketWatch",
        "credibility": 90,
        "bias": "center",
        "specialties": ["finance", "markets", "investing"]
    },
    "coindesk": {
        "name": "CoinDesk",
        "credibility": 85,
        "bias": "center",
        "specialties": ["crypto", "blockchain", "defi"]
    },
    "cointelegraph": {
        "name": "Cointelegraph",
        "credibility": 80,
        "bias": "center",
        "specialties": ["crypto", "blockchain", "nft"]
    },
    "arstechnica": {
        "name": "Ars Technica",
        "credibility": 88,
        "bias": "center",
        "specialties": ["technology", "science", "innovation"]
    },
    "theguardian": {
        "name": "The Guardian",
        "credibility": 85,
        "bias": "left",
        "specialties": ["business", "politics", "environment"]
    },
    "economist": {
        "name": "The Economist",
        "credibility": 92,
        "bias": "center",
        "specialties": ["economics", "politics", "international"]
    },
    "reuters": {
        "name": "Reuters",
        "credibility": 93,
        "bias": "center",
        "specialties": ["finance", "business", "international"]
    },
    "ft": {
        "name": "Financial Times",
        "credibility": 94,
        "bias": "center",
        "specialties": ["finance", "business", "economics"]
    }
}

# Export main configuration
__all__ = [
    'RSS_FEEDS_CONFIG',
    'PREMIUM_RSS_FEEDS', 
    'FINANCIAL_RSS_FEEDS',
    'CRYPTO_RSS_FEEDS',
    'TECH_RSS_FEEDS',
    'ALTERNATIVE_RSS_FEEDS',
    'BACKUP_RSS_FEEDS',
    'FEED_METADATA'
]

# Legacy constants for backward compatibility
DEFAULT_MAX_ARTICLES = 100
DEFAULT_ARTICLES_PER_FEED = 6
DEFAULT_TIMEOUT_SECONDS = 300
DEFAULT_RETRY_ATTEMPTS = 3

# Feed metadata for enhanced processing (legacy support)
RSS_FEED_METADATA: Dict[str, Dict[str, Any]] = {
    "https://feeds.bloomberg.com/markets/news.rss": {
        "source": "Bloomberg",
        "category": "financial",
        "reliability": 95,
        "update_frequency": "hourly"
    },
    "https://www.coindesk.com/arc/outboundfeeds/rss/": {
        "source": "CoinDesk",
        "category": "cryptocurrency", 
        "reliability": 85,
        "update_frequency": "hourly"
    },
    "https://feeds.bbc.co.uk/news/business/rss.xml": {
        "source": "BBC Business",
        "category": "financial",
        "reliability": 95,
        "update_frequency": "hourly"
    }
}

# Feed categories for organization (legacy support)
FEED_CATEGORIES = {
    "financial": FINANCIAL_RSS_FEEDS,
    "cryptocurrency": CRYPTO_RSS_FEEDS,
    "technology": TECH_RSS_FEEDS,
    "general": ALTERNATIVE_RSS_FEEDS
} 