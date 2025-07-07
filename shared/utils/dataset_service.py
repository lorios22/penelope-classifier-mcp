"""
Dataset Service for Penelope News Classification System

This service manages the loading and access to various datasets used by the
classification system, including financial keywords, crypto mappings, source
credibility ratings, and bias information.

Classes:
    DatasetService: Main service for managing datasets
    
Features:
    - Financial keywords categorization
    - Crypto symbol mapping
    - Source credibility ratings (MBFC)
    - Political bias ratings (AllSides)
    - Lazy loading and caching
    - Error handling for missing datasets

Author: Claude AI Assistant
Date: 2025-01-18
Version: 1.0
"""

import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from loguru import logger
from urllib.parse import urlparse

class DatasetService:
    """
    Service for managing and accessing classification datasets.
    
    This service provides centralized access to all datasets used by the
    news classification system, including financial keywords, crypto mappings,
    source credibility ratings, and bias information.
    """
    
    def __init__(self, datasets_dir: str = "data/datasets"):
        """
        Initialize the DatasetService.
        
        Args:
            datasets_dir (str): Path to the datasets directory
        """
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize dataset caches
        self._financial_keywords: Optional[Dict[str, List[str]]] = None
        self._crypto_mapping: Optional[Dict[str, str]] = None
        self._mbfc_ratings: Optional[Dict[str, Dict[str, Any]]] = None
        self._allsides_bias: Optional[Dict[str, Dict[str, Any]]] = None
        
        # Combined keyword sets for faster lookups
        self._all_financial_keywords: Optional[Set[str]] = None
        self._all_crypto_keywords: Optional[Set[str]] = None
        
        logger.info(f"DatasetService initialized with datasets directory: {self.datasets_dir}")
    
    def get_financial_keywords(self) -> Dict[str, List[str]]:
        """
        Get financial keywords categorized by type.
        
        Returns:
            Dict[str, List[str]]: Dictionary with categories as keys and keyword lists as values
        """
        if self._financial_keywords is None:
            self._load_financial_keywords()
        return self._financial_keywords or {}
    
    def get_crypto_mapping(self) -> Dict[str, str]:
        """
        Get cryptocurrency symbol mapping.
        
        Returns:
            Dict[str, str]: Dictionary mapping crypto symbols to canonical names
        """
        if self._crypto_mapping is None:
            self._load_crypto_mapping()
        return self._crypto_mapping or {}
    
    def get_all_financial_keywords(self) -> Set[str]:
        """
        Get all financial keywords as a flat set for fast lookups.
        
        Returns:
            Set[str]: Set of all financial keywords
        """
        if self._all_financial_keywords is None:
            keywords_dict = self.get_financial_keywords()
            self._all_financial_keywords = set()
            for category_keywords in keywords_dict.values():
                self._all_financial_keywords.update(category_keywords)
        return self._all_financial_keywords
    
    def get_all_crypto_keywords(self) -> Set[str]:
        """
        Get all crypto keywords (from both financial keywords and crypto mapping).
        
        Returns:
            Set[str]: Set of all crypto-related keywords
        """
        if self._all_crypto_keywords is None:
            # Get crypto keywords from financial keywords
            financial_keywords = self.get_financial_keywords()
            crypto_from_financial = set(financial_keywords.get('crypto', []))
            
            # Get crypto keywords from crypto mapping
            crypto_mapping = self.get_crypto_mapping()
            crypto_from_mapping = set(crypto_mapping.keys()) | set(crypto_mapping.values())
            
            # Combine both sets
            self._all_crypto_keywords = crypto_from_financial | crypto_from_mapping
        return self._all_crypto_keywords
    
    def get_source_credibility(self, source_url: str) -> Dict[str, Any]:
        """
        Get credibility information for a news source.
        
        Args:
            source_url (str): URL of the news source
            
        Returns:
            Dict[str, Any]: Credibility information including scores and ratings
        """
        if self._mbfc_ratings is None:
            self._load_mbfc_ratings()
        
        domain = self._extract_domain(source_url)
        return self._mbfc_ratings.get(domain, {
            "found": False,
            "credibility_score": 50,
            "bias_rating": "unknown",
            "factual_rating": "unknown"
        })
    
    def get_source_bias(self, source_url: str) -> Dict[str, Any]:
        """
        Get political bias information for a news source.
        
        Args:
            source_url (str): URL of the news source
            
        Returns:
            Dict[str, Any]: Bias information including political lean and confidence
        """
        if self._allsides_bias is None:
            self._load_allsides_bias()
        
        domain = self._extract_domain(source_url)
        return self._allsides_bias.get(domain, {
            "found": False,
            "bias": "unknown",
            "confidence": "unknown",
            "political_lean": "center"
        })
    
    def is_financial_content(self, text: str) -> bool:
        """
        Check if text contains financial keywords.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            bool: True if text contains financial keywords
        """
        text_lower = text.lower()
        financial_keywords = self.get_all_financial_keywords()
        
        return any(keyword in text_lower for keyword in financial_keywords)
    
    def is_crypto_content(self, text: str) -> bool:
        """
        Check if text contains crypto-related keywords.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            bool: True if text contains crypto keywords
        """
        text_lower = text.lower()
        crypto_keywords = self.get_all_crypto_keywords()
        
        return any(keyword in text_lower for keyword in crypto_keywords)
    
    def find_crypto_mentions(self, text: str) -> List[str]:
        """
        Find cryptocurrency mentions in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            List[str]: List of cryptocurrency mentions found
        """
        text_lower = text.lower()
        crypto_mapping = self.get_crypto_mapping()
        mentions = []
        
        for symbol, canonical_name in crypto_mapping.items():
            if symbol in text_lower:
                mentions.append(canonical_name)
        
        return list(set(mentions))  # Remove duplicates
    
    def get_financial_keyword_categories(self, text: str) -> Dict[str, List[str]]:
        """
        Get financial keyword categories found in text.
        
        Args:
            text (str): Text to analyze
            
        Returns:
            Dict[str, List[str]]: Dictionary with categories and found keywords
        """
        text_lower = text.lower()
        financial_keywords = self.get_financial_keywords()
        found_categories = {}
        
        for category, keywords in financial_keywords.items():
            found_keywords = [keyword for keyword in keywords if keyword in text_lower]
            if found_keywords:
                found_categories[category] = found_keywords
        
        return found_categories
    
    def _load_financial_keywords(self):
        """Load financial keywords from JSON file."""
        try:
            keywords_file = self.datasets_dir / "financial_keywords.json"
            if keywords_file.exists():
                with open(keywords_file, 'r', encoding='utf-8') as f:
                    self._financial_keywords = json.load(f)
                logger.info(f"✅ Financial keywords loaded: {len(self._financial_keywords)} categories")
            else:
                logger.warning("❌ Financial keywords file not found, using defaults")
                self._financial_keywords = self._get_default_financial_keywords()
        except Exception as e:
            logger.error(f"Error loading financial keywords: {e}")
            self._financial_keywords = self._get_default_financial_keywords()
    
    def _load_crypto_mapping(self):
        """Load crypto symbol mapping from JSON file."""
        try:
            mapping_file = self.datasets_dir / "crypto_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, 'r', encoding='utf-8') as f:
                    self._crypto_mapping = json.load(f)
                logger.info(f"✅ Crypto mapping loaded: {len(self._crypto_mapping)} symbols")
            else:
                logger.warning("❌ Crypto mapping file not found, using defaults")
                self._crypto_mapping = self._get_default_crypto_mapping()
        except Exception as e:
            logger.error(f"Error loading crypto mapping: {e}")
            self._crypto_mapping = self._get_default_crypto_mapping()
    
    def _load_mbfc_ratings(self):
        """Load MBFC credibility ratings from CSV file."""
        try:
            ratings_file = self.datasets_dir / "mbfc_ratings.csv"
            if ratings_file.exists():
                df = pd.read_csv(ratings_file)
                self._mbfc_ratings = {}
                
                for _, row in df.iterrows():
                    url = row.get('url', '') or row.get('URL', '') or row.get('website', '')
                    if pd.notna(url):
                        domain = self._extract_domain(url)
                        if domain:
                            self._mbfc_ratings[domain] = {
                                "found": True,
                                "name": row.get('name', domain),
                                "credibility_score": self._calculate_credibility_score(row),
                                "bias_rating": row.get('bias', 'unknown'),
                                "factual_rating": row.get('factual', 'unknown'),
                                "notes": row.get('notes', ''),
                                "country": row.get('country', 'unknown')
                            }
                
                logger.info(f"✅ MBFC ratings loaded: {len(self._mbfc_ratings)} sources")
            else:
                logger.warning("❌ MBFC ratings file not found")
                self._mbfc_ratings = {}
        except Exception as e:
            logger.error(f"Error loading MBFC ratings: {e}")
            self._mbfc_ratings = {}
    
    def _load_allsides_bias(self):
        """Load AllSides bias ratings from CSV file."""
        try:
            bias_file = self.datasets_dir / "allsides_bias.csv"
            if bias_file.exists():
                df = pd.read_csv(bias_file)
                self._allsides_bias = {}
                
                for _, row in df.iterrows():
                    url = row.get('url', '') or row.get('URL', '') or row.get('website', '')
                    if pd.notna(url):
                        domain = self._extract_domain(url)
                        if domain:
                            self._allsides_bias[domain] = {
                                "found": True,
                                "name": row.get('name', domain),
                                "bias": row.get('bias', 'unknown'),
                                "confidence": row.get('confidence', 'unknown'),
                                "political_lean": self._normalize_political_lean(row.get('bias', 'unknown')),
                                "rating": row.get('rating', ''),
                                "agree_disagree": row.get('agree_disagree', '')
                            }
                
                logger.info(f"✅ AllSides bias loaded: {len(self._allsides_bias)} sources")
            else:
                logger.warning("❌ AllSides bias file not found")
                self._allsides_bias = {}
        except Exception as e:
            logger.error(f"Error loading AllSides bias: {e}")
            self._allsides_bias = {}
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            # Remove www. prefix
            if domain.startswith('www.'):
                domain = domain[4:]
            return domain
        except:
            return ""
    
    def _calculate_credibility_score(self, row: pd.Series) -> int:
        """Calculate credibility score from MBFC data."""
        # This is a simplified scoring system
        factual = str(row.get('factual', '')).lower()
        bias = str(row.get('bias', '')).lower()
        
        score = 50  # Base score
        
        # Factual rating adjustments
        if 'very high' in factual or 'high' in factual:
            score += 30
        elif 'mostly factual' in factual or 'mixed' in factual:
            score += 10
        elif 'low' in factual:
            score -= 20
        
        # Bias adjustments (less extreme bias is better)
        if 'center' in bias or 'least' in bias:
            score += 10
        elif 'mixed' in bias:
            score += 5
        elif 'strong' in bias or 'extreme' in bias:
            score -= 15
        
        return max(0, min(100, score))
    
    def _normalize_political_lean(self, bias: str) -> str:
        """Normalize political bias to standard categories."""
        bias_lower = str(bias).lower()
        
        if 'left' in bias_lower:
            return 'left'
        elif 'right' in bias_lower:
            return 'right'
        elif 'center' in bias_lower:
            return 'center'
        else:
            return 'unknown'
    
    def _get_default_financial_keywords(self) -> Dict[str, List[str]]:
        """Get default financial keywords if file is not available."""
        return {
            "stocks": ["stock", "share", "equity", "market", "nasdaq", "dow", "bull", "bear"],
            "crypto": ["bitcoin", "ethereum", "crypto", "blockchain", "defi", "nft"],
            "economy": ["inflation", "recession", "gdp", "fed", "interest", "rate"],
            "sentiment": ["bullish", "bearish", "positive", "negative", "optimistic", "pessimistic"]
        }
    
    def _get_default_crypto_mapping(self) -> Dict[str, str]:
        """Get default crypto mapping if file is not available."""
        return {
            "bitcoin": "bitcoin",
            "btc": "bitcoin",
            "ethereum": "ethereum",
            "eth": "ethereum",
            "dogecoin": "dogecoin",
            "doge": "dogecoin"
        }
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """
        Get information about loaded datasets.
        
        Returns:
            Dict[str, Any]: Information about all datasets
        """
        return {
            "datasets_directory": str(self.datasets_dir),
            "financial_keywords": {
                "loaded": self._financial_keywords is not None,
                "categories": len(self._financial_keywords) if self._financial_keywords else 0,
                "total_keywords": len(self.get_all_financial_keywords()) if self._financial_keywords else 0
            },
            "crypto_mapping": {
                "loaded": self._crypto_mapping is not None,
                "symbols": len(self._crypto_mapping) if self._crypto_mapping else 0
            },
            "mbfc_ratings": {
                "loaded": self._mbfc_ratings is not None,
                "sources": len(self._mbfc_ratings) if self._mbfc_ratings else 0
            },
            "allsides_bias": {
                "loaded": self._allsides_bias is not None,
                "sources": len(self._allsides_bias) if self._allsides_bias else 0
            }
        } 