#!/usr/bin/env python3
"""
Enhanced News Classifier with Advanced Features
===============================================

Advanced news classification system with comprehensive analysis including:
- Topic classification and categorization
- Named entity recognition (NER)
- Enhanced credibility scoring
- Market impact assessment
- Geopolitical risk analysis
- Sentiment intensity scoring
- Content quality metrics
- Source reliability evaluation

Author: Claude
Date: 2025-07-03
Version: 2.0
"""

import asyncio
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from loguru import logger
import pandas as pd
from collections import Counter
import spacy
from textblob import TextBlob

# Import existing MCPs
try:
    from ..integrations.coingecko_free import CoinGeckoFreeMCP
    from ..integrations.mbfc_free import MBFCFreeMCP
    from ..integrations.allsides_free import AllSidesFreeMCP
    from .finbert_classifier import FinBERTClassifier
    from .news_classifier import NewsClassifier
except ImportError:
    from integrations.coingecko_free import CoinGeckoFreeMCP
    from integrations.mbfc_free import MBFCFreeMCP
    from integrations.allsides_free import AllSidesFreeMCP
    from finbert_classifier import FinBERTClassifier
    from news_classifier import NewsClassifier

class EnhancedNewsClassifier:
    """Enhanced news classifier with advanced features"""
    
    def __init__(self, config: Dict = None):
        # Initialize base classifier
        self.base_classifier = NewsClassifier(config)
        
        # Initialize enhanced features
        self.topic_categories = {
            'financial_markets': ['market', 'stock', 'trading', 'investment', 'portfolio', 'earnings', 'revenue', 'profit'],
            'cryptocurrency': ['bitcoin', 'ethereum', 'crypto', 'blockchain', 'defi', 'nft', 'digital asset'],
            'technology': ['ai', 'artificial intelligence', 'machine learning', 'cloud', 'software', 'tech', 'innovation'],
            'geopolitics': ['trade war', 'sanctions', 'diplomatic', 'international', 'foreign policy', 'conflict'],
            'economic_policy': ['federal reserve', 'interest rate', 'inflation', 'gdp', 'economic growth', 'recession'],
            'energy': ['oil', 'gas', 'renewable', 'energy', 'petroleum', 'solar', 'wind'],
            'healthcare': ['pharmaceutical', 'biotech', 'medical', 'health', 'vaccine', 'drug'],
            'regulatory': ['regulation', 'compliance', 'policy', 'law', 'legal', 'government'],
            'corporate': ['merger', 'acquisition', 'ipo', 'management', 'leadership', 'strategy'],
            'cybersecurity': ['cyber', 'security', 'hack', 'breach', 'vulnerability', 'malware']
        }
        
        # Market impact indicators
        self.market_impact_keywords = {
            'high_impact': ['federal reserve', 'interest rate', 'inflation', 'gdp', 'earnings', 'merger', 'acquisition'],
            'medium_impact': ['market', 'stock', 'trading', 'revenue', 'profit', 'investment'],
            'low_impact': ['announcement', 'update', 'news', 'report', 'statement']
        }
        
        # Credibility indicators
        self.credibility_indicators = {
            'positive': ['research', 'analysis', 'data', 'study', 'report', 'evidence', 'official'],
            'negative': ['rumor', 'speculation', 'unconfirmed', 'allegedly', 'sources say', 'leaked'],
            'neutral': ['according to', 'reported', 'announced', 'stated', 'confirmed']
        }
        
        # Try to load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
            self.ner_available = True
        except OSError:
            logger.warning("SpaCy model not available. NER features will be limited.")
            self.nlp = None
            self.ner_available = False
        
        logger.info("✅ Enhanced News Classifier initialized")
    
    async def classify_news_enhanced(self, news_item: Dict) -> Dict:
        """Enhanced news classification with advanced features"""
        try:
            # Get base classification
            base_result = await self.base_classifier.classify_news(news_item)
            
            # Extract text for analysis
            title = news_item.get("title", "")
            content = news_item.get("content", "")
            full_text = f"{title}. {content}".strip()
            
            # Perform enhanced analysis
            enhanced_features = await self._perform_enhanced_analysis(full_text, news_item)
            
            # Combine results
            base_result.update(enhanced_features)
            
            # Calculate enhanced scores
            base_result.update(self._calculate_enhanced_scores(base_result))
            
            return base_result
            
        except Exception as e:
            logger.error(f"Error in enhanced classification: {e}")
            return await self.base_classifier.classify_news(news_item)
    
    async def _perform_enhanced_analysis(self, text: str, news_item: Dict) -> Dict:
        """Perform enhanced analysis on the text"""
        analysis = {}
        
        # Topic classification
        analysis.update(self._classify_topics(text))
        
        # Named entity recognition
        analysis.update(self._extract_entities(text))
        
        # Market impact assessment
        analysis.update(self._assess_market_impact(text))
        
        # Content quality metrics
        analysis.update(self._assess_content_quality(text, news_item))
        
        # Sentiment intensity
        analysis.update(self._assess_sentiment_intensity(text))
        
        # Credibility indicators
        analysis.update(self._assess_credibility_indicators(text))
        
        # Geopolitical risk
        analysis.update(self._assess_geopolitical_risk(text))
        
        return analysis
    
    def _classify_topics(self, text: str) -> Dict:
        """Classify news topics using keyword analysis"""
        text_lower = text.lower()
        topic_scores = {}
        
        for category, keywords in self.topic_categories.items():
            score = 0
            matched_keywords = []
            
            for keyword in keywords:
                count = text_lower.count(keyword)
                if count > 0:
                    score += count
                    matched_keywords.append(keyword)
            
            if score > 0:
                topic_scores[category] = {
                    'score': score,
                    'keywords': matched_keywords
                }
        
        # Determine primary topic
        primary_topic = max(topic_scores.keys(), key=lambda x: topic_scores[x]['score']) if topic_scores else 'general'
        
        return {
            'topic_classification': {
                'primary_topic': primary_topic,
                'topic_scores': topic_scores,
                'topic_count': len(topic_scores)
            }
        }
    
    def _extract_entities(self, text: str) -> Dict:
        """Extract named entities from text"""
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'money': [],
            'dates': [],
            'misc': []
        }
        
        if self.ner_available and self.nlp:
            try:
                doc = self.nlp(text)
                
                for ent in doc.ents:
                    if ent.label_ in ['PERSON']:
                        entities['persons'].append(ent.text)
                    elif ent.label_ in ['ORG']:
                        entities['organizations'].append(ent.text)
                    elif ent.label_ in ['GPE', 'LOC']:
                        entities['locations'].append(ent.text)
                    elif ent.label_ in ['MONEY']:
                        entities['money'].append(ent.text)
                    elif ent.label_ in ['DATE', 'TIME']:
                        entities['dates'].append(ent.text)
                    else:
                        entities['misc'].append(ent.text)
                
                # Remove duplicates
                for key in entities:
                    entities[key] = list(set(entities[key]))
                
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
        
        return {
            'named_entities': entities,
            'entity_count': sum(len(v) for v in entities.values())
        }
    
    def _assess_market_impact(self, text: str) -> Dict:
        """Assess potential market impact of the news"""
        text_lower = text.lower()
        impact_score = 0
        impact_level = 'low'
        impact_keywords = []
        
        # Check for high impact keywords
        for keyword in self.market_impact_keywords['high_impact']:
            count = text_lower.count(keyword)
            if count > 0:
                impact_score += count * 3
                impact_keywords.append(keyword)
        
        # Check for medium impact keywords
        for keyword in self.market_impact_keywords['medium_impact']:
            count = text_lower.count(keyword)
            if count > 0:
                impact_score += count * 2
                impact_keywords.append(keyword)
        
        # Check for low impact keywords
        for keyword in self.market_impact_keywords['low_impact']:
            count = text_lower.count(keyword)
            if count > 0:
                impact_score += count
                impact_keywords.append(keyword)
        
        # Determine impact level
        if impact_score >= 10:
            impact_level = 'high'
        elif impact_score >= 5:
            impact_level = 'medium'
        else:
            impact_level = 'low'
        
        return {
            'market_impact': {
                'score': impact_score,
                'level': impact_level,
                'keywords': list(set(impact_keywords))
            }
        }
    
    def _assess_content_quality(self, text: str, news_item: Dict) -> Dict:
        """Assess content quality metrics"""
        # Basic metrics
        word_count = len(text.split())
        sentence_count = len(text.split('.'))
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Check for quality indicators
        has_quotes = '"' in text or "'" in text
        has_numbers = bool(re.search(r'\d+', text))
        has_sources = any(word in text.lower() for word in ['according to', 'source', 'reported', 'confirmed'])
        
        # Calculate quality score
        quality_score = 0
        
        # Length quality
        if word_count > 300:
            quality_score += 2
        elif word_count > 150:
            quality_score += 1
        
        # Structure quality
        if 10 <= avg_sentence_length <= 25:
            quality_score += 1
        
        # Content quality
        if has_quotes:
            quality_score += 1
        if has_numbers:
            quality_score += 1
        if has_sources:
            quality_score += 1
        
        quality_level = 'high' if quality_score >= 5 else 'medium' if quality_score >= 3 else 'low'
        
        return {
            'content_quality': {
                'word_count': word_count,
                'sentence_count': sentence_count,
                'avg_sentence_length': avg_sentence_length,
                'has_quotes': has_quotes,
                'has_numbers': has_numbers,
                'has_sources': has_sources,
                'quality_score': quality_score,
                'quality_level': quality_level
            }
        }
    
    def _assess_sentiment_intensity(self, text: str) -> Dict:
        """Assess sentiment intensity using TextBlob"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Determine intensity
            if abs(polarity) > 0.6:
                intensity = 'high'
            elif abs(polarity) > 0.3:
                intensity = 'medium'
            else:
                intensity = 'low'
            
            return {
                'sentiment_intensity': {
                    'polarity': polarity,
                    'subjectivity': subjectivity,
                    'intensity': intensity
                }
            }
        except Exception as e:
            logger.warning(f"Sentiment intensity assessment failed: {e}")
            return {
                'sentiment_intensity': {
                    'polarity': 0.0,
                    'subjectivity': 0.0,
                    'intensity': 'unknown'
                }
            }
    
    def _assess_credibility_indicators(self, text: str) -> Dict:
        """Assess credibility indicators in the text"""
        text_lower = text.lower()
        
        positive_count = sum(text_lower.count(indicator) for indicator in self.credibility_indicators['positive'])
        negative_count = sum(text_lower.count(indicator) for indicator in self.credibility_indicators['negative'])
        neutral_count = sum(text_lower.count(indicator) for indicator in self.credibility_indicators['neutral'])
        
        # Calculate credibility score
        credibility_score = positive_count * 2 + neutral_count - negative_count * 2
        
        # Determine credibility level
        if credibility_score >= 3:
            credibility_level = 'high'
        elif credibility_score >= 0:
            credibility_level = 'medium'
        else:
            credibility_level = 'low'
        
        return {
            'credibility_indicators': {
                'positive_indicators': positive_count,
                'negative_indicators': negative_count,
                'neutral_indicators': neutral_count,
                'credibility_score': credibility_score,
                'credibility_level': credibility_level
            }
        }
    
    def _assess_geopolitical_risk(self, text: str) -> Dict:
        """Assess geopolitical risk factors"""
        text_lower = text.lower()
        
        geopolitical_keywords = [
            'war', 'conflict', 'sanctions', 'trade war', 'tariff', 'diplomatic',
            'military', 'tension', 'crisis', 'embargo', 'blockade', 'alliance'
        ]
        
        risk_score = 0
        risk_keywords = []
        
        for keyword in geopolitical_keywords:
            count = text_lower.count(keyword)
            if count > 0:
                risk_score += count
                risk_keywords.append(keyword)
        
        # Determine risk level
        if risk_score >= 3:
            risk_level = 'high'
        elif risk_score >= 1:
            risk_level = 'medium'
        else:
            risk_level = 'low'
        
        return {
            'geopolitical_risk': {
                'risk_score': risk_score,
                'risk_level': risk_level,
                'risk_keywords': list(set(risk_keywords))
            }
        }
    
    def _calculate_enhanced_scores(self, classification: Dict) -> Dict:
        """Calculate enhanced composite scores"""
        # Overall confidence score
        confidence_factors = []
        
        # Base confidence
        if classification.get('finbert_confidence'):
            confidence_factors.append(classification['finbert_confidence'])
        
        # Content quality factor
        content_quality = classification.get('content_quality', {})
        if content_quality.get('quality_score', 0) >= 4:
            confidence_factors.append(0.8)
        elif content_quality.get('quality_score', 0) >= 2:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.4)
        
        # Credibility factor
        credibility = classification.get('credibility_indicators', {})
        if credibility.get('credibility_level') == 'high':
            confidence_factors.append(0.9)
        elif credibility.get('credibility_level') == 'medium':
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        
        # Market relevance score
        market_impact = classification.get('market_impact', {})
        topic_classification = classification.get('topic_classification', {})
        
        market_relevance = 0.0
        if market_impact.get('level') == 'high':
            market_relevance += 0.4
        elif market_impact.get('level') == 'medium':
            market_relevance += 0.2
        
        # Add topic relevance
        financial_topics = ['financial_markets', 'cryptocurrency', 'economic_policy', 'corporate']
        if topic_classification.get('primary_topic') in financial_topics:
            market_relevance += 0.3
        
        market_relevance = min(market_relevance, 1.0)
        
        return {
            'enhanced_scores': {
                'overall_confidence': overall_confidence,
                'market_relevance': market_relevance,
                'composite_score': (overall_confidence + market_relevance) / 2
            }
        }
    
    def get_classification_summary(self, classification: Dict) -> Dict:
        """Generate a summary of the classification results"""
        return {
            'classification_summary': {
                'primary_topic': classification.get('topic_classification', {}).get('primary_topic', 'general'),
                'sentiment': classification.get('finbert_sentiment', 'neutral'),
                'market_impact': classification.get('market_impact', {}).get('level', 'low'),
                'credibility': classification.get('credibility_indicators', {}).get('credibility_level', 'medium'),
                'quality': classification.get('content_quality', {}).get('quality_level', 'medium'),
                'confidence': classification.get('enhanced_scores', {}).get('overall_confidence', 0.5),
                'market_relevance': classification.get('enhanced_scores', {}).get('market_relevance', 0.0)
            }
        }


async def main():
    """Test the enhanced classifier"""
    classifier = EnhancedNewsClassifier()
    
    # Test article
    test_article = {
        'title': 'Federal Reserve Announces Interest Rate Decision Affecting Bitcoin and Stock Markets',
        'content': 'The Federal Reserve announced today a significant interest rate decision that is expected to impact cryptocurrency markets and traditional stock trading. According to official sources, the decision comes after careful analysis of economic indicators and inflation data. Bitcoin prices have already shown volatility in response to the announcement, while major stock indices are experiencing mixed reactions.',
        'source_url': 'https://example.com/news',
        'published_date': '2025-07-03'
    }
    
    result = await classifier.classify_news_enhanced(test_article)
    
    print("Enhanced Classification Results:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main()) 