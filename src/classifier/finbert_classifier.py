#!/usr/bin/env python3
"""
FinBERT Classifier - Modelo local gratuito para análisis de sentiment financiero
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
import numpy as np
import re
from datetime import datetime

class FinBERTClassifier:
    """Clasificador FinBERT para sentiment financiero"""
    
    MODEL_NAME = "ProsusAI/finbert"
    
    def __init__(self, model_cache_dir: str = "data/models", max_length: int = 512):
        self.model_cache_dir = Path(model_cache_dir)
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.classifier = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Cargar keywords financieras
        self.financial_keywords = self._load_financial_keywords()
        
        # Inicializar modelo
        self._initialize_model()
    
    def _load_financial_keywords(self) -> Dict:
        """Cargar keywords financieras desde archivo"""
        try:
            import json
            keywords_file = Path("data/datasets/financial_keywords.json")
            
            if keywords_file.exists():
                with open(keywords_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            else:
                logger.warning("Financial keywords file not found, using defaults")
                return self._get_default_keywords()
        except Exception as e:
            logger.error(f"Error loading financial keywords: {e}")
            return self._get_default_keywords()
    
    def _get_default_keywords(self) -> Dict:
        """Keywords financieras por defecto"""
        return {
            "crypto": ["bitcoin", "ethereum", "crypto", "blockchain", "defi"],
            "stocks": ["stock", "share", "equity", "market", "trading"],
            "economy": ["inflation", "recession", "gdp", "fed", "interest"],
            "sentiment": ["bullish", "bearish", "optimistic", "pessimistic"]
        }
    
    def _initialize_model(self):
        """Inicializar modelo FinBERT"""
        try:
            logger.info("🤖 Inicializando FinBERT...")
            
            # Configurar cache
            cache_dir = str(self.model_cache_dir)
            
            # Cargar tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.MODEL_NAME,
                cache_dir=cache_dir,
                model_max_length=self.max_length
            )
            
            # Cargar modelo
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.MODEL_NAME,
                cache_dir=cache_dir
            )
            
            # Mover a device apropiado
            self.model.to(self.device)
            
            # Crear pipeline
            self.classifier = pipeline(
                "text-classification",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )
            
            logger.info(f"✅ FinBERT inicializado en {self.device}")
            
        except Exception as e:
            logger.error(f"Error inicializando FinBERT: {e}")
            self.classifier = None
    
    def is_financial_text(self, text: str) -> bool:
        """Determinar si el texto es financiero"""
        text_lower = text.lower()
        
        # Contar keywords financieras
        financial_count = 0
        for category, keywords in self.financial_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    financial_count += 1
        
        # Considerar financiero si tiene al menos 2 keywords
        return financial_count >= 2
    
    def preprocess_text(self, text: str) -> str:
        """Preprocesar texto para FinBERT"""
        if not text:
            return ""
        
        # Limpiar texto
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        text = re.sub(r'@[^\s]+', '', text)
        text = re.sub(r'#[^\s]+', '', text)
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Limitar longitud
        if len(text) > self.max_length * 4:  # Aproximadamente 4 chars por token
            text = text[:self.max_length * 4]
        
        return text.strip()
    
    def classify_sentiment(self, text: str) -> Dict:
        """Clasificar sentiment financiero de un texto"""
        if not self.classifier:
            return {
                "error": "FinBERT not initialized",
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {}
            }
        
        if not text or not text.strip():
            return {
                "error": "Empty text",
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {}
            }
        
        try:
            # Preprocesar texto
            processed_text = self.preprocess_text(text)
            
            if not processed_text:
                return {
                    "error": "Text became empty after preprocessing",
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "scores": {}
                }
            
            # Clasificar con FinBERT
            results = self.classifier(processed_text)
            
            # Procesar resultados
            scores = {}
            max_score = 0
            predicted_sentiment = "neutral"
            
            for result in results[0]:
                label = result['label'].lower()
                score = result['score']
                scores[label] = score
                
                if score > max_score:
                    max_score = score
                    predicted_sentiment = label
            
            # Normalizar sentiment labels
            sentiment_map = {
                "positive": "bullish",
                "negative": "bearish",
                "neutral": "neutral"
            }
            
            normalized_sentiment = sentiment_map.get(predicted_sentiment, predicted_sentiment)
            
            return {
                "sentiment": normalized_sentiment,
                "confidence": max_score,
                "scores": scores,
                "is_financial": self.is_financial_text(text),
                "text_length": len(processed_text),
                "model": "finbert"
            }
            
        except Exception as e:
            logger.error(f"Error clasificando sentiment: {e}")
            return {
                "error": str(e),
                "sentiment": "neutral",
                "confidence": 0.0,
                "scores": {}
            }
    
    def batch_classify(self, texts: List[str]) -> List[Dict]:
        """Clasificar múltiples textos en batch"""
        results = []
        
        for text in texts:
            result = self.classify_sentiment(text)
            results.append(result)
        
        return results
    
    def analyze_news_sentiment(self, title: str, content: str = "") -> Dict:
        """Analizar sentiment de noticia completa"""
        # Combinar título y contenido
        full_text = f"{title}. {content}".strip()
        
        # Clasificar sentiment
        sentiment_result = self.classify_sentiment(full_text)
        
        # Análisis adicional
        title_sentiment = self.classify_sentiment(title) if title else {}
        content_sentiment = self.classify_sentiment(content) if content else {}
        
        return {
            "overall_sentiment": sentiment_result,
            "title_sentiment": title_sentiment,
            "content_sentiment": content_sentiment,
            "is_financial_news": self.is_financial_text(full_text),
            "financial_keywords": self._extract_financial_keywords(full_text),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _extract_financial_keywords(self, text: str) -> Dict:
        """Extraer keywords financieras del texto"""
        text_lower = text.lower()
        found_keywords = {}
        
        for category, keywords in self.financial_keywords.items():
            found_in_category = []
            for keyword in keywords:
                if keyword in text_lower:
                    found_in_category.append(keyword)
            
            if found_in_category:
                found_keywords[category] = found_in_category
        
        return found_keywords
    
    def get_sentiment_summary(self, classifications: List[Dict]) -> Dict:
        """Obtener resumen de sentiment de múltiples clasificaciones"""
        if not classifications:
            return {
                "total_texts": 0,
                "sentiment_distribution": {},
                "avg_confidence": 0.0,
                "dominant_sentiment": "neutral"
            }
        
        # Contar sentiments
        sentiment_counts = {}
        confidences = []
        financial_count = 0
        
        for classification in classifications:
            if "error" in classification:
                continue
            
            sentiment = classification.get("sentiment", "neutral")
            confidence = classification.get("confidence", 0.0)
            is_financial = classification.get("is_financial", False)
            
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            confidences.append(confidence)
            
            if is_financial:
                financial_count += 1
        
        avg_confidence = np.mean(confidences) if confidences else 0.0
        dominant_sentiment = max(sentiment_counts, key=sentiment_counts.get) if sentiment_counts else "neutral"
        
        return {
            "total_texts": len(classifications),
            "sentiment_distribution": sentiment_counts,
            "avg_confidence": avg_confidence,
            "dominant_sentiment": dominant_sentiment,
            "financial_texts": financial_count,
            "financial_percentage": (financial_count / len(classifications)) * 100 if classifications else 0
        }
    
    def get_model_info(self) -> Dict:
        """Obtener información del modelo"""
        return {
            "model_name": self.MODEL_NAME,
            "device": self.device,
            "max_length": self.max_length,
            "cache_dir": str(self.model_cache_dir),
            "initialized": self.classifier is not None,
            "financial_keywords_loaded": len(self.financial_keywords) > 0
        }

# Función de prueba
def test_finbert_classifier():
    """Función de prueba para FinBERT Classifier"""
    classifier = FinBERTClassifier()
    
    if not classifier.classifier:
        print("❌ FinBERT no inicializado")
        return
    
    # Tests de ejemplo
    test_texts = [
        "Bitcoin is going to the moon! Great investment opportunity.",
        "The stock market crashed today, investors are panicking.",
        "Federal Reserve announces interest rate hike.",
        "Apple reports strong quarterly earnings.",
        "Cryptocurrency market shows bearish trends.",
        "This is just a normal sentence about cats."
    ]
    
    print("🧪 Testing FinBERT Classifier...")
    print(f"Model info: {classifier.get_model_info()}")
    
    for text in test_texts:
        result = classifier.classify_sentiment(text)
        print(f"\n📄 Text: {text[:50]}...")
        print(f"   Sentiment: {result.get('sentiment', 'error')}")
        print(f"   Confidence: {result.get('confidence', 0):.3f}")
        print(f"   Is Financial: {result.get('is_financial', False)}")
        
        if 'scores' in result:
            for label, score in result['scores'].items():
                print(f"   {label}: {score:.3f}")
    
    # Test batch
    print(f"\n📊 Batch Analysis:")
    batch_results = classifier.batch_classify(test_texts)
    summary = classifier.get_sentiment_summary(batch_results)
    print(f"   Total texts: {summary['total_texts']}")
    print(f"   Dominant sentiment: {summary['dominant_sentiment']}")
    print(f"   Avg confidence: {summary['avg_confidence']:.3f}")
    print(f"   Financial texts: {summary['financial_texts']}")

if __name__ == "__main__":
    test_finbert_classifier() 