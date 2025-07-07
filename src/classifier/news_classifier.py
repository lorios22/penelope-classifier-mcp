#!/usr/bin/env python3
"""
Clasificador Principal de Noticias - Combina todos los MCPs gratuitos
"""

import asyncio
import csv
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
from loguru import logger
import pandas as pd
from dotenv import load_dotenv

# Importar MCPs gratuitos
try:
    from ..integrations.coingecko_free import CoinGeckoFreeMCP
    from ..integrations.enhanced_fact_checker import EnhancedFactChecker
    from ..integrations.mbfc_free import MBFCFreeMCP
    from ..integrations.allsides_free import AllSidesFreeMCP
    from .finbert_classifier import FinBERTClassifier
except ImportError:
    # Imports absolutos para cuando se ejecuta directamente
    from integrations.coingecko_free import CoinGeckoFreeMCP
    from integrations.enhanced_fact_checker import EnhancedFactChecker
    from integrations.mbfc_free import MBFCFreeMCP
    from integrations.allsides_free import AllSidesFreeMCP
    from finbert_classifier import FinBERTClassifier

load_dotenv()

class NewsClassifier:
    """Clasificador principal que combina todos los MCPs gratuitos"""
    
    def __init__(self, config: Dict = None):
        self.config = config or self._load_config()
        
        # Inicializar MCPs
        self.coingecko = CoinGeckoFreeMCP()
        self.fact_checker = EnhancedFactChecker()
        self.mbfc = MBFCFreeMCP()
        self.allsides = AllSidesFreeMCP()
        self.finbert = FinBERTClassifier()
        
        # Configurar output
        self.output_dir = Path(self.config.get("output_dir", "data/results"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurar logging
        self._setup_logging()
        
        logger.info("🚀 NewsClassifier inicializado con todos los MCPs gratuitos")
    
    def _load_config(self) -> Dict:
        """Cargar configuración desde variables de entorno"""
        return {
            "output_dir": os.getenv("OUTPUT_DIR", "data/results"),
            "csv_output_file": os.getenv("CSV_OUTPUT_FILE", "news_classification_results.csv"),
            "enable_fact_checking": os.getenv("ENABLE_FACT_CHECKING", "true").lower() == "true",
            "enable_market_data": os.getenv("ENABLE_MARKET_DATA", "true").lower() == "true",
            "enable_bias_detection": os.getenv("ENABLE_BIAS_DETECTION", "true").lower() == "true",
            "enable_finbert": os.getenv("ENABLE_FINBERT_CLASSIFICATION", "true").lower() == "true",
            "request_timeout": float(os.getenv("REQUEST_TIMEOUT", "30.0")),
            "max_concurrent": int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
        }
    
    def _setup_logging(self):
        """Configurar logging"""
        log_level = os.getenv("LOG_LEVEL", "INFO")
        logger.remove()
        logger.add(
            "logs/news_classifier.log",
            rotation="10 MB",
            retention="7 days",
            level=log_level,
            format="{time} | {level} | {message}"
        )
        logger.add(
            lambda msg: print(msg),
            level=log_level,
            format="{time} | {level} | {message}"
        )
    
    async def classify_news(self, news_item: Dict) -> Dict:
        """Clasificar una noticia usando todos los MCPs disponibles"""
        try:
            # Extraer información básica
            title = news_item.get("title", "")
            content = news_item.get("content", "")
            source_url = news_item.get("source_url", "")
            published_date = news_item.get("published_date", "")
            
            # Combinar texto para análisis
            full_text = f"{title}. {content}".strip()
            
            # Realizar análisis en paralelo
            tasks = []
            
            # 1. Análisis de sentiment financiero con FinBERT
            if self.config["enable_finbert"]:
                tasks.append(self._analyze_finbert_sentiment(full_text))
            
            # 2. Análisis de bias de fuente
            if self.config["enable_bias_detection"]:
                tasks.append(self._analyze_source_bias(source_url))
            
            # 3. Fact-checking
            if self.config["enable_fact_checking"]:
                tasks.append(self._analyze_fact_checking(full_text, source_url))
            
            # 4. Análisis de mercado crypto
            if self.config["enable_market_data"]:
                tasks.append(self._analyze_crypto_market(full_text))
            
            # Ejecutar análisis en paralelo
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados
            finbert_result = results[0] if len(results) > 0 else {}
            bias_result = results[1] if len(results) > 1 else {}
            fact_check_result = results[2] if len(results) > 2 else {}
            crypto_result = results[3] if len(results) > 3 else {}
            
            # Combinar todos los análisis
            classification = {
                "timestamp": datetime.now().isoformat(),
                "title": title,
                "content": content[:500] + "..." if len(content) > 500 else content,
                "source_url": source_url,
                "published_date": published_date,
                
                # Análisis FinBERT
                "finbert_sentiment": finbert_result.get("sentiment", "neutral"),
                "finbert_confidence": finbert_result.get("confidence", 0.0),
                "is_financial_news": finbert_result.get("is_financial", False),
                
                # Análisis de bias
                "source_bias": bias_result.get("bias", "unknown"),
                "source_credibility": bias_result.get("credibility", 50),
                "political_lean": bias_result.get("political_lean", "center"),
                
                # Fact-checking
                "fact_check_score": fact_check_result.get("score", 50),
                "fact_check_classification": fact_check_result.get("classification", "unverified"),
                "fact_checks_found": fact_check_result.get("count", 0),
                
                # Análisis crypto
                "crypto_mentions": crypto_result.get("mentions", []),
                "crypto_sentiment": crypto_result.get("sentiment", "neutral"),
                "crypto_price_data": crypto_result.get("price_data", {}),
                
                # Clasificación final
                "overall_classification": "",
                "confidence_score": 0.0,
                "reliability_score": 0.0
            }
            
            # Generar clasificación final
            classification.update(self._generate_final_classification(classification))
            
            return classification
            
        except Exception as e:
            logger.error(f"Error clasificando noticia: {e}")
            return self._get_error_classification(news_item, str(e))
    
    async def _analyze_finbert_sentiment(self, text: str) -> Dict:
        """Analizar sentiment con FinBERT"""
        try:
            if not text:
                return {}
            
            result = self.finbert.classify_sentiment(text)
            return {
                "sentiment": result.get("sentiment", "neutral"),
                "confidence": result.get("confidence", 0.0),
                "is_financial": result.get("is_financial", False),
                "scores": result.get("scores", {})
            }
        except Exception as e:
            logger.error(f"Error FinBERT: {e}")
            return {}
    
    async def _analyze_source_bias(self, source_url: str) -> Dict:
        """Analizar bias de fuente"""
        try:
            if not source_url:
                return {}
            
            # Análisis MBFC
            mbfc_result = self.mbfc.get_source_analysis(source_url)
            
            # Análisis AllSides
            allsides_result = self.allsides.get_bias_analysis(source_url)
            
            return {
                "bias": mbfc_result.get("bias_rating", "unknown"),
                "credibility": mbfc_result.get("credibility_score", 50),
                "political_lean": allsides_result.get("political_lean", "center"),
                "mbfc_found": mbfc_result.get("found", False),
                "allsides_found": allsides_result.get("found", False)
            }
        except Exception as e:
            logger.error(f"Error análisis bias: {e}")
            return {}
    
    async def _analyze_fact_checking(self, text: str, source_url: str = "") -> Dict:
        """Analizar fact-checking con sistema mejorado"""
        try:
            if not text:
                return {}
            
            result = await self.fact_checker.check_claims(text, source_url)
            reliability = self.fact_checker.analyze_factual_reliability(
                result.get("fact_checks", [])
            )
            
            return {
                "score": reliability.get("reliability_score", 50),
                "classification": reliability.get("classification", "unverified"),
                "count": result.get("claims_found", 0),
                "confidence": reliability.get("confidence", "low")
            }
        except Exception as e:
            logger.error(f"Error fact-checking: {e}")
            return {}
    
    async def _analyze_crypto_market(self, text: str) -> Dict:
        """Analizar menciones crypto y mercado"""
        try:
            if not text:
                return {}
            
            # Extraer menciones crypto
            mentions = self.coingecko.extract_crypto_mentions(text)
            
            if not mentions:
                return {"mentions": [], "sentiment": "neutral"}
            
            # Analizar sentiment crypto
            crypto_analysis = await self.coingecko.analyze_crypto_sentiment(text)
            
            return {
                "mentions": mentions,
                "sentiment": crypto_analysis.get("recommendation", "neutral"),
                "price_data": crypto_analysis.get("price_data", {}),
                "analysis": crypto_analysis.get("analysis", "no_crypto_detected")
            }
        except Exception as e:
            logger.error(f"Error análisis crypto: {e}")
            return {}
    
    def _generate_final_classification(self, classification: Dict) -> Dict:
        """Generar clasificación final combinando todos los análisis"""
        try:
            # Calcular confidence score
            confidence_factors = []
            
            if classification.get("finbert_confidence", 0) > 0:
                confidence_factors.append(classification["finbert_confidence"])
            
            if classification.get("source_credibility", 0) > 50:
                confidence_factors.append(classification["source_credibility"] / 100)
            
            if classification.get("fact_check_score", 0) > 50:
                confidence_factors.append(classification["fact_check_score"] / 100)
            
            avg_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
            # Calcular reliability score
            reliability_factors = [
                classification.get("source_credibility", 50),
                classification.get("fact_check_score", 50)
            ]
            
            avg_reliability = sum(reliability_factors) / len(reliability_factors)
            
            # Generar clasificación final
            if avg_reliability >= 80 and avg_confidence >= 0.8:
                final_classification = "highly_reliable"
            elif avg_reliability >= 60 and avg_confidence >= 0.6:
                final_classification = "reliable"
            elif avg_reliability >= 40 and avg_confidence >= 0.4:
                final_classification = "moderate"
            elif avg_reliability >= 20:
                final_classification = "questionable"
            else:
                final_classification = "unreliable"
            
            return {
                "overall_classification": final_classification,
                "confidence_score": avg_confidence,
                "reliability_score": avg_reliability
            }
            
        except Exception as e:
            logger.error(f"Error generando clasificación final: {e}")
            return {
                "overall_classification": "error",
                "confidence_score": 0.0,
                "reliability_score": 0.0
            }
    
    def _get_error_classification(self, news_item: Dict, error: str) -> Dict:
        """Generar clasificación de error"""
        return {
            "timestamp": datetime.now().isoformat(),
            "title": news_item.get("title", ""),
            "content": news_item.get("content", "")[:500],
            "source_url": news_item.get("source_url", ""),
            "published_date": news_item.get("published_date", ""),
            "error": error,
            "overall_classification": "error",
            "confidence_score": 0.0,
            "reliability_score": 0.0
        }
    
    async def classify_news_batch(self, news_items: List[Dict]) -> List[Dict]:
        """Clasificar múltiples noticias en batch"""
        logger.info(f"🔄 Clasificando {len(news_items)} noticias...")
        
        results = []
        semaphore = asyncio.Semaphore(self.config["max_concurrent"])
        
        async def classify_with_semaphore(item):
            async with semaphore:
                return await self.classify_news(item)
        
        # Ejecutar clasificaciones en paralelo con límite
        tasks = [classify_with_semaphore(item) for item in news_items]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Procesar resultados
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error en noticia {i}: {result}")
                processed_results.append(self._get_error_classification(news_items[i], str(result)))
            else:
                processed_results.append(result)
        
        logger.info(f"✅ Clasificación completada: {len(processed_results)} noticias")
        return processed_results
    
    def save_to_csv(self, classifications: List[Dict], filename: str = None) -> str:
        """Guardar clasificaciones en CSV"""
        try:
            if not classifications:
                logger.warning("No hay clasificaciones para guardar")
                return ""
            
            filename = filename or self.config["csv_output_file"]
            filepath = self.output_dir / filename
            
            # Definir columnas CSV
            fieldnames = [
                "timestamp", "title", "content", "source_url", "published_date",
                "finbert_sentiment", "finbert_confidence", "is_financial_news",
                "source_bias", "source_credibility", "political_lean",
                "fact_check_score", "fact_check_classification", "fact_checks_found",
                "crypto_mentions", "crypto_sentiment", "crypto_price_data",
                "overall_classification", "confidence_score", "reliability_score",
                "error"
            ]
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                
                for classification in classifications:
                    # Convertir listas y dicts a strings para CSV
                    row = classification.copy()
                    
                    if isinstance(row.get("crypto_mentions"), list):
                        row["crypto_mentions"] = ", ".join(row["crypto_mentions"])
                    
                    if isinstance(row.get("crypto_price_data"), dict):
                        row["crypto_price_data"] = json.dumps(row["crypto_price_data"])
                    
                    writer.writerow(row)
            
            logger.info(f"📊 Resultados guardados en: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error guardando CSV: {e}")
            return ""
    
    def get_summary_stats(self, classifications: List[Dict]) -> Dict:
        """Obtener estadísticas resumen"""
        if not classifications:
            return {}
        
        # Contar clasificaciones
        classification_counts = {}
        sentiment_counts = {}
        bias_counts = {}
        financial_count = 0
        crypto_count = 0
        fact_checked_count = 0
        
        for c in classifications:
            if c.get("error"):
                continue
            
            # Clasificación general
            overall = c.get("overall_classification", "unknown")
            classification_counts[overall] = classification_counts.get(overall, 0) + 1
            
            # Sentiment
            sentiment = c.get("finbert_sentiment", "neutral")
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            
            # Bias
            bias = c.get("source_bias", "unknown")
            bias_counts[bias] = bias_counts.get(bias, 0) + 1
            
            # Contadores
            if c.get("is_financial_news"):
                financial_count += 1
            
            if c.get("crypto_mentions"):
                crypto_count += 1
            
            if c.get("fact_checks_found", 0) > 0:
                fact_checked_count += 1
        
        return {
            "total_news": len(classifications),
            "classification_distribution": classification_counts,
            "sentiment_distribution": sentiment_counts,
            "bias_distribution": bias_counts,
            "financial_news": financial_count,
            "crypto_news": crypto_count,
            "fact_checked_news": fact_checked_count,
            "financial_percentage": (financial_count / len(classifications)) * 100,
            "crypto_percentage": (crypto_count / len(classifications)) * 100,
            "fact_checked_percentage": (fact_checked_count / len(classifications)) * 100
        }

# Funciones utilitarias
def load_news_from_json(filepath: str) -> List[Dict]:
    """Cargar noticias desde archivo JSON"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "articles" in data:
            return data["articles"]
        else:
            return [data]
    except Exception as e:
        logger.error(f"Error cargando JSON: {e}")
        return []

def create_sample_news() -> List[Dict]:
    """Crear noticias de ejemplo para testing"""
    return [
        {
            "title": "Bitcoin Surges to New All-Time High Amid Institutional Adoption",
            "content": "Bitcoin reached a new all-time high today as major institutions continue to adopt cryptocurrency. The bullish trend is expected to continue as more companies add Bitcoin to their balance sheets.",
            "source_url": "https://www.coindesk.com",
            "published_date": "2024-01-15T10:30:00Z"
        },
        {
            "title": "Federal Reserve Announces Interest Rate Hike",
            "content": "The Federal Reserve announced a 0.25% interest rate hike today, citing inflation concerns. Markets reacted negatively to the news with major indices falling.",
            "source_url": "https://www.reuters.com",
            "published_date": "2024-01-15T14:00:00Z"
        },
        {
            "title": "Climate Change Hoax Claims Debunked by Scientists",
            "content": "A new study published in Nature provides overwhelming evidence that climate change is real and caused by human activities. The research debunks common conspiracy theories.",
            "source_url": "https://www.bbc.com",
            "published_date": "2024-01-15T09:15:00Z"
        },
        {
            "title": "Tesla Stock Plummets After Earnings Miss",
            "content": "Tesla shares fell 15% in after-hours trading following disappointing quarterly earnings. The electric vehicle maker cited supply chain challenges and increased competition.",
            "source_url": "https://www.cnbc.com",
            "published_date": "2024-01-15T16:45:00Z"
        }
    ]

# Función principal para testing
async def main():
    """Función principal para testing"""
    classifier = NewsClassifier()
    
    # Crear noticias de ejemplo
    sample_news = create_sample_news()
    
    # Clasificar noticias
    results = await classifier.classify_news_batch(sample_news)
    
    # Guardar resultados
    csv_path = classifier.save_to_csv(results)
    
    # Mostrar estadísticas
    stats = classifier.get_summary_stats(results)
    
    print("🎉 Clasificación completada!")
    print(f"📊 Total noticias: {stats['total_news']}")
    print(f"💰 Noticias financieras: {stats['financial_percentage']:.1f}%")
    print(f"🪙 Noticias crypto: {stats['crypto_percentage']:.1f}%")
    print(f"🔍 Noticias fact-checked: {stats['fact_checked_percentage']:.1f}%")
    print(f"📁 Archivo CSV: {csv_path}")

if __name__ == "__main__":
    asyncio.run(main()) 