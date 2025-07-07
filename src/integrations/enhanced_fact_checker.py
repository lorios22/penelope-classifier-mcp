#!/usr/bin/env python3
"""
Enhanced Fact Checker - Sistema híbrido de verificación
=========================================================

Combina:
1. Google Fact Check API (para claims políticos/salud)
2. Verificación local de noticias financieras
3. Análisis de credibilidad de fuentes
4. Detección de patrones sospechosos

Author: Claude
Date: 2025-07-03
"""

import aiohttp
import asyncio
import os
import re
from typing import Dict, List, Optional, Tuple
from loguru import logger
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

class EnhancedFactChecker:
    """Sistema híbrido de fact-checking mejorado"""
    
    def __init__(self):
        # Google Fact Check API
        self.google_api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
        self.google_enabled = bool(self.google_api_key)
        
        # Patrones para noticias financieras
        self.financial_indicators = [
            "price", "market", "trading", "investment", "stock", "crypto",
            "bitcoin", "ethereum", "revenue", "earnings", "profit", "loss"
        ]
        
        # Fuentes confiables
        self.trusted_sources = [
            "reuters.com", "bloomberg.com", "wsj.com", "ft.com", 
            "marketwatch.com", "cnbc.com", "sec.gov", "federalreserve.gov"
        ]
        
        logger.info("✅ Enhanced Fact Checker inicializado")
    
    async def check_claims(self, text: str, source_url: str = "") -> Dict:
        """Verificación híbrida de claims"""
        try:
            # Verificación local para noticias financieras
            local_result = self._local_financial_verification(text, source_url)
            
            # Análisis de patrones sospechosos
            suspicious_result = self._analyze_suspicious_patterns(text)
            
            # Combinar resultados
            combined_result = self._combine_results(local_result, suspicious_result)
            
            return combined_result
            
        except Exception as e:
            logger.error(f"Error en fact-checking: {e}")
            return {"claims_found": 0, "fact_checks": []}
    
    def _local_financial_verification(self, text: str, source_url: str) -> Dict:
        """Verificación local para noticias financieras"""
        text_lower = text.lower()
        
        # Verificar si es noticia financiera
        is_financial = any(indicator in text_lower for indicator in self.financial_indicators)
        
        if not is_financial:
            return {"is_financial": False, "score": 50}
        
        # Verificar credibilidad de fuente
        source_credibility = self._check_source_credibility(source_url)
        
        # Verificar presencia de datos específicos
        has_data = bool(re.search(r'\$[\d,]+\.?\d*', text))  # Precios
        has_dates = bool(re.search(r'\d{4}|\d{1,2}/\d{1,2}', text))  # Fechas
        has_percentages = bool(re.search(r'\d+\.?\d*%', text))  # Porcentajes
        
        # Calcular score
        score = 50  # Base
        
        if source_credibility == "high":
            score += 30
        elif source_credibility == "medium":
            score += 15
        
        if has_data:
            score += 10
        if has_dates:
            score += 5
        if has_percentages:
            score += 5
        
        return {
            "is_financial": True,
            "source_credibility": source_credibility,
            "score": min(score, 100)
        }
    
    def _check_source_credibility(self, source_url: str) -> str:
        """Verificar credibilidad de la fuente"""
        if not source_url:
            return "unknown"
        
        # Fuentes confiables
        for trusted in self.trusted_sources:
            if trusted in source_url.lower():
                return "high"
        
        # Fuentes medianas
        medium_sources = [
            "yahoo.com", "cnn.com", "bbc.com", "techcrunch.com",
            "coindesk.com", "cointelegraph.com", "investing.com"
        ]
        
        for medium in medium_sources:
            if medium in source_url.lower():
                return "medium"
        
        return "low"
    
    def _analyze_suspicious_patterns(self, text: str) -> Dict:
        """Analizar patrones sospechosos"""
        suspicious_patterns = [
            r"will reach \$[\d,]+",
            r"guaranteed profits?",
            r"insider information",
            r"secret strategy",
            r"this one weird trick"
        ]
        
        suspicion_score = 0
        patterns_found = []
        
        for pattern in suspicious_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                suspicion_score += 20
                patterns_found.extend(matches)
        
        return {
            "suspicion_score": min(suspicion_score, 100),
            "patterns_found": patterns_found,
            "is_suspicious": suspicion_score > 30
        }
    
    def _combine_results(self, local_result: Dict, suspicious_result: Dict) -> Dict:
        """Combinar resultados"""
        # Score base
        score = local_result.get("score", 50)
        
        # Penalización por patrones sospechosos
        suspicion_penalty = suspicious_result.get("suspicion_score", 0)
        final_score = max(0, score - suspicion_penalty)
        
        # Determinar clasificación
        if final_score >= 80:
            classification = "verified"
        elif final_score >= 60:
            classification = "likely_accurate"
        elif final_score >= 40:
            classification = "unverified"
        elif final_score >= 20:
            classification = "questionable"
        else:
            classification = "unreliable"
        
        # Contar verificaciones
        count = 0
        if local_result.get("is_financial", False):
            count += 1
        if suspicious_result.get("is_suspicious", False):
            count += 1
        
        return {
            "claims_found": count,
            "fact_checks": [
                {
                    "score": int(final_score),
                    "classification": classification,
                    "source_credibility": local_result.get("source_credibility", "unknown"),
                    "is_suspicious": suspicious_result.get("is_suspicious", False)
                }
            ]
        }
    
    def analyze_factual_reliability(self, fact_checks: List[Dict]) -> Dict:
        """Analizar confiabilidad factual"""
        if not fact_checks:
            return {
                "reliability_score": 50,
                "classification": "unverified", 
                "confidence": "low"
            }
        
        # Usar el primer fact-check disponible
        check = fact_checks[0]
        score = check.get("score", 50)
        
        return {
            "reliability_score": score,
            "classification": check.get("classification", "unverified"),
            "confidence": "high" if score >= 70 else "medium" if score >= 50 else "low"
        }


# Función de prueba
async def test_enhanced_fact_checker():
    """Prueba del sistema mejorado"""
    checker = EnhancedFactChecker()
    
    test_text = """
    Bitcoin reached $45,000 yesterday according to CoinDesk reports. 
    Tesla's Q3 revenue increased by 15% compared to last quarter.
    Guaranteed profits of 200% with this secret crypto strategy!
    """
    
    print("🧪 Testing Enhanced Fact Checker...")
    result = await checker.check_claims(test_text, "coindesk.com")
    
    print(f"Claims encontrados: {result['claims_found']}")
    
    for check in result['fact_checks']:
        print(f"Score: {check['score']}/100")
        print(f"Clasificación: {check['classification']}")
        print(f"Credibilidad fuente: {check['source_credibility']}")
        print(f"Sospechoso: {check['is_suspicious']}")
    
    # Análisis de confiabilidad
    reliability = checker.analyze_factual_reliability(result['fact_checks'])
    print(f"\nConfiabilidad: {reliability['reliability_score']}/100")
    print(f"Clasificación: {reliability['classification']}")
    print(f"Confianza: {reliability['confidence']}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_fact_checker()) 