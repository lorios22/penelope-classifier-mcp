#!/usr/bin/env python3
"""
Google Fact Check MCP - API gratuita (1000 requests/día)
"""

import aiohttp
import asyncio
import os
import re
from typing import Dict, List, Optional
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

class GoogleFactCheckFreeMCP:
    """Google Fact Check API - completamente gratuita"""
    
    BASE_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"
    
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_FACT_CHECK_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ GOOGLE_FACT_CHECK_API_KEY no configurada - fact checking deshabilitado")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ Google Fact Check API configurada")
    
    async def check_claims(self, text: str, max_queries: int = 3) -> Dict:
        """Verificar claims en el texto - 1000 requests/día gratis"""
        if not self.enabled:
            return {"claims_found": 0, "fact_checks": [], "error": "api_key_missing"}
        
        try:
            # Extraer frases clave para búsqueda
            search_queries = self._extract_search_terms(text, max_queries)
            
            if not search_queries:
                return {"claims_found": 0, "fact_checks": [], "queries_used": []}
            
            fact_checks = []
            successful_queries = []
            
            for query in search_queries:
                result = await self._search_fact_checks(query)
                if result:
                    fact_checks.extend(result)
                    successful_queries.append(query)
                
                # Pequeña pausa para respetar rate limits
                await asyncio.sleep(0.1)
            
            # Eliminar duplicados y limitar resultados
            unique_checks = self._deduplicate_fact_checks(fact_checks)
            
            return {
                "claims_found": len(unique_checks),
                "fact_checks": unique_checks[:5],  # Top 5 resultados
                "queries_used": successful_queries,
                "total_queries": len(search_queries)
            }
            
        except Exception as e:
            logger.error(f"Error Fact Check: {e}")
            return {"claims_found": 0, "fact_checks": [], "error": str(e)}
    
    async def _search_fact_checks(self, query: str) -> List[Dict]:
        """Buscar fact-checks para una query específica"""
        try:
            params = {
                "query": query,
                "key": self.api_key,
                "pageSize": 10,
                "languageCode": "en"  # Puedes cambiar a "es" para español
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.BASE_URL, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        claims = data.get("claims", [])
                        return self._parse_fact_checks(claims)
                    elif response.status == 429:
                        logger.warning("Rate limit alcanzado para Google Fact Check")
                        return []
                    else:
                        logger.error(f"Error API Fact Check: {response.status}")
                        return []
        
        except Exception as e:
            logger.error(f"Error buscando fact-checks: {e}")
            return []
    
    def _extract_search_terms(self, text: str, max_terms: int = 3) -> List[str]:
        """Extraer términos de búsqueda del texto"""
        try:
            # Limpiar el texto
            text = re.sub(r'[^\w\s]', '', text)
            
            # Dividir en oraciones
            sentences = re.split(r'[.!?]', text)
            
            # Filtrar oraciones relevantes
            relevant_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20 and len(sentence) < 200:
                    # Buscar oraciones con claims potenciales
                    claim_indicators = [
                        "according to", "reports", "claims", "states", "says",
                        "announced", "confirmed", "revealed", "discovered",
                        "study shows", "research", "data", "statistics"
                    ]
                    
                    if any(indicator in sentence.lower() for indicator in claim_indicators):
                        relevant_sentences.append(sentence)
            
            # Si no hay oraciones con indicadores, usar las más largas
            if not relevant_sentences:
                relevant_sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
            
            # Limitar y retornar
            return relevant_sentences[:max_terms]
            
        except Exception as e:
            logger.error(f"Error extrayendo términos: {e}")
            return []
    
    def _parse_fact_checks(self, claims: List[Dict]) -> List[Dict]:
        """Parsear resultados de fact-check"""
        parsed = []
        
        for claim in claims:
            try:
                claim_text = claim.get("text", "")
                
                # Procesar reviews
                for review in claim.get("claimReview", []):
                    parsed_review = {
                        "claim_text": claim_text,
                        "rating": review.get("textualRating", ""),
                        "title": review.get("title", ""),
                        "url": review.get("url", ""),
                        "publisher": review.get("publisher", {}).get("name", ""),
                        "review_date": review.get("reviewDate", ""),
                        "language": review.get("languageCode", ""),
                        "credibility_score": self._calculate_credibility_score(review)
                    }
                    parsed.append(parsed_review)
                    
            except Exception as e:
                logger.error(f"Error parseando claim: {e}")
                continue
        
        return parsed
    
    def _calculate_credibility_score(self, review: Dict) -> int:
        """Calcular score de credibilidad 0-100"""
        rating = review.get("textualRating", "").lower()
        
        # Mapping de ratings a scores
        rating_scores = {
            "true": 100,
            "mostly true": 85,
            "half true": 50,
            "mostly false": 25,
            "false": 10,
            "pants on fire": 0,
            "misleading": 30,
            "unsubstantiated": 40,
            "correct": 95,
            "incorrect": 15,
            "disputed": 45
        }
        
        # Buscar match exacto
        for key, score in rating_scores.items():
            if key in rating:
                return score
        
        # Si no hay match, intentar detectar patrones
        if "true" in rating and "false" not in rating:
            return 80
        elif "false" in rating and "true" not in rating:
            return 20
        else:
            return 50  # neutral/desconocido
    
    def _deduplicate_fact_checks(self, fact_checks: List[Dict]) -> List[Dict]:
        """Eliminar fact-checks duplicados"""
        seen_urls = set()
        unique_checks = []
        
        for check in fact_checks:
            url = check.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_checks.append(check)
        
        return unique_checks
    
    def analyze_factual_reliability(self, fact_checks: List[Dict]) -> Dict:
        """Analizar confiabilidad general basada en fact-checks"""
        if not fact_checks:
            return {
                "reliability_score": 50,  # neutral
                "classification": "unverified",
                "confidence": "low"
            }
        
        # Calcular score promedio
        total_score = sum(check.get("credibility_score", 50) for check in fact_checks)
        avg_score = total_score / len(fact_checks)
        
        # Clasificar
        if avg_score >= 80:
            classification = "highly_reliable"
            confidence = "high"
        elif avg_score >= 60:
            classification = "mostly_reliable"
            confidence = "medium"
        elif avg_score >= 40:
            classification = "mixed_reliability"
            confidence = "medium"
        elif avg_score >= 20:
            classification = "mostly_unreliable"
            confidence = "medium"
        else:
            classification = "unreliable"
            confidence = "high"
        
        return {
            "reliability_score": int(avg_score),
            "classification": classification,
            "confidence": confidence,
            "fact_checks_count": len(fact_checks)
        }

# Función de prueba
async def test_google_fact_check():
    """Función de prueba para Google Fact Check"""
    mcp = GoogleFactCheckFreeMCP()
    
    if not mcp.enabled:
        print("❌ Google Fact Check API no disponible - configurar GOOGLE_API_KEY")
        return
    
    # Test con texto de ejemplo
    test_text = """
    According to recent reports, the COVID-19 vaccine contains microchips. 
    Scientists have discovered that climate change is a hoax created by China.
    Studies show that drinking water helps with hydration.
    """
    
    print("🧪 Testing Google Fact Check MCP...")
    result = await mcp.check_claims(test_text)
    
    print(f"Claims encontradas: {result['claims_found']}")
    print(f"Queries utilizadas: {result['total_queries']}")
    
    for i, check in enumerate(result['fact_checks'], 1):
        print(f"\n{i}. {check['claim_text'][:100]}...")
        print(f"   Rating: {check['rating']}")
        print(f"   Score: {check['credibility_score']}")
        print(f"   Publisher: {check['publisher']}")
    
    # Analizar confiabilidad general
    reliability = mcp.analyze_factual_reliability(result['fact_checks'])
    print(f"\n📊 Análisis de confiabilidad:")
    print(f"   Score: {reliability['reliability_score']}/100")
    print(f"   Clasificación: {reliability['classification']}")
    print(f"   Confianza: {reliability['confidence']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_google_fact_check()) 