#!/usr/bin/env python3
"""
AllSides MCP - Dataset público gratuito para análisis de bias político
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

class AllSidesFreeMCP:
    """AllSides usando dataset público gratuito"""
    
    def __init__(self, datasets_dir: str = "data/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.allsides_data = None
        self.bias_map = {}
        self.political_spectrum = {}
        self.load_dataset()
    
    def load_dataset(self):
        """Cargar dataset AllSides"""
        try:
            dataset_path = self.datasets_dir / "allsides_bias.csv"
            
            if not dataset_path.exists():
                logger.warning(f"❌ AllSides dataset no encontrado: {dataset_path}")
                return
            
            # Cargar CSV con diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.allsides_data = pd.read_csv(dataset_path, encoding=encoding)
                    logger.info(f"✅ AllSides dataset cargado con {encoding}: {len(self.allsides_data)} fuentes")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.allsides_data is None:
                logger.error("❌ No se pudo cargar el dataset AllSides")
                return
            
            # Procesar datos
            self._process_dataset()
            
        except Exception as e:
            logger.error(f"Error cargando AllSides dataset: {e}")
    
    def _process_dataset(self):
        """Procesar y limpiar dataset"""
        if self.allsides_data is None:
            return
        
        try:
            self.bias_map = {}
            
            for idx, row in self.allsides_data.iterrows():
                # Obtener URL y información
                url = row.get('url', '') or row.get('URL', '') or row.get('website', '')
                
                if pd.isna(url) or not url:
                    continue
                
                domain = self._extract_domain(url)
                if not domain:
                    continue
                
                # Extraer información de bias
                bias = self._normalize_bias(row)
                confidence = self._normalize_confidence(row)
                
                self.bias_map[domain] = {
                    "name": row.get('name', '') or row.get('source', '') or domain,
                    "bias": bias,
                    "confidence": confidence,
                    "rating": row.get('rating', '') or row.get('allsides_rating', ''),
                    "agree_disagree": row.get('agree_disagree', ''),
                    "community_rating": row.get('community_rating', ''),
                    "url": url,
                    "bias_score": self._calculate_bias_score(bias),
                    "reliability_score": self._calculate_reliability_score(confidence)
                }
            
            logger.info(f"✅ Procesadas {len(self.bias_map)} fuentes de AllSides")
            
            # Calcular espectro político
            self._calculate_political_spectrum()
            
        except Exception as e:
            logger.error(f"Error procesando AllSides dataset: {e}")
    
    def _extract_domain(self, url: str) -> str:
        """Extraer dominio de URL"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remover www
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
            
        except Exception as e:
            logger.error(f"Error extrayendo dominio de {url}: {e}")
            return ""
    
    def _normalize_bias(self, row: Dict) -> str:
        """Normalizar valores de bias político"""
        # Buscar columnas de bias
        bias_columns = ['bias', 'political_bias', 'lean', 'allsides_rating']
        bias_value = ""
        
        for col in bias_columns:
            if col in row and not pd.isna(row[col]):
                bias_value = str(row[col]).lower()
                break
        
        if not bias_value:
            return "center"
        
        # Normalizar valores de AllSides
        bias_mapping = {
            'left': 'left',
            'lean left': 'lean_left',
            'center': 'center',
            'lean right': 'lean_right',
            'right': 'right',
            'mixed': 'mixed'
        }
        
        for key, value in bias_mapping.items():
            if key in bias_value:
                return value
        
        return "center"  # default
    
    def _normalize_confidence(self, row: Dict) -> str:
        """Normalizar valores de confianza"""
        confidence_columns = ['confidence', 'reliability', 'certainty']
        confidence_value = ""
        
        for col in confidence_columns:
            if col in row and not pd.isna(row[col]):
                confidence_value = str(row[col]).lower()
                break
        
        if not confidence_value:
            return "medium"
        
        if "high" in confidence_value:
            return "high"
        elif "low" in confidence_value:
            return "low"
        else:
            return "medium"
    
    def _calculate_bias_score(self, bias: str) -> int:
        """Calcular score numérico de bias (-100 a 100)"""
        bias_scores = {
            "left": -80,
            "lean_left": -40,
            "center": 0,
            "lean_right": 40,
            "right": 80,
            "mixed": 0
        }
        return bias_scores.get(bias, 0)
    
    def _calculate_reliability_score(self, confidence: str) -> int:
        """Calcular score de confiabilidad (0-100)"""
        confidence_scores = {
            "high": 85,
            "medium": 60,
            "low": 35
        }
        return confidence_scores.get(confidence, 60)
    
    def _calculate_political_spectrum(self):
        """Calcular distribución en el espectro político"""
        if not self.bias_map:
            return
        
        spectrum_counts = {}
        total_sources = 0
        
        for domain, data in self.bias_map.items():
            bias = data["bias"]
            spectrum_counts[bias] = spectrum_counts.get(bias, 0) + 1
            total_sources += 1
        
        # Calcular porcentajes
        self.political_spectrum = {
            "distribution": spectrum_counts,
            "percentages": {
                bias: (count / total_sources) * 100 
                for bias, count in spectrum_counts.items()
            } if total_sources > 0 else {},
            "total_sources": total_sources
        }
        
        logger.info(f"📊 Espectro político: {spectrum_counts}")
    
    def get_bias_analysis(self, source_url: str) -> Dict:
        """Obtener análisis de bias para una fuente"""
        if not source_url:
            return self._get_empty_analysis()
        
        domain = self._extract_domain(source_url)
        
        if domain in self.bias_map:
            source_data = self.bias_map[domain]
            return {
                "found": True,
                "domain": domain,
                "name": source_data["name"],
                "political_bias": source_data["bias"],
                "confidence": source_data["confidence"],
                "bias_score": source_data["bias_score"],
                "reliability_score": source_data["reliability_score"],
                "rating": source_data["rating"],
                "agree_disagree": source_data["agree_disagree"],
                "community_rating": source_data["community_rating"],
                "source": "allsides_dataset",
                "political_lean": self._get_political_lean(source_data["bias_score"])
            }
        
        return self._get_empty_analysis(domain)
    
    def _get_empty_analysis(self, domain: str = "") -> Dict:
        """Retornar análisis vacío para fuentes desconocidas"""
        return {
            "found": False,
            "domain": domain,
            "name": domain,
            "political_bias": "unknown",
            "confidence": "low",
            "bias_score": 0,
            "reliability_score": 50,
            "rating": "unknown",
            "agree_disagree": "unknown",
            "community_rating": "unknown",
            "source": "allsides_dataset",
            "political_lean": "center"
        }
    
    def _get_political_lean(self, bias_score: int) -> str:
        """Convertir score a descripción política"""
        if bias_score <= -60:
            return "strongly_left"
        elif bias_score <= -20:
            return "moderately_left"
        elif bias_score <= 20:
            return "center"
        elif bias_score <= 60:
            return "moderately_right"
        else:
            return "strongly_right"
    
    def batch_analyze_bias(self, urls: List[str]) -> Dict:
        """Analizar bias político de múltiples fuentes"""
        results = {}
        
        for url in urls:
            domain = self._extract_domain(url)
            if domain:
                results[domain] = self.get_bias_analysis(url)
        
        return results
    
    def get_political_summary(self, urls: List[str]) -> Dict:
        """Obtener resumen político de múltiples fuentes"""
        analyses = self.batch_analyze_bias(urls)
        
        if not analyses:
            return {
                "total_sources": 0,
                "political_distribution": {},
                "avg_bias_score": 0,
                "political_lean": "center"
            }
        
        # Contar distribución política
        political_counts = {}
        bias_scores = []
        found_sources = 0
        
        for domain, analysis in analyses.items():
            if analysis["found"]:
                found_sources += 1
                bias = analysis["political_bias"]
                score = analysis["bias_score"]
                
                political_counts[bias] = political_counts.get(bias, 0) + 1
                bias_scores.append(score)
        
        avg_bias = np.mean(bias_scores) if bias_scores else 0
        
        return {
            "total_sources": len(analyses),
            "found_sources": found_sources,
            "coverage_percentage": (found_sources / len(analyses)) * 100 if analyses else 0,
            "political_distribution": political_counts,
            "avg_bias_score": avg_bias,
            "political_lean": self._get_political_lean(int(avg_bias)),
            "dominant_bias": max(political_counts, key=political_counts.get) if political_counts else "unknown"
        }
    
    def analyze_political_balance(self, urls: List[str]) -> Dict:
        """Analizar balance político de un conjunto de fuentes"""
        summary = self.get_political_summary(urls)
        
        left_count = summary["political_distribution"].get("left", 0) + \
                    summary["political_distribution"].get("lean_left", 0)
        
        right_count = summary["political_distribution"].get("right", 0) + \
                     summary["political_distribution"].get("lean_right", 0)
        
        center_count = summary["political_distribution"].get("center", 0)
        
        total_partisan = left_count + right_count
        balance_score = 0
        
        if total_partisan > 0:
            balance_score = abs(left_count - right_count) / total_partisan
        
        # Clasificar balance
        if balance_score <= 0.2:
            balance_classification = "well_balanced"
        elif balance_score <= 0.5:
            balance_classification = "somewhat_balanced"
        else:
            balance_classification = "imbalanced"
        
        return {
            "left_sources": left_count,
            "right_sources": right_count,
            "center_sources": center_count,
            "balance_score": balance_score,
            "balance_classification": balance_classification,
            "total_sources": summary["found_sources"],
            "recommendation": self._get_balance_recommendation(balance_classification, left_count, right_count)
        }
    
    def _get_balance_recommendation(self, classification: str, left: int, right: int) -> str:
        """Generar recomendación de balance"""
        if classification == "well_balanced":
            return "Sources show good political balance"
        elif left > right:
            return "Consider adding more conservative sources for balance"
        elif right > left:
            return "Consider adding more liberal sources for balance"
        else:
            return "Consider adding more diverse political perspectives"
    
    def get_dataset_info(self) -> Dict:
        """Obtener información sobre el dataset"""
        return {
            "loaded": self.allsides_data is not None,
            "total_sources": len(self.bias_map),
            "dataset_path": str(self.datasets_dir / "allsides_bias.csv"),
            "political_spectrum": self.political_spectrum
        }

# Función de prueba
def test_allsides_mcp():
    """Función de prueba para AllSides MCP"""
    mcp = AllSidesFreeMCP()
    
    if not mcp.bias_map:
        print("❌ AllSides dataset no cargado")
        return
    
    # Test con URLs de ejemplo
    test_urls = [
        "https://www.cnn.com",
        "https://www.foxnews.com",
        "https://www.reuters.com",
        "https://www.msnbc.com",
        "https://www.wsj.com",
        "https://www.bbc.com"
    ]
    
    print("🧪 Testing AllSides MCP...")
    print(f"Dataset info: {mcp.get_dataset_info()}")
    
    for url in test_urls:
        analysis = mcp.get_bias_analysis(url)
        print(f"\n🔍 {url}")
        print(f"   Found: {analysis['found']}")
        print(f"   Political bias: {analysis['political_bias']}")
        print(f"   Confidence: {analysis['confidence']}")
        print(f"   Bias score: {analysis['bias_score']}")
        print(f"   Political lean: {analysis['political_lean']}")
    
    # Test resumen político
    print(f"\n📊 Political Summary:")
    summary = mcp.get_political_summary(test_urls)
    print(f"   Total sources: {summary['total_sources']}")
    print(f"   Found sources: {summary['found_sources']}")
    print(f"   Avg bias score: {summary['avg_bias_score']:.1f}")
    print(f"   Political lean: {summary['political_lean']}")
    print(f"   Dominant bias: {summary['dominant_bias']}")
    
    # Test balance político
    print(f"\n⚖️ Political Balance:")
    balance = mcp.analyze_political_balance(test_urls)
    print(f"   Left sources: {balance['left_sources']}")
    print(f"   Right sources: {balance['right_sources']}")
    print(f"   Center sources: {balance['center_sources']}")
    print(f"   Balance: {balance['balance_classification']}")
    print(f"   Recommendation: {balance['recommendation']}")

if __name__ == "__main__":
    test_allsides_mcp() 