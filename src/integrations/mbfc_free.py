#!/usr/bin/env python3
"""
MBFC (Media Bias/Fact Check) MCP - Dataset público gratuito
"""

import pandas as pd
import numpy as np
from urllib.parse import urlparse
from typing import Dict, Optional, List
from pathlib import Path
from loguru import logger
import re

class MBFCFreeMCP:
    """MBFC usando dataset público gratuito"""
    
    def __init__(self, datasets_dir: str = "data/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.mbfc_data = None
        self.bias_map = {}
        self.domain_stats = {}
        self.load_dataset()
    
    def load_dataset(self):
        """Cargar dataset MBFC"""
        try:
            dataset_path = self.datasets_dir / "mbfc_ratings.csv"
            
            if not dataset_path.exists():
                logger.warning(f"❌ MBFC dataset no encontrado: {dataset_path}")
                return
            
            # Cargar CSV con diferentes encodings
            encodings = ['utf-8', 'latin-1', 'cp1252']
            
            for encoding in encodings:
                try:
                    self.mbfc_data = pd.read_csv(dataset_path, encoding=encoding)
                    logger.info(f"✅ MBFC dataset cargado con {encoding}: {len(self.mbfc_data)} fuentes")
                    break
                except UnicodeDecodeError:
                    continue
            
            if self.mbfc_data is None:
                logger.error("❌ No se pudo cargar el dataset MBFC")
                return
            
            # Procesar datos
            self._process_dataset()
            
        except Exception as e:
            logger.error(f"Error cargando MBFC dataset: {e}")
    
    def _process_dataset(self):
        """Procesar y limpiar dataset"""
        if self.mbfc_data is None:
            return
        
        try:
            # Crear lookup por dominio
            self.bias_map = {}
            
            for idx, row in self.mbfc_data.iterrows():
                # Obtener URL
                url = row.get('url', '') or row.get('URL', '') or row.get('website', '')
                
                if pd.isna(url) or not url:
                    continue
                
                domain = self._extract_domain(url)
                if not domain:
                    continue
                
                # Extraer información de bias
                bias = self._normalize_bias(row)
                factual = self._normalize_factual(row)
                
                self.bias_map[domain] = {
                    "name": row.get('name', '') or row.get('source', '') or domain,
                    "bias": bias,
                    "factual": factual,
                    "credibility_score": self._calculate_credibility_score(row),
                    "notes": row.get('notes', '') or row.get('description', ''),
                    "country": row.get('country', '') or row.get('location', ''),
                    "media_type": row.get('media_type', '') or row.get('type', ''),
                    "url": url
                }
            
            logger.info(f"✅ Procesadas {len(self.bias_map)} fuentes de MBFC")
            
            # Calcular estadísticas
            self._calculate_stats()
            
        except Exception as e:
            logger.error(f"Error procesando dataset: {e}")
    
    def _extract_domain(self, url: str) -> str:
        """Extraer dominio de URL"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            
            # Remover www y subdominios comunes
            if domain.startswith('www.'):
                domain = domain[4:]
            
            return domain
            
        except Exception as e:
            logger.error(f"Error extrayendo dominio de {url}: {e}")
            return ""
    
    def _normalize_bias(self, row: Dict) -> str:
        """Normalizar valores de bias"""
        # Buscar columnas de bias
        bias_columns = ['bias', 'political_bias', 'lean', 'orientation']
        bias_value = ""
        
        for col in bias_columns:
            if col in row and not pd.isna(row[col]):
                bias_value = str(row[col]).lower()
                break
        
        # Normalizar valores
        if not bias_value:
            return "center"
        
        bias_mapping = {
            "left": "left",
            "left-center": "lean_left",
            "lean left": "lean_left",
            "center": "center",
            "least biased": "center",
            "right-center": "lean_right",
            "lean right": "lean_right",
            "right": "right",
            "extreme left": "extreme_left",
            "extreme right": "extreme_right",
            "conspiracy": "conspiracy",
            "pseudoscience": "pseudoscience",
            "pro-science": "pro_science",
            "satire": "satire"
        }
        
        for key, value in bias_mapping.items():
            if key in bias_value:
                return value
        
        return "center"  # default
    
    def _normalize_factual(self, row: Dict) -> str:
        """Normalizar valores de factual reporting"""
        # Buscar columnas de factual
        factual_columns = ['factual', 'factual_reporting', 'reliability', 'accuracy']
        factual_value = ""
        
        for col in factual_columns:
            if col in row and not pd.isna(row[col]):
                factual_value = str(row[col]).lower()
                break
        
        if not factual_value:
            return "mixed"
        
        factual_mapping = {
            "very high": "very_high",
            "high": "high",
            "mostly factual": "mostly_factual",
            "mixed": "mixed",
            "low": "low",
            "very low": "very_low"
        }
        
        for key, value in factual_mapping.items():
            if key in factual_value:
                return value
        
        return "mixed"  # default
    
    def _calculate_credibility_score(self, row: Dict) -> int:
        """Calcular score de credibilidad 0-100"""
        try:
            # Score basado en factual reporting
            factual = self._normalize_factual(row)
            factual_scores = {
                "very_high": 95,
                "high": 85,
                "mostly_factual": 75,
                "mixed": 50,
                "low": 25,
                "very_low": 10
            }
            
            base_score = factual_scores.get(factual, 50)
            
            # Ajustar por bias extremo
            bias = self._normalize_bias(row)
            if bias in ["conspiracy", "pseudoscience"]:
                base_score = min(base_score, 20)
            elif bias in ["extreme_left", "extreme_right"]:
                base_score = min(base_score, 40)
            
            return base_score
            
        except Exception as e:
            logger.error(f"Error calculando credibilidad: {e}")
            return 50
    
    def _calculate_stats(self):
        """Calcular estadísticas del dataset"""
        if not self.bias_map:
            return
        
        # Estadísticas por bias
        bias_counts = {}
        factual_counts = {}
        credibility_scores = []
        
        for domain, data in self.bias_map.items():
            bias = data["bias"]
            factual = data["factual"]
            score = data["credibility_score"]
            
            bias_counts[bias] = bias_counts.get(bias, 0) + 1
            factual_counts[factual] = factual_counts.get(factual, 0) + 1
            credibility_scores.append(score)
        
        self.domain_stats = {
            "total_sources": len(self.bias_map),
            "bias_distribution": bias_counts,
            "factual_distribution": factual_counts,
            "avg_credibility": np.mean(credibility_scores) if credibility_scores else 50,
            "median_credibility": np.median(credibility_scores) if credibility_scores else 50
        }
        
        logger.info(f"📊 Stats: {self.domain_stats['total_sources']} fuentes, avg credibility: {self.domain_stats['avg_credibility']:.1f}")
    
    def get_source_analysis(self, source_url: str) -> Dict:
        """Analizar fuente por URL - completamente offline"""
        if not source_url:
            return self._get_empty_analysis()
        
        domain = self._extract_domain(source_url)
        
        if domain in self.bias_map:
            source_data = self.bias_map[domain]
            return {
                "found": True,
                "domain": domain,
                "name": source_data["name"],
                "bias_rating": source_data["bias"],
                "factual_rating": source_data["factual"],
                "credibility_score": source_data["credibility_score"],
                "notes": source_data["notes"],
                "country": source_data["country"],
                "media_type": source_data["media_type"],
                "source": "mbfc_dataset",
                "reliability_level": self._get_reliability_level(source_data["credibility_score"])
            }
        
        return self._get_empty_analysis(domain)
    
    def _get_empty_analysis(self, domain: str = "") -> Dict:
        """Retornar análisis vacío para fuentes desconocidas"""
        return {
            "found": False,
            "domain": domain,
            "name": domain,
            "bias_rating": "unknown",
            "factual_rating": "unknown",
            "credibility_score": 50,  # neutral default
            "notes": "Source not found in MBFC database",
            "country": "unknown",
            "media_type": "unknown",
            "source": "mbfc_dataset",
            "reliability_level": "unknown"
        }
    
    def _get_reliability_level(self, score: int) -> str:
        """Convertir score a nivel de confiabilidad"""
        if score >= 85:
            return "very_high"
        elif score >= 70:
            return "high"
        elif score >= 55:
            return "medium"
        elif score >= 40:
            return "low"
        else:
            return "very_low"
    
    def batch_analyze_sources(self, urls: List[str]) -> Dict:
        """Analizar múltiples fuentes en batch"""
        results = {}
        
        for url in urls:
            domain = self._extract_domain(url)
            if domain:
                results[domain] = self.get_source_analysis(url)
        
        return results
    
    def get_bias_summary(self, urls: List[str]) -> Dict:
        """Obtener resumen de bias para múltiples fuentes"""
        analyses = self.batch_analyze_sources(urls)
        
        if not analyses:
            return {"total_sources": 0, "bias_distribution": {}, "avg_credibility": 50}
        
        # Contar bias y calcular promedios
        bias_counts = {}
        credibility_scores = []
        found_sources = 0
        
        for domain, analysis in analyses.items():
            if analysis["found"]:
                found_sources += 1
                bias = analysis["bias_rating"]
                score = analysis["credibility_score"]
                
                bias_counts[bias] = bias_counts.get(bias, 0) + 1
                credibility_scores.append(score)
        
        avg_credibility = np.mean(credibility_scores) if credibility_scores else 50
        
        return {
            "total_sources": len(analyses),
            "found_sources": found_sources,
            "coverage_percentage": (found_sources / len(analyses)) * 100 if analyses else 0,
            "bias_distribution": bias_counts,
            "avg_credibility": avg_credibility,
            "dominant_bias": max(bias_counts, key=bias_counts.get) if bias_counts else "unknown"
        }
    
    def get_dataset_info(self) -> Dict:
        """Obtener información sobre el dataset"""
        return {
            "loaded": self.mbfc_data is not None,
            "total_sources": len(self.bias_map),
            "dataset_path": str(self.datasets_dir / "mbfc_ratings.csv"),
            "stats": self.domain_stats
        }

# Función de prueba
def test_mbfc_mcp():
    """Función de prueba para MBFC MCP"""
    mcp = MBFCFreeMCP()
    
    if not mcp.bias_map:
        print("❌ MBFC dataset no cargado")
        return
    
    # Test con URLs de ejemplo
    test_urls = [
        "https://www.cnn.com",
        "https://www.foxnews.com",
        "https://www.bbc.com",
        "https://www.breitbart.com",
        "https://www.huffpost.com",
        "https://example-unknown-site.com"
    ]
    
    print("🧪 Testing MBFC MCP...")
    print(f"Dataset info: {mcp.get_dataset_info()}")
    
    for url in test_urls:
        analysis = mcp.get_source_analysis(url)
        print(f"\n🔍 {url}")
        print(f"   Found: {analysis['found']}")
        print(f"   Bias: {analysis['bias_rating']}")
        print(f"   Factual: {analysis['factual_rating']}")
        print(f"   Credibility: {analysis['credibility_score']}/100")
        print(f"   Reliability: {analysis['reliability_level']}")
    
    # Test resumen
    print(f"\n📊 Bias Summary:")
    summary = mcp.get_bias_summary(test_urls)
    print(f"   Total sources: {summary['total_sources']}")
    print(f"   Found sources: {summary['found_sources']}")
    print(f"   Coverage: {summary['coverage_percentage']:.1f}%")
    print(f"   Avg credibility: {summary['avg_credibility']:.1f}")
    print(f"   Dominant bias: {summary['dominant_bias']}")

if __name__ == "__main__":
    test_mbfc_mcp() 