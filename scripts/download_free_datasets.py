#!/usr/bin/env python3
"""
Script para descargar datasets gratuitos para el clasificador de noticias
"""

import requests
import pandas as pd
import json
import os
from typing import Dict, List
from pathlib import Path

class DatasetDownloader:
    """Descargador de datasets gratuitos"""
    
    def __init__(self, datasets_dir: str = "data/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)
        
    def download_mbfc_dataset(self) -> bool:
        """Descargar dataset MBFC gratuito"""
        print("📊 Descargando MBFC dataset...")
        url = "https://raw.githubusercontent.com/BurhanUlTayyab/MediaBiasFactCheck/main/mbfc.csv"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = self.datasets_dir / "mbfc_ratings.csv"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            print(f"✅ MBFC dataset descargado: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error descargando MBFC dataset: {e}")
            return False
    
    def download_allsides_dataset(self) -> bool:
        """Descargar dataset AllSides gratuito"""
        print("📊 Descargando AllSides dataset...")
        url = "https://raw.githubusercontent.com/josephrussell/media_bias/master/media_bias_data.csv"
        
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            filepath = self.datasets_dir / "allsides_bias.csv"
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(response.text)
                
            print(f"✅ AllSides dataset descargado: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error descargando AllSides dataset: {e}")
            return False
    
    def create_crypto_mapping(self) -> bool:
        """Crear mapeo de criptomonedas para CoinGecko"""
        print("🪙 Creando mapeo de criptomonedas...")
        
        crypto_map = {
            "bitcoin": "bitcoin",
            "btc": "bitcoin",
            "ethereum": "ethereum", 
            "eth": "ethereum",
            "cardano": "cardano",
            "ada": "cardano",
            "solana": "solana",
            "sol": "solana",
            "dogecoin": "dogecoin",
            "doge": "dogecoin",
            "polkadot": "polkadot",
            "dot": "polkadot",
            "chainlink": "chainlink",
            "link": "chainlink",
            "polygon": "polygon",
            "matic": "polygon",
            "uniswap": "uniswap",
            "uni": "uniswap",
            "avalanche": "avalanche",
            "avax": "avalanche",
            "cosmos": "cosmos",
            "atom": "cosmos",
            "near": "near",
            "algo": "algorand",
            "algorand": "algorand",
            "ftm": "fantom",
            "fantom": "fantom"
        }
        
        try:
            filepath = self.datasets_dir / "crypto_mapping.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(crypto_map, f, indent=2)
                
            print(f"✅ Crypto mapping creado: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error creando crypto mapping: {e}")
            return False
    
    def create_financial_keywords(self) -> bool:
        """Crear lista de palabras clave financieras"""
        print("💰 Creando keywords financieras...")
        
        financial_keywords = {
            "crypto": [
                "bitcoin", "ethereum", "crypto", "blockchain", "defi", 
                "nft", "mining", "wallet", "exchange", "trading", "hodl",
                "altcoin", "stablecoin", "yield", "staking", "liquidity"
            ],
            "stocks": [
                "stock", "share", "equity", "market", "nasdaq", "s&p",
                "dow", "bull", "bear", "dividend", "earnings", "ipo",
                "merger", "acquisition", "portfolio", "volatility"
            ],
            "economy": [
                "inflation", "recession", "gdp", "fed", "interest", "rate",
                "employment", "unemployment", "cpi", "retail", "consumer",
                "spending", "debt", "deficit", "surplus", "trade"
            ],
            "sentiment": [
                "bullish", "bearish", "optimistic", "pessimistic", "positive",
                "negative", "uncertain", "confident", "worried", "excited",
                "concerned", "hopeful", "fearful", "greedy", "panic"
            ]
        }
        
        try:
            filepath = self.datasets_dir / "financial_keywords.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(financial_keywords, f, indent=2)
                
            print(f"✅ Financial keywords creadas: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error creando keywords: {e}")
            return False
    
    def verify_datasets(self) -> Dict[str, bool]:
        """Verificar que todos los datasets estén disponibles"""
        print("🔍 Verificando datasets...")
        
        required_files = [
            "mbfc_ratings.csv",
            "allsides_bias.csv", 
            "crypto_mapping.json",
            "financial_keywords.json"
        ]
        
        status = {}
        for filename in required_files:
            filepath = self.datasets_dir / filename
            exists = filepath.exists()
            status[filename] = exists
            
            if exists:
                print(f"✅ {filename} - OK")
            else:
                print(f"❌ {filename} - MISSING")
        
        return status
    
    def download_all(self) -> bool:
        """Descargar todos los datasets necesarios"""
        print("🚀 Iniciando descarga de datasets gratuitos...")
        
        success = True
        
        # Descargar datasets
        if not self.download_mbfc_dataset():
            success = False
            
        if not self.download_allsides_dataset():
            success = False
            
        # Crear mappings
        if not self.create_crypto_mapping():
            success = False
            
        if not self.create_financial_keywords():
            success = False
        
        # Verificar resultados
        status = self.verify_datasets()
        missing = [f for f, exists in status.items() if not exists]
        
        if missing:
            print(f"❌ Datasets faltantes: {missing}")
            success = False
        else:
            print("✅ Todos los datasets descargados correctamente")
        
        return success

def main():
    """Función principal"""
    downloader = DatasetDownloader()
    
    if downloader.download_all():
        print("\n🎉 Setup de datasets completado exitosamente!")
        print("📁 Archivos disponibles:")
        status = downloader.verify_datasets()
        for filename, exists in status.items():
            if exists:
                filepath = downloader.datasets_dir / filename
                print(f"   {filename} ({filepath.stat().st_size} bytes)")
    else:
        print("\n❌ Error en setup de datasets")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 