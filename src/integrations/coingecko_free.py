#!/usr/bin/env python3
"""
CoinGecko MCP - Completamente gratuita sin API keys
"""

import aiohttp
import json
import asyncio
from typing import Dict, List, Optional
from pathlib import Path
from loguru import logger

class CoinGeckoFreeMCP:
    """CoinGecko MCP completamente gratuita - sin API key"""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, datasets_dir: str = "data/datasets"):
        self.datasets_dir = Path(datasets_dir)
        self.crypto_map = {}
        self.load_crypto_mapping()
    
    def load_crypto_mapping(self):
        """Cargar mapeo de cryptos desde archivo"""
        try:
            mapping_file = self.datasets_dir / "crypto_mapping.json"
            if mapping_file.exists():
                with open(mapping_file, "r", encoding="utf-8") as f:
                    self.crypto_map = json.load(f)
                logger.info(f"✅ Crypto mapping cargado: {len(self.crypto_map)} cryptos")
            else:
                logger.warning("❌ Archivo crypto_mapping.json no encontrado")
        except Exception as e:
            logger.error(f"Error cargando crypto mapping: {e}")
    
    async def get_crypto_prices(self, symbols: List[str]) -> Dict:
        """Obtener precios de cryptos - 50 requests/min gratis"""
        try:
            # Convertir símbolos a IDs de CoinGecko
            coin_ids = []
            for symbol in symbols:
                symbol_lower = symbol.lower()
                if symbol_lower in self.crypto_map:
                    coin_id = self.crypto_map[symbol_lower]
                    if coin_id not in coin_ids:
                        coin_ids.append(coin_id)
            
            if not coin_ids:
                logger.warning("No se encontraron cryptos válidas")
                return {}
            
            # API gratuita - sin autenticación
            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": ",".join(coin_ids),
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_market_cap": "true",
                "include_24hr_vol": "true"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = self._format_price_data(data)
                        logger.info(f"✅ Precios obtenidos para {len(result)} cryptos")
                        return result
                    else:
                        logger.error(f"Error API CoinGecko: {response.status}")
            
            return {}
            
        except Exception as e:
            logger.error(f"Error CoinGecko: {e}")
            return {}
    
    async def get_trending_cryptos(self) -> Dict:
        """Obtener cryptos trending - completamente gratis"""
        try:
            url = f"{self.BASE_URL}/search/trending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_trending_data(data)
                    else:
                        logger.error(f"Error trending API: {response.status}")
            
            return {}
            
        except Exception as e:
            logger.error(f"Error trending cryptos: {e}")
            return {}
    
    async def get_market_data(self, coin_id: str) -> Dict:
        """Obtener datos de mercado para una crypto específica"""
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}"
            params = {
                "localization": "false",
                "tickers": "false",
                "market_data": "true",
                "community_data": "false",
                "developer_data": "false",
                "sparkline": "false"
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=30) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._format_market_data(data)
                    else:
                        logger.error(f"Error market data API: {response.status}")
            
            return {}
            
        except Exception as e:
            logger.error(f"Error market data: {e}")
            return {}
    
    def _format_price_data(self, raw_data: Dict) -> Dict:
        """Formatear datos de precios"""
        formatted = {}
        for coin_id, data in raw_data.items():
            formatted[coin_id] = {
                "symbol": coin_id,
                "price_usd": data.get("usd", 0),
                "change_24h": data.get("usd_24h_change", 0),
                "market_cap": data.get("usd_market_cap", 0),
                "volume_24h": data.get("usd_24h_vol", 0),
                "last_updated": "real-time",
                "source": "coingecko_free"
            }
        return formatted
    
    def _format_trending_data(self, raw_data: Dict) -> Dict:
        """Formatear datos trending"""
        try:
            trending_coins = []
            for coin_data in raw_data.get("coins", []):
                coin = coin_data.get("item", {})
                trending_coins.append({
                    "id": coin.get("id", ""),
                    "name": coin.get("name", ""),
                    "symbol": coin.get("symbol", ""),
                    "market_cap_rank": coin.get("market_cap_rank", 0),
                    "score": coin.get("score", 0)
                })
            
            return {
                "trending_coins": trending_coins[:10],  # Top 10
                "count": len(trending_coins),
                "source": "coingecko_trending"
            }
        except Exception as e:
            logger.error(f"Error formateando trending: {e}")
            return {}
    
    def _format_market_data(self, raw_data: Dict) -> Dict:
        """Formatear datos de mercado"""
        try:
            market_data = raw_data.get("market_data", {})
            
            return {
                "id": raw_data.get("id", ""),
                "name": raw_data.get("name", ""),
                "symbol": raw_data.get("symbol", ""),
                "current_price": market_data.get("current_price", {}).get("usd", 0),
                "market_cap": market_data.get("market_cap", {}).get("usd", 0),
                "market_cap_rank": market_data.get("market_cap_rank", 0),
                "total_volume": market_data.get("total_volume", {}).get("usd", 0),
                "high_24h": market_data.get("high_24h", {}).get("usd", 0),
                "low_24h": market_data.get("low_24h", {}).get("usd", 0),
                "price_change_24h": market_data.get("price_change_24h", 0),
                "price_change_percentage_24h": market_data.get("price_change_percentage_24h", 0),
                "circulating_supply": market_data.get("circulating_supply", 0),
                "total_supply": market_data.get("total_supply", 0),
                "max_supply": market_data.get("max_supply", 0),
                "ath": market_data.get("ath", {}).get("usd", 0),
                "ath_change_percentage": market_data.get("ath_change_percentage", {}).get("usd", 0),
                "atl": market_data.get("atl", {}).get("usd", 0),
                "atl_change_percentage": market_data.get("atl_change_percentage", {}).get("usd", 0),
                "source": "coingecko_market"
            }
        except Exception as e:
            logger.error(f"Error formateando market data: {e}")
            return {}
    
    def extract_crypto_mentions(self, text: str) -> List[str]:
        """Extraer menciones de cryptos del texto"""
        mentions = []
        text_lower = text.lower()
        
        for symbol in self.crypto_map:
            if symbol in text_lower:
                mentions.append(symbol)
        
        return list(set(mentions))  # Eliminar duplicados
    
    async def analyze_crypto_sentiment(self, text: str) -> Dict:
        """Analizar sentiment de cryptos mencionadas en el texto"""
        try:
            # Extraer cryptos mencionadas
            mentioned_cryptos = self.extract_crypto_mentions(text)
            
            if not mentioned_cryptos:
                return {"mentioned_cryptos": [], "analysis": "no_crypto_detected"}
            
            # Obtener precios actuales
            price_data = await self.get_crypto_prices(mentioned_cryptos)
            
            # Combinar con análisis de texto
            analysis = {
                "mentioned_cryptos": mentioned_cryptos,
                "price_data": price_data,
                "analysis": "crypto_detected",
                "recommendation": self._generate_crypto_recommendation(text, price_data)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error crypto sentiment: {e}")
            return {"mentioned_cryptos": [], "analysis": "error"}
    
    def _generate_crypto_recommendation(self, text: str, price_data: Dict) -> str:
        """Generar recomendación basada en el texto y precios"""
        text_lower = text.lower()
        
        # Palabras positivas
        positive_words = ["buy", "bull", "bullish", "up", "rise", "gain", "profit", "pump"]
        negative_words = ["sell", "bear", "bearish", "down", "fall", "loss", "dump", "crash"]
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "bullish_sentiment"
        elif negative_count > positive_count:
            return "bearish_sentiment"
        else:
            return "neutral_sentiment"

# Función de prueba
async def test_coingecko_mcp():
    """Función de prueba para CoinGecko MCP"""
    mcp = CoinGeckoFreeMCP()
    
    # Test 1: Precios de cryptos
    print("🧪 Test 1: Precios de cryptos")
    prices = await mcp.get_crypto_prices(["bitcoin", "ethereum", "solana"])
    print(f"Precios obtenidos: {len(prices)} cryptos")
    
    # Test 2: Trending cryptos
    print("\n🧪 Test 2: Trending cryptos")
    trending = await mcp.get_trending_cryptos()
    print(f"Trending: {trending.get('count', 0)} cryptos")
    
    # Test 3: Análisis de texto
    print("\n🧪 Test 3: Análisis de texto")
    text = "Bitcoin is going to the moon! Ethereum looks bullish too."
    analysis = await mcp.analyze_crypto_sentiment(text)
    print(f"Análisis: {analysis.get('analysis', 'error')}")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_coingecko_mcp()) 