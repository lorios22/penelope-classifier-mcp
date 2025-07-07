#!/usr/bin/env python3
"""
Servidor MCP Principal para Clasificación de Noticias
"""

import asyncio
import json
from typing import Dict, List, Any
from mcp.server import Server
from mcp.types import Tool, TextContent
from loguru import logger
from dotenv import load_dotenv
import os

# Importar clasificador principal
from classifier.news_classifier import NewsClassifier, create_sample_news

load_dotenv()

class NewsClassificationMCPServer:
    """Servidor MCP para clasificación de noticias"""
    
    def __init__(self):
        self.server = Server("penelope-news-classifier")
        self.classifier = NewsClassifier()
        self._setup_tools()
        logger.info("🚀 MCP Server inicializado")
    
    def _setup_tools(self):
        """Configurar herramientas MCP"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name="classify_news",
                    description="Clasificar una noticia usando todos los MCPs gratuitos (FinBERT, MBFC, AllSides, Google Fact Check, CoinGecko)",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "description": "Título de la noticia"},
                            "content": {"type": "string", "description": "Contenido de la noticia"},
                            "source_url": {"type": "string", "description": "URL de la fuente"},
                            "published_date": {"type": "string", "description": "Fecha de publicación (ISO format)"}
                        },
                        "required": ["title"]
                    }
                ),
                Tool(
                    name="classify_news_batch",
                    description="Clasificar múltiples noticias en batch y guardar en CSV",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "news_items": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "content": {"type": "string"},
                                        "source_url": {"type": "string"},
                                        "published_date": {"type": "string"}
                                    },
                                    "required": ["title"]
                                }
                            },
                            "output_filename": {"type": "string", "description": "Nombre del archivo CSV de salida"}
                        },
                        "required": ["news_items"]
                    }
                ),
                Tool(
                    name="test_with_sample_news",
                    description="Probar el sistema con noticias de ejemplo",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "count": {"type": "integer", "description": "Número de noticias de ejemplo (default: 4)"}
                        }
                    }
                ),
                Tool(
                    name="get_system_status",
                    description="Obtener estado de todos los MCPs y componentes",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            try:
                if name == "classify_news":
                    return await self._classify_single_news(arguments)
                elif name == "classify_news_batch":
                    return await self._classify_news_batch(arguments)
                elif name == "test_with_sample_news":
                    return await self._test_with_sample_news(arguments)
                elif name == "get_system_status":
                    return await self._get_system_status()
                else:
                    return [TextContent(type="text", text=f"Herramienta desconocida: {name}")]
            except Exception as e:
                logger.error(f"Error en herramienta {name}: {e}")
                return [TextContent(type="text", text=f"Error: {str(e)}")]
    
    async def _classify_single_news(self, args: Dict) -> List[TextContent]:
        """Clasificar una sola noticia"""
        news_item = {
            "title": args.get("title", ""),
            "content": args.get("content", ""),
            "source_url": args.get("source_url", ""),
            "published_date": args.get("published_date", "")
        }
        
        result = await self.classifier.classify_news(news_item)
        
        # Formatear resultado
        formatted_result = f"""
🔍 CLASIFICACIÓN DE NOTICIA

📰 Título: {result.get('title', 'N/A')}
🌐 Fuente: {result.get('source_url', 'N/A')}
📅 Fecha: {result.get('published_date', 'N/A')}

📊 ANÁLISIS FINANCIERO (FinBERT):
• Sentiment: {result.get('finbert_sentiment', 'neutral')}
• Confianza: {result.get('finbert_confidence', 0):.3f}
• Es noticia financiera: {'Sí' if result.get('is_financial_news') else 'No'}

🎯 ANÁLISIS DE BIAS:
• Bias político: {result.get('source_bias', 'unknown')}
• Credibilidad: {result.get('source_credibility', 50)}/100
• Inclinación política: {result.get('political_lean', 'center')}

✅ FACT-CHECKING:
• Score: {result.get('fact_check_score', 50)}/100
• Clasificación: {result.get('fact_check_classification', 'unverified')}
• Claims verificadas: {result.get('fact_checks_found', 0)}

🪙 ANÁLISIS CRYPTO:
• Menciones: {', '.join(result.get('crypto_mentions', [])) if result.get('crypto_mentions') else 'Ninguna'}
• Sentiment crypto: {result.get('crypto_sentiment', 'neutral')}

🏆 CLASIFICACIÓN FINAL:
• Clasificación: {result.get('overall_classification', 'unknown')}
• Confianza: {result.get('confidence_score', 0):.3f}
• Confiabilidad: {result.get('reliability_score', 0):.1f}/100

{('❌ Error: ' + result.get('error', '')) if result.get('error') else ''}
"""
        
        return [TextContent(type="text", text=formatted_result)]
    
    async def _classify_news_batch(self, args: Dict) -> List[TextContent]:
        """Clasificar múltiples noticias"""
        news_items = args.get("news_items", [])
        output_filename = args.get("output_filename", "news_classification_results.csv")
        
        if not news_items:
            return [TextContent(type="text", text="❌ No se proporcionaron noticias para clasificar")]
        
        # Clasificar noticias
        results = await self.classifier.classify_news_batch(news_items)
        
        # Guardar en CSV
        csv_path = self.classifier.save_to_csv(results, output_filename)
        
        # Obtener estadísticas
        stats = self.classifier.get_summary_stats(results)
        
        formatted_result = f"""
🎉 CLASIFICACIÓN BATCH COMPLETADA

📊 ESTADÍSTICAS:
• Total noticias: {stats.get('total_news', 0)}
• Noticias financieras: {stats.get('financial_percentage', 0):.1f}%
• Noticias crypto: {stats.get('crypto_percentage', 0):.1f}%
• Noticias fact-checked: {stats.get('fact_checked_percentage', 0):.1f}%

📁 ARCHIVO CSV: {csv_path}

🔍 DISTRIBUCIÓN POR CLASIFICACIÓN:
{self._format_distribution(stats.get('classification_distribution', {}))}

📈 DISTRIBUCIÓN POR SENTIMENT:
{self._format_distribution(stats.get('sentiment_distribution', {}))}

🎯 DISTRIBUCIÓN POR BIAS:
{self._format_distribution(stats.get('bias_distribution', {}))}
"""
        
        return [TextContent(type="text", text=formatted_result)]
    
    async def _test_with_sample_news(self, args: Dict) -> List[TextContent]:
        """Probar con noticias de ejemplo"""
        count = args.get("count", 4)
        
        # Crear noticias de ejemplo
        sample_news = create_sample_news()[:count]
        
        # Clasificar
        results = await self.classifier.classify_news_batch(sample_news)
        
        # Guardar
        csv_path = self.classifier.save_to_csv(results, "sample_news_test.csv")
        
        # Estadísticas
        stats = self.classifier.get_summary_stats(results)
        
        formatted_result = f"""
🧪 PRUEBA CON NOTICIAS DE EJEMPLO

📊 RESULTADOS:
• Noticias procesadas: {len(results)}
• Noticias financieras: {stats.get('financial_news', 0)}
• Noticias crypto: {stats.get('crypto_news', 0)}
• Noticias fact-checked: {stats.get('fact_checked_news', 0)}

📁 Archivo CSV: {csv_path}

🔍 CLASIFICACIONES:
{self._format_distribution(stats.get('classification_distribution', {}))}

✅ Sistema funcionando correctamente!
"""
        
        return [TextContent(type="text", text=formatted_result)]
    
    async def _get_system_status(self) -> List[TextContent]:
        """Obtener estado del sistema"""
        try:
            # Verificar estado de MCPs
            finbert_status = self.classifier.finbert.get_model_info()
            mbfc_status = self.classifier.mbfc.get_dataset_info()
            allsides_status = self.classifier.allsides.get_dataset_info()
            
            status_text = f"""
🚀 ESTADO DEL SISTEMA PENELOPE NEWS CLASSIFIER

🤖 FINBERT (Sentiment Financiero):
• Modelo: {finbert_status.get('model_name', 'N/A')}
• Device: {finbert_status.get('device', 'N/A')}
• Inicializado: {'✅' if finbert_status.get('initialized') else '❌'}

🎯 MBFC (Media Bias/Fact Check):
• Fuentes cargadas: {mbfc_status.get('total_sources', 0)}
• Dataset cargado: {'✅' if mbfc_status.get('loaded') else '❌'}

⚖️ ALLSIDES (Bias Político):
• Fuentes cargadas: {allsides_status.get('total_sources', 0)}
• Dataset cargado: {'✅' if allsides_status.get('loaded') else '❌'}

🔍 GOOGLE FACT CHECK:
• Habilitado: {'✅' if self.classifier.fact_checker.enabled else '❌'}
• API Key: {'✅' if self.classifier.fact_checker.api_key else '❌'}

🪙 COINGECKO (Crypto Data):
• Cryptos mapeadas: {len(self.classifier.coingecko.crypto_map)}
• Funcional: {'✅' if self.classifier.coingecko.crypto_map else '❌'}

📊 CONFIGURACIÓN:
• Output dir: {self.classifier.config['output_dir']}
• CSV filename: {self.classifier.config['csv_output_file']}
• Max concurrent: {self.classifier.config['max_concurrent']}

🎉 ESTADO GENERAL: {'✅ OPERATIVO' if self._is_system_operational() else '❌ PROBLEMAS DETECTADOS'}
"""
            
            return [TextContent(type="text", text=status_text)]
            
        except Exception as e:
            return [TextContent(type="text", text=f"❌ Error obteniendo estado: {str(e)}")]
    
    def _format_distribution(self, distribution: Dict) -> str:
        """Formatear distribución para mostrar"""
        if not distribution:
            return "• No hay datos"
        
        lines = []
        for key, value in distribution.items():
            lines.append(f"• {key}: {value}")
        
        return "\n".join(lines)
    
    def _is_system_operational(self) -> bool:
        """Verificar si el sistema está operativo"""
        try:
            # Verificar componentes críticos
            finbert_ok = self.classifier.finbert.classifier is not None
            mbfc_ok = len(self.classifier.mbfc.bias_map) > 0
            allsides_ok = len(self.classifier.allsides.bias_map) > 0
            
            return finbert_ok and (mbfc_ok or allsides_ok)
        except:
            return False
    
    def run(self, transport: str = "stdio"):
        """Ejecutar servidor MCP"""
        logger.info(f"🚀 Iniciando servidor MCP en modo {transport}")
        
        if transport == "stdio":
            import mcp.server.stdio
            mcp.server.stdio.run_server(self.server)
        else:
            logger.error(f"Transporte no soportado: {transport}")

# Función principal
def main():
    """Función principal"""
    server = NewsClassificationMCPServer()
    server.run()

if __name__ == "__main__":
    main() 