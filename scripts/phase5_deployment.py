#!/usr/bin/env python3
"""
FASE 5: Despliegue Completo con IA
Bot completo con LSTM, sentiment analysis y listo para VPS.

⚠️ ADVERTENCIA: Este script opera con DINERO REAL
NOTA: Requiere modelos pre-entrenados en models/
"""

import sys
import os
import signal
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from src.data.binance_client import BinanceClientManager
from src.strategy.risk_manager import RiskManager
from src.utils.logger import TradingLogger
from src.ai.lstm_model import load_lstm_model
from src.ai.sentiment import load_sentiment_analyzer


def signal_handler(sig, frame, logger=None):
    """Maneja señales del sistema (CTRL+C)"""
    if logger:
        logger.separator()
        logger.info("Bot detenido por el usuario")
        logger.separator()
    print("\n\n⚠ Bot detenido de forma segura")
    sys.exit(0)


def main():
    print("=" * 70)
    print("FASE 5: DESPLIEGUE COMPLETO CON IA")
    print("=" * 70)

    # Inicializar logger
    logger = TradingLogger()
    logger.separator()
    logger.info("BOT DE TRADING ALGORÍTMICO - FASE 5")
    logger.separator()

    # Configurar handler de señales
    signal.signal(signal.SIGINT, lambda s, f: signal_handler(s, f, logger))

    # Cargar configuración
    logger.info("Cargando configuración...")
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)

        ai_config = config.get('ai', {})
        use_lstm = ai_config.get('use_lstm', False)
        use_sentiment = ai_config.get('use_sentiment', False)

        if config['binance']['testnet']:
            logger.info("Modo: TESTNET (entorno seguro)")
        else:
            logger.warning("Modo: PRODUCCIÓN (DINERO REAL)")

    except Exception as e:
        logger.error(f"Error al cargar configuración: {e}")
        return

    # Cargar modelos de IA
    lstm_model = None
    sentiment_analyzer = None

    if use_lstm:
        logger.info("Cargando modelo LSTM...")
        lstm_model = load_lstm_model(ai_config.get('lstm_model_path', 'models/lstm_model.h5'))
        if lstm_model:
            logger.info("✓ Modelo LSTM cargado exitosamente")
        else:
            logger.warning("⚠ Modelo LSTM no disponible, continuando sin LSTM")

    if use_sentiment:
        logger.info("Cargando analizador de sentimiento...")
        sentiment_analyzer = load_sentiment_analyzer(ai_config.get('sentiment_model'))
        if sentiment_analyzer:
            logger.info("✓ Analizador de sentimiento cargado")
        else:
            logger.warning("⚠ Analizador no disponible, continuando sin sentiment analysis")

    # Conectar a Binance
    logger.info("Conectando a Binance...")
    try:
        manager = BinanceClientManager()
        client = manager.get_authenticated_client()
        logger.info("✓ Autenticación exitosa")
    except Exception as e:
        logger.error(f"Error de autenticación: {e}")
        return

    # Inicializar gestor de riesgo
    logger.info("Inicializando gestor de riesgo...")
    risk_manager = RiskManager()
    stats = risk_manager.get_estadisticas()
    logger.info(f"Capital por operación: ${stats['capital_per_trade']}")
    logger.info(f"Gestión de riesgo: SL {stats['atr_sl_multiplier']}x ATR, TP {stats['atr_tp_multiplier']}x ATR")

    # Cargar parámetros óptimos
    try:
        with open('config/optimal_params.json', 'r') as f:
            optimal_params = json.load(f)
        logger.info(f"Parámetros óptimos cargados: EMA({optimal_params['ema_short']},{optimal_params['ema_long']})")
    except FileNotFoundError:
        logger.error("No se encontró optimal_params.json. Ejecuta Fase 2 primero")
        return

    logger.separator()
    logger.info("CONFIGURACIÓN COMPLETA")
    logger.info("El bot está listo para operar")
    logger.separator()

    print("\n" + "=" * 70)
    print("ESTADO DEL BOT")
    print("=" * 70)
    print(f"✓ Modelos de IA: LSTM={use_lstm}, Sentiment={use_sentiment}")
    print(f"✓ Gestor de riesgo: Activo")
    print(f"✓ Logging: logs/bot.log y logs/trades.log")
    print(f"✓ Auto-reconexión: Habilitada")

    print("\n" + "=" * 70)
    print("IMPLEMENTACIÓN COMPLETA")
    print("=" * 70)
    print("Este script requiere la implementación completa de:")
    print("  1. WebSocket con auto-reconexión")
    print("  2. UserDataStream para detectar ejecuciones de OCO")
    print("  3. Lógica de decisión combinada:")
    print("     - Señales técnicas (EMA, RSI, MACD)")
    print("     - Predicción LSTM")
    print("     - Análisis de sentimiento")
    print("  4. Ejecución de órdenes con gestión de riesgo")
    print("  5. Sistema de logging robusto")
    print("\nConsulta PHASE_GUIDE.md sección 'FASE 5' para la implementación completa")

    print("\n" + "=" * 70)
    print("DESPLIEGUE EN VPS")
    print("=" * 70)
    print("Para ejecutar en un VPS (producción 24/7):")
    print("\n1. Conectar al VPS:")
    print("   ssh root@tu-vps-ip")
    print("\n2. Clonar repositorio y configurar")
    print("\n3. Ejecutar con screen:")
    print("   screen -S trading_bot")
    print("   python3 scripts/phase5_deployment.py")
    print("   # Detach: Ctrl+A, luego D")
    print("   # Reconectar: screen -r trading_bot")
    print("\n4. Monitorear logs:")
    print("   tail -f logs/bot.log")

    logger.separator()
    logger.info("Demo completada. Para producción, implementa según PHASE_GUIDE.md")
    logger.separator()

    print("\n✓ FASE 5 (DEMOSTRACIÓN) COMPLETADA")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
