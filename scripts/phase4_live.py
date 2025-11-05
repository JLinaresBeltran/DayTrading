#!/usr/bin/env python3
"""
FASE 4: Ejecución Semiautomática (Trading Real)
Ejecuta órdenes reales con capital mínimo y gestión de riesgo.

⚠️ ADVERTENCIA: Este script opera con DINERO REAL
NOTA: Este es un script esqueleto. Ver PHASE_GUIDE.md para implementación completa
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from src.data.binance_client import BinanceClientManager
from src.strategy.risk_manager import RiskManager
from src.utils.logger import TradingLogger


def main():
    print("=" * 70)
    print("⚠️  FASE 4: TRADING REAL CON DINERO REAL  ⚠️")
    print("=" * 70)

    # Advertencia de seguridad
    print("\n⚠️  ADVERTENCIAS IMPORTANTES:")
    print("   1. Este bot ejecutará órdenes REALES en Binance")
    print("   2. Asegúrate de estar en TESTNET (testnet: true en config.json)")
    print("   3. Capital por operación: $15 USD")
    print("   4. Gestión de riesgo: Stop Loss + Take Profit automáticos")
    print("   5. Límite de pérdida diaria: 5% del capital total")

    respuesta = input("\n¿Estás seguro de continuar? (escribe 'SI' para confirmar): ")
    if respuesta != 'SI':
        print("\n✓ Operación cancelada. Esto fue una buena decisión si no estabas seguro.")
        return

    # Verificar configuración
    print("\n1. Verificando configuración...")
    try:
        with open('config/config.json', 'r') as f:
            config = json.load(f)

        if config['binance']['testnet']:
            print("   ✓ TESTNET activado (entorno seguro)")
        else:
            print("   ⚠️  PRODUCCIÓN activada (DINERO REAL)")
            respuesta2 = input("   ¿Continuar con dinero real? (escribe 'CONFIRMO'): ")
            if respuesta2 != 'CONFIRMO':
                print("\n✓ Operación cancelada")
                return

    except Exception as e:
        print(f"   ❌ Error al leer configuración: {e}")
        return

    # Cargar parámetros óptimos
    print("\n2. Cargando parámetros óptimos...")
    try:
        with open('config/optimal_params.json', 'r') as f:
            optimal_params = json.load(f)
        print(f"   ✓ Parámetros cargados")
    except FileNotFoundError:
        print("   ❌ No se encontró optimal_params.json")
        print("   Ejecuta primero: python scripts/phase2_backtest.py")
        return

    # Crear cliente autenticado
    print("\n3. Conectando a Binance con credenciales...")
    try:
        manager = BinanceClientManager()
        client = manager.get_authenticated_client()
        print("   ✓ Autenticación exitosa")
    except Exception as e:
        print(f"   ❌ Error de autenticación: {e}")
        print("   Verifica tus credenciales en config/config.json")
        return

    # Inicializar gestión de riesgo
    print("\n4. Inicializando gestor de riesgo...")
    risk_manager = RiskManager()
    stats = risk_manager.get_estadisticas()
    print(f"   ✓ Capital por operación: ${stats['capital_per_trade']}")
    print(f"   ✓ Stop Loss: {stats['atr_sl_multiplier']}x ATR")
    print(f"   ✓ Take Profit: {stats['atr_tp_multiplier']}x ATR")
    print(f"   ✓ Máx. posiciones: {stats['max_open_positions']}")
    print(f"   ✓ Máx. pérdida diaria: {stats['max_daily_loss_pct']}%")

    # Inicializar logger
    print("\n5. Inicializando sistema de logging...")
    logger = TradingLogger()
    logger.separator()
    logger.info("BOT DE TRADING - FASE 4 INICIADO")
    logger.separator()

    print("\n6. Iniciando bot...")
    print("   ⚠️  IMPLEMENTACIÓN COMPLETA EN PHASE_GUIDE.md")
    print("\n   El bot debería:")
    print("   - Conectarse a WebSocket de datos en vivo")
    print("   - Calcular indicadores en cada vela")
    print("   - Generar señales")
    print("   - Ejecutar órdenes cuando señal == COMPRA:")
    print("     • Calcular tamaño de posición")
    print("     • Calcular SL y TP basado en ATR")
    print("     • Ejecutar orden MARKET de compra")
    print("     • Colocar orden OCO (TP + SL)")
    print("   - Registrar todas las operaciones en logs")

    logger.info("DEMO MODE: Bot no ejecutará órdenes reales")
    logger.info("Consulta PHASE_GUIDE.md para implementación completa")

    print("\n" + "=" * 70)
    print("✓ FASE 4 (DEMOSTRACIÓN) COMPLETADA")
    print("=" * 70)
    print("\nPróximos pasos:")
    print("  - Implementar según PHASE_GUIDE.md sección 'FASE 4'")
    print("  - PROBAR PRIMERO EN TESTNET durante 1 semana")
    print("  - Monitorear logs en logs/bot.log y logs/trades.log")
    print("  - Cuando funcione bien: Ejecutar Fase 5 (con IA)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Bot detenido por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
