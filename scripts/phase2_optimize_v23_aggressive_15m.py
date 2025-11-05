#!/usr/bin/env python3
"""
ITERACIÓN 23: OPTIMIZACIÓN ULTRA-AGRESIVA EN 15M (DONCHIAN + EMA)
===================================================================

CONTEXTO:
- Iteración 22: PF 2.18 (éxito técnico) pero solo 32 trades (fracaso en frecuencia)
- Objetivo v23: Forzar alta frecuencia (> 150 trades) manteniendo PF rentable (> 1.1)

HIPÓTESIS:
Al reducir los períodos de Donchian y EMA a niveles de "scalping" (Donchian 5-7,
EMA 20-30), podemos aumentar drásticamente el Num Trades a > 150, manteniendo un
Profit Factor rentable.

ESTRATEGIA:
- 2 Capas: Filtro EMA + Breakout Donchian (generar_senales_donchian_ema)
- Gestión de riesgo: SL y TP basados en ATR (run_backtest_with_sl_tp)

PARÁMETROS A OPTIMIZAR:
- donchian_period: [5, 7, 10] (Ultra-agresivos/rápidos)
- ema_filter_period: [25, 50, 75] (Filtros de tendencia cortos)
- sl_multiplier: [2.0, 3.0, 4.0] (Stop Loss)
- tp_multiplier: [2.0, 3.0, 4.0] (Take Profit - R:R más simétricos)

Total de combinaciones: 3 × 3 × 3 × 3 = 81 configuraciones

CRITERIOS DE ÉXITO:
- Profit Factor > 1.1 (rentabilidad)
- Num Trades > 150 (alta frecuencia)
- Ambos deben cumplirse simultáneamente

DATOS:
- Activo: ETHUSDT
- Timeframe: 15m
- Período: 1 año (datos del archivo CSV ya descargado)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

# Configurar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical_simple import agregar_indicadores  # Versión sin pandas-ta
from src.strategy.signal_generator import generar_senales_donchian_ema
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger("phase2_optimize_v23", "logs/phase2_optimize_v23.log")


def main():
    logger.info("=" * 80)
    logger.info("ITERACIÓN 23: OPTIMIZACIÓN ULTRA-AGRESIVA EN 15M")
    logger.info("=" * 80)
    logger.info("\nHipótesis: Parámetros ultra-agresivos → Alta frecuencia (>150 trades)")
    logger.info("Objetivo: PF > 1.1 AND Num Trades > 150")

    # =========================================================================
    # 1. CARGAR DATOS HISTÓRICOS DE 15M
    # =========================================================================
    logger.info("\n1. Cargando datos históricos de ETHUSDT 15m...")

    # Intentar cargar desde archivo CSV primero
    csv_file = project_root / "ETHUSDT_15m_OHLCV_2025-11-05.csv"

    if csv_file.exists():
        logger.info(f"   ✓ Cargando desde archivo: {csv_file.name}")
        df = pd.read_csv(csv_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"   ✓ {len(df):,} velas cargadas desde CSV")
    else:
        # Si no existe el archivo, descargar desde API
        logger.info("   Descargando desde Binance API...")
        try:
            client = BinanceClientManager().get_public_client()
            df = obtener_datos_binance(client, "ETHUSDT", "15m", "1 year ago UTC")
            logger.info(f"   ✓ {len(df):,} velas descargadas desde API")
        except Exception as e:
            logger.error(f"   ✗ Error descargando datos: {e}")
            return

    # =========================================================================
    # 2. DEFINIR GRID DE PARÁMETROS (ULTRA-AGRESIVOS)
    # =========================================================================
    logger.info("\n2. Definiendo Grid de Parámetros Ultra-Agresivos...")

    param_grid = {
        'donchian_period': [5, 7, 10],        # Períodos muy cortos para alta frecuencia
        'ema_filter_period': [25, 50, 75],    # Filtros de tendencia de corto plazo
        'sl_multiplier': [2.0, 3.0, 4.0],     # Stop Loss en múltiplos de ATR
        'tp_multiplier': [2.0, 3.0, 4.0]      # Take Profit en múltiplos de ATR
    }

    total_combinations = len(list(ParameterGrid(param_grid)))
    logger.info(f"   ✓ Total de combinaciones: {total_combinations}")
    logger.info(f"   ✓ Parámetros:")
    for param, values in param_grid.items():
        logger.info(f"      - {param}: {values}")

    # =========================================================================
    # 3. GRID SEARCH: ITERAR SOBRE TODAS LAS COMBINACIONES
    # =========================================================================
    logger.info("\n3. Ejecutando Grid Search...")
    logger.info("   (Esto puede tardar varios minutos)\n")

    results_list = []

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        try:
            # Mostrar progreso
            if i % 10 == 0 or i == 1:
                logger.info(f"   Evaluando combinación {i}/{total_combinations}...")

            # ================================================================
            # 3.1. AGREGAR INDICADORES CON PARÁMETROS ESPECÍFICOS
            # ================================================================
            df_test = df.copy()

            # Config para agregar indicadores
            indicator_config = {
                'ema_filter': params['ema_filter_period'],    # EMA filtro de tendencia
                'donchian_period': params['donchian_period'],  # Canal de Donchian
                'atr_length': 14,                              # ATR estándar para SL/TP
                'rsi_period': 14,                              # RSI (no se usa pero se calcula)
                'bb_length': 20,                               # Bollinger (no se usa)
                'bb_std': 2,
                'macd_fast': 12,                               # MACD (no se usa)
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': 14,                                 # Estocástico (no se usa)
                'stoch_d': 3,
                'stoch_smooth': 3
            }

            df_test = agregar_indicadores(df_test, config=indicator_config)

            # ================================================================
            # 3.2. GENERAR SEÑALES CON ESTRATEGIA DONCHIAN + EMA
            # ================================================================
            signal_config = {
                'donchian_period': params['donchian_period'],
                'ema_filter_period': params['ema_filter_period']
            }

            df_test = generar_senales_donchian_ema(df_test, config=signal_config)

            # ================================================================
            # 3.3. EJECUTAR BACKTEST CON SL Y TP
            # ================================================================
            backtester = VectorizedBacktester(
                df=df_test,
                initial_capital=10000,
                commission=0.00075,   # Binance 0.075%
                slippage=0.0005       # Slippage 0.05%
            )

            # Ejecutar backtest con SL y TP parametrizables
            backtester.run_backtest_with_sl_tp(
                atr_column='ATRr_14',
                sl_multiplier=params['sl_multiplier'],
                tp_multiplier=params['tp_multiplier']
            )

            # Calcular métricas
            metrics = backtester.calculate_metrics()

            # ================================================================
            # 3.4. GUARDAR RESULTADOS DE ESTA COMBINACIÓN
            # ================================================================
            result = {
                # Parámetros
                'donchian_period': params['donchian_period'],
                'ema_filter_period': params['ema_filter_period'],
                'sl_multiplier': params['sl_multiplier'],
                'tp_multiplier': params['tp_multiplier'],

                # Métricas clave
                'profit_factor': metrics.get('profit_factor', 0),
                'num_trades': metrics.get('num_trades', 0),
                'win_rate_pct': metrics.get('win_rate_pct', 0),
                'total_return_pct': metrics.get('total_return_pct', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                'final_value': metrics.get('final_value', 0),
                'annual_return_pct': metrics.get('annual_return_pct', 0),
                'sortino_ratio': metrics.get('sortino_ratio', 0),
                'calmar_ratio': metrics.get('calmar_ratio', 0)
            }

            results_list.append(result)

        except Exception as e:
            logger.error(f"   ✗ Error en combinación {i}: {e}")
            continue

    # =========================================================================
    # 4. GUARDAR TODOS LOS RESULTADOS EN CSV
    # =========================================================================
    logger.info(f"\n4. Guardando resultados completos...")

    results_df = pd.DataFrame(results_list)

    # Ordenar por Profit Factor (descendente)
    results_df = results_df.sort_values('profit_factor', ascending=False)

    # Guardar CSV completo
    output_file = project_root / "backtest_results_eth_v23_aggressive_15m.csv"
    results_df.to_csv(output_file, index=False)

    logger.info(f"   ✓ Resultados guardados: {output_file.name}")
    logger.info(f"   ✓ Total de combinaciones evaluadas: {len(results_df)}")

    # =========================================================================
    # 5. FILTRAR Y MOSTRAR TOP 10
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("REPORTE FINAL: CRITERIOS DE ÉXITO")
    logger.info("=" * 80)

    # Definir columnas para display
    display_cols = [
        'donchian_period', 'ema_filter_period', 'sl_multiplier', 'tp_multiplier',
        'profit_factor', 'num_trades', 'win_rate_pct', 'total_return_pct',
        'sharpe_ratio', 'max_drawdown_pct'
    ]

    # Filtrar por criterios de éxito
    filtered = results_df[
        (results_df['profit_factor'] > 1.1) &
        (results_df['num_trades'] > 150)
    ]

    logger.info(f"\nCombinaciones que cumplen AMBOS criterios:")
    logger.info(f"  - Profit Factor > 1.1: ✓")
    logger.info(f"  - Num Trades > 150: ✓")
    logger.info(f"  - Total encontradas: {len(filtered)}")

    if len(filtered) > 0:
        logger.info("\n" + "=" * 80)
        logger.info("TOP 10 CONFIGURACIONES (ordenadas por Profit Factor)")
        logger.info("=" * 80)

        top10 = filtered.head(10)

        # Mostrar tabla con pandas
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)

        logger.info("\n" + top10[display_cols].to_string(index=False))

        # Mostrar detalles del #1
        logger.info("\n" + "=" * 80)
        logger.info("MEJOR CONFIGURACIÓN (#1)")
        logger.info("=" * 80)

        best = top10.iloc[0]
        logger.info(f"\nParámetros Optimizados:")
        logger.info(f"  - Donchian Period: {int(best['donchian_period'])}")
        logger.info(f"  - EMA Filter Period: {int(best['ema_filter_period'])}")
        logger.info(f"  - SL Multiplier: {best['sl_multiplier']:.1f}x ATR")
        logger.info(f"  - TP Multiplier: {best['tp_multiplier']:.1f}x ATR")
        logger.info(f"  - Risk:Reward Ratio: 1:{best['tp_multiplier']/best['sl_multiplier']:.2f}")

        logger.info(f"\nMétricas de Performance:")
        logger.info(f"  - Profit Factor: {best['profit_factor']:.2f} ✓")
        logger.info(f"  - Num Trades: {int(best['num_trades'])} ✓")
        logger.info(f"  - Win Rate: {best['win_rate_pct']:.1f}%")
        logger.info(f"  - Total Return: {best['total_return_pct']:+.2f}%")
        logger.info(f"  - Annual Return: {best['annual_return_pct']:+.2f}%")
        logger.info(f"  - Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        logger.info(f"  - Max Drawdown: {best['max_drawdown_pct']:.2f}%")
        logger.info(f"  - Final Value: ${best['final_value']:,.2f}")

        logger.info("\n✓✓✓ ITERACIÓN 23 APROBADA ✓✓✓")
        logger.info("Encontramos configuraciones que cumplen ambos objetivos:")
        logger.info("  1. Alta frecuencia (> 150 trades)")
        logger.info("  2. Rentabilidad sostenible (PF > 1.1)")

    else:
        logger.info("\n✗✗✗ ITERACIÓN 23 SIN APROBACIÓN ✗✗✗")
        logger.info("No se encontraron configuraciones que cumplan AMBOS criterios.")

        # Mostrar las mejores por separado
        logger.info("\n" + "-" * 80)
        logger.info("ANÁLISIS SEPARADO:")
        logger.info("-" * 80)

        # Mejores por PF (sin filtro de trades)
        best_pf = results_df.head(5)
        logger.info("\nTop 5 por Profit Factor (sin filtro de trades):")
        logger.info(best_pf[display_cols].to_string(index=False))

        # Mejores por frecuencia (sin filtro de PF)
        best_freq = results_df[results_df['num_trades'] > 150].head(5)
        if len(best_freq) > 0:
            logger.info("\nTop 5 con alta frecuencia (> 150 trades):")
            logger.info(best_freq[display_cols].to_string(index=False))
        else:
            logger.info("\nNinguna configuración alcanzó > 150 trades.")
            logger.info("Configuraciones con más trades:")
            highest_trades = results_df.nlargest(5, 'num_trades')
            logger.info(highest_trades[display_cols].to_string(index=False))

    # =========================================================================
    # 6. CONCLUSIÓN
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EJECUCIÓN COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"\nResultados guardados en: {output_file.name}")
    logger.info("Próximos pasos:")
    logger.info("  1. Revisar el CSV completo para análisis detallado")
    logger.info("  2. Si se cumplieron los criterios, proceder con v24")
    logger.info("  3. Si no, considerar:")
    logger.info("     - Ajustar objetivo de frecuencia (¿100 trades en lugar de 150?)")
    logger.info("     - Probar timeframe más corto (5m o 1m)")
    logger.info("     - Evaluar estrategia de múltiples activos")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Operación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error fatal: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
