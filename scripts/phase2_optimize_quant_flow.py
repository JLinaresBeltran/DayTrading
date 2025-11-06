#!/usr/bin/env python3
"""
FASE 2: OPTIMIZACIÓN - ESTRATEGIA QUANT-FLOW
Estrategia Multi-Timeframe (M15 + H1) con Pullbacks y gestión avanzada de TP.

Este script optimiza los parámetros de la estrategia Quant-Flow usando Grid Search.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import json

from src.indicators.technical_simple import agregar_indicadores
from src.strategy.signal_generator_quant_flow import generar_senales_quant_flow
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger('phase2_optimize_quant_flow', 'logs/optimize_quant_flow.log')


def cargar_datos_timeframe(symbol='ETHUSDT', timeframe='15m'):
    """
    Carga datos históricos desde CSV.

    Args:
        symbol: Par de trading
        timeframe: Timeframe ('15m' o '1h')

    Returns:
        DataFrame con columnas OHLCV
    """
    data_dir = 'data'
    pattern = f"{symbol}_{timeframe}_OHLCV"

    logger.info(f"Buscando datos para {symbol} {timeframe}...")

    if not os.path.exists(data_dir):
        logger.error(f"Directorio {data_dir} no existe")
        return None

    files = [f for f in os.listdir(data_dir) if f.startswith(pattern) and f.endswith('.csv')]

    if not files:
        logger.error(f"No se encontró archivo CSV para {symbol} {timeframe}")
        return None

    filepath = os.path.join(data_dir, sorted(files)[-1])
    logger.info(f"Cargando datos desde: {filepath}")

    df = pd.read_csv(filepath)

    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Archivo CSV no tiene las columnas requeridas")
        return None

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"✓ Datos cargados: {len(df)} velas")

    return df


def optimizar_estrategia_quant_flow():
    """
    Optimización Grid Search para estrategia Quant-Flow.
    """
    print("=" * 80)
    print("FASE 2: OPTIMIZACIÓN - ESTRATEGIA QUANT-FLOW V2")
    print("MTF con ADX, filtro de fin de semana y gestión avanzada de TP")
    print("=" * 80)
    print()

    # ============================================
    # 1. CARGAR DATOS DE 2 TIMEFRAMES
    # ============================================
    print("📊 Paso 1: Cargando datos históricos...")

    df_m15 = cargar_datos_timeframe('ETHUSDT', '15m')
    df_h1 = cargar_datos_timeframe('ETHUSDT', '1h')

    if df_m15 is None or df_h1 is None:
        logger.error("No se pudieron cargar los datos")
        print("\n❌ ERROR: No se encontraron los archivos de datos necesarios.")
        print("\nEjecuta primero: python scripts/download_mtf_data.py")
        return

    print(f"  ✓ Datos 15m: {len(df_m15)} velas")
    print(f"  ✓ Datos 1h: {len(df_h1)} velas")
    print()

    # ============================================
    # 2. DEFINIR GRID DE PARÁMETROS
    # ============================================
    print("🔧 Paso 2: Definiendo grid de parámetros...")

    param_grid = {
        'ema_pullback': [21, 34, 50],
        'ema_trend_h1': [100, 150, 200],
        'adx_period': [14],
        'adx_threshold': [15, 20, 25],
        'rsi_period': [14],
        'atr_period': [14],
        'rsi_long_min': [40, 45],
        'rsi_long_max': [60, 65],
        'rsi_short_min': [35, 40],
        'rsi_short_max': [50, 55],
        'atr_multiplier': [1.5, 2.0, 2.5]
    }

    # Generar combinaciones
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    total_combinations = len(combinations)
    print(f"  ✓ Total de combinaciones a probar: {total_combinations}")
    print()

    # ============================================
    # 3. OPTIMIZACIÓN GRID SEARCH
    # ============================================
    print("🔍 Paso 3: Iniciando Grid Search...")
    print()

    resultados = []
    capital_inicial = 10000
    comision = 0.00075

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))

        # Progreso
        if idx % 10 == 0 or idx == 1:
            print(f"  Progreso: {idx}/{total_combinations} ({100*idx/total_combinations:.1f}%)")

        try:
            # ==========================================
            # 3.1. Calcular indicadores
            # ==========================================
            df_m15_test = df_m15.copy()
            df_h1_test = df_h1.copy()

            # Indicadores M15
            config_m15 = {
                'ema_short': params['ema_pullback'],
                'rsi_period': params['rsi_period'],
                'atr_length': params['atr_period'],
                'vwap': True
            }

            # Indicadores H1
            config_h1 = {
                'ema_trend': params['ema_trend_h1'],
                'adx_period': params['adx_period']
            }

            df_m15_test = agregar_indicadores(df_m15_test, config_m15)
            df_h1_test = agregar_indicadores(df_h1_test, config_h1)

            # ==========================================
            # 3.2. Generar señales
            # ==========================================
            df_signals = generar_senales_quant_flow(df_m15_test, df_h1_test, params)

            if df_signals is None or 'señal' not in df_signals.columns:
                logger.warning(f"Combinación {idx}: No se generaron señales")
                continue

            # ==========================================
            # 3.3. Ejecutar backtest con gestión avanzada de TP
            # ==========================================
            backtester = VectorizedBacktester(
                df=df_signals,
                initial_capital=capital_inicial,
                commission=comision
            )

            atr_col = f"ATRr_{params['atr_period']}"
            results = backtester.run_backtest_quant_flow(
                atr_column=atr_col,
                atr_multiplier=params['atr_multiplier']
            )

            # ==========================================
            # 3.4. Calcular métricas
            # ==========================================
            metricas = backtester.calculate_metrics()

            # Agregar parámetros al resultado
            resultado = {
                'id': idx,
                **params,
                **metricas
            }

            resultados.append(resultado)

        except Exception as e:
            logger.error(f"Error en combinación {idx}: {str(e)}")
            continue

    print()
    print(f"✓ Grid Search completado: {len(resultados)} combinaciones evaluadas")
    print()

    # ============================================
    # 4. GUARDAR RESULTADOS
    # ============================================
    print("💾 Paso 4: Guardando resultados...")

    if len(resultados) == 0:
        print("❌ No se obtuvieron resultados válidos")
        return

    df_resultados = pd.DataFrame(resultados)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/optimize_quant_flow_{timestamp}.csv'

    os.makedirs('results', exist_ok=True)
    df_resultados.to_csv(output_file, index=False)

    print(f"  ✓ Resultados guardados en: {output_file}")
    print()

    # ============================================
    # 5. GENERAR REPORTE
    # ============================================
    print("=" * 80)
    print("📊 REPORTE DE OPTIMIZACIÓN - QUANT-FLOW V2")
    print("=" * 80)
    print()

    # Filtrar resultados rentables
    df_rentables = df_resultados[
        (df_resultados['Profit Factor'] > 1.1) &
        (df_resultados['Total Trades'] >= 30)
    ].copy()

    if len(df_rentables) == 0:
        print("⚠️  No se encontraron configuraciones rentables (PF > 1.1 y Trades >= 30)")
        print()
        print("Mejores 10 configuraciones por Profit Factor:")
        print()
        df_top10 = df_resultados.nlargest(10, 'Profit Factor')
    else:
        print(f"✓ Se encontraron {len(df_rentables)} configuraciones rentables")
        print()
        print("Top 10 configuraciones rentables:")
        print()
        df_top10 = df_rentables.nlargest(10, 'Profit Factor')

    # Mostrar Top 10
    for idx, row in df_top10.iterrows():
        print(f"Rank #{idx + 1}")
        print(f"  Parámetros:")
        print(f"    - EMA Pullback: {int(row['ema_pullback'])}")
        print(f"    - EMA Trend H1: {int(row['ema_trend_h1'])}")
        print(f"    - ADX Threshold: {int(row['adx_threshold'])}")
        print(f"    - RSI Long: {int(row['rsi_long_min'])}-{int(row['rsi_long_max'])}")
        print(f"    - RSI Short: {int(row['rsi_short_min'])}-{int(row['rsi_short_max'])}")
        print(f"    - ATR Multiplier: {row['atr_multiplier']:.1f}")
        print(f"  Métricas:")
        print(f"    - Profit Factor: {row['Profit Factor']:.2f}")
        print(f"    - Total Return: {row['Total Return (%)']:.2f}%")
        print(f"    - Sharpe Ratio: {row['Sharpe Ratio']:.2f}")
        print(f"    - Max Drawdown: {row['Max Drawdown (%)']:.2f}%")
        print(f"    - Win Rate: {row['Win Rate (%)']:.2f}%")
        print(f"    - Total Trades: {int(row['Total Trades'])}")
        print()

    # ============================================
    # 6. GUARDAR MEJOR CONFIGURACIÓN
    # ============================================
    if len(df_rentables) > 0:
        mejor = df_rentables.nlargest(1, 'Profit Factor').iloc[0]

        config_optima = {
            'estrategia': 'quant_flow_v2',
            'descripcion': 'Estrategia Quant-Flow v2: MTF con ADX y filtro de fin de semana',
            'timestamp': timestamp,
            'parametros': {
                'ema_pullback': int(mejor['ema_pullback']),
                'ema_trend_h1': int(mejor['ema_trend_h1']),
                'adx_period': int(mejor['adx_period']),
                'adx_threshold': int(mejor['adx_threshold']),
                'rsi_period': int(mejor['rsi_period']),
                'atr_period': int(mejor['atr_period']),
                'rsi_long_min': int(mejor['rsi_long_min']),
                'rsi_long_max': int(mejor['rsi_long_max']),
                'rsi_short_min': int(mejor['rsi_short_min']),
                'rsi_short_max': int(mejor['rsi_short_max']),
                'atr_multiplier': float(mejor['atr_multiplier'])
            },
            'metricas': {
                'profit_factor': float(mejor['Profit Factor']),
                'total_return_pct': float(mejor['Total Return (%)']),
                'sharpe_ratio': float(mejor['Sharpe Ratio']),
                'max_drawdown_pct': float(mejor['Max Drawdown (%)']),
                'win_rate_pct': float(mejor['Win Rate (%)']),
                'total_trades': int(mejor['Total Trades'])
            }
        }

        config_file = 'config/optimal_params_quant_flow_v2.json'
        with open(config_file, 'w') as f:
            json.dump(config_optima, f, indent=4)

        print("=" * 80)
        print(f"✓ Configuración óptima guardada en: {config_file}")
        print("=" * 80)
    else:
        print("=" * 80)
        print("⚠️  No se guardó configuración óptima (ninguna cumplió criterios)")
        print("=" * 80)

    print()
    print("🎉 Optimización completada exitosamente")


if __name__ == "__main__":
    optimizar_estrategia_quant_flow()
