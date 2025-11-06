#!/usr/bin/env python3
"""
FASE 2: OPTIMIZACIÓN - ITERACIÓN 001
Estrategia Multi-Timeframe Bidireccional (15m + 1h)

Este script optimiza los parámetros de la estrategia MTF usando Grid Search.
Requiere datos históricos en dos timeframes (15m y 1h).
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
from src.strategy.signal_generator import generar_senales_mtf_v001
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger('phase2_optimize_mtf_v001', 'logs/optimize_mtf_v001.log')


def cargar_datos_timeframe(symbol='ETHUSDT', timeframe='15m'):
    """
    Carga datos históricos desde CSV.

    Args:
        symbol: Par de trading (ej: 'ETHUSDT')
        timeframe: Timeframe ('15m' o '1h')

    Returns:
        DataFrame con columnas OHLCV
    """
    # Buscar archivo CSV con el patrón esperado
    data_dir = 'data'
    pattern = f"{symbol}_{timeframe}_OHLCV"

    logger.info(f"Buscando datos para {symbol} {timeframe}...")

    if not os.path.exists(data_dir):
        logger.error(f"Directorio {data_dir} no existe")
        return None

    # Buscar archivo que coincida con el patrón
    files = [f for f in os.listdir(data_dir) if f.startswith(pattern) and f.endswith('.csv')]

    if not files:
        logger.error(f"No se encontró archivo CSV para {symbol} {timeframe}")
        logger.info(f"Buscando patrón: {pattern}*.csv en {data_dir}/")
        return None

    # Tomar el archivo más reciente si hay varios
    filepath = os.path.join(data_dir, sorted(files)[-1])
    logger.info(f"Cargando datos desde: {filepath}")

    df = pd.read_csv(filepath)

    # Verificar columnas requeridas
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Archivo CSV no tiene las columnas requeridas: {required_cols}")
        return None

    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    logger.info(f"✓ Datos cargados: {len(df)} velas desde {df['timestamp'].min()} hasta {df['timestamp'].max()}")

    return df


def optimizar_estrategia_mtf():
    """
    Optimización Grid Search para estrategia Multi-Timeframe Bidireccional.
    """
    print("=" * 80)
    print("FASE 2: OPTIMIZACIÓN - ITERACIÓN 001")
    print("Estrategia Multi-Timeframe Bidireccional (15m + 1h)")
    print("=" * 80)
    print()

    # ============================================
    # 1. CARGAR DATOS DE 2 TIMEFRAMES
    # ============================================
    print("📊 Paso 1: Cargando datos históricos...")

    df_m15 = cargar_datos_timeframe('ETHUSDT', '15m')
    df_h1 = cargar_datos_timeframe('ETHUSDT', '1h')

    if df_m15 is None or df_h1 is None:
        logger.error("No se pudieron cargar los datos. Abortando optimización.")
        print("\n❌ ERROR: No se encontraron los archivos de datos necesarios.")
        print("\nAsegúrate de tener los siguientes archivos en data/:")
        print("  - ETHUSDT_15m_OHLCV_*.csv")
        print("  - ETHUSDT_1h_OHLCV_*.csv")
        print("\nPuedes descargarlos con:")
        print("  python scripts/download_mtf_data.py")
        return

    print(f"  ✓ Datos 15m: {len(df_m15)} velas")
    print(f"  ✓ Datos 1h: {len(df_h1)} velas")
    print()

    # ============================================
    # 2. DEFINIR GRID DE PARÁMETROS
    # ============================================
    print("🔧 Paso 2: Definiendo grid de parámetros...")

    param_grid = {
        'ema_fast_m15': [9, 12, 15],
        'ema_slow_m15': [21, 26, 30],
        'ema_trend_h1': [100, 150, 200],
        'atr_period': [14, 20],
        'atr_lookback': [3, 5, 7],
        'atr_multiplier': [2.0, 2.5, 3.0]
    }

    # Generar todas las combinaciones
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(product(*values))

    total_combinations = len(combinations)
    print(f"  ✓ Total de combinaciones a probar: {total_combinations}")
    print()

    # ============================================
    # 3. CALCULAR INDICADORES BASE
    # ============================================
    print("📈 Paso 3: Calculando indicadores base...")

    # Para 15m: calcularemos EMA_fast, EMA_slow y ATR para cada combinación
    # Para 1h: calcularemos EMA_trend para cada combinación

    print("  ✓ Los indicadores se calcularán dinámicamente para cada combinación")
    print()

    # ============================================
    # 4. OPTIMIZACIÓN GRID SEARCH
    # ============================================
    print("🔍 Paso 4: Iniciando Grid Search...")
    print()

    resultados = []
    capital_inicial = 10000
    comision = 0.00075

    for idx, combo in enumerate(combinations, 1):
        params = dict(zip(keys, combo))

        # Mostrar progreso cada 10 combinaciones
        if idx % 10 == 0 or idx == 1:
            print(f"  Progreso: {idx}/{total_combinations} ({100*idx/total_combinations:.1f}%)")

        try:
            # ============================================
            # 4.1. Calcular indicadores para esta combinación
            # ============================================

            # Copiar DataFrames para no modificar originales
            df_m15_test = df_m15.copy()
            df_h1_test = df_h1.copy()

            # Configuración de indicadores para 15m
            config_m15 = {
                'ema_short': params['ema_fast_m15'],
                'ema_long': params['ema_slow_m15'],
                'atr_length': params['atr_period']
            }

            # Configuración de indicadores para 1h
            config_h1 = {
                'ema_trend': params['ema_trend_h1']
            }

            # Agregar indicadores
            df_m15_test = agregar_indicadores(df_m15_test, config_m15)
            df_h1_test = agregar_indicadores(df_h1_test, config_h1)

            # ============================================
            # 4.2. Generar señales MTF
            # ============================================

            df_signals = generar_senales_mtf_v001(df_m15_test, df_h1_test, params)

            # Verificar que se generaron señales
            if df_signals is None or 'señal' not in df_signals.columns:
                logger.warning(f"Combinación {idx}: No se generaron señales")
                continue

            # ============================================
            # 4.3. Ejecutar backtest bidireccional
            # ============================================

            backtester = VectorizedBacktester(
                df=df_signals,
                initial_capital=capital_inicial,
                commission=comision
            )

            # Usar el método bidireccional v2
            atr_col = f"ATRr_{params['atr_period']}"
            results = backtester.run_backtest_bidirectional_v2(
                atr_column=atr_col,
                atr_multiplier=params['atr_multiplier']
            )

            # ============================================
            # 4.4. Calcular métricas
            # ============================================

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
    print(f"✓ Grid Search completado: {len(resultados)} combinaciones evaluadas exitosamente")
    print()

    # ============================================
    # 5. GUARDAR RESULTADOS
    # ============================================
    print("💾 Paso 5: Guardando resultados...")

    if len(resultados) == 0:
        print("❌ No se obtuvieron resultados válidos")
        return

    df_resultados = pd.DataFrame(resultados)

    # Guardar CSV completo
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'results/optimize_mtf_v001_{timestamp}.csv'

    os.makedirs('results', exist_ok=True)
    df_resultados.to_csv(output_file, index=False)

    print(f"  ✓ Resultados guardados en: {output_file}")
    print()

    # ============================================
    # 6. GENERAR REPORTE
    # ============================================
    print("=" * 80)
    print("📊 REPORTE DE OPTIMIZACIÓN")
    print("=" * 80)
    print()

    # Filtrar resultados rentables
    df_rentables = df_resultados[
        (df_resultados['Profit Factor'] > 1.1) &
        (df_resultados['Total Trades'] >= 50)
    ].copy()

    if len(df_rentables) == 0:
        print("⚠️  No se encontraron configuraciones rentables (PF > 1.1 y Trades >= 50)")
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
        print(f"Rank #{row.name + 1}")
        print(f"  Parámetros:")
        print(f"    - EMA Fast M15: {int(row['ema_fast_m15'])}")
        print(f"    - EMA Slow M15: {int(row['ema_slow_m15'])}")
        print(f"    - EMA Trend H1: {int(row['ema_trend_h1'])}")
        print(f"    - ATR Period: {int(row['atr_period'])}")
        print(f"    - ATR Lookback: {int(row['atr_lookback'])}")
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
    # 7. GUARDAR MEJOR CONFIGURACIÓN
    # ============================================
    if len(df_rentables) > 0:
        mejor = df_rentables.nlargest(1, 'Profit Factor').iloc[0]

        config_optima = {
            'estrategia': 'mtf_v001',
            'descripcion': 'Estrategia Multi-Timeframe Bidireccional 15m+1h',
            'timestamp': timestamp,
            'parametros': {
                'ema_fast_m15': int(mejor['ema_fast_m15']),
                'ema_slow_m15': int(mejor['ema_slow_m15']),
                'ema_trend_h1': int(mejor['ema_trend_h1']),
                'atr_period': int(mejor['atr_period']),
                'atr_lookback': int(mejor['atr_lookback']),
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

        config_file = 'config/optimal_params_mtf_v001.json'
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
    optimizar_estrategia_mtf()
