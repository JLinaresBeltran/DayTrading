"""
BÚSQUEDA EXHAUSTIVA DE ESTRATEGIAS ÓPTIMAS

Este script prueba miles de combinaciones de estrategias para encontrar
aquellas que cumplan con criterios específicos de rentabilidad y riesgo.

Criterios objetivo:
- Profit Factor >= 2.0 (ratio 1:2)
- Retorno Total >= 100%
- Max Drawdown <= 12%
- Número de Trades > 220

Autor: Claude Code
Fecha: 2025-11-06
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product
import json

# Add src to path
sys.path.append('/home/user/DayTrading')

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.advanced_signals import generate_signals_multi_indicator
from src.backtest.engine import VectorizedBacktester


# ============================================================================
# CONFIGURACIÓN DE BÚSQUEDA
# ============================================================================

SYMBOL = 'ETHUSDT'
INTERVAL = '15m'
START_DATE = '365 days ago UTC'
INITIAL_CAPITAL = 10000

# Criterios de filtrado
CRITERIA = {
    'min_profit_factor': 2.0,      # Objetivo: 1:2 ratio
    'min_return_pct': 100.0,       # Objetivo: 100% retorno
    'max_drawdown_pct': 12.0,      # Objetivo: Max DD <= 12%
    'min_num_trades': 220,         # Objetivo: > 220 trades
    'min_win_rate_pct': 35.0,      # Adicional: Win Rate >= 35%
    'min_sharpe_ratio': 0.5,       # Adicional: Sharpe >= 0.5
}

# ============================================================================
# GRID DE PARÁMETROS (ESTRATEGIAS A PROBAR)
# ============================================================================

STRATEGY_GRID = {
    # ========================================
    # TIPO DE ESTRATEGIA (REDUCIDO)
    # ========================================
    'regime_type': ['ema_adx', 'sma_adx', 'none'],  # Los más prometedores con ADX
    'regime_period': [100, 150],  # Períodos medios
    'regime_direction': ['long_only', 'hybrid'],  # Enfoque en largos e híbrido

    # ========================================
    # PARÁMETROS DE ADX
    # ========================================
    'adx_period': [14],
    'adx_threshold': [25],  # Valor óptimo estándar

    # ========================================
    # INDICADORES DE ENTRADA (combinaciones REDUCIDAS)
    # ========================================
    'entry_combinations': [
        # Estrategias EMA (las más probadas)
        ['ema_cross'],
        ['ema_cross', 'rsi'],

        # Estrategias VWMA (alternativa volumen)
        ['vwma_cross'],
        ['vwma_cross', 'rsi'],

        # Estrategias MACD
        ['macd', 'rsi'],

        # Estrategias Donchian
        ['donchian', 'rsi'],

        # Estrategias Bollinger
        ['bb', 'rsi'],

        # Estrategias Supertrend (nuevo - prioritario)
        ['supertrend'],
        ['supertrend', 'rsi'],
    ],

    # ========================================
    # PARÁMETROS DE EMAS (REDUCIDO)
    # ========================================
    'ema_fast': [12, 15],  # Valores más usados
    'ema_slow': [21, 26],

    # ========================================
    # PARÁMETROS DE RSI
    # ========================================
    'rsi_period': [14],
    'rsi_oversold': [30],  # Valor estándar
    'rsi_overbought': [70],  # Valor estándar

    # ========================================
    # PARÁMETROS DE MACD (ESTÁNDAR)
    # ========================================
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],

    # ========================================
    # PARÁMETROS DE BOLLINGER (ESTÁNDAR)
    # ========================================
    'bb_length': [20],
    'bb_std': [2.0],

    # ========================================
    # PARÁMETROS DE DONCHIAN
    # ========================================
    'donchian_period': [20, 30],

    # ========================================
    # PARÁMETROS DE VWMA (REDUCIDO)
    # ========================================
    'vwma_fast': [12, 15],
    'vwma_slow': [21, 26],

    # ========================================
    # PARÁMETROS DE SUPERTREND
    # ========================================
    'supertrend_length': [10, 14],
    'supertrend_multiplier': [2.0, 3.0],

    # ========================================
    # FILTROS ADICIONALES (SIMPLIFICADO)
    # ========================================
    'use_volume_filter': [False, True],
    'volume_ma_period': [20],
    'use_atr_filter': [False],  # Simplificado
    'atr_min_threshold': [0.5],  # Valor único

    # ========================================
    # GESTIÓN DE RIESGO (RATIOS SL:TP REDUCIDOS)
    # ========================================
    'atr_period': [14],  # Solo estándar
    'atr_multiplier_combinations': [
        {'sl': 2.0, 'tp': 4.0},   # Ratio 1:2
        {'sl': 2.0, 'tp': 6.0},   # Ratio 1:3
        {'sl': 2.5, 'tp': 5.0},   # Ratio 1:2
        {'sl': 3.0, 'tp': 6.0},   # Ratio 1:2
    ],
}


# ============================================================================
# FUNCIONES AUXILIARES
# ============================================================================

def get_relevant_params_for_indicators(entry_indicators):
    """
    Determina qué parámetros son relevantes para una combinación de indicadores.

    Esto evita generar combinaciones inútiles (ej: probar parámetros de VWMA
    cuando la estrategia solo usa EMA).
    """
    params = {
        # Siempre incluir parámetros base
        'regime_type': STRATEGY_GRID['regime_type'],
        'regime_period': STRATEGY_GRID['regime_period'],
        'regime_direction': STRATEGY_GRID['regime_direction'],
        'adx_period': STRATEGY_GRID['adx_period'],
        'adx_threshold': STRATEGY_GRID['adx_threshold'],
        'use_volume_filter': STRATEGY_GRID['use_volume_filter'],
        'volume_ma_period': STRATEGY_GRID['volume_ma_period'],
        'use_atr_filter': STRATEGY_GRID['use_atr_filter'],
        'atr_min_threshold': STRATEGY_GRID['atr_min_threshold'],
        'atr_period': STRATEGY_GRID['atr_period'],
        'atr_multiplier_combinations': STRATEGY_GRID['atr_multiplier_combinations'],
    }

    # Agregar parámetros específicos según indicadores usados
    for indicator in entry_indicators:
        if indicator == 'ema_cross':
            params['ema_fast'] = STRATEGY_GRID['ema_fast']
            params['ema_slow'] = STRATEGY_GRID['ema_slow']
        elif indicator == 'vwma_cross':
            params['vwma_fast'] = STRATEGY_GRID['vwma_fast']
            params['vwma_slow'] = STRATEGY_GRID['vwma_slow']
        elif indicator == 'rsi':
            params['rsi_period'] = STRATEGY_GRID['rsi_period']
            params['rsi_oversold'] = STRATEGY_GRID['rsi_oversold']
            params['rsi_overbought'] = STRATEGY_GRID['rsi_overbought']
        elif indicator == 'macd':
            params['macd_fast'] = STRATEGY_GRID['macd_fast']
            params['macd_slow'] = STRATEGY_GRID['macd_slow']
            params['macd_signal'] = STRATEGY_GRID['macd_signal']
        elif indicator == 'bb':
            params['bb_length'] = STRATEGY_GRID['bb_length']
            params['bb_std'] = STRATEGY_GRID['bb_std']
        elif indicator == 'donchian':
            params['donchian_period'] = STRATEGY_GRID['donchian_period']
        elif indicator == 'supertrend':
            params['supertrend_length'] = STRATEGY_GRID['supertrend_length']
            params['supertrend_multiplier'] = STRATEGY_GRID['supertrend_multiplier']

    return params


def generate_strategy_configs():
    """
    Genera configuraciones de estrategias de forma INTELIGENTE.

    Solo incluye parámetros relevantes para cada combinación de indicadores,
    evitando la explosión combinatoria de probar parámetros innecesarios.
    """
    configs = []

    # Iterar sobre cada entry_combination
    for entry_combination in STRATEGY_GRID['entry_combinations']:

        # Obtener solo los parámetros relevantes para esta combinación
        relevant_params = get_relevant_params_for_indicators(entry_combination)

        # Generar producto cartesiano SOLO de parámetros relevantes
        keys = list(relevant_params.keys())
        values = [relevant_params[k] for k in keys]

        for combination in product(*values):
            config = dict(zip(keys, combination))

            # Extraer entry_indicators y atr_multipliers
            config['entry_indicators'] = entry_combination
            atr_mult = config.pop('atr_multiplier_combinations')
            config['sl_atr_multiplier'] = atr_mult['sl']
            config['tp_atr_multiplier'] = atr_mult['tp']

            configs.append(config)

    return configs


def calculate_total_combinations():
    """
    Calcula el número REAL de combinaciones a probar.

    Tiene en cuenta que solo se prueban parámetros relevantes para cada
    combinación de indicadores.
    """
    return len(generate_strategy_configs())


def test_strategy(df_base, config, strategy_id):
    """
    Prueba una configuración de estrategia y devuelve métricas.

    Args:
        df_base: DataFrame base con indicadores pre-calculados
        config: Configuración de la estrategia
        strategy_id: ID único de la estrategia

    Returns:
        Dict con métricas de la estrategia
    """
    try:
        # Generar señales
        df = generate_signals_multi_indicator(df_base.copy(), config)

        # Si no hay señales, retornar None
        if df['señal'].sum() == 0:
            return None

        # Ejecutar backtest con Stop Loss dinámico
        backtester = VectorizedBacktester(
            df=df,
            initial_capital=INITIAL_CAPITAL,
            commission=0.00075,
            slippage=0.0005
        )

        atr_col = f"ATRr_{config['atr_period']}"
        backtester.run_backtest_with_stop_loss(
            atr_column=atr_col,
            atr_multiplier=config['sl_atr_multiplier']
        )

        # Calcular métricas
        metrics = backtester.calculate_metrics()

        # Agregar metadatos de la estrategia
        result = {
            'id': strategy_id,
            **config,
            **metrics
        }

        return result

    except Exception as e:
        print(f"  ✗ Error en estrategia {strategy_id}: {str(e)}")
        return None


def filter_results(results, criteria):
    """Filtra resultados por criterios de éxito."""
    df = pd.DataFrame(results)

    filtered = df[
        (df['profit_factor'] >= criteria['min_profit_factor']) &
        (df['total_return_pct'] >= criteria['min_return_pct']) &
        (df['max_drawdown_pct'] <= criteria['max_drawdown_pct']) &
        (df['num_trades'] > criteria['min_num_trades']) &
        (df['win_rate_pct'] >= criteria['min_win_rate_pct']) &
        (df['sharpe_ratio'] >= criteria['min_sharpe_ratio'])
    ]

    return filtered.sort_values('profit_factor', ascending=False)


def print_progress(current, total, start_time):
    """Imprime progreso de la búsqueda."""
    elapsed = (datetime.now() - start_time).total_seconds()
    pct = (current / total) * 100
    rate = current / elapsed if elapsed > 0 else 0
    eta = (total - current) / rate if rate > 0 else 0

    print(f"  [{current}/{total}] {pct:.1f}% | {rate:.1f} tests/sec | ETA: {eta/60:.1f} min", end='\r')


# ============================================================================
# MAIN: EJECUCIÓN DE BÚSQUEDA EXHAUSTIVA
# ============================================================================

def main():
    print("="*80)
    print("BÚSQUEDA EXHAUSTIVA DE ESTRATEGIAS ÓPTIMAS")
    print("="*80)
    print(f"\nObjetivo: Encontrar estrategias con:")
    print(f"  • Profit Factor >= {CRITERIA['min_profit_factor']}")
    print(f"  • Retorno Total >= {CRITERIA['min_return_pct']}%")
    print(f"  • Max Drawdown <= {CRITERIA['max_drawdown_pct']}%")
    print(f"  • Número de Trades > {CRITERIA['min_num_trades']}")
    print(f"  • Win Rate >= {CRITERIA['min_win_rate_pct']}%")
    print(f"  • Sharpe Ratio >= {CRITERIA['min_sharpe_ratio']}")

    # Calcular número de combinaciones
    total_combinations = calculate_total_combinations()
    print(f"\n📊 Combinaciones totales a probar: {total_combinations:,}")
    print(f"⏱️  Tiempo estimado: {total_combinations/10/60:.1f} minutos (asumiendo 10 tests/sec)\n")

    # Confirmar ejecución
    response = input("¿Deseas continuar? (y/n): ")
    if response.lower() != 'y':
        print("❌ Búsqueda cancelada.")
        return

    print("\n" + "="*80)
    print("FASE 1: DESCARGA DE DATOS Y CÁLCULO DE INDICADORES")
    print("="*80)

    # Conectar a Binance
    print(f"\n1. Conectando a Binance...")
    manager = BinanceClientManager()
    client = manager.get_public_client()

    # Descargar datos
    print(f"2. Descargando datos {SYMBOL} {INTERVAL}...")
    df_base = obtener_datos_binance(
        client=client,
        simbolo=SYMBOL,
        intervalo=INTERVAL,
        inicio=START_DATE
    )
    print(f"   ✓ {len(df_base)} velas descargadas")

    # Calcular TODOS los indicadores necesarios
    print(f"3. Calculando indicadores...")
    indicator_config = {
        'ema_periods': [9, 12, 15, 21, 26, 30, 50, 100, 150, 200],
        'sma_periods': [50, 100, 150, 200],
        'rsi_period': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'bb_length': 20,
        'bb_std': 2.0,
        'atr_length': 14,  # Solo ATR_14 primero
        'stoch_k': 14,
        'stoch_d': 3,
        'stoch_smooth': 3,
        'donchian_period': [20, 30],
    }
    df_base = agregar_indicadores(df_base, config=indicator_config)

    # Calcular indicadores adicionales manualmente
    print(f"   Calculando ADX, ATR_20 y Supertrend...")

    # ATR_20
    df_base.ta.atr(length=20, append=True)

    # ADX (Average Directional Index)
    df_base.ta.adx(length=14, append=True)

    # Supertrend (múltiples configuraciones)
    df_base.ta.supertrend(length=10, multiplier=2.0, append=True)
    df_base.ta.supertrend(length=10, multiplier=3.0, append=True)
    df_base.ta.supertrend(length=14, multiplier=2.0, append=True)
    df_base.ta.supertrend(length=14, multiplier=3.0, append=True)

    print(f"   ✓ Indicadores calculados (EMA, SMA, RSI, MACD, BB, ATR, Donchian, ADX, Supertrend)")

    print("\n" + "="*80)
    print("FASE 2: GENERACIÓN Y PRUEBA DE ESTRATEGIAS")
    print("="*80)

    # Generar configuraciones
    print(f"\n4. Generando configuraciones de estrategias...")
    configs = generate_strategy_configs()
    print(f"   ✓ {len(configs):,} configuraciones generadas")

    # Probar todas las estrategias
    print(f"\n5. Ejecutando backtests...")
    results = []
    start_time = datetime.now()

    for i, config in enumerate(configs, 1):
        result = test_strategy(df_base, config, strategy_id=i)
        if result is not None:
            results.append(result)

        # Mostrar progreso cada 10 estrategias
        if i % 10 == 0:
            print_progress(i, len(configs), start_time)

    # Limpiar línea de progreso
    print(" " * 100, end='\r')

    elapsed_time = (datetime.now() - start_time).total_seconds()
    print(f"\n   ✓ {len(results):,} estrategias completadas en {elapsed_time/60:.1f} minutos")

    print("\n" + "="*80)
    print("FASE 3: FILTRADO Y ANÁLISIS DE RESULTADOS")
    print("="*80)

    # Guardar TODOS los resultados
    print(f"\n6. Guardando resultados completos...")
    all_results_df = pd.DataFrame(results)
    all_results_df.to_csv('results/exhaustive_search_all.csv', index=False)
    print(f"   ✓ Guardado: results/exhaustive_search_all.csv ({len(all_results_df)} estrategias)")

    # Filtrar por criterios
    print(f"\n7. Filtrando estrategias por criterios de éxito...")
    best_strategies = filter_results(results, CRITERIA)
    print(f"   ✓ {len(best_strategies)} estrategias cumplen TODOS los criterios")

    if len(best_strategies) > 0:
        # Guardar mejores estrategias
        best_strategies.to_csv('results/exhaustive_search_best.csv', index=False)
        print(f"   ✓ Guardado: results/exhaustive_search_best.csv")

        print("\n" + "="*80)
        print("🏆 TOP 10 MEJORES ESTRATEGIAS")
        print("="*80)

        for idx, row in best_strategies.head(10).iterrows():
            print(f"\n#{int(row['id'])} - PF: {row['profit_factor']:.2f} | Return: {row['total_return_pct']:.1f}% | DD: {row['max_drawdown_pct']:.1f}%")
            print(f"   Config: {row['regime_type'].upper()}({row['regime_period']}) + {', '.join(row['entry_indicators'])}")
            print(f"   SL:TP = {row['sl_atr_multiplier']}:{row['tp_atr_multiplier']} | Trades: {int(row['num_trades'])} | WR: {row['win_rate_pct']:.1f}%")

    else:
        print("\n⚠️  NINGUNA ESTRATEGIA CUMPLE TODOS LOS CRITERIOS")
        print("\nProbando con criterios relajados...")

        # Criterios relajados
        relaxed_criteria = {
            'min_profit_factor': 1.5,
            'min_return_pct': 50.0,
            'max_drawdown_pct': 15.0,
            'min_num_trades': 150,
            'min_win_rate_pct': 30.0,
            'min_sharpe_ratio': 0.3,
        }

        relaxed_best = filter_results(results, relaxed_criteria)
        print(f"   ✓ {len(relaxed_best)} estrategias cumplen criterios relajados")

        if len(relaxed_best) > 0:
            relaxed_best.to_csv('results/exhaustive_search_relaxed.csv', index=False)
            print(f"   ✓ Guardado: results/exhaustive_search_relaxed.csv")

            print("\n" + "="*80)
            print("🥈 TOP 10 MEJORES ESTRATEGIAS (Criterios Relajados)")
            print("="*80)

            for idx, row in relaxed_best.head(10).iterrows():
                print(f"\n#{int(row['id'])} - PF: {row['profit_factor']:.2f} | Return: {row['total_return_pct']:.1f}% | DD: {row['max_drawdown_pct']:.1f}%")
                print(f"   Config: {row['regime_type'].upper()}({row['regime_period']}) + {', '.join(row['entry_indicators'])}")
                print(f"   SL:TP = {row['sl_atr_multiplier']}:{row['tp_atr_multiplier']} | Trades: {int(row['num_trades'])} | WR: {row['win_rate_pct']:.1f}%")

    # Estadísticas generales
    print("\n" + "="*80)
    print("📈 ESTADÍSTICAS GENERALES")
    print("="*80)

    stats_df = pd.DataFrame(results)
    print(f"\nProfit Factor:")
    print(f"  • Máximo: {stats_df['profit_factor'].max():.2f}")
    print(f"  • Promedio: {stats_df['profit_factor'].mean():.2f}")
    print(f"  • Con PF >= 1.5: {(stats_df['profit_factor'] >= 1.5).sum()} ({(stats_df['profit_factor'] >= 1.5).sum() / len(stats_df) * 100:.1f}%)")
    print(f"  • Con PF >= 2.0: {(stats_df['profit_factor'] >= 2.0).sum()} ({(stats_df['profit_factor'] >= 2.0).sum() / len(stats_df) * 100:.1f}%)")

    print(f"\nRetorno Total:")
    print(f"  • Máximo: {stats_df['total_return_pct'].max():.1f}%")
    print(f"  • Promedio: {stats_df['total_return_pct'].mean():.1f}%")
    print(f"  • Con Return >= 100%: {(stats_df['total_return_pct'] >= 100).sum()} ({(stats_df['total_return_pct'] >= 100).sum() / len(stats_df) * 100:.1f}%)")

    print(f"\nMax Drawdown:")
    print(f"  • Mejor (menor): {stats_df['max_drawdown_pct'].min():.1f}%")
    print(f"  • Promedio: {stats_df['max_drawdown_pct'].mean():.1f}%")
    print(f"  • Con DD <= 12%: {(stats_df['max_drawdown_pct'] <= 12).sum()} ({(stats_df['max_drawdown_pct'] <= 12).sum() / len(stats_df) * 100:.1f}%)")

    print("\n" + "="*80)
    print("✅ BÚSQUEDA EXHAUSTIVA COMPLETADA")
    print("="*80)
    print(f"\nArchivos generados:")
    print(f"  • results/exhaustive_search_all.csv - Todas las estrategias")
    if len(best_strategies) > 0:
        print(f"  • results/exhaustive_search_best.csv - Mejores estrategias")
    if len(relaxed_best) > 0:
        print(f"  • results/exhaustive_search_relaxed.csv - Criterios relajados")

    print(f"\nTiempo total: {elapsed_time/60:.1f} minutos")
    print()


if __name__ == "__main__":
    main()
