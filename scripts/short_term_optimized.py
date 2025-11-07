"""
BÚSQUEDA OPTIMIZADA PARA TRADING DE CORTO PLAZO - VERSIÓN REDUCIDA

Objetivos:
- Win Rate: 60-80%
- Risk/Reward: 1:1 a 1:5
- Profit Factor: > 1.5
- Max Drawdown: < 15%
- Número de Trades: > 200
- Retorno: Consistente y Positivo

OPTIMIZACIÓN: Grid reducido pero inteligente (~5000 combinaciones = ~10 minutos)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

sys.path.append('/Users/jhonathan/BotDayTrading')

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.advanced_signals import generate_signals_multi_indicator
from src.backtest.engine import VectorizedBacktester

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SYMBOL = 'ETHUSDT'
INTERVAL = '15m'  # Solo 15m (mejor balance frecuencia/ruido)
START_DATE = '365 days ago UTC'
INITIAL_CAPITAL = 10000

# Criterios objetivo
TARGET_CRITERIA = {
    'min_win_rate': 60.0,
    'max_win_rate': 80.0,
    'min_profit_factor': 1.5,
    'max_drawdown_pct': 15.0,
    'min_trades': 200,
    'min_return_pct': 10.0,
}

# ============================================================================
# GRID OPTIMIZADO (solo las mejores combinaciones)
# ============================================================================

OPTIMIZED_GRID = {
    'regime_type': ['none'],
    'regime_period': [100],
    'regime_direction': ['hybrid'],  # Long y Short

    # ADX opcional
    'adx_period': [14],
    'adx_threshold': [20, 25],

    # SOLO las 3 mejores combinaciones de indicadores
    'entry_indicators': [
        ['supertrend', 'rsi'],      # Ya probado, funciona
        ['ema_cross', 'rsi'],       # Alta frecuencia
        ['supertrend', 'rsi', 'macd'],  # Triple confirmación
    ],

    # Supertrend optimizado
    'supertrend_length': [7, 10],
    'supertrend_multiplier': [1.5, 2.0],

    # RSI optimizado
    'rsi_period': [14],
    'rsi_oversold': [25, 30],
    'rsi_overbought': [70, 75],

    # EMA Cross
    'ema_fast': [9, 12],
    'ema_slow': [21, 26],

    # MACD estándar
    'macd_fast': [12],
    'macd_slow': [26],
    'macd_signal': [9],

    # Filtros
    'use_volume_filter': [True, False],
    'volume_ma_period': [20],
    'use_atr_filter': [False],
    'atr_min_threshold': [0.3],

    # Gestión de riesgo - SOLO los mejores ratios
    'atr_period': [14],
    'atr_multiplier_combinations': [
        {'sl': 1.0, 'tp': 1.0},   # 1:1
        {'sl': 1.0, 'tp': 1.5},   # 1:1.5
        {'sl': 1.0, 'tp': 2.0},   # 1:2
        {'sl': 1.5, 'tp': 3.0},   # 1:2
        {'sl': 1.0, 'tp': 3.0},   # 1:3
        {'sl': 1.0, 'tp': 4.0},   # 1:4
        {'sl': 1.0, 'tp': 5.0},   # 1:5
    ],
}


def generate_configs():
    """Genera configuraciones del grid."""
    configs = []
    special_keys = ['entry_indicators', 'atr_multiplier_combinations']
    regular_keys = [k for k in OPTIMIZED_GRID.keys() if k not in special_keys]
    regular_values = [OPTIMIZED_GRID[k] for k in regular_keys]

    for combination in product(*regular_values):
        base_config = dict(zip(regular_keys, combination))

        for entry_indicators in OPTIMIZED_GRID['entry_indicators']:
            config = base_config.copy()
            config['entry_indicators'] = entry_indicators

            for atr_mult in OPTIMIZED_GRID['atr_multiplier_combinations']:
                test_config = config.copy()
                test_config['sl_atr_multiplier'] = atr_mult['sl']
                test_config['tp_atr_multiplier'] = atr_mult['tp']
                configs.append(test_config)

    return configs


def test_strategy(df_base, config, strategy_id):
    """Prueba una configuración de estrategia."""
    try:
        df = generate_signals_multi_indicator(df_base.copy(), config)

        if df['señal'].abs().sum() == 0:
            return None

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

        metrics = backtester.calculate_metrics()
        rr_ratio = config['tp_atr_multiplier'] / config['sl_atr_multiplier']

        result = {
            'id': strategy_id,
            'rr_ratio': rr_ratio,
            **config,
            **metrics
        }

        return result

    except Exception as e:
        return None


def meets_criteria(result):
    """Verifica si cumple criterios."""
    return (
        result['win_rate_pct'] >= TARGET_CRITERIA['min_win_rate'] and
        result['win_rate_pct'] <= TARGET_CRITERIA['max_win_rate'] and
        result['profit_factor'] >= TARGET_CRITERIA['min_profit_factor'] and
        result['max_drawdown_pct'] <= TARGET_CRITERIA['max_drawdown_pct'] and
        result['num_trades'] >= TARGET_CRITERIA['min_trades'] and
        result['total_return_pct'] >= TARGET_CRITERIA['min_return_pct']
    )


def main():
    print("="*80)
    print("🎯 BÚSQUEDA OPTIMIZADA: TRADING DE CORTO PLAZO (VERSIÓN REDUCIDA)")
    print("="*80)
    print(f"\n📊 Objetivos:")
    print(f"  • Win Rate: {TARGET_CRITERIA['min_win_rate']}-{TARGET_CRITERIA['max_win_rate']}%")
    print(f"  • Profit Factor: >= {TARGET_CRITERIA['min_profit_factor']}")
    print(f"  • Max Drawdown: <= {TARGET_CRITERIA['max_drawdown_pct']}%")
    print(f"  • Número de Trades: >= {TARGET_CRITERIA['min_trades']}")
    print(f"  • Retorno: >= {TARGET_CRITERIA['min_return_pct']}%")

    configs = generate_configs()
    print(f"\n📊 Configuraciones a probar: {len(configs)}")
    print(f"⏱️  Tiempo estimado: {len(configs)/10/60:.1f} minutos")

    response = input("\n¿Continuar? (y/n): ")
    if response.lower() != 'y':
        print("❌ Cancelado.")
        return

    print("\n" + "="*80)
    print("FASE 1: DESCARGA DE DATOS")
    print("="*80)

    manager = BinanceClientManager()
    client = manager.get_public_client()

    print(f"Descargando {SYMBOL} {INTERVAL}...")
    df_base = obtener_datos_binance(
        client=client,
        simbolo=SYMBOL,
        intervalo=INTERVAL,
        inicio=START_DATE
    )

    print(f"✅ Datos: {len(df_base)} registros")
    print("\nCalculando indicadores...")

    df_base = agregar_indicadores(df_base)
    df_base.ta.adx(length=14, append=True)
    df_base.ta.atr(length=14, append=True)

    # Supertrend
    for length in [7, 10]:
        for mult in [1.5, 2.0]:
            df_base.ta.supertrend(length=length, multiplier=mult, append=True)

    # EMAs
    for period in [9, 12, 21, 26]:
        if f"EMA_{period}" not in df_base.columns:
            df_base.ta.ema(length=period, append=True)

    print(f"✅ Indicadores calculados: {len(df_base.columns)} columnas")

    print("\n" + "="*80)
    print(f"FASE 2: PROBANDO {len(configs)} ESTRATEGIAS")
    print("="*80)

    results = []
    start_time = datetime.now()
    candidates = 0

    for idx, config in enumerate(configs, 1):
        result = test_strategy(df_base, config, idx)

        if result:
            results.append(result)

            if meets_criteria(result):
                candidates += 1
                print(f"\n🎯 CANDIDATO #{candidates}! ID {idx}")
                print(f"   WR: {result['win_rate_pct']:.1f}% | PF: {result['profit_factor']:.2f} | Ret: {result['total_return_pct']:.1f}%")
                print(f"   Trades: {result['num_trades']} | DD: {result['max_drawdown_pct']:.1f}% | RR: 1:{result['rr_ratio']:.1f}")

        if idx % 50 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(configs) - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{len(configs)}] {idx/len(configs)*100:.1f}% | {rate:.1f} t/s | ETA: {eta/60:.1f}min | Candidatos: {candidates}", end='\r')

    print("\n\n" + "="*80)
    print("FASE 3: RESULTADOS")
    print("="*80)

    if not results:
        print("❌ Sin resultados")
        return

    df_results = pd.DataFrame(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_file = f'results/short_term_all_{timestamp}.csv'
    df_results.to_csv(all_file, index=False)
    print(f"\n💾 Todos los resultados: {all_file}")

    best = df_results[df_results.apply(meets_criteria, axis=1)].sort_values('profit_factor', ascending=False)

    if len(best) > 0:
        print(f"\n🏆 ESTRATEGIAS QUE CUMPLEN CRITERIOS: {len(best)}")
        print("="*80)

        best_file = f'results/short_term_best_{timestamp}.csv'
        best.to_csv(best_file, index=False)
        print(f"💾 Mejores estrategias: {best_file}\n")

        print("📊 TOP 10:")
        for idx, (i, row) in enumerate(best.head(10).iterrows(), 1):
            print(f"\n{'─'*80}")
            print(f"#{idx} - ID {int(row['id'])}")
            print(f"  📈 PF: {row['profit_factor']:.2f} | 💰 Ret: {row['total_return_pct']:.1f}% | 🎯 WR: {row['win_rate_pct']:.1f}%")
            print(f"  📉 DD: {row['max_drawdown_pct']:.1f}% | 📊 Trades: {int(row['num_trades'])} | ⚖️ RR: 1:{row['rr_ratio']:.1f}")
            print(f"  🔧 {row['entry_indicators']}")

            if 'supertrend' in row['entry_indicators']:
                print(f"     ST: L={int(row['supertrend_length'])}, M={row['supertrend_multiplier']}")
            if 'rsi' in row['entry_indicators']:
                print(f"     RSI: {int(row['rsi_oversold'])}/{int(row['rsi_overbought'])}")
            if 'ema_cross' in row['entry_indicators']:
                print(f"     EMA: {int(row['ema_fast'])}/{int(row['ema_slow'])}")

            print(f"     SL:TP = {row['sl_atr_multiplier']}:{row['tp_atr_multiplier']} ATR")

    else:
        print("\n⚠️ No se encontraron estrategias que cumplan TODOS los criterios")
        print("\n📊 TOP 5 por diferentes métricas:\n")

        print("🏆 Por Profit Factor:")
        for idx, (i, row) in enumerate(df_results.nlargest(5, 'profit_factor').iterrows(), 1):
            print(f"  {idx}. ID {int(row['id'])}: PF={row['profit_factor']:.2f}, WR={row['win_rate_pct']:.1f}%, Trades={int(row['num_trades'])}, DD={row['max_drawdown_pct']:.1f}%")

        print("\n🎯 Por Win Rate:")
        for idx, (i, row) in enumerate(df_results.nlargest(5, 'win_rate_pct').iterrows(), 1):
            print(f"  {idx}. ID {int(row['id'])}: WR={row['win_rate_pct']:.1f}%, PF={row['profit_factor']:.2f}, Trades={int(row['num_trades'])}, DD={row['max_drawdown_pct']:.1f}%")

        print("\n📊 Por Número de Trades:")
        for idx, (i, row) in enumerate(df_results.nlargest(5, 'num_trades').iterrows(), 1):
            print(f"  {idx}. ID {int(row['id'])}: Trades={int(row['num_trades'])}, WR={row['win_rate_pct']:.1f}%, PF={row['profit_factor']:.2f}, DD={row['max_drawdown_pct']:.1f}%")

        print("\n💰 Por Retorno:")
        for idx, (i, row) in enumerate(df_results.nlargest(5, 'total_return_pct').iterrows(), 1):
            print(f"  {idx}. ID {int(row['id'])}: Ret={row['total_return_pct']:.1f}%, WR={row['win_rate_pct']:.1f}%, PF={row['profit_factor']:.2f}, Trades={int(row['num_trades'])}")

    print("\n" + "="*80)
    print("✅ COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
