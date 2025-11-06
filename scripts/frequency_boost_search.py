"""
BÚSQUEDA DIRIGIDA: Supertrend + RSI con frecuencia moderada

Objetivo: Encontrar configuraciones que logren:
- 30-80 trades (frecuencia moderada)
- PF >= 1.5 (calidad aceptable)
- Retorno >= 20%
- Max DD <= 20%

Estrategia: Probar parámetros MÁS PERMISIVOS no probados en búsqueda original
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from itertools import product

sys.path.append('/home/user/DayTrading')

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.advanced_signals import generate_signals_multi_indicator
from src.backtest.engine import VectorizedBacktester

# Configuración
SYMBOL = 'ETHUSDT'
INTERVAL = '15m'
START_DATE = '365 days ago UTC'
INITIAL_CAPITAL = 10000

# Criterios objetivo
TARGET_CRITERIA = {
    'min_trades': 30,
    'max_trades': 80,
    'min_profit_factor': 1.5,
    'min_return_pct': 20.0,
    'max_drawdown_pct': 20.0,
}

# Grid ESPECÍFICO para aumentar frecuencia manteniendo calidad
FREQUENCY_BOOST_GRID = {
    # Sin filtros de régimen restrictivos
    'regime_type': ['none'],
    'regime_period': [100],
    'regime_direction': ['long_only', 'hybrid'],

    # Sin ADX (es muy restrictivo)
    'adx_period': [14],
    'adx_threshold': [25],

    # Indicadores: Solo Supertrend + RSI
    'entry_indicators': [['supertrend', 'rsi']],

    # SUPERTREND más sensible (genera más señales)
    'supertrend_length': [7, 10, 14],  # Períodos más cortos
    'supertrend_multiplier': [1.5, 2.0, 2.5],  # Multiplicadores más bajos = más señales

    # RSI más permisivo (genera más señales)
    'rsi_period': [14],
    'rsi_oversold': [25, 30, 35],  # Más permisivo (antes solo 30)
    'rsi_overbought': [65, 70, 75],  # Más permisivo (antes solo 70)

    # SIN filtros adicionales (son restrictivos)
    'use_volume_filter': [False],
    'volume_ma_period': [20],
    'use_atr_filter': [False],
    'atr_min_threshold': [0.5],

    # Gestión de riesgo: Mantener ratios que funcionaron
    'atr_period': [14],
    'atr_multiplier_combinations': [
        {'sl': 2.0, 'tp': 4.0},   # Ratio 1:2
        {'sl': 2.5, 'tp': 5.0},   # Ratio 1:2 (el mejor)
        {'sl': 2.0, 'tp': 6.0},   # Ratio 1:3
    ],
}


def generate_configs():
    """Genera configuraciones de estrategias."""
    configs = []

    keys = [k for k in FREQUENCY_BOOST_GRID.keys() if k != 'entry_indicators' and k != 'atr_multiplier_combinations']
    values = [FREQUENCY_BOOST_GRID[k] for k in keys]

    for combination in product(*values):
        config = dict(zip(keys, combination))

        # Agregar entry_indicators
        config['entry_indicators'] = FREQUENCY_BOOST_GRID['entry_indicators'][0]

        # Probar cada ratio SL:TP
        for atr_mult in FREQUENCY_BOOST_GRID['atr_multiplier_combinations']:
            test_config = config.copy()
            test_config['sl_atr_multiplier'] = atr_mult['sl']
            test_config['tp_atr_multiplier'] = atr_mult['tp']
            configs.append(test_config)

    return configs


def test_strategy(df_base, config, strategy_id):
    """Prueba una configuración de estrategia."""
    try:
        df = generate_signals_multi_indicator(df_base.copy(), config)

        if df['señal'].sum() == 0:
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

        result = {
            'id': strategy_id,
            **config,
            **metrics
        }

        return result

    except Exception as e:
        print(f"  ✗ Error en estrategia {strategy_id}: {str(e)}")
        return None


def main():
    print("="*80)
    print("🎯 BÚSQUEDA DIRIGIDA: Frecuencia Moderada con Calidad")
    print("="*80)
    print(f"\nObjetivo:")
    print(f"  • Trades: {TARGET_CRITERIA['min_trades']}-{TARGET_CRITERIA['max_trades']}")
    print(f"  • Profit Factor >= {TARGET_CRITERIA['min_profit_factor']}")
    print(f"  • Retorno >= {TARGET_CRITERIA['min_return_pct']}%")
    print(f"  • Max Drawdown <= {TARGET_CRITERIA['max_drawdown_pct']}%")

    # Generar configuraciones
    configs = generate_configs()
    print(f"\n📊 Configuraciones a probar: {len(configs)}")
    print(f"⏱️  Tiempo estimado: {len(configs)/10/60:.1f} minutos\n")

    response = input("¿Deseas continuar? (y/n): ")
    if response.lower() != 'y':
        print("❌ Búsqueda cancelada.")
        return

    print("\n" + "="*80)
    print("FASE 1: DESCARGA DE DATOS")
    print("="*80)

    # Conectar y descargar
    manager = BinanceClientManager()
    client = manager.get_public_client()

    print(f"Descargando {SYMBOL} {INTERVAL}...")
    df_base = obtener_datos_binance(
        client=client,
        simbolo=SYMBOL,
        intervalo=INTERVAL,
        inicio=START_DATE
    )

    print(f"Datos descargados: {len(df_base)} registros")

    # Calcular indicadores
    print("\nCalculando indicadores...")
    df_base = agregar_indicadores(df_base)  # Usar parámetros por defecto

    # Calcular indicadores adicionales manualmente
    df_base.ta.atr(length=20, append=True)
    df_base.ta.adx(length=14, append=True)

    # Supertrend para todos los parámetros
    for length in [7, 10, 14]:
        for mult in [1.5, 2.0, 2.5]:
            df_base.ta.supertrend(length=length, multiplier=mult, append=True)

    print(f"Indicadores calculados. Columnas: {len(df_base.columns)}")

    # FASE 2: Probar estrategias
    print("\n" + "="*80)
    print("FASE 2: PROBANDO ESTRATEGIAS")
    print("="*80)

    results = []
    start_time = datetime.now()

    for idx, config in enumerate(configs, 1):
        result = test_strategy(df_base, config, idx)

        if result:
            results.append(result)

            # Mostrar si cumple criterios
            if (result['num_trades'] >= TARGET_CRITERIA['min_trades'] and
                result['num_trades'] <= TARGET_CRITERIA['max_trades'] and
                result['profit_factor'] >= TARGET_CRITERIA['min_profit_factor'] and
                result['total_return_pct'] >= TARGET_CRITERIA['min_return_pct'] and
                result['max_drawdown_pct'] <= TARGET_CRITERIA['max_drawdown_pct']):

                print(f"\n🎯 ¡CANDIDATO ENCONTRADO! ID {idx}")
                print(f"   PF: {result['profit_factor']:.2f} | Ret: {result['total_return_pct']:.2f}% | DD: {result['max_drawdown_pct']:.2f}%")
                print(f"   Trades: {result['num_trades']} | WR: {result['win_rate_pct']:.1f}%")

        # Progreso
        if idx % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(configs) - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{len(configs)}] {idx/len(configs)*100:.1f}% | {rate:.1f} tests/sec | ETA: {eta/60:.1f} min", end='\r')

    print("\n\n" + "="*80)
    print("FASE 3: ANÁLISIS DE RESULTADOS")
    print("="*80)

    if not results:
        print("\n❌ No se encontraron estrategias válidas")
        return

    df_results = pd.DataFrame(results)

    # Guardar todos los resultados
    df_results.to_csv('results/frequency_boost_all.csv', index=False)
    print(f"\n💾 Guardados {len(df_results)} resultados en: results/frequency_boost_all.csv")

    # Filtrar por criterios
    best = df_results[
        (df_results['num_trades'] >= TARGET_CRITERIA['min_trades']) &
        (df_results['num_trades'] <= TARGET_CRITERIA['max_trades']) &
        (df_results['profit_factor'] >= TARGET_CRITERIA['min_profit_factor']) &
        (df_results['total_return_pct'] >= TARGET_CRITERIA['min_return_pct']) &
        (df_results['max_drawdown_pct'] <= TARGET_CRITERIA['max_drawdown_pct'])
    ].sort_values('profit_factor', ascending=False)

    if len(best) > 0:
        print(f"\n🏆 ¡ESTRATEGIAS QUE CUMPLEN CRITERIOS: {len(best)}!")
        print("="*80)

        best.to_csv('results/frequency_boost_best.csv', index=False)
        print(f"💾 Guardadas en: results/frequency_boost_best.csv\n")

        # Mostrar top 5
        for idx, (i, row) in enumerate(best.head(5).iterrows(), 1):
            print(f"\n{'─'*80}")
            print(f"#{idx} - Estrategia ID {int(row['id'])}")
            print(f"{'─'*80}")
            print(f"  📈 Profit Factor: {row['profit_factor']:.2f}")
            print(f"  💰 Retorno: {row['total_return_pct']:.2f}%")
            print(f"  📉 Max Drawdown: {row['max_drawdown_pct']:.2f}%")
            print(f"  📊 Trades: {int(row['num_trades'])} | Win Rate: {row['win_rate_pct']:.1f}%")
            print(f"  📐 Sharpe: {row['sharpe_ratio']:.2f}")
            print(f"\n  🔧 Configuración:")
            print(f"     - Supertrend: length={int(row['supertrend_length'])}, multiplier={row['supertrend_multiplier']}")
            print(f"     - RSI: oversold={int(row['rsi_oversold'])}, overbought={int(row['rsi_overbought'])}")
            print(f"     - Dirección: {row['regime_direction']}")
            print(f"     - SL:TP: {row['sl_atr_multiplier']}:{row['tp_atr_multiplier']} (ratio 1:{row['tp_atr_multiplier']/row['sl_atr_multiplier']:.1f})")
    else:
        print(f"\n⚠️  No se encontraron estrategias que cumplan TODOS los criterios")
        print("\n📊 Análisis de las mejores estrategias encontradas:")

        # Mostrar las mejores por PF
        top_pf = df_results[df_results['num_trades'] > 0].sort_values('profit_factor', ascending=False).head(5)

        for idx, (i, row) in enumerate(top_pf.iterrows(), 1):
            print(f"\n#{idx} - ID {int(row['id'])}")
            print(f"   PF: {row['profit_factor']:.2f} | Ret: {row['total_return_pct']:.2f}% | DD: {row['max_drawdown_pct']:.2f}%")
            print(f"   Trades: {int(row['num_trades'])} | WR: {row['win_rate_pct']:.1f}%")
            print(f"   Supertrend: L={int(row['supertrend_length'])}, M={row['supertrend_multiplier']}")
            print(f"   RSI: OS={int(row['rsi_oversold'])}, OB={int(row['rsi_overbought'])}")


if __name__ == "__main__":
    main()
