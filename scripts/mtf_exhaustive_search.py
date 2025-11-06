"""
BÚSQUEDA EXHAUSTIVA MULTI-TIMEFRAME (MTF)

Objetivo: Encontrar estrategias con:
- 200+ trades/año (alta frecuencia)
- DD <12% (bajo riesgo)
- PF >= 1.8 (buena calidad)
- Retorno >= 70%

Estrategia MTF:
- Timeframe Superior: Filtra tendencia (1h/4h)
- Timeframe Operación: Genera entradas (5m/15m)
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
from src.strategy.advanced_signals import generate_signals_multi_timeframe
from src.backtest.engine import VectorizedBacktester

# Configuración
SYMBOL = 'ETHUSDT'
START_DATE = '365 days ago UTC'
INITIAL_CAPITAL = 10000

# Criterios objetivo
TARGET_CRITERIA = {
    'min_trades': 200,
    'max_drawdown_pct': 12.0,
    'min_profit_factor': 1.8,
    'min_return_pct': 70.0,
}

# Grid de búsqueda MTF
MTF_SEARCH_GRID = {
    # ========================================
    # COMBINACIONES DE TIMEFRAMES
    # ========================================
    'tf_combinations': [
        #{'higher': '4h', 'trade': '15m'},   # Muy conservador
        {'higher': '1h', 'trade': '15m'},   # Conservador
        {'higher': '1h', 'trade': '5m'},    # ⭐ RECOMENDADO
        #{'higher': '15m', 'trade': '5m'},   # Agresivo
    ],

    # ========================================
    # FILTRO TIMEFRAME SUPERIOR
    # ========================================
    'htf_ema_fast': [50],
    'htf_ema_slow': [100, 200],
    'htf_adx_threshold': [20, 25, 30],
    'htf_use_rsi_filter': [False],  # Simplificar inicialmente

    # ========================================
    # SEÑALES TIMEFRAME OPERACIÓN
    # ========================================
    'entry_indicators': [
        ['supertrend', 'rsi'],
    ],

    # Supertrend
    'supertrend_length': [7, 10],
    'supertrend_multiplier': [1.5, 2.0],

    # RSI
    'rsi_period': [14],
    'rsi_oversold': [30, 35],
    'rsi_overbought': [65, 70],

    # ========================================
    # GESTIÓN DE RIESGO
    # ========================================
    'atr_period': [14],
    'sl_atr_multiplier': [2.0, 2.5],
    'tp_atr_multiplier': [4.0, 5.0],

    # ========================================
    # FILTROS ADICIONALES (simplificado)
    # ========================================
    'use_volume_filter': [False],
    'use_atr_filter': [False],
}


def generate_mtf_configs():
    """Genera todas las configuraciones MTF a probar."""
    configs = []

    # Para cada combinación de timeframes
    for tf_comb in MTF_SEARCH_GRID['tf_combinations']:
        higher_tf = tf_comb['higher']
        trade_tf = tf_comb['trade']

        # Parámetros a combinar
        params = {
            'htf_ema_fast': MTF_SEARCH_GRID['htf_ema_fast'],
            'htf_ema_slow': MTF_SEARCH_GRID['htf_ema_slow'],
            'htf_adx_threshold': MTF_SEARCH_GRID['htf_adx_threshold'],
            'htf_use_rsi_filter': MTF_SEARCH_GRID['htf_use_rsi_filter'],
            'supertrend_length': MTF_SEARCH_GRID['supertrend_length'],
            'supertrend_multiplier': MTF_SEARCH_GRID['supertrend_multiplier'],
            'rsi_oversold': MTF_SEARCH_GRID['rsi_oversold'],
            'rsi_overbought': MTF_SEARCH_GRID['rsi_overbought'],
            'sl_atr_multiplier': MTF_SEARCH_GRID['sl_atr_multiplier'],
            'tp_atr_multiplier': MTF_SEARCH_GRID['tp_atr_multiplier'],
        }

        keys = list(params.keys())
        values = [params[k] for k in keys]

        # Generar producto cartesiano
        for combination in product(*values):
            config = dict(zip(keys, combination))

            # Agregar metadata
            config['higher_tf'] = higher_tf
            config['trade_tf'] = trade_tf
            config['entry_indicators'] = MTF_SEARCH_GRID['entry_indicators'][0]
            config['rsi_period'] = 14
            config['atr_period'] = 14
            config['use_volume_filter'] = False
            config['use_atr_filter'] = False
            config['htf_adx_period'] = 14

            configs.append(config)

    return configs


def test_mtf_strategy(df_higher, df_trade, config, strategy_id):
    """Prueba una configuración MTF."""
    try:
        # Generar señales MTF
        df_signals = generate_signals_multi_timeframe(df_trade, df_higher, config)

        if df_signals['señal'].sum() == 0:
            return None

        # Ejecutar backtest
        backtester = VectorizedBacktester(
            df=df_signals,
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
    print("🎯 BÚSQUEDA EXHAUSTIVA MULTI-TIMEFRAME (MTF)")
    print("="*80)
    print(f"\nObjetivo:")
    print(f"  • Trades: >= {TARGET_CRITERIA['min_trades']}")
    print(f"  • Max DD: <= {TARGET_CRITERIA['max_drawdown_pct']}%")
    print(f"  • Profit Factor: >= {TARGET_CRITERIA['min_profit_factor']}")
    print(f"  • Retorno: >= {TARGET_CRITERIA['min_return_pct']}%")

    # Generar configuraciones
    configs = generate_mtf_configs()
    print(f"\n📊 Configuraciones MTF a probar: {len(configs)}")

    # Mostrar combinaciones de TF
    tf_combs = set([(c['higher_tf'], c['trade_tf']) for c in configs])
    print(f"\n📈 Combinaciones de Timeframes:")
    for htf, ttf in tf_combs:
        count = len([c for c in configs if c['higher_tf'] == htf and c['trade_tf'] == ttf])
        print(f"   • {htf} → {ttf}: {count} configuraciones")

    print(f"\n⏱️  Tiempo estimado: {len(configs)/5/60:.1f} minutos (asumiendo 5 tests/sec)\n")

    response = input("¿Deseas continuar? (y/n): ")
    if response.lower() != 'y':
        print("❌ Búsqueda cancelada.")
        return

    print("\n" + "="*80)
    print("FASE 1: DESCARGA DE DATOS")
    print("="*80)

    manager = BinanceClientManager()
    client = manager.get_public_client()

    # Descargar datos para todos los timeframes necesarios
    timeframes_needed = set()
    for config in configs:
        timeframes_needed.add(config['higher_tf'])
        timeframes_needed.add(config['trade_tf'])

    print(f"\nTimeframes a descargar: {sorted(timeframes_needed)}")

    data_cache = {}
    for tf in sorted(timeframes_needed):
        print(f"\nDescargando {SYMBOL} {tf}...")
        df = obtener_datos_binance(
            client=client,
            simbolo=SYMBOL,
            intervalo=tf,
            inicio=START_DATE
        )
        print(f"  Descargados: {len(df)} registros")

        # Calcular indicadores
        print(f"  Calculando indicadores para {tf}...")
        df = agregar_indicadores(df)

        # Indicadores adicionales
        df.ta.atr(length=20, append=True)
        df.ta.adx(length=14, append=True)

        # Supertrend
        for length in [7, 10]:
            for mult in [1.5, 2.0]:
                df.ta.supertrend(length=length, multiplier=mult, append=True)

        # EMAs para higher TF
        for period in [50, 100, 200]:
            df.ta.ema(length=period, append=True)

        data_cache[tf] = df
        print(f"  ✓ Listo: {len(df.columns)} columnas")

    # FASE 2: Probar estrategias
    print("\n" + "="*80)
    print("FASE 2: PROBANDO ESTRATEGIAS MTF")
    print("="*80)

    results = []
    start_time = datetime.now()
    candidates_found = 0

    for idx, config in enumerate(configs, 1):
        # Obtener datos de ambos timeframes
        df_higher = data_cache[config['higher_tf']].copy()
        df_trade = data_cache[config['trade_tf']].copy()

        result = test_mtf_strategy(df_higher, df_trade, config, idx)

        if result:
            results.append(result)

            # Verificar si cumple criterios
            if (result['num_trades'] >= TARGET_CRITERIA['min_trades'] and
                result['max_drawdown_pct'] <= TARGET_CRITERIA['max_drawdown_pct'] and
                result['profit_factor'] >= TARGET_CRITERIA['min_profit_factor'] and
                result['total_return_pct'] >= TARGET_CRITERIA['min_return_pct']):

                candidates_found += 1
                print(f"\n🎯 CANDIDATO #{candidates_found} - ID {idx}")
                print(f"   {config['higher_tf']} → {config['trade_tf']}")
                print(f"   PF: {result['profit_factor']:.2f} | Ret: {result['total_return_pct']:.2f}% | DD: {result['max_drawdown_pct']:.2f}%")
                print(f"   Trades: {result['num_trades']} | WR: {result['win_rate_pct']:.1f}%")

        # Progreso
        if idx % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = idx / elapsed if elapsed > 0 else 0
            eta = (len(configs) - idx) / rate if rate > 0 else 0
            print(f"  [{idx}/{len(configs)}] {idx/len(configs)*100:.1f}% | {rate:.1f} tests/sec | ETA: {eta/60:.1f} min | Candidatos: {candidates_found}", end='\r')

    print("\n\n" + "="*80)
    print("FASE 3: ANÁLISIS DE RESULTADOS")
    print("="*80)

    if not results:
        print("\n❌ No se generaron resultados válidos")
        return

    df_results = pd.DataFrame(results)

    # Guardar todos
    df_results.to_csv('results/mtf_search_all.csv', index=False)
    print(f"\n💾 Guardados {len(df_results)} resultados en: results/mtf_search_all.csv")

    # Filtrar por criterios
    best = df_results[
        (df_results['num_trades'] >= TARGET_CRITERIA['min_trades']) &
        (df_results['max_drawdown_pct'] <= TARGET_CRITERIA['max_drawdown_pct']) &
        (df_results['profit_factor'] >= TARGET_CRITERIA['min_profit_factor']) &
        (df_results['total_return_pct'] >= TARGET_CRITERIA['min_return_pct'])
    ].sort_values('total_return_pct', ascending=False)

    if len(best) > 0:
        print(f"\n🏆 ¡ESTRATEGIAS MTF QUE CUMPLEN CRITERIOS: {len(best)}!")
        print("="*80)

        best.to_csv('results/mtf_search_best.csv', index=False)
        print(f"💾 Guardadas en: results/mtf_search_best.csv\n")

        # Mostrar top 5
        for idx, (i, row) in enumerate(best.head(5).iterrows(), 1):
            print(f"\n{'─'*80}")
            print(f"#{idx} - Estrategia ID {int(row['id'])} | {row['higher_tf']} → {row['trade_tf']}")
            print(f"{'─'*80}")
            print(f"  📈 MÉTRICAS:")
            print(f"     • Profit Factor: {row['profit_factor']:.2f}")
            print(f"     • Retorno: {row['total_return_pct']:.2f}%")
            print(f"     • Max DD: {row['max_drawdown_pct']:.2f}%")
            print(f"     • Trades: {int(row['num_trades'])} | WR: {row['win_rate_pct']:.1f}%")
            print(f"     • Sharpe: {row['sharpe_ratio']:.2f}")

            print(f"\n  🔧 CONFIG TIMEFRAME SUPERIOR ({row['higher_tf']}):")
            print(f"     • EMA: {int(row['htf_ema_fast'])} vs {int(row['htf_ema_slow'])}")
            print(f"     • ADX threshold: {int(row['htf_adx_threshold'])}")

            print(f"\n  🔧 CONFIG TIMEFRAME OPERACIÓN ({row['trade_tf']}):")
            print(f"     • Supertrend: ({int(row['supertrend_length'])}, {row['supertrend_multiplier']})")
            print(f"     • RSI: {int(row['rsi_oversold'])}/{int(row['rsi_overbought'])}")
            print(f"     • SL:TP: {row['sl_atr_multiplier']}:{row['tp_atr_multiplier']}")

    else:
        print(f"\n⚠️  No se encontraron estrategias que cumplan TODOS los criterios")
        print("\n📊 Mejores estrategias por cada métrica:")

        valid = df_results[df_results['num_trades'] > 0]

        if len(valid) > 0:
            # Mejor por PF
            best_pf = valid.loc[valid['profit_factor'].idxmax()]
            print(f"\n🏆 Mejor Profit Factor: {best_pf['profit_factor']:.2f}")
            print(f"   {best_pf['higher_tf']} → {best_pf['trade_tf']}")
            print(f"   Ret: {best_pf['total_return_pct']:.2f}% | DD: {best_pf['max_drawdown_pct']:.2f}% | Trades: {int(best_pf['num_trades'])}")

            # Mejor por retorno
            best_ret = valid.loc[valid['total_return_pct'].idxmax()]
            print(f"\n💰 Mejor Retorno: {best_ret['total_return_pct']:.2f}%")
            print(f"   {best_ret['higher_tf']} → {best_ret['trade_tf']}")
            print(f"   PF: {best_ret['profit_factor']:.2f} | DD: {best_ret['max_drawdown_pct']:.2f}% | Trades: {int(best_ret['num_trades'])}")

            # Mejor por trades
            best_trades = valid.loc[valid['num_trades'].idxmax()]
            print(f"\n📊 Más Trades: {int(best_trades['num_trades'])}")
            print(f"   {best_trades['higher_tf']} → {best_trades['trade_tf']}")
            print(f"   PF: {best_trades['profit_factor']:.2f} | Ret: {best_trades['total_return_pct']:.2f}% | DD: {best_trades['max_drawdown_pct']:.2f}%")

            # Mejor DD
            best_dd = valid.loc[valid['max_drawdown_pct'].idxmin()]
            print(f"\n📉 Mejor Drawdown: {best_dd['max_drawdown_pct']:.2f}%")
            print(f"   {best_dd['higher_tf']} → {best_dd['trade_tf']}")
            print(f"   PF: {best_dd['profit_factor']:.2f} | Ret: {best_dd['total_return_pct']:.2f}% | Trades: {int(best_dd['num_trades'])}")

    print("\n" + "="*80)
    print("✅ BÚSQUEDA MTF COMPLETADA")
    print("="*80)


if __name__ == "__main__":
    main()
