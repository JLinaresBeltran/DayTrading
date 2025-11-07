"""
BÚSQUEDA OPTIMIZADA PARA TRADING DE CORTO PLAZO RENTABLE

Objetivos Específicos:
- Win Rate: 60-80%
- Risk/Reward: 1:1 a 1:5
- Profit Factor: > 1.5
- Max Drawdown: < 15%
- Número de Trades: > 200
- Retorno: Consistente y Positivo

Estrategia:
1. Usar timeframes más cortos (5m, 15m) para mayor frecuencia
2. Combinar múltiples indicadores para mejorar win rate
3. Stop Loss más ajustado (1-1.5 ATR) para trades de corto plazo
4. Take Profit escalonado (1:1, 1:2, 1:3, 1:4, 1:5)
5. Filtros de volatilidad y momentum para calidad de señales
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
INTERVALS = ['5m', '15m']  # Probar ambos timeframes
START_DATE = '365 days ago UTC'
INITIAL_CAPITAL = 10000

# Criterios objetivo
TARGET_CRITERIA = {
    'min_win_rate': 60.0,      # 60% mínimo
    'max_win_rate': 80.0,      # 80% máximo
    'min_profit_factor': 1.5,  # > 1.5
    'max_drawdown_pct': 15.0,  # < 15%
    'min_trades': 200,         # > 200 trades
    'min_return_pct': 10.0,    # Al menos 10% retorno
    'min_rr_ratio': 1.0,       # Risk/Reward mínimo 1:1
    'max_rr_ratio': 5.0,       # Risk/Reward máximo 1:5
}

# ============================================================================
# GRID DE BÚSQUEDA OPTIMIZADO PARA CORTO PLAZO
# ============================================================================

SHORT_TERM_GRID = {
    # SIN filtros de régimen (queremos TODAS las oportunidades)
    'regime_type': ['none'],
    'regime_period': [100],
    'regime_direction': ['hybrid'],  # Long y Short para más oportunidades

    # ADX: Filtro de tendencia (pero más permisivo)
    'adx_period': [14],
    'adx_threshold': [20, 25],  # Más bajo = más señales

    # ========================================================================
    # COMBINACIONES DE INDICADORES (probar múltiples combinaciones)
    # ========================================================================
    'entry_indicators': [
        # Combinación 1: Supertrend + RSI (ya probado, funciona bien)
        ['supertrend', 'rsi'],

        # Combinación 2: EMA Cross + RSI (clásico, alta frecuencia)
        ['ema_cross', 'rsi'],

        # Combinación 3: MACD + RSI (confirmación doble)
        ['macd', 'rsi'],

        # Combinación 4: Supertrend + MACD (tendencia + momentum)
        ['supertrend', 'macd'],

        # Combinación 5: Triple confirmación (más restrictivo pero mejor win rate)
        ['supertrend', 'rsi', 'macd'],
    ],

    # ========================================================================
    # SUPERTREND (más sensible para corto plazo)
    # ========================================================================
    'supertrend_length': [5, 7, 10],       # Períodos más cortos = más señales
    'supertrend_multiplier': [1.0, 1.5, 2.0],  # Multiplicadores bajos = más sensible

    # ========================================================================
    # RSI (más permisivo para más entradas)
    # ========================================================================
    'rsi_period': [7, 14],  # RSI corto y estándar
    'rsi_oversold': [20, 25, 30],   # Más permisivo
    'rsi_overbought': [70, 75, 80], # Más permisivo

    # ========================================================================
    # EMA CROSS (para estrategias que lo usen)
    # ========================================================================
    'ema_fast': [5, 9, 12],
    'ema_slow': [21, 26, 50],

    # ========================================================================
    # MACD (parámetros estándar y rápidos)
    # ========================================================================
    'macd_fast': [8, 12],
    'macd_slow': [21, 26],
    'macd_signal': [9],

    # ========================================================================
    # FILTROS ADICIONALES (usar volumen para calidad de señales)
    # ========================================================================
    'use_volume_filter': [True, False],  # Probar con y sin
    'volume_ma_period': [20],

    # SIN filtro ATR (es muy restrictivo)
    'use_atr_filter': [False],
    'atr_min_threshold': [0.3],

    # ========================================================================
    # GESTIÓN DE RIESGO - STOP LOSS Y TAKE PROFIT
    # Para trading de corto plazo: SL más ajustado
    # ========================================================================
    'atr_period': [14],

    # Combinaciones SL:TP optimizadas para corto plazo
    'atr_multiplier_combinations': [
        # Risk/Reward 1:1 (más frecuencia, win rate alto)
        {'sl': 1.0, 'tp': 1.0},

        # Risk/Reward 1:1.5
        {'sl': 1.0, 'tp': 1.5},

        # Risk/Reward 1:2 (equilibrado)
        {'sl': 1.0, 'tp': 2.0},
        {'sl': 1.5, 'tp': 3.0},

        # Risk/Reward 1:3 (más ambicioso)
        {'sl': 1.0, 'tp': 3.0},
        {'sl': 1.5, 'tp': 4.5},

        # Risk/Reward 1:4
        {'sl': 1.0, 'tp': 4.0},

        # Risk/Reward 1:5 (máximo)
        {'sl': 1.0, 'tp': 5.0},
    ],
}


def generate_configs():
    """Genera todas las configuraciones posibles del grid."""
    configs = []

    # Obtener todas las claves excepto las que requieren manejo especial
    special_keys = ['entry_indicators', 'atr_multiplier_combinations']
    regular_keys = [k for k in SHORT_TERM_GRID.keys() if k not in special_keys]

    # Generar todas las combinaciones de parámetros regulares
    regular_values = [SHORT_TERM_GRID[k] for k in regular_keys]

    for combination in product(*regular_values):
        base_config = dict(zip(regular_keys, combination))

        # Para cada combinación de entry_indicators
        for entry_indicators in SHORT_TERM_GRID['entry_indicators']:
            config = base_config.copy()
            config['entry_indicators'] = entry_indicators

            # Para cada ratio SL:TP
            for atr_mult in SHORT_TERM_GRID['atr_multiplier_combinations']:
                test_config = config.copy()
                test_config['sl_atr_multiplier'] = atr_mult['sl']
                test_config['tp_atr_multiplier'] = atr_mult['tp']
                configs.append(test_config)

    return configs


def test_strategy(df_base, config, strategy_id):
    """Prueba una configuración de estrategia."""
    try:
        # Generar señales
        df = generate_signals_multi_indicator(df_base.copy(), config)

        # Si no hay señales, saltar
        if df['señal'].abs().sum() == 0:
            return None

        # Ejecutar backtest
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

        # Calcular ratio Risk/Reward
        rr_ratio = config['tp_atr_multiplier'] / config['sl_atr_multiplier']

        result = {
            'id': strategy_id,
            'rr_ratio': rr_ratio,
            **config,
            **metrics
        }

        return result

    except Exception as e:
        print(f"  ✗ Error en estrategia {strategy_id}: {str(e)}")
        return None


def meets_criteria(result):
    """Verifica si una estrategia cumple con los criterios objetivo."""
    return (
        result['win_rate_pct'] >= TARGET_CRITERIA['min_win_rate'] and
        result['win_rate_pct'] <= TARGET_CRITERIA['max_win_rate'] and
        result['profit_factor'] >= TARGET_CRITERIA['min_profit_factor'] and
        result['max_drawdown_pct'] <= TARGET_CRITERIA['max_drawdown_pct'] and
        result['num_trades'] >= TARGET_CRITERIA['min_trades'] and
        result['total_return_pct'] >= TARGET_CRITERIA['min_return_pct'] and
        result['rr_ratio'] >= TARGET_CRITERIA['min_rr_ratio'] and
        result['rr_ratio'] <= TARGET_CRITERIA['max_rr_ratio']
    )


def main():
    print("="*80)
    print("🎯 BÚSQUEDA OPTIMIZADA: TRADING DE CORTO PLAZO RENTABLE")
    print("="*80)
    print(f"\n📊 Objetivos:")
    print(f"  • Win Rate: {TARGET_CRITERIA['min_win_rate']}-{TARGET_CRITERIA['max_win_rate']}%")
    print(f"  • Profit Factor: >= {TARGET_CRITERIA['min_profit_factor']}")
    print(f"  • Max Drawdown: <= {TARGET_CRITERIA['max_drawdown_pct']}%")
    print(f"  • Número de Trades: >= {TARGET_CRITERIA['min_trades']}")
    print(f"  • Risk/Reward: {TARGET_CRITERIA['min_rr_ratio']}:1 a {TARGET_CRITERIA['max_rr_ratio']}:1")
    print(f"  • Retorno: >= {TARGET_CRITERIA['min_return_pct']}%")

    # Generar configuraciones
    configs = generate_configs()
    total_configs = len(configs) * len(INTERVALS)

    print(f"\n📊 Configuraciones a probar:")
    print(f"  • Estrategias por timeframe: {len(configs)}")
    print(f"  • Timeframes: {len(INTERVALS)}")
    print(f"  • TOTAL: {total_configs}")
    print(f"  • Tiempo estimado: {total_configs/10/60:.1f} minutos")

    response = input("\n¿Deseas continuar? (y/n): ")
    if response.lower() != 'y':
        print("❌ Búsqueda cancelada.")
        return

    # Conectar a Binance
    print("\n" + "="*80)
    print("FASE 1: CONEXIÓN Y DESCARGA DE DATOS")
    print("="*80)

    manager = BinanceClientManager()
    client = manager.get_public_client()

    all_results = []

    # Probar cada timeframe
    for interval in INTERVALS:
        print(f"\n{'='*80}")
        print(f"TIMEFRAME: {interval}")
        print(f"{'='*80}")

        # Descargar datos
        print(f"Descargando {SYMBOL} {interval}...")
        df_base = obtener_datos_binance(
            client=client,
            simbolo=SYMBOL,
            intervalo=interval,
            inicio=START_DATE
        )

        print(f"Datos descargados: {len(df_base)} registros")

        # Calcular indicadores base
        print("Calculando indicadores...")
        df_base = agregar_indicadores(df_base)

        # Calcular indicadores adicionales
        print("Calculando indicadores adicionales...")

        # ADX
        df_base.ta.adx(length=14, append=True)

        # ATR
        df_base.ta.atr(length=14, append=True)

        # Supertrend (todos los parámetros)
        for length in [5, 7, 10]:
            for mult in [1.0, 1.5, 2.0]:
                df_base.ta.supertrend(length=length, multiplier=mult, append=True)

        # EMAs adicionales
        for period in [5, 12, 26, 50]:
            if f"EMA_{period}" not in df_base.columns:
                df_base.ta.ema(length=period, append=True)

        # RSI adicional (7 períodos)
        df_base.ta.rsi(length=7, append=True)

        # MACD adicional
        df_base.ta.macd(fast=8, slow=21, signal=9, append=True)

        print(f"Indicadores calculados. Columnas totales: {len(df_base.columns)}")

        # FASE 2: Probar estrategias
        print("\n" + "="*80)
        print(f"FASE 2: PROBANDO {len(configs)} ESTRATEGIAS EN {interval}")
        print("="*80)

        interval_results = []
        start_time = datetime.now()
        candidates_found = 0

        for idx, config in enumerate(configs, 1):
            # Agregar timeframe al config
            config_with_tf = config.copy()
            config_with_tf['timeframe'] = interval

            result = test_strategy(df_base, config_with_tf, f"{interval}_{idx}")

            if result:
                result['timeframe'] = interval
                interval_results.append(result)

                # Verificar si cumple criterios
                if meets_criteria(result):
                    candidates_found += 1
                    print(f"\n🎯 ¡CANDIDATO #{candidates_found}! ID {result['id']}")
                    print(f"   WR: {result['win_rate_pct']:.1f}% | PF: {result['profit_factor']:.2f} | Ret: {result['total_return_pct']:.1f}%")
                    print(f"   Trades: {result['num_trades']} | DD: {result['max_drawdown_pct']:.1f}% | RR: 1:{result['rr_ratio']:.1f}")
                    print(f"   Indicadores: {result['entry_indicators']}")

            # Progreso
            if idx % 100 == 0 or idx == len(configs):
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = idx / elapsed if elapsed > 0 else 0
                eta = (len(configs) - idx) / rate if rate > 0 else 0
                print(f"  [{idx}/{len(configs)}] {idx/len(configs)*100:.1f}% | {rate:.1f} tests/sec | ETA: {eta/60:.1f} min | Candidatos: {candidates_found}", end='\r')

        print(f"\n\n✅ Completado {interval}: {len(interval_results)} resultados válidos, {candidates_found} candidatos")
        all_results.extend(interval_results)

    # ========================================================================
    # FASE 3: ANÁLISIS DE RESULTADOS
    # ========================================================================
    print("\n" + "="*80)
    print("FASE 3: ANÁLISIS DE RESULTADOS")
    print("="*80)

    if not all_results:
        print("\n❌ No se encontraron estrategias válidas")
        return

    df_results = pd.DataFrame(all_results)

    # Guardar todos los resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_file = f'results/short_term_all_{timestamp}.csv'
    df_results.to_csv(all_file, index=False)
    print(f"\n💾 Guardados {len(df_results)} resultados en: {all_file}")

    # Filtrar estrategias que cumplen criterios
    best = df_results[df_results.apply(meets_criteria, axis=1)].sort_values('profit_factor', ascending=False)

    if len(best) > 0:
        print(f"\n🏆 ¡ESTRATEGIAS QUE CUMPLEN TODOS LOS CRITERIOS: {len(best)}!")
        print("="*80)

        best_file = f'results/short_term_best_{timestamp}.csv'
        best.to_csv(best_file, index=False)
        print(f"💾 Guardadas en: {best_file}\n")

        # Mostrar top 10
        print("\n📊 TOP 10 ESTRATEGIAS:")
        print("="*80)

        for idx, (i, row) in enumerate(best.head(10).iterrows(), 1):
            print(f"\n{'─'*80}")
            print(f"#{idx} - Estrategia {row['id']} ({row['timeframe']})")
            print(f"{'─'*80}")
            print(f"  📈 Profit Factor: {row['profit_factor']:.2f}")
            print(f"  💰 Retorno: {row['total_return_pct']:.2f}%")
            print(f"  🎯 Win Rate: {row['win_rate_pct']:.1f}%")
            print(f"  📉 Max Drawdown: {row['max_drawdown_pct']:.2f}%")
            print(f"  📊 Trades: {int(row['num_trades'])}")
            print(f"  ⚖️  Risk/Reward: 1:{row['rr_ratio']:.1f}")
            print(f"  📐 Sharpe: {row['sharpe_ratio']:.2f}")
            print(f"\n  🔧 Configuración:")
            print(f"     - Indicadores: {row['entry_indicators']}")

            if 'supertrend' in row['entry_indicators']:
                print(f"     - Supertrend: length={int(row['supertrend_length'])}, mult={row['supertrend_multiplier']}")
            if 'rsi' in row['entry_indicators']:
                print(f"     - RSI: period={int(row['rsi_period'])}, OS={int(row['rsi_oversold'])}, OB={int(row['rsi_overbought'])}")
            if 'ema_cross' in row['entry_indicators']:
                print(f"     - EMA Cross: fast={int(row['ema_fast'])}, slow={int(row['ema_slow'])}")
            if 'macd' in row['entry_indicators']:
                print(f"     - MACD: {int(row['macd_fast'])}/{int(row['macd_slow'])}/{int(row['macd_signal'])}")

            print(f"     - SL:TP: {row['sl_atr_multiplier']}:{row['tp_atr_multiplier']} ATR")
            print(f"     - Volumen Filter: {row['use_volume_filter']}")

    else:
        print(f"\n⚠️  No se encontraron estrategias que cumplan TODOS los criterios")
        print("\n📊 Análisis de las mejores estrategias encontradas:")

        # Analizar por métrica
        print("\n🏆 TOP 5 por Profit Factor:")
        top_pf = df_results.nlargest(5, 'profit_factor')
        for idx, (i, row) in enumerate(top_pf.iterrows(), 1):
            print(f"  {idx}. {row['id']} ({row['timeframe']}): PF={row['profit_factor']:.2f}, WR={row['win_rate_pct']:.1f}%, Trades={int(row['num_trades'])}, DD={row['max_drawdown_pct']:.1f}%")

        print("\n🎯 TOP 5 por Win Rate:")
        top_wr = df_results.nlargest(5, 'win_rate_pct')
        for idx, (i, row) in enumerate(top_wr.iterrows(), 1):
            print(f"  {idx}. {row['id']} ({row['timeframe']}): WR={row['win_rate_pct']:.1f}%, PF={row['profit_factor']:.2f}, Trades={int(row['num_trades'])}, DD={row['max_drawdown_pct']:.1f}%")

        print("\n📊 TOP 5 por Número de Trades:")
        top_trades = df_results.nlargest(5, 'num_trades')
        for idx, (i, row) in enumerate(top_trades.iterrows(), 1):
            print(f"  {idx}. {row['id']} ({row['timeframe']}): Trades={int(row['num_trades'])}, WR={row['win_rate_pct']:.1f}%, PF={row['profit_factor']:.2f}, DD={row['max_drawdown_pct']:.1f}%")

        print("\n💰 TOP 5 por Retorno:")
        top_ret = df_results.nlargest(5, 'total_return_pct')
        for idx, (i, row) in enumerate(top_ret.iterrows(), 1):
            print(f"  {idx}. {row['id']} ({row['timeframe']}): Ret={row['total_return_pct']:.1f}%, WR={row['win_rate_pct']:.1f}%, PF={row['profit_factor']:.2f}, Trades={int(row['num_trades'])}")

    print("\n" + "="*80)
    print("✅ BÚSQUEDA COMPLETADA")
    print("="*80)


if __name__ == "__main__":
    main()
