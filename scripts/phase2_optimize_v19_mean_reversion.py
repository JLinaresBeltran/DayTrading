#!/usr/bin/env python3
"""
ITERACIÓN 19: OPTIMIZACIÓN DE ESTRATEGIA MEAN REVERSION
========================================================

Objetivo: Encontrar la combinación óptima de Stop Loss y Take Profit
para lograr alta frecuencia (>500 trades/año) con rentabilidad sostenible.

Estrategia: Mean Reversion con filtro EMA_200
- LONG: Sobreventa (BB lower + RSI <30) en tendencia alcista
- SHORT: Sobrecompra (BB upper + RSI >70) en tendencia bajista

Optimización: Grid Search de SL y TP (multiplicadores ATR)
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from itertools import product

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_fetcher import obtener_datos_binance
from src.data.binance_client import BinanceClientManager
from src.indicators.technical import agregar_indicadores
from src.strategy.signal_generator import generar_senales_mean_reversion_v19
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Logger
logger = setup_logger("phase2_optimize_v19", "logs/phase2_optimize_v19.log")


def main():
    logger.info("=" * 80)
    logger.info("ITERACIÓN 19: OPTIMIZACIÓN MEAN REVERSION")
    logger.info("=" * 80)

    # ========================================
    # 1. CARGAR DATOS
    # ========================================
    logger.info("\n1. Cargando datos ETH/USDT 15m...")

    DATA_FILE = project_root / "data/ETHUSDT_15m_OHLCV_2025-11-05.csv"

    try:
        df = pd.read_csv(DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"   ✓ {len(df):,} velas cargadas")
        logger.info(f"   Período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # ========================================
    # 2. CALCULAR INDICADORES
    # ========================================
    logger.info("\n2. Calculando indicadores (EMA_200, BB, RSI, ATR)...")

    try:
        config_indicadores = {
            'ema_trend': 200,
            'bb_length': 20,
            'bb_std': 2,
            'rsi_period': 14,
            'atr_length': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3
        }

        df = agregar_indicadores(df, config=config_indicadores)
        logger.info(f"   ✓ Indicadores calculados")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # ========================================
    # 3. GENERAR SEÑALES v19
    # ========================================
    logger.info("\n3. Generando señales Mean Reversion v19...")

    try:
        df = generar_senales_mean_reversion_v19(
            df,
            bb_period=20,
            bb_std=2,
            rsi_period=14,
            rsi_oversold=30,
            rsi_overbought=70
        )

        num_buy = (df['señal'] == 1).sum()
        num_sell = (df['señal'] == -1).sum()
        total_signals = num_buy + num_sell

        logger.info(f"   ✓ Señales generadas:")
        logger.info(f"      LONG:  {num_buy:>6} señales")
        logger.info(f"      SHORT: {num_sell:>6} señales")
        logger.info(f"      Total: {total_signals:>6} señales")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 4. GRID SEARCH DE SL Y TP
    # ========================================
    logger.info("\n4. Iniciando Grid Search de SL y TP...")

    # Definir rangos de parámetros
    sl_multipliers = [1.5, 2.0, 2.5, 3.0]
    tp_multipliers = [1.0, 1.5, 2.0, 2.5, 3.0]

    total_combinations = len(sl_multipliers) * len(tp_multipliers)
    logger.info(f"   Combinaciones a probar: {total_combinations}")
    logger.info(f"   SL multipliers: {sl_multipliers}")
    logger.info(f"   TP multipliers: {tp_multipliers}")

    results_list = []
    current_combo = 0

    for sl_mult, tp_mult in product(sl_multipliers, tp_multipliers):
        current_combo += 1
        logger.info(f"\n   [{current_combo}/{total_combinations}] Probando SL={sl_mult}, TP={tp_mult}...")

        try:
            # Crear backtester
            backtester = VectorizedBacktester(
                df=df.copy(),
                initial_capital=10000,
                commission=0.00075,
                slippage=0.0005
            )

            # Ejecutar backtest con SL y TP
            backtester.run_backtest_with_sl_tp(
                atr_column='ATRr_14',
                atr_sl_multiplier=sl_mult,
                atr_tp_multiplier=tp_mult
            )

            # Calcular métricas
            metrics = backtester.calculate_metrics()

            # Guardar resultados
            result = {
                'sl_multiplier': sl_mult,
                'tp_multiplier': tp_mult,
                'ratio_rr': tp_mult / sl_mult,
                'profit_factor': metrics['profit_factor'],
                'win_rate_pct': metrics['win_rate_pct'],
                'num_trades': metrics['num_trades'],
                'total_return_pct': metrics['total_return_pct'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'max_drawdown_pct': metrics['max_drawdown_pct'],
                'avg_win': metrics.get('avg_win', 0),
                'avg_loss': metrics.get('avg_loss', 0)
            }

            results_list.append(result)

            logger.info(f"      PF: {metrics['profit_factor']:.2f} | "
                       f"WR: {metrics['win_rate_pct']:.1f}% | "
                       f"Trades: {metrics['num_trades']} | "
                       f"Return: {metrics['total_return_pct']:+.2f}%")

        except Exception as e:
            logger.error(f"      ✗ Error: {e}")
            continue

    # ========================================
    # 5. ANALIZAR RESULTADOS
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info("RESULTADOS DE OPTIMIZACIÓN")
    logger.info(f"{'=' * 80}")

    if not results_list:
        logger.error("No se obtuvieron resultados válidos")
        return

    # Convertir a DataFrame
    results_df = pd.DataFrame(results_list)

    # Ordenar por Profit Factor
    results_df_sorted = results_df.sort_values('profit_factor', ascending=False)

    # ========================================
    # 6. TOP 10 COMBINACIONES
    # ========================================
    logger.info("\n📊 TOP 10 COMBINACIONES (Ordenadas por Profit Factor):")
    logger.info(f"{'-' * 80}")

    top_10 = results_df_sorted.head(10)

    for idx, row in top_10.iterrows():
        logger.info(f"\n#{list(top_10.index).index(idx) + 1}:")
        logger.info(f"  SL: {row['sl_multiplier']:.1f}x  |  TP: {row['tp_multiplier']:.1f}x  |  R:R = 1:{row['ratio_rr']:.2f}")
        logger.info(f"  Profit Factor:  {row['profit_factor']:>6.2f}")
        logger.info(f"  Win Rate:       {row['win_rate_pct']:>6.2f}%")
        logger.info(f"  Num Trades:     {row['num_trades']:>6}")
        logger.info(f"  Return:         {row['total_return_pct']:>+6.2f}%")
        logger.info(f"  Sharpe:         {row['sharpe_ratio']:>6.2f}")
        logger.info(f"  Max DD:         {row['max_drawdown_pct']:>6.2f}%")

    # ========================================
    # 7. ANÁLISIS DE CRITERIOS DE ÉXITO
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info("ANÁLISIS DE CRITERIOS DE ÉXITO")
    logger.info(f"{'=' * 80}")

    # Filtrar combinaciones que cumplen los criterios
    criterios = results_df[
        (results_df['num_trades'] > 500) &
        (results_df['profit_factor'] > 1.10) &
        (results_df['win_rate_pct'] > 40)
    ]

    logger.info(f"\nCriterios buscados:")
    logger.info(f"  • Num Trades > 500  (alta frecuencia)")
    logger.info(f"  • Profit Factor > 1.10")
    logger.info(f"  • Win Rate > 40%")

    if len(criterios) > 0:
        logger.info(f"\n✅ {len(criterios)} combinación(es) cumple(n) TODOS los criterios:")

        for idx, row in criterios.iterrows():
            logger.info(f"\n  🎯 SL={row['sl_multiplier']:.1f}x, TP={row['tp_multiplier']:.1f}x:")
            logger.info(f"     PF={row['profit_factor']:.2f}, WR={row['win_rate_pct']:.1f}%, Trades={row['num_trades']}")

    else:
        logger.info(f"\n⚠️  Ninguna combinación cumple TODOS los criterios")

        # Analizar por criterio individual
        high_freq = results_df[results_df['num_trades'] > 500]
        good_pf = results_df[results_df['profit_factor'] > 1.10]
        good_wr = results_df[results_df['win_rate_pct'] > 40]

        logger.info(f"\n  Alta frecuencia (>500 trades): {len(high_freq)} combinaciones")
        logger.info(f"  Buen Profit Factor (>1.10): {len(good_pf)} combinaciones")
        logger.info(f"  Buen Win Rate (>40%): {len(good_wr)} combinaciones")

    # ========================================
    # 8. MEJOR COMBINACIÓN OVERALL
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info("RECOMENDACIÓN FINAL")
    logger.info(f"{'=' * 80}")

    best = results_df_sorted.iloc[0]

    logger.info(f"\n🏆 MEJOR COMBINACIÓN (por Profit Factor):")
    logger.info(f"   SL Multiplier: {best['sl_multiplier']:.1f}x ATR")
    logger.info(f"   TP Multiplier: {best['tp_multiplier']:.1f}x ATR")
    logger.info(f"   Ratio R:R: 1:{best['ratio_rr']:.2f}")
    logger.info(f"\n   Métricas:")
    logger.info(f"   - Profit Factor: {best['profit_factor']:.2f}")
    logger.info(f"   - Win Rate: {best['win_rate_pct']:.2f}%")
    logger.info(f"   - Num Trades: {best['num_trades']}")
    logger.info(f"   - Return: {best['total_return_pct']:+.2f}%")
    logger.info(f"   - Sharpe: {best['sharpe_ratio']:.2f}")

    # Evaluación
    if best['num_trades'] > 500:
        logger.info(f"\n   ✅ Alta frecuencia lograda ({best['num_trades']} > 500)")
    else:
        logger.info(f"\n   ⚠️  Frecuencia baja ({best['num_trades']} < 500)")

    if best['profit_factor'] > 1.10:
        logger.info(f"   ✅ Profit Factor excelente ({best['profit_factor']:.2f} > 1.10)")
    else:
        logger.info(f"   ⚠️  Profit Factor bajo ({best['profit_factor']:.2f} < 1.10)")

    if best['win_rate_pct'] > 40:
        logger.info(f"   ✅ Win Rate alto ({best['win_rate_pct']:.2f}% > 40%)")
    else:
        logger.info(f"   ⚠️  Win Rate bajo ({best['win_rate_pct']:.2f}% < 40%)")

    # ========================================
    # 9. GUARDAR RESULTADOS
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info("Guardando resultados...")
    logger.info(f"{'=' * 80}")

    try:
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Guardar tabla completa
        results_path = results_dir / f"optimization_v19_{timestamp}.csv"
        results_df_sorted.to_csv(results_path, index=False)
        logger.info(f"\n✓ Resultados completos: {results_path.name}")

        # Guardar top 10
        top10_path = results_dir / f"optimization_v19_top10_{timestamp}.csv"
        top_10.to_csv(top10_path, index=False)
        logger.info(f"✓ Top 10: {top10_path.name}")

    except Exception as e:
        logger.error(f"✗ Error al guardar: {e}")

    logger.info(f"\n{'=' * 80}")
    logger.info("OPTIMIZACIÓN COMPLETADA")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
