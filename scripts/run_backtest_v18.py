#!/usr/bin/env python3
"""
ITERACIÓN 18: BACKTEST CON DATOS REALES ETH 15m
================================================

Ejecuta el backtest de la estrategia Donchian + Filtro EMA_200 (v18)
con datos reales de ETH/USDT en timeframe de 15 minutos.
"""

import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indicators.technical import agregar_indicadores
from src.strategy.signal_generator import generar_senales_donchian_filtrado_v18
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger("phase2_backtest_v18_eth15m", "logs/phase2_backtest_v18_eth15m.log")


def main():
    logger.info("=" * 80)
    logger.info("ITERACIÓN 18: DONCHIAN + FILTRO EMA_200 - ETH 15m (DATOS REALES)")
    logger.info("=" * 80)

    # ========================================
    # CONFIGURACIÓN
    # ========================================
    DONCHIAN_PERIOD = 20
    ATR_MULTIPLIER = 4.0
    DATA_FILE = project_root / "data/ETHUSDT_15m_OHLCV_2025-11-05.csv"

    logger.info(f"\nCONFIGURACIÓN:")
    logger.info(f"  Archivo: {DATA_FILE.name}")
    logger.info(f"  Timeframe: 15 minutos (Day Trading)")
    logger.info(f"  Donchian Period: {DONCHIAN_PERIOD}")
    logger.info(f"  ATR Multiplier: {ATR_MULTIPLIER}")

    # ========================================
    # 1. CARGAR DATOS
    # ========================================
    logger.info(f"\n1. Cargando datos desde CSV...")

    try:
        df = pd.read_csv(DATA_FILE)
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        logger.info(f"   ✓ {len(df):,} velas cargadas")
        logger.info(f"   Período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
        logger.info(f"   Precio: ${df['close'].min():.2f} → ${df['close'].max():.2f}")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # ========================================
    # 2. CALCULAR INDICADORES
    # ========================================
    logger.info(f"\n2. Calculando indicadores técnicos...")

    try:
        config_indicadores = {
            'ema_trend': 200,
            'rsi_period': 14,
            'bb_length': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_length': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'donchian_period': DONCHIAN_PERIOD
        }

        df = agregar_indicadores(df, config=config_indicadores)
        logger.info(f"   ✓ Indicadores calculados ({len(df.columns)} columnas)")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 3. GENERAR SEÑALES v18
    # ========================================
    logger.info(f"\n3. Generando señales Donchian + Filtro EMA_200...")

    try:
        df = generar_senales_donchian_filtrado_v18(
            df,
            donchian_period=DONCHIAN_PERIOD,
            config={'ema_trend': 200, 'atr_length': 14}
        )

        num_buy = (df['señal'] == 1).sum()
        num_sell = (df['señal'] == -1).sum()
        num_neutral = (df['señal'] == 0).sum()

        logger.info(f"   ✓ Señales generadas:")
        logger.info(f"      LONG:    {num_buy:>6} ({num_buy/len(df)*100:>5.2f}%)")
        logger.info(f"      SHORT:   {num_sell:>6} ({num_sell/len(df)*100:>5.2f}%)")
        logger.info(f"      NEUTRAL: {num_neutral:>6} ({num_neutral/len(df)*100:>5.2f}%)")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 4. BACKTESTING
    # ========================================
    logger.info(f"\n4. Ejecutando backtesting (ATR multiplier = {ATR_MULTIPLIER})...")

    try:
        backtester = VectorizedBacktester(
            df=df,
            initial_capital=10000,
            commission=0.00075,
            slippage=0.0005
        )

        results = backtester.run_backtest_with_stop_loss(
            atr_column='ATRr_14',
            atr_multiplier=ATR_MULTIPLIER
        )

        logger.info(f"   ✓ Backtest completado")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 5. MÉTRICAS
    # ========================================
    logger.info(f"\n5. Calculando métricas...")

    try:
        metrics = backtester.calculate_metrics()

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 6. REPORTE
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"REPORTE ITERACIÓN 18 - ETH 15m (DATOS REALES)")
    logger.info(f"{'=' * 80}")

    logger.info(f"\nDATOS:")
    logger.info(f"  Símbolo: ETH/USDT")
    logger.info(f"  Timeframe: 15 minutos")
    logger.info(f"  Velas: {len(df):,}")
    logger.info(f"  Período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    logger.info(f"\nESTRATEGIA:")
    logger.info(f"  v18: Donchian Breakout + Filtro EMA_200 Bilateral")
    logger.info(f"  LONG: close > EMA_200 AND breakout canal superior")
    logger.info(f"  SHORT: close < EMA_200 AND breakout canal inferior")

    logger.info(f"\nRENDIMIENTO:")
    logger.info(f"  Capital Inicial:     ${metrics['initial_capital']:>10,.2f}")
    logger.info(f"  Capital Final:       ${metrics['final_value']:>10,.2f}")
    logger.info(f"  Retorno Total:       {metrics['total_return_pct']:>10.2f}%")
    logger.info(f"  Num Trades:          {metrics['num_trades']:>10}")
    logger.info(f"  Win Rate:            {metrics['win_rate_pct']:>10.2f}%")
    logger.info(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
    logger.info(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%")

    if metrics.get('avg_win') and metrics.get('avg_loss'):
        logger.info(f"  Ganancia Promedio:   ${metrics['avg_win']:>10,.2f}")
        logger.info(f"  Pérdida Promedio:    ${metrics['avg_loss']:>10,.2f}")

    # ========================================
    # 7. COMPARACIÓN CON v17
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPARACIÓN: v18 (Filtrado) vs v17 (Sin filtro SHORT)")
    logger.info(f"{'=' * 80}")

    v17 = {'profit_factor': 1.03, 'win_rate': 19.23, 'num_trades': 26}

    logger.info(f"\nMÉTRICA                v17              v18            Delta")
    logger.info(f"{'-' * 80}")

    pf_delta = metrics['profit_factor'] - v17['profit_factor']
    logger.info(f"Profit Factor        {v17['profit_factor']:>6.2f}         {metrics['profit_factor']:>6.2f}         {pf_delta:>+6.2f} {'✓' if pf_delta > 0 else '✗'}")

    wr_delta = metrics['win_rate_pct'] - v17['win_rate']
    logger.info(f"Win Rate (%)         {v17['win_rate']:>6.2f}         {metrics['win_rate_pct']:>6.2f}         {wr_delta:>+6.2f} {'✓' if wr_delta > 0 else '✗'}")

    nt_delta = metrics['num_trades'] - v17['num_trades']
    logger.info(f"Num Trades           {v17['num_trades']:>6}         {metrics['num_trades']:>6}         {nt_delta:>+6} {'↓' if nt_delta < 0 else '↑'}")

    # ========================================
    # 8. EVALUACIÓN
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"EVALUACIÓN DE LA HIPÓTESIS")
    logger.info(f"{'=' * 80}")

    logger.info(f"\nHIPÓTESIS:")
    logger.info(f"  El filtro EMA_200 bilateral mejorará Profit Factor y Win Rate")
    logger.info(f"  al evitar señales en mercados laterales o contra-tendencia.")

    success_pf = metrics['profit_factor'] > v17['profit_factor']
    success_wr = metrics['win_rate_pct'] > v17['win_rate']

    logger.info(f"\nRESULTADOS:")
    logger.info(f"  ✓ Profit Factor mejoró: {success_pf} ({pf_delta:+.2f})")
    logger.info(f"  ✓ Win Rate mejoró: {success_wr} ({wr_delta:+.2f}%)")
    logger.info(f"  ✓ Trades: {'Redujo' if nt_delta < 0 else 'Aumentó'} ({nt_delta:+d})")

    if success_pf and success_wr:
        logger.info(f"\n{'✓' * 40}")
        logger.info(f"✓✓✓ HIPÓTESIS CONFIRMADA ✓✓✓")
        logger.info(f"✓ El filtro EMA_200 bilateral mejoró AMBAS métricas clave")
        logger.info(f"✓ La estrategia v18 es SUPERIOR a v17")
        logger.info(f"{'✓' * 40}")
    elif success_pf or success_wr:
        logger.info(f"\n{'~' * 40}")
        logger.info(f"~~~ HIPÓTESIS PARCIALMENTE CONFIRMADA ~~~")
        logger.info(f"~ Mejoró {'Profit Factor' if success_pf else 'Win Rate'}")
        logger.info(f"~ Requiere ajustes adicionales")
        logger.info(f"{'~' * 40}")
    else:
        logger.info(f"\n{'✗' * 40}")
        logger.info(f"✗✗✗ HIPÓTESIS RECHAZADA ✗✗✗")
        logger.info(f"✗ El filtro EMA_200 bilateral NO mejoró las métricas")
        logger.info(f"✗ v17 sigue siendo superior")
        logger.info(f"{'✗' * 40}")

    # ========================================
    # 9. GUARDAR RESULTADOS
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Guardando resultados...")
    logger.info(f"{'=' * 80}")

    try:
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Señales
        signals_path = results_dir / f"signals_v18_eth15m_{ts}.csv"
        df[['timestamp', 'close', 'EMA_200', f'DONCHI_h_{DONCHIAN_PERIOD}',
            f'DONCHI_l_{DONCHIAN_PERIOD}', 'ATRr_14', 'señal', 'position']].to_csv(
            signals_path, index=False
        )
        logger.info(f"✓ Señales: {signals_path.name}")

        # Trades
        if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
            trades_path = results_dir / f"trades_v18_eth15m_{ts}.csv"
            backtester.trades_log.to_csv(trades_path, index=False)
            logger.info(f"✓ Trades: {trades_path.name}")

            logger.info(f"\n   Primeros 10 trades:")
            print(backtester.trades_log.head(10).to_string(index=False))

    except Exception as e:
        logger.error(f"✗ Error: {e}")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"BACKTEST COMPLETADO")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
