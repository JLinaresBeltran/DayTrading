#!/usr/bin/env python3
"""
ITERACIÓN 18: DONCHIAN BREAKOUT CON FILTRO DE TENDENCIA EMA_200 (TEST CON DATOS SINTÉTICOS)
============================================================================================

Este script de prueba demuestra la estrategia v18 usando datos sintéticos.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

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
logger = setup_logger("phase2_backtest_v18_test", "logs/phase2_backtest_v18_test.log")


def generar_datos_sinteticos(num_candles=5000, precio_inicial=2000, tendencia='mixta'):
    """
    Genera datos sintéticos OHLCV para pruebas.

    Args:
        num_candles: Número de velas a generar
        precio_inicial: Precio inicial
        tendencia: 'alcista', 'bajista', o 'mixta'
    """
    np.random.seed(42)

    # Generar timestamps
    start_date = datetime.now() - timedelta(minutes=15 * num_candles)
    timestamps = [start_date + timedelta(minutes=15 * i) for i in range(num_candles)]

    # Generar precios con tendencia
    prices = [precio_inicial]

    for i in range(1, num_candles):
        # Volatilidad base
        volatility = 0.01

        # Agregar tendencia
        if tendencia == 'alcista':
            drift = 0.0002
        elif tendencia == 'bajista':
            drift = -0.0002
        else:  # mixta
            # Crear ciclos de tendencia
            cycle = i / 1000
            drift = 0.0003 * np.sin(cycle)

        # Movimiento aleatorio con deriva
        change = np.random.normal(drift, volatility)
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)

    # Generar OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        # High y Low basados en volatilidad
        high_factor = np.random.uniform(1.001, 1.01)
        low_factor = np.random.uniform(0.99, 0.999)

        high = close * high_factor
        low = close * low_factor

        # Open cerca del close anterior
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] + np.random.uniform(-5, 5)

        # Volume aleatorio
        volume = np.random.uniform(1000000, 5000000)

        data.append({
            'timestamp': ts,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    df = pd.DataFrame(data)
    return df


def main():
    """
    Función principal para ejecutar el backtest de prueba de la Iteración 18.
    """
    logger.info("=" * 80)
    logger.info("ITERACIÓN 18: DONCHIAN BREAKOUT CON FILTRO EMA_200 (TEST)")
    logger.info("=" * 80)

    # ========================================
    # CONFIGURACIÓN
    # ========================================
    DONCHIAN_PERIOD = 20
    ATR_MULTIPLIER = 4.0

    logger.info(f"\nCONFIGURACIÓN:")
    logger.info(f"  Datos: Sintéticos (5000 velas)")
    logger.info(f"  Timeframe: 15m")
    logger.info(f"  Donchian Period: {DONCHIAN_PERIOD}")
    logger.info(f"  ATR Multiplier: {ATR_MULTIPLIER}")

    # ========================================
    # 1. GENERAR DATOS SINTÉTICOS
    # ========================================
    logger.info(f"\n1. Generando datos sintéticos...")
    df = generar_datos_sinteticos(num_candles=5000, precio_inicial=2000, tendencia='mixta')
    logger.info(f"   ✓ {len(df):,} velas generadas")
    logger.info(f"   Rango de precios: ${df['close'].min():.2f} → ${df['close'].max():.2f}")

    # ========================================
    # 2. CALCULAR INDICADORES
    # ========================================
    logger.info(f"\n2. Calculando indicadores técnicos...")
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
    logger.info(f"   ✓ Indicadores calculados")

    # ========================================
    # 3. GENERAR SEÑALES v18
    # ========================================
    logger.info(f"\n3. Generando señales con estrategia Donchian + Filtro EMA_200...")
    df = generar_senales_donchian_filtrado_v18(
        df,
        donchian_period=DONCHIAN_PERIOD,
        config={'ema_trend': 200, 'atr_length': 14}
    )

    num_buy_signals = (df['señal'] == 1).sum()
    num_sell_signals = (df['señal'] == -1).sum()
    num_neutral = (df['señal'] == 0).sum()
    total_signals = num_buy_signals + num_sell_signals

    logger.info(f"   ✓ Señales generadas:")
    logger.info(f"      COMPRA (LONG):  {num_buy_signals:>5} ({num_buy_signals/len(df)*100:>5.2f}%)")
    logger.info(f"      VENTA (SHORT):  {num_sell_signals:>5} ({num_sell_signals/len(df)*100:>5.2f}%)")
    logger.info(f"      NEUTRAL:        {num_neutral:>5} ({num_neutral/len(df)*100:>5.2f}%)")

    # ========================================
    # 4. EJECUTAR BACKTESTING
    # ========================================
    logger.info(f"\n4. Ejecutando backtesting con Stop Loss ATR (multiplier={ATR_MULTIPLIER})...")
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

    logger.info(f"   ✓ Backtesting completado")

    # ========================================
    # 5. CALCULAR MÉTRICAS
    # ========================================
    logger.info(f"\n5. Calculando métricas de rendimiento...")
    metrics = backtester.calculate_metrics()

    # ========================================
    # 6. REPORTE DE RESULTADOS
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"REPORTE ITERACIÓN 18: DONCHIAN + FILTRO EMA_200 (DATOS SINTÉTICOS)")
    logger.info(f"{'=' * 80}")

    logger.info(f"\nESTRATEGIA:")
    logger.info(f"  Tipo: Donchian Breakout con Filtro EMA_200")
    logger.info(f"  Filtro: Solo LONG si close > EMA_200, Solo SHORT si close < EMA_200")

    logger.info(f"\nSEÑALES:")
    logger.info(f"  COMPRA (LONG):  {num_buy_signals}")
    logger.info(f"  VENTA (SHORT):  {num_sell_signals}")
    logger.info(f"  Total activas:  {total_signals}")

    logger.info(f"\nRENDIMIENTO:")
    logger.info(f"  Capital Inicial:     ${metrics['initial_capital']:>10,.2f}")
    logger.info(f"  Capital Final:       ${metrics['final_value']:>10,.2f}")
    logger.info(f"  Retorno Total:       {metrics['total_return_pct']:>10.2f}%")
    logger.info(f"  Num Trades:          {metrics['num_trades']:>10}")
    logger.info(f"  Win Rate:            {metrics['win_rate_pct']:>10.2f}%")
    logger.info(f"  Profit Factor:       {metrics['profit_factor']:>10.2f}")
    logger.info(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>10.2f}")
    logger.info(f"  Max Drawdown:        {metrics['max_drawdown_pct']:>10.2f}%")

    # ========================================
    # 7. INTERPRETACIÓN DE RESULTADOS
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"ANÁLISIS DE LA ESTRATEGIA v18")
    logger.info(f"{'=' * 80}")

    logger.info(f"\nCARACTERÍSTICAS DE LA ESTRATEGIA:")
    logger.info(f"  1. Filtro de Régimen: EMA_200 determina si estamos en tendencia alcista o bajista")
    logger.info(f"  2. Señales de Entrada:")
    logger.info(f"     - LONG: Breakout del canal superior + precio > EMA_200")
    logger.info(f"     - SHORT: Breakout del canal inferior + precio < EMA_200")
    logger.info(f"  3. Gestión de Riesgo: Stop Loss ATR dinámico (4.0x)")

    logger.info(f"\nDIFERENCIA CON v17:")
    logger.info(f"  - v17: LONG con filtro, SHORT sin filtro → Win Rate: 19.23%, PF: 1.03")
    logger.info(f"  - v18: LONG con filtro, SHORT con filtro → Win Rate: {metrics['win_rate_pct']:.2f}%, PF: {metrics['profit_factor']:.2f}")

    if metrics['profit_factor'] > 1.03 and metrics['win_rate_pct'] > 19.23:
        logger.info(f"\n{'✓' * 40}")
        logger.info(f"✓✓✓ ESTRATEGIA v18 MEJORADA ✓✓✓")
        logger.info(f"✓ El filtro EMA_200 bilateral mejoró las métricas")
        logger.info(f"{'✓' * 40}")
    else:
        logger.info(f"\nRESULTADO: La estrategia v18 con filtro EMA_200 necesita ajustes")

    # ========================================
    # 8. GUARDAR RESULTADOS
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Guardando resultados de prueba...")
    logger.info(f"{'=' * 80}")

    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Guardar señales
    signals_path = results_dir / f"signals_v18_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_to_save = df[['timestamp', 'close', 'EMA_200', f'DONCHI_h_{DONCHIAN_PERIOD}',
                     f'DONCHI_l_{DONCHIAN_PERIOD}', 'ATRr_14', 'señal', 'position']].copy()
    df_to_save.to_csv(signals_path, index=False)
    logger.info(f"✓ Señales guardadas: {signals_path.name}")

    if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
        trades_path = results_dir / f"trades_log_v18_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        backtester.trades_log.to_csv(trades_path, index=False)
        logger.info(f"✓ Trades guardados: {trades_path.name}")

        logger.info(f"\n   Primeros 5 trades:")
        logger.info(f"\n{backtester.trades_log.head().to_string(index=False)}")

    logger.info(f"\n{'=' * 80}")
    logger.info(f"PRUEBA COMPLETADA")
    logger.info(f"{'=' * 80}")
    logger.info(f"\nNOTA: Esta es una prueba con datos sintéticos.")
    logger.info(f"Para ejecutar con datos reales de Binance, asegúrate de:")
    logger.info(f"  1. Tener conexión a internet")
    logger.info(f"  2. Configurar credenciales de Binance si es necesario")
    logger.info(f"  3. Ejecutar: python scripts/phase2_backtest_v18.py")


if __name__ == "__main__":
    main()
