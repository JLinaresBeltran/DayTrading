#!/usr/bin/env python3
"""
ITERACIÓN 18: DONCHIAN BREAKOUT CON FILTRO DE TENDENCIA EMA_200
=================================================================

HIPÓTESIS:
La estrategia Donchian Breakout v17 (Profit Factor 1.03, Win Rate 19.23%, 26 trades)
solo será rentable si opera a favor de la tendencia principal, filtrando las señales
en mercados laterales o contra-tendencia mediante el filtro EMA_200.

ESTRATEGIA:
- COMPRA (LONG): Precio cruza canal superior de Donchian Y precio > EMA_200
- VENTA (SHORT): Precio cruza canal inferior de Donchian Y precio < EMA_200
- NEUTRAL: Mercado lateral (precio entre bandas Donchian o sin filtro de tendencia)

GESTIÓN DE RIESGO:
- Stop Loss dinámico: ATR × 4.0 (mejor parámetro de v17)
- Capital por trade: Según configuración

CRITERIOS DE ÉXITO vs Iteración 17:
- Profit Factor > 1.03
- Win Rate > 19.23%
- Num Trades: Se espera menor cantidad pero con mayor calidad
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Agregar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.signal_generator import generar_senales_donchian_filtrado_v18
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger("phase2_backtest_v18", "logs/phase2_backtest_v18.log")


def main():
    """
    Función principal para ejecutar el backtest de la Iteración 18.
    """
    logger.info("=" * 80)
    logger.info("ITERACIÓN 18: DONCHIAN BREAKOUT CON FILTRO EMA_200")
    logger.info("=" * 80)

    # ========================================
    # CONFIGURACIÓN
    # ========================================
    SYMBOL = "ETHUSDT"
    TIMEFRAME = "15m"
    DONCHIAN_PERIOD = 20  # Período del Canal de Donchian
    ATR_MULTIPLIER = 4.0  # Mejor parámetro de v17

    logger.info(f"\nCONFIGURACIÓN:")
    logger.info(f"  Símbolo: {SYMBOL}")
    logger.info(f"  Timeframe: {TIMEFRAME}")
    logger.info(f"  Donchian Period: {DONCHIAN_PERIOD}")
    logger.info(f"  ATR Multiplier: {ATR_MULTIPLIER}")

    # ========================================
    # 1. DESCARGAR DATOS
    # ========================================
    logger.info(f"\n1. Descargando datos de {SYMBOL} ({TIMEFRAME})...")
    try:
        manager = BinanceClientManager()
        client = manager.get_public_client()

        df = obtener_datos_binance(
            client=client,
            simbolo=SYMBOL,
            intervalo=TIMEFRAME,
            inicio='1 year ago UTC'
        )

        logger.info(f"   ✓ {len(df):,} velas descargadas")
        logger.info(f"   Rango: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    except Exception as e:
        logger.error(f"   ✗ Error al descargar datos: {e}")
        return

    # ========================================
    # 2. CALCULAR INDICADORES
    # ========================================
    logger.info(f"\n2. Calculando indicadores técnicos...")
    try:
        # Configuración de indicadores para v18
        config_indicadores = {
            'ema_trend': 200,           # EMA de tendencia para filtro de régimen
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
            'donchian_period': DONCHIAN_PERIOD  # Canal de Donchian
        }

        df = agregar_indicadores(df, config=config_indicadores)
        logger.info(f"   ✓ Indicadores calculados: {len(df.columns)} columnas")

        # Verificar que EMA_200 y Canales de Donchian existan
        required_indicators = ['EMA_200', f'DONCHI_h_{DONCHIAN_PERIOD}', f'DONCHI_l_{DONCHIAN_PERIOD}', 'ATRr_14']
        missing_indicators = [ind for ind in required_indicators if ind not in df.columns]

        if missing_indicators:
            logger.error(f"   ✗ Indicadores faltantes: {missing_indicators}")
            logger.error(f"   Columnas disponibles: {df.columns.tolist()}")
            return

        logger.info(f"   ✓ Indicadores requeridos verificados: {required_indicators}")

    except Exception as e:
        logger.error(f"   ✗ Error al calcular indicadores: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 3. GENERAR SEÑALES v18
    # ========================================
    logger.info(f"\n3. Generando señales con estrategia Donchian + Filtro EMA_200...")
    try:
        df = generar_senales_donchian_filtrado_v18(
            df,
            donchian_period=DONCHIAN_PERIOD,
            config={'ema_trend': 200, 'atr_length': 14}
        )

        # Estadísticas de señales
        num_buy_signals = (df['señal'] == 1).sum()
        num_sell_signals = (df['señal'] == -1).sum()
        num_neutral = (df['señal'] == 0).sum()
        total_signals = num_buy_signals + num_sell_signals

        logger.info(f"   ✓ Señales generadas:")
        logger.info(f"      COMPRA (LONG):  {num_buy_signals:>5} ({num_buy_signals/len(df)*100:>5.2f}%)")
        logger.info(f"      VENTA (SHORT):  {num_sell_signals:>5} ({num_sell_signals/len(df)*100:>5.2f}%)")
        logger.info(f"      NEUTRAL:        {num_neutral:>5} ({num_neutral/len(df)*100:>5.2f}%)")
        logger.info(f"      Total señales activas: {total_signals}")

    except Exception as e:
        logger.error(f"   ✗ Error al generar señales: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 4. EJECUTAR BACKTESTING
    # ========================================
    logger.info(f"\n4. Ejecutando backtesting con Stop Loss ATR (multiplier={ATR_MULTIPLIER})...")
    try:
        backtester = VectorizedBacktester(
            df=df,
            initial_capital=10000,
            commission=0.00075,   # 0.075% comisión Binance
            slippage=0.0005       # 0.05% slippage
        )

        # Ejecutar backtest con Stop Loss ATR
        results = backtester.run_backtest_with_stop_loss(
            atr_column='ATRr_14',
            atr_multiplier=ATR_MULTIPLIER
        )

        logger.info(f"   ✓ Backtesting completado")

    except Exception as e:
        logger.error(f"   ✗ Error al ejecutar backtesting: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 5. CALCULAR MÉTRICAS
    # ========================================
    logger.info(f"\n5. Calculando métricas de rendimiento...")
    try:
        metrics = backtester.calculate_metrics()
        logger.info(f"   ✓ Métricas calculadas")

    except Exception as e:
        logger.error(f"   ✗ Error al calcular métricas: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================
    # 6. REPORTE DE RESULTADOS
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"REPORTE ITERACIÓN 18: DONCHIAN + FILTRO EMA_200")
    logger.info(f"{'=' * 80}")

    logger.info(f"\nDATOS:")
    logger.info(f"  Símbolo: {SYMBOL}")
    logger.info(f"  Timeframe: {TIMEFRAME}")
    logger.info(f"  Velas: {len(df):,}")
    logger.info(f"  Período: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    logger.info(f"\nESTRATEGIA:")
    logger.info(f"  Tipo: Donchian Breakout con Filtro EMA_200")
    logger.info(f"  Donchian Period: {DONCHIAN_PERIOD}")
    logger.info(f"  EMA Trend: 200")
    logger.info(f"  ATR Multiplier: {ATR_MULTIPLIER}")

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

    if metrics.get('avg_win') is not None and metrics.get('avg_loss') is not None:
        logger.info(f"  Ganancia Promedio:   ${metrics['avg_win']:>10,.2f}")
        logger.info(f"  Pérdida Promedio:    ${metrics['avg_loss']:>10,.2f}")

    # ========================================
    # 7. COMPARACIÓN CON ITERACIÓN 17
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"COMPARACIÓN: Iteración 18 vs Iteración 17")
    logger.info(f"{'=' * 80}")

    # Métricas de referencia de v17 (según el usuario)
    v17_metrics = {
        'profit_factor': 1.03,
        'win_rate': 19.23,
        'num_trades': 26
    }

    logger.info(f"\nMÉTRICA                IT17 (Baseline)    IT18 (Actual)      Mejora")
    logger.info(f"{'-' * 80}")

    # Profit Factor
    pf_delta = metrics['profit_factor'] - v17_metrics['profit_factor']
    pf_status = "✓" if pf_delta > 0 else "✗"
    logger.info(f"Profit Factor          {v17_metrics['profit_factor']:>10.2f}      {metrics['profit_factor']:>10.2f}      {pf_delta:>+7.2f} {pf_status}")

    # Win Rate
    wr_delta = metrics['win_rate_pct'] - v17_metrics['win_rate']
    wr_status = "✓" if wr_delta > 0 else "✗"
    logger.info(f"Win Rate (%)           {v17_metrics['win_rate']:>10.2f}      {metrics['win_rate_pct']:>10.2f}      {wr_delta:>+7.2f} {wr_status}")

    # Num Trades
    nt_delta = metrics['num_trades'] - v17_metrics['num_trades']
    nt_status = "↓" if nt_delta < 0 else "↑"
    logger.info(f"Num Trades             {v17_metrics['num_trades']:>10}      {metrics['num_trades']:>10}      {nt_delta:>+7} {nt_status}")

    # ========================================
    # 8. CRITERIOS DE ÉXITO
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"CRITERIOS DE ÉXITO")
    logger.info(f"{'=' * 80}")

    success_profit_factor = metrics['profit_factor'] > v17_metrics['profit_factor']
    success_win_rate = metrics['win_rate_pct'] > v17_metrics['win_rate']
    success_overall = success_profit_factor and success_win_rate

    logger.info(f"\nProfit Factor > {v17_metrics['profit_factor']}: {metrics['profit_factor']:.2f} {'✓ APROBADO' if success_profit_factor else '✗ RECHAZADO'}")
    logger.info(f"Win Rate > {v17_metrics['win_rate']:.2f}%: {metrics['win_rate_pct']:.2f}% {'✓ APROBADO' if success_win_rate else '✗ RECHAZADO'}")

    if success_overall:
        logger.info(f"\n{'✓' * 40}")
        logger.info(f"✓✓✓ ITERACIÓN 18 APROBADA ✓✓✓")
        logger.info(f"✓ El filtro EMA_200 mejoró significativamente la estrategia Donchian")
        logger.info(f"{'✓' * 40}")
    else:
        logger.info(f"\n{'✗' * 40}")
        logger.info(f"✗✗✗ ITERACIÓN 18 NO APROBADA ✗✗✗")
        logger.info(f"✗ El filtro EMA_200 no mejoró la estrategia Donchian como se esperaba")
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

        # Guardar log de trades si existe
        if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            trades_path = results_dir / f"trades_log_v18_{timestamp}.csv"
            backtester.trades_log.to_csv(trades_path, index=False)
            logger.info(f"✓ Log de trades guardado: {trades_path.name}")

            # Mostrar algunas trades de ejemplo
            logger.info(f"\n   Ejemplo de trades (primeras 5):")
            logger.info(f"\n{backtester.trades_log.head().to_string(index=False)}")

        # Guardar señales para análisis
        signals_path = results_dir / f"signals_v18_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        df_to_save = df[['timestamp', 'close', 'EMA_200', f'DONCHI_h_{DONCHIAN_PERIOD}',
                         f'DONCHI_l_{DONCHIAN_PERIOD}', 'ATRr_14', 'señal', 'position']].copy()
        df_to_save.to_csv(signals_path, index=False)
        logger.info(f"✓ Señales guardadas: {signals_path.name}")

        logger.info(f"\n✓ Todos los resultados guardados en: {results_dir}/")

    except Exception as e:
        logger.error(f"✗ Error al guardar resultados: {e}")
        import traceback
        logger.error(traceback.format_exc())

    # ========================================
    # FIN
    # ========================================
    logger.info(f"\n{'=' * 80}")
    logger.info(f"EJECUCIÓN COMPLETADA")
    logger.info(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
