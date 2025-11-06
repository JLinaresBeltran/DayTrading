#!/usr/bin/env python3
"""
ITERACIÓN 27: ESTRATEGIA STOCHASTIC CROSSOVER CON FILTRO DE TENDENCIA EMA EN TIMEFRAME 5M
============================================================================================

CONTEXTO:
- Iteraciones v24, v25, v26: TODAS las estrategias anteriores fracasaron en 5m
  * v24 (Donchian Breakout): PF 0.78 ✗
  * v25 (EMA Pullback): PF 0.81 ✗
  * v26 (MACD Crossover): PF 0.80 ✗
- Problema común: Alta frecuencia (>150 trades) pero NO rentable (PF < 1.0)
- Conclusión: El timeframe de 5m es extremadamente ruidoso

HIPÓTESIS V27 (ÚLTIMA ESTRATEGIA DE DAY TRADING EN 5M):
El Oscilador Estocástico puede identificar reversiones rápidas en zonas extremas
(sobreventa/sobrecompra), proporcionando el "edge" que necesitamos cuando se opera
a favor de la tendencia principal (EMA).

ESTRATEGIA: STOCHASTIC CROSSOVER CON FILTRO DE TENDENCIA
- Usa el cruce de Stochastic (%K y %D) en zonas extremas (< 20 o > 80)
- Filtra las señales según tendencia de EMA para operar solo a favor del flujo
- Stochastic es más sensible que MACD para identificar reversiones rápidas

LÓGICA DE SEÑALES:

COMPRA:
  1. Precio[t] > EMA_Filtro[t] (Tendencia alcista)
  2. Stoch_K[t] < 20 (Zona de sobreventa)
  3. Stoch_K[t] cruza por encima de Stoch_D[t] (Cruce alcista)
     (Stoch_K[t-1] <= Stoch_D[t-1] AND Stoch_K[t] > Stoch_D[t])

VENTA:
  1. Precio[t] < EMA_Filtro[t] (Tendencia bajista)
  2. Stoch_K[t] > 80 (Zona de sobrecompra)
  3. Stoch_K[t] cruza por debajo de Stoch_D[t] (Cruce bajista)
     (Stoch_K[t-1] >= Stoch_D[t-1] AND Stoch_K[t] < Stoch_D[t])

PARÁMETROS A OPTIMIZAR (Grid Search):
- ema_filter_periodo: [150, 200] (Filtros de tendencia largos)
- stoch_k: [14] (Estándar Stochastic)
- stoch_d: [3] (Estándar Stochastic)
- stoch_smooth: [3] (Estándar Stochastic)
- sl_multiplier: [2.0, 3.0] (Stops ajustados para 5m)
- tp_multiplier: [2.0, 3.0, 4.0]

Total de combinaciones: 2 × 1 × 1 × 1 × 2 × 3 = 12 configuraciones

CRITERIOS DE ÉXITO:
- Profit Factor > 1.1 (umbral mínimo de rentabilidad)
- Num Trades > 150 (alta frecuencia - objetivo Day Trading)
- Ambos deben cumplirse simultáneamente

DECISIÓN CRÍTICA:
Si esta iteración falla (PF < 1.1), habremos probado las CUATRO familias principales
de indicadores técnicos en 5m:
  1. Breakout (Donchian) ✗
  2. Pullback (EMA) ✗
  3. Momentum Crossover (MACD) ✗
  4. Oscillator Crossover (Stochastic) ← ÚLTIMA OPORTUNIDAD

Conclusión: El objetivo de Day Trading NO es viable con nuestro sistema en 5m.
Próximo paso: Cambiar a timeframe 15m o pivotear completamente la estrategia.

DATOS:
- Activo: ETHUSDT
- Timeframe: 5m
- Período: 1 año (datos ya descargados)
"""

import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid

# Configurar path del proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical_simple import agregar_indicadores
from src.strategy.signal_generator import generar_senales_stoch_crossover_v27
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger("phase2_optimize_v27", "logs/phase2_optimize_v27.log")


def main():
    logger.info("=" * 80)
    logger.info("ITERACIÓN 27: ESTRATEGIA STOCHASTIC CROSSOVER EN TIMEFRAME 5M")
    logger.info("=" * 80)
    logger.info("\nHipótesis: Stochastic en zonas extremas → ÚLTIMA OPORTUNIDAD en 5m")
    logger.info("Objetivo: PF > 1.1 AND Num Trades > 150")

    # ========================================
    # 1. CARGAR DATOS HISTÓRICOS
    # ========================================
    logger.info("\n1. Cargando datos históricos de ETHUSDT 5m...")

    # Ruta del archivo CSV con datos previamente descargados
    data_file = project_root / 'data' / 'ETHUSDT_5m_OHLCV_2025-11-05.csv'

    if data_file.exists():
        logger.info(f"   ✓ Cargando desde archivo: {data_file.name}")
        df = pd.read_csv(data_file)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        logger.info(f"   ✓ {len(df):,} velas cargadas desde CSV")
        logger.info(f"   ✓ Período: {df['timestamp'].iloc[0]} hasta {df['timestamp'].iloc[-1]}")
    else:
        logger.error(f"   ✗ Archivo no encontrado: {data_file}")
        logger.info("   → Descargando datos desde Binance...")

        manager = BinanceClientManager()
        client = manager.get_public_client()

        df = obtener_datos_binance(
            client=client,
            simbolo='ETHUSDT',
            intervalo='5m',
            inicio='1 year ago UTC'
        )
        logger.info(f"   ✓ {len(df):,} velas descargadas")

    # ========================================
    # 2. DEFINIR GRID DE PARÁMETROS
    # ========================================
    logger.info("\n2. Definiendo Grid de Parámetros (Estrategia Stochastic Crossover)...")

    param_grid = {
        'ema_filter_periodo': [150, 200],
        'stoch_k': [14],
        'stoch_d': [3],
        'stoch_smooth': [3],
        'sl_multiplier': [2.0, 3.0],
        'tp_multiplier': [2.0, 3.0, 4.0]
    }

    grid = list(ParameterGrid(param_grid))
    logger.info(f"   ✓ Total de combinaciones: {len(grid)}")
    logger.info(f"   ✓ Parámetros:")
    for key, values in param_grid.items():
        logger.info(f"      - {key}: {values}")

    # ========================================
    # 3. EJECUTAR GRID SEARCH
    # ========================================
    logger.info("\n3. Ejecutando Grid Search...")
    logger.info("   (Esto puede tardar varios minutos)\n")

    results = []
    total = len(grid)

    for idx, params in enumerate(grid, start=1):
        if idx % 5 == 1:
            logger.info(f"   Evaluando combinación {idx}/{total}...")

        try:
            # 3.1. Configurar indicadores según los parámetros actuales
            indicator_config = {
                'atr_length': 14,
                'rsi_period': 14,
                'bb_length': 20,
                'bb_std': 2,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,
                'stoch_k': params['stoch_k'],
                'stoch_d': params['stoch_d'],
                'stoch_smooth': params['stoch_smooth']
            }

            # Añadir EMA de filtro dinámicamente
            if params['ema_filter_periodo'] == 150:
                indicator_config['ema_filter'] = 150
            elif params['ema_filter_periodo'] == 200:
                indicator_config['ema_trend'] = 200

            # 3.2. Calcular indicadores
            df_indicators = agregar_indicadores(df.copy(), config=indicator_config)

            # 3.3. Generar señales con la estrategia Stochastic Crossover v27
            strategy_config = {
                'ema_filter_periodo': params['ema_filter_periodo'],
                'stoch_k': params['stoch_k'],
                'stoch_d': params['stoch_d'],
                'stoch_smooth': params['stoch_smooth']
            }
            df_signals = generar_senales_stoch_crossover_v27(df_indicators, config=strategy_config)

            # 3.4. Ejecutar backtest con Stop Loss y Take Profit
            backtester = VectorizedBacktester(
                df=df_signals,
                initial_capital=10000,
                commission=0.00075,  # 0.075% comisión Binance
                slippage=0.0005      # 0.05% slippage
            )

            # Ejecutar backtest con SL y TP parametrizables
            backtester.run_backtest_with_sl_tp(
                atr_column='ATRr_14',
                sl_multiplier=params['sl_multiplier'],
                tp_multiplier=params['tp_multiplier']
            )

            metrics = backtester.calculate_metrics()

            # 3.5. Guardar resultados
            result_row = {
                'ema_filter_periodo': params['ema_filter_periodo'],
                'stoch_k': params['stoch_k'],
                'stoch_d': params['stoch_d'],
                'stoch_smooth': params['stoch_smooth'],
                'sl_multiplier': params['sl_multiplier'],
                'tp_multiplier': params['tp_multiplier'],
                'profit_factor': metrics.get('profit_factor', 0),
                'num_trades': metrics.get('num_trades', 0),
                'win_rate_pct': metrics.get('win_rate_pct', 0),
                'total_return_pct': metrics.get('total_return_pct', 0),
                'sharpe_ratio': metrics.get('sharpe_ratio', 0),
                'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
                'final_value': metrics.get('final_value', 0),
                'annual_return_pct': metrics.get('annual_return_pct', 0)
            }

            results.append(result_row)

        except Exception as e:
            logger.error(f"   ✗ Error en combinación {idx}: {e}")
            continue

    # ========================================
    # 4. GUARDAR RESULTADOS
    # ========================================
    logger.info("\n4. Guardando resultados completos...")

    df_results = pd.DataFrame(results)
    output_file = project_root / 'backtest_results_eth_v27_stoch_5m.csv'
    df_results.to_csv(output_file, index=False)

    logger.info(f"   ✓ Resultados guardados: {output_file.name}")
    logger.info(f"   ✓ Total de combinaciones evaluadas: {len(df_results)}")

    # ========================================
    # 5. REPORTE FINAL
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("REPORTE FINAL: CRITERIOS DE ÉXITO")
    logger.info("=" * 80)

    # Verificar si hay resultados
    if len(df_results) == 0:
        logger.error("\n✗ No se pudieron evaluar ninguna de las combinaciones.")
        logger.info("   Por favor revisa los errores arriba.")
        return

    # Filtrar combinaciones que cumplen AMBOS criterios
    success_criteria = (df_results['profit_factor'] > 1.1) & (df_results['num_trades'] > 150)
    df_success = df_results[success_criteria].sort_values('profit_factor', ascending=False)

    logger.info("\nCombinaciones que cumplen AMBOS criterios:")
    logger.info("  - Profit Factor > 1.1: ✓")
    logger.info("  - Num Trades > 150: ✓")
    logger.info(f"  - Total encontradas: {len(df_success)}")

    if len(df_success) > 0:
        logger.info("\n" + "✓" * 40 + " ITERACIÓN 27 APROBADA " + "✓" * 40)
        logger.info("\n🎉 ¡ÉXITO! Hemos encontrado configuraciones rentables en 5m")
        logger.info("\nTop 10 configuraciones que cumplen AMBOS criterios:")
        logger.info("-" * 80)

        top_10 = df_success.head(10)
        display_cols = [
            'ema_filter_periodo', 'stoch_k', 'stoch_d', 'stoch_smooth',
            'sl_multiplier', 'tp_multiplier', 'profit_factor', 'num_trades',
            'win_rate_pct', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct'
        ]
        logger.info(top_10[display_cols].to_string(index=False))

        logger.info("\n" + "=" * 80)
        logger.info("MEJOR CONFIGURACIÓN ENCONTRADA:")
        logger.info("=" * 80)
        best = df_success.iloc[0]
        logger.info(f"EMA Filtro Período: {best['ema_filter_periodo']:.0f}")
        logger.info(f"Stochastic K/D/Smooth: {best['stoch_k']:.0f}/{best['stoch_d']:.0f}/{best['stoch_smooth']:.0f}")
        logger.info(f"SL Multiplier: {best['sl_multiplier']:.1f}x ATR")
        logger.info(f"TP Multiplier: {best['tp_multiplier']:.1f}x ATR")
        logger.info(f"")
        logger.info(f"Profit Factor: {best['profit_factor']:.2f}")
        logger.info(f"Número de Trades: {best['num_trades']:.0f}")
        logger.info(f"Win Rate: {best['win_rate_pct']:.2f}%")
        logger.info(f"Retorno Total: {best['total_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {best['max_drawdown_pct']:.2f}%")

    else:
        logger.info("\n" + "✗" * 40 + " ITERACIÓN 27 SIN APROBACIÓN " + "✗" * 40)
        logger.info("No se encontraron configuraciones que cumplan AMBOS criterios.")
        logger.info("\n" + "🔴" * 40)
        logger.info("CONCLUSIÓN CRÍTICA:")
        logger.info("🔴" * 40)
        logger.info("\nHemos agotado las CUATRO familias principales de estrategias en 5m:")
        logger.info("  1. ✗ v24 - Donchian Breakout (PF: 0.78)")
        logger.info("  2. ✗ v25 - EMA Pullback (PF: 0.81)")
        logger.info("  3. ✗ v26 - MACD Crossover (PF: 0.80)")
        logger.info("  4. ✗ v27 - Stochastic Crossover (evaluado)")
        logger.info("\nTODAS las estrategias generan alta frecuencia (>150 trades) pero")
        logger.info("NINGUNA es rentable (todas PF < 1.0 = pérdidas sistemáticas).")

        # Análisis separado para diagnóstico
        logger.info("\n" + "-" * 80)
        logger.info("ANÁLISIS SEPARADO:")
        logger.info("-" * 80)

        # Top 10 por Profit Factor (sin filtro de trades)
        logger.info("\nTop 10 por Profit Factor (sin filtro de trades):")
        top_pf = df_results.sort_values('profit_factor', ascending=False).head(10)
        display_cols = [
            'ema_filter_periodo', 'stoch_k', 'stoch_d', 'stoch_smooth',
            'sl_multiplier', 'tp_multiplier', 'profit_factor', 'num_trades',
            'win_rate_pct', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct'
        ]
        logger.info(top_pf[display_cols].to_string(index=False))

        # Top 10 con alta frecuencia (>150 trades)
        logger.info("\nTop 10 con alta frecuencia (> 150 trades):")
        high_freq = df_results[df_results['num_trades'] > 150].sort_values('profit_factor', ascending=False).head(10)
        if len(high_freq) > 0:
            logger.info(high_freq[display_cols].to_string(index=False))
        else:
            logger.info("   ✗ Ninguna configuración generó más de 150 trades")

        # Trade-off análisis
        logger.info("\n" + "-" * 80)
        logger.info("TRADE-OFF: ¿Qué tan cerca estamos?")
        logger.info("-" * 80)

        # Configuraciones con PF > 1.1 (sin considerar frecuencia)
        df_pf_ok = df_results[df_results['profit_factor'] > 1.1]
        logger.info(f"\nConfiguraciones con PF > 1.1: {len(df_pf_ok)}")
        if len(df_pf_ok) > 0:
            logger.info(f"   - Mejor num_trades entre ellas: {df_pf_ok['num_trades'].max():.0f}")
            logger.info(f"   - Necesitamos: > 150 trades")

        # Configuraciones con > 150 trades (sin considerar PF)
        df_trades_ok = df_results[df_results['num_trades'] > 150]
        logger.info(f"\nConfiguraciones con > 150 trades: {len(df_trades_ok)}")
        if len(df_trades_ok) > 0:
            logger.info(f"   - Mejor PF entre ellas: {df_trades_ok['profit_factor'].max():.2f}")
            logger.info(f"   - Necesitamos: PF > 1.1")

        logger.info("\n" + "=" * 80)
        logger.info("RECOMENDACIÓN FINAL:")
        logger.info("=" * 80)
        logger.info("\nEl timeframe de 5 minutos es demasiado ruidoso para ETHUSDT.")
        logger.info("Próximos pasos:")
        logger.info("  1. PIVOTAR a timeframe 15m (menos ruido, señales más confiables)")
        logger.info("  2. Evaluar BTC en lugar de ETH (mayor liquidez)")
        logger.info("  3. Considerar estrategias Mean Reversion (RSI)")
        logger.info("  4. Aceptar que Day Trading agresivo (5m) NO es viable con este sistema")

    # ========================================
    # 6. CONCLUSIÓN
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("EJECUCIÓN COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"\nResultados guardados en: {output_file.name}")

    if len(df_success) > 0:
        logger.info("\nPróximos pasos:")
        logger.info("  1. Implementar la mejor configuración en paper trading (Fase 3)")
        logger.info("  2. Validar en datos out-of-sample")
        logger.info("  3. Preparar para trading en vivo (Fase 4)")
    else:
        logger.info("\nPróximos pasos:")
        logger.info("  1. Revisar el CSV completo para análisis detallado")
        logger.info("  2. Cambiar a timeframe 15m para la próxima iteración")
        logger.info("  3. Re-evaluar el objetivo de Day Trading")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠ Operación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
