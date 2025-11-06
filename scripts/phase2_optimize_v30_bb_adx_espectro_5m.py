#!/usr/bin/env python3
"""
ITERACIÓN 30: BB BREAKOUT CON TRIPLE FILTRO (ADX + EMA ESPECTRO) EN TIMEFRAME 5M
==================================================================================

CONTEXTO - PRUEBA DEFINITIVA DEL EDGE:
Las iteraciones v24-v28 fracasaron consistentemente en 5m:
  * v24 (Donchian): PF 0.78 ✗
  * v25 (EMA Pullback): PF 0.81 ✗
  * v26 (MACD Cross): PF 0.80 ✗
  * v27 (Stochastic): PF 0.80 ✗
  * v28 (BB + Multi-EMA): PF 0.74-0.85 ✗

DIAGNÓSTICO FINAL:
El problema NO es solo el filtro EMA, sino la falta de filtro de MOMENTUM.
Resultado común de v24-v28:
- Todas generan >150 trades (alta frecuencia ✓)
- Todas tienen PF < 1.0 (pérdidas sistemáticas ✗)
- Retornos: -75% a -97% (whipsaws en consolidaciones)

HIPÓTESIS V30 (ÚLTIMA PRUEBA CON INDICADORES TÉCNICOS):
ADX > 15 filtra consolidaciones laterales (principal causa de whipsaws),
permitiendo operar SOLO cuando hay momentum real.

INNOVACIÓN - TRIPLE FILTRO:
1. **ADX > 15:** Filtra mercado lateral (solo opera cuando hay tendencia)
2. **EMA [21, 50, 100, 150, 200]:** Valida dirección de micro-tendencia
3. **BB Breakout:** Detecta expansión de volatilidad

ESTRATEGIA: BB BREAKOUT + ADX FILTER + ESPECTRO EMA COMPLETO
El ADX es la pieza faltante que eliminará los whipsaws laterales.

COMPRA:
  1. ADX[t] > 15 (Momentum confirmado - NO lateral)
  2. Precio[t] > EMA_Filtro[t] (Micro-tendencia alcista)
  3. Precio[t] cruza por encima de BB_Upper (Breakout alcista)

VENTA:
  1. ADX[t] > 15 (Momentum confirmado - NO lateral)
  2. Precio[t] < EMA_Filtro[t] (Micro-tendencia bajista)
  3. Precio[t] cruza por debajo de BB_Lower (Breakout bajista)

PARÁMETROS A OPTIMIZAR (Grid Search):
- ema_filter_periodo: [21, 50, 100, 150, 200] (Espectro completo)
- bb_length: [20] (Estándar)
- bb_std: [2.0] (Estándar)
- adx_period: [14] (Estándar)
- adx_threshold: [15] (Filtro de no-consolidación fijo)
- sl_multiplier: [3.0, 4.0]
- tp_multiplier: [3.0, 4.0]

Total de combinaciones: 5 × 1 × 1 × 1 × 1 × 2 × 2 = 20 configuraciones

CRITERIOS DE ÉXITO (AJUSTADOS):
- Profit Factor > 1.1 (rentabilidad mínima)
- Num Trades > 100 (reducido debido al filtro ADX que elimina señales laterales)
- Ambos deben cumplirse simultáneamente

DECISIÓN FINAL:
Si encontramos PF > 1.1 → ¡ÉXITO! Habremos encontrado el EDGE para Day Trading
Si NO encontramos PF > 1.1 → El objetivo de Day Trading en 5m NO es viable

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
from src.strategy.signal_generator import generar_senales_bb_adx_filter_v30
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger("phase2_optimize_v30", "logs/phase2_optimize_v30.log")


def main():
    logger.info("=" * 80)
    logger.info("ITERACIÓN 30: BB + TRIPLE FILTRO (ADX + EMA ESPECTRO) EN 5M")
    logger.info("=" * 80)
    logger.info("\nHipótesis: ADX > 15 elimina whipsaws → PRUEBA DEFINITIVA DEL EDGE")
    logger.info("Objetivo: PF > 1.1 AND Num Trades > 100")

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
    logger.info("\n2. Definiendo Grid de Parámetros (Triple Filtro: ADX + BB + EMA)...")

    param_grid = {
        'ema_filter_periodo': [21, 50, 100, 150, 200],  # Espectro completo
        'bb_length': [20],
        'bb_std': [2.0],
        'adx_period': [14],
        'adx_threshold': [15],
        'sl_multiplier': [3.0, 4.0],
        'tp_multiplier': [3.0, 4.0]
    }

    grid = list(ParameterGrid(param_grid))
    logger.info(f"   ✓ Total de combinaciones: {len(grid)}")
    logger.info(f"   ✓ Parámetros:")
    for key, values in param_grid.items():
        logger.info(f"      - {key}: {values}")
    logger.info("\n   🔑 INNOVACIÓN: Filtro ADX > 15 para eliminar whipsaws laterales")

    # ========================================
    # 3. EJECUTAR GRID SEARCH
    # ========================================
    logger.info("\n3. Ejecutando Grid Search...")
    logger.info("   (Probando Triple Filtro en espectro completo de EMAs)\n")

    results = []
    total = len(grid)

    for idx, params in enumerate(grid, start=1):
        if idx % 5 == 1:
            logger.info(f"   Evaluando combinación {idx}/{total}...")

        try:
            # 3.1. Configurar indicadores según los parámetros actuales
            indicator_config = {
                'atr_length': 14,
                'bb_length': int(params['bb_length']),
                'bb_std': params['bb_std'],
                'adx_period': int(params['adx_period'])
            }

            # Añadir EMA de filtro dinámicamente según el período
            ema_periodo = params['ema_filter_periodo']
            if ema_periodo == 21:
                indicator_config['ema_short'] = 21
            elif ema_periodo == 50:
                indicator_config['ema_long'] = 50
            elif ema_periodo == 100:
                indicator_config['ema_filter'] = 100
            elif ema_periodo == 150:
                indicator_config['ema_filter'] = 150
            elif ema_periodo == 200:
                indicator_config['ema_trend'] = 200

            # 3.2. Calcular indicadores
            df_indicators = agregar_indicadores(df.copy(), config=indicator_config)

            # 3.3. Generar señales con la estrategia BB + ADX Filter v30
            strategy_config = {
                'ema_filter_periodo': params['ema_filter_periodo'],
                'bb_length': int(params['bb_length']),
                'bb_std': params['bb_std'],
                'adx_period': int(params['adx_period']),
                'adx_threshold': params['adx_threshold']
            }
            df_signals = generar_senales_bb_adx_filter_v30(df_indicators, config=strategy_config)

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
                'bb_length': params['bb_length'],
                'bb_std': params['bb_std'],
                'adx_period': params['adx_period'],
                'adx_threshold': params['adx_threshold'],
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
    output_file = project_root / 'backtest_results_eth_v30_bb_adx_espectro_5m.csv'
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
    success_criteria = (df_results['profit_factor'] > 1.1) & (df_results['num_trades'] > 100)
    df_success = df_results[success_criteria].sort_values('profit_factor', ascending=False)

    logger.info("\nCombinaciones que cumplen AMBOS criterios:")
    logger.info("  - Profit Factor > 1.1: ✓")
    logger.info("  - Num Trades > 100: ✓")
    logger.info(f"  - Total encontradas: {len(df_success)}")

    if len(df_success) > 0:
        logger.info("\n" + "🎉" * 40)
        logger.info("¡¡¡ ÉXITO !!! HEMOS ENCONTRADO EL EDGE PARA DAY TRADING EN 5M")
        logger.info("🎉" * 40)

        logger.info("\nTop 10 configuraciones que cumplen AMBOS criterios:")
        logger.info("-" * 80)

        top_10 = df_success.head(10)
        display_cols = [
            'ema_filter_periodo', 'bb_length', 'bb_std', 'adx_threshold',
            'sl_multiplier', 'tp_multiplier', 'profit_factor', 'num_trades',
            'win_rate_pct', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct'
        ]
        logger.info(top_10[display_cols].to_string(index=False))

        logger.info("\n" + "=" * 80)
        logger.info("MEJOR CONFIGURACIÓN ENCONTRADA:")
        logger.info("=" * 80)
        best = df_success.iloc[0]
        logger.info(f"EMA Filtro Período: {best['ema_filter_periodo']:.0f}")
        logger.info(f"Bollinger Bands: BB({best['bb_length']:.0f}, {best['bb_std']:.1f})")
        logger.info(f"ADX Threshold: {best['adx_threshold']:.0f}")
        logger.info(f"SL Multiplier: {best['sl_multiplier']:.1f}x ATR")
        logger.info(f"TP Multiplier: {best['tp_multiplier']:.1f}x ATR")
        logger.info(f"")
        logger.info(f"✅ Profit Factor: {best['profit_factor']:.2f}")
        logger.info(f"✅ Número de Trades: {best['num_trades']:.0f}")
        logger.info(f"✅ Win Rate: {best['win_rate_pct']:.2f}%")
        logger.info(f"✅ Retorno Total: {best['total_return_pct']:.2f}%")
        logger.info(f"✅ Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        logger.info(f"✅ Max Drawdown: {best['max_drawdown_pct']:.2f}%")

        logger.info("\n" + "=" * 80)
        logger.info("ANÁLISIS: ¿CUÁL FILTRO EMA FUNCIONÓ MEJOR?")
        logger.info("=" * 80)
        logger.info("\nComparación por tipo de filtro EMA:")

        for ema_period in [21, 50, 100, 150, 200]:
            ema_configs = df_results[df_results['ema_filter_periodo'] == ema_period]
            if len(ema_configs) > 0:
                best_pf = ema_configs['profit_factor'].max()
                avg_trades = ema_configs['num_trades'].mean()
                profitable = len(ema_configs[ema_configs['profit_factor'] > 1.1])
                logger.info(f"  EMA {ema_period:3d}: PF máx = {best_pf:.2f}, Trades promedio = {avg_trades:.0f}, Rentables = {profitable}/4")

        logger.info("\n" + "=" * 80)
        logger.info("CONCLUSIÓN:")
        logger.info("=" * 80)
        logger.info("\n✅ El filtro ADX > 15 fue la pieza faltante.")
        logger.info("✅ Eliminar consolidaciones laterales mejoró drásticamente el PF.")
        logger.info("✅ Day Trading en 5m ES VIABLE con el Triple Filtro.")

    else:
        logger.info("\n" + "❌" * 40)
        logger.info("ITERACIÓN 30 SIN APROBACIÓN - FIN DE LA EXPERIMENTACIÓN EN 5M")
        logger.info("❌" * 40)

        logger.info("\nEl filtro ADX NO resolvió el problema de whipsaws.")
        logger.info("\nHemos probado TODO en 5m:")
        logger.info("  1. ✗ Breakout (Donchian) - PF 0.78")
        logger.info("  2. ✗ Pullback (EMA) - PF 0.81")
        logger.info("  3. ✗ Momentum Cross (MACD) - PF 0.80")
        logger.info("  4. ✗ Oscillator (Stochastic) - PF 0.80")
        logger.info("  5. ✗ Volatility (BB + Multi-EMA) - PF 0.74-0.85")
        logger.info("  6. ✗ Triple Filter (BB + ADX + EMA) - evaluado")

        # Análisis separado
        logger.info("\n" + "-" * 80)
        logger.info("ANÁLISIS POR FILTRO EMA (con ADX > 15):")
        logger.info("-" * 80)

        for ema_period in [21, 50, 100, 150, 200]:
            ema_configs = df_results[df_results['ema_filter_periodo'] == ema_period]
            if len(ema_configs) > 0:
                best = ema_configs.sort_values('profit_factor', ascending=False).iloc[0]
                logger.info(f"\n  EMA {ema_period}:")
                logger.info(f"    Mejor PF: {best['profit_factor']:.2f}")
                logger.info(f"    Num Trades: {best['num_trades']:.0f}")
                logger.info(f"    Win Rate: {best['win_rate_pct']:.2f}%")
                logger.info(f"    Retorno: {best['total_return_pct']:.2f}%")

        # Top 10 global
        logger.info("\n" + "-" * 80)
        logger.info("Top 10 configuraciones por Profit Factor:")
        logger.info("-" * 80)
        top_pf = df_results.sort_values('profit_factor', ascending=False).head(10)
        display_cols = [
            'ema_filter_periodo', 'adx_threshold', 'sl_multiplier', 'tp_multiplier',
            'profit_factor', 'num_trades', 'win_rate_pct', 'total_return_pct'
        ]
        logger.info(top_pf[display_cols].to_string(index=False))

        # Impacto del ADX
        logger.info("\n" + "-" * 80)
        logger.info("IMPACTO DEL FILTRO ADX:")
        logger.info("-" * 80)
        logger.info("\nComparación con v28 (sin ADX):")
        logger.info("  v28 (sin ADX): PF máximo = 0.85")
        logger.info(f"  v30 (con ADX): PF máximo = {df_results['profit_factor'].max():.2f}")
        diff = df_results['profit_factor'].max() - 0.85
        logger.info(f"  Mejora: {diff:+.2f} puntos de PF")

        if diff > 0:
            logger.info("\n  ✓ El ADX mejoró el PF, pero NO fue suficiente para PF > 1.1")
        else:
            logger.info("\n  ✗ El ADX NO mejoró el PF (incluso empeoró)")

        logger.info("\n" + "=" * 80)
        logger.info("CONCLUSIÓN DEFINITIVA:")
        logger.info("=" * 80)
        logger.info("\n⛔ Day Trading en 5m NO es viable para ETHUSDT con indicadores técnicos.")
        logger.info("\nPRÓXIMOS PASOS OBLIGATORIOS:")
        logger.info("  1. 🔄 Pivotar a timeframe 15m (Iteración 31)")
        logger.info("  2. 🪙 Evaluar BTC en lugar de ETH")
        logger.info("  3. 🤖 Considerar Machine Learning / AI")
        logger.info("  4. 📊 Estrategias adaptativas (no estáticas)")

    # ========================================
    # 6. CONCLUSIÓN
    # ========================================
    logger.info("\n" + "=" * 80)
    logger.info("EJECUCIÓN COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"\nResultados guardados en: {output_file.name}")

    if len(df_success) > 0:
        logger.info("\nPróximos pasos:")
        logger.info("  1. Validar en datos out-of-sample")
        logger.info("  2. Implementar en paper trading (Fase 3)")
        logger.info("  3. Preparar para trading en vivo (Fase 4)")
    else:
        logger.info("\nPróximos pasos:")
        logger.info("  1. Implementar Iteración 31 en timeframe 15m")
        logger.info("  2. Abandonar definitivamente el objetivo de 5m")


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
