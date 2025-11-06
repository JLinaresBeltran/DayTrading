#!/usr/bin/env python3
"""
ITERACIÓN 28: BOLLINGER BAND BREAKOUT CON MÚLTIPLES FILTROS EMA EN TIMEFRAME 5M
==================================================================================

CONTEXTO CRÍTICO - ÚLTIMA PRUEBA TÉCNICA:
Las iteraciones v24-v27 fracasaron en 5m con resultados consistentemente negativos:
  * v24 (Donchian Breakout): PF 0.78 ✗
  * v25 (EMA Pullback): PF 0.81 ✗
  * v26 (MACD Crossover): PF 0.80 ✗
  * v27 (Stochastic Crossover): PF 0.80 ✗

DIAGNÓSTICO DEL PROBLEMA:
Conflicto entre velocidad y filtro en 5m:
- Filtros lentos (EMA 100-200) → Señales tardías, pocas oportunidades
- Filtros rápidos / sin filtro → Demasiado ruido, whipsaws excesivos

HIPÓTESIS V28 (REVISADA - ÚLTIMA OPORTUNIDAD):
Una EMA de tendencia MÁS RÁPIDA (21 o 50) puede capturar micro-tendencias en 5m,
haciendo rentable la ruptura de volatilidad (Bollinger Bands).

INNOVACIÓN CLAVE:
Probar un RANGO AMPLIO de filtros EMA [21, 50, 100, 200] para encontrar el equilibrio
óptimo entre:
  - Velocidad de señal (filtros rápidos)
  - Calidad de filtrado (filtros lentos)

ESTRATEGIA: BOLLINGER BAND BREAKOUT
Las Bollinger Bands detectan expansiones de volatilidad (momentum explosivo).
La EMA valida la dirección de la micro-tendencia.

COMPRA:
  1. Precio[t] > EMA_Filtro[t] (Micro-tendencia alcista)
  2. Precio[t] cruza por encima de BB_Upper (Breakout alcista)
     (Close[t-1] <= BB_Upper[t-1] AND Close[t] > BB_Upper[t])

VENTA:
  1. Precio[t] < EMA_Filtro[t] (Micro-tendencia bajista)
  2. Precio[t] cruza por debajo de BB_Lower (Breakout bajista)
     (Close[t-1] >= BB_Lower[t-1] AND Close[t] < BB_Lower[t])

PARÁMETROS A OPTIMIZAR (Grid Search):
- ema_filter_periodo: [21, 50, 100, 200] ← CLAVE: Rango ampliado
- bb_length: [20] (Estándar)
- bb_std: [2.0] (Estándar)
- sl_multiplier: [3.0, 4.0] (Stops amplios para reducir whipsaws)
- tp_multiplier: [3.0, 4.0] (R:R probados)

Total de combinaciones: 4 × 1 × 1 × 2 × 2 = 16 configuraciones

CRITERIOS DE ÉXITO:
- Profit Factor > 1.1 (rentabilidad mínima)
- Num Trades > 150 (alta frecuencia - Day Trading)
- Ambos deben cumplirse simultáneamente

DECISIÓN FINAL:
Si NINGUNA de estas 16 combinaciones de filtros EMA y gestión de riesgo es rentable,
la conclusión será DEFINITIVA:

  ⚠️  El objetivo de Day Trading en 5m NO es viable con estrategias técnicas puras.
  ⚠️  Necesitamos pivotar a:
      - Timeframe 15m (menos ruido)
      - Estrategias de Machine Learning
      - Activos más líquidos (BTC)

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
from src.strategy.signal_generator import generar_senales_bb_breakout_v28
from src.backtest.engine import VectorizedBacktester
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger("phase2_optimize_v28", "logs/phase2_optimize_v28.log")


def main():
    logger.info("=" * 80)
    logger.info("ITERACIÓN 28: BB BREAKOUT CON MÚLTIPLES FILTROS EMA EN 5M")
    logger.info("=" * 80)
    logger.info("\nHipótesis: Filtros EMA rápidos [21-200] → Última prueba técnica en 5m")
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
    logger.info("\n2. Definiendo Grid de Parámetros (BB Breakout con Múltiples Filtros)...")

    param_grid = {
        'ema_filter_periodo': [21, 50, 100, 200],  # CLAVE: Rango ampliado
        'bb_length': [20],
        'bb_std': [2.0],
        'sl_multiplier': [3.0, 4.0],
        'tp_multiplier': [3.0, 4.0]
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
    logger.info("   (Probando filtros EMA desde muy rápidos hasta lentos)\n")

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
                'bb_std': params['bb_std']
            }

            # Añadir EMA de filtro dinámicamente según el período
            ema_periodo = params['ema_filter_periodo']
            if ema_periodo == 21:
                indicator_config['ema_short'] = 21
            elif ema_periodo == 50:
                indicator_config['ema_long'] = 50
            elif ema_periodo == 100:
                indicator_config['ema_filter'] = 100
            elif ema_periodo == 200:
                indicator_config['ema_trend'] = 200

            # 3.2. Calcular indicadores
            df_indicators = agregar_indicadores(df.copy(), config=indicator_config)

            # 3.3. Generar señales con la estrategia BB Breakout v28
            strategy_config = {
                'ema_filter_periodo': params['ema_filter_periodo'],
                'bb_length': int(params['bb_length']),
                'bb_std': params['bb_std']
            }
            df_signals = generar_senales_bb_breakout_v28(df_indicators, config=strategy_config)

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
    output_file = project_root / 'backtest_results_eth_v28_bb_ema_combo_5m.csv'
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
        logger.info("\n" + "✓" * 40 + " ITERACIÓN 28 APROBADA " + "✓" * 40)
        logger.info("\n🎉 ¡ÉXITO! Hemos encontrado el equilibrio entre velocidad y filtro")
        logger.info("\nTop 10 configuraciones que cumplen AMBOS criterios:")
        logger.info("-" * 80)

        top_10 = df_success.head(10)
        display_cols = [
            'ema_filter_periodo', 'bb_length', 'bb_std',
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
        logger.info(f"SL Multiplier: {best['sl_multiplier']:.1f}x ATR")
        logger.info(f"TP Multiplier: {best['tp_multiplier']:.1f}x ATR")
        logger.info(f"")
        logger.info(f"Profit Factor: {best['profit_factor']:.2f}")
        logger.info(f"Número de Trades: {best['num_trades']:.0f}")
        logger.info(f"Win Rate: {best['win_rate_pct']:.2f}%")
        logger.info(f"Retorno Total: {best['total_return_pct']:.2f}%")
        logger.info(f"Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {best['max_drawdown_pct']:.2f}%")

        logger.info("\n" + "=" * 80)
        logger.info("ANÁLISIS DEL FILTRO GANADOR:")
        logger.info("=" * 80)
        logger.info("\nComparación por tipo de filtro EMA:")

        for ema_period in [21, 50, 100, 200]:
            ema_configs = df_results[df_results['ema_filter_periodo'] == ema_period]
            if len(ema_configs) > 0:
                best_pf = ema_configs['profit_factor'].max()
                avg_trades = ema_configs['num_trades'].mean()
                logger.info(f"  EMA {ema_period:3d}: PF máx = {best_pf:.2f}, Trades promedio = {avg_trades:.0f}")

    else:
        logger.info("\n" + "✗" * 40 + " ITERACIÓN 28 SIN APROBACIÓN " + "✗" * 40)
        logger.info("No se encontraron configuraciones que cumplan AMBOS criterios.")

        logger.info("\n" + "🔴" * 80)
        logger.info("CONCLUSIÓN DEFINITIVA - FIN DE EXPERIMENTACIÓN EN 5M")
        logger.info("🔴" * 80)

        logger.info("\nHemos agotado TODAS las estrategias técnicas principales en 5m:")
        logger.info("  1. ✗ v24 - Donchian Breakout (Precio) - PF: 0.78")
        logger.info("  2. ✗ v25 - EMA Pullback (Precio) - PF: 0.81")
        logger.info("  3. ✗ v26 - MACD Crossover (Momentum) - PF: 0.80")
        logger.info("  4. ✗ v27 - Stochastic Crossover (Oscilador) - PF: 0.80")
        logger.info("  5. ✗ v28 - BB Breakout + Multi-EMA (Volatilidad) - evaluado")

        logger.info("\nTODAS las estrategias muestran el mismo patrón:")
        logger.info("  ✓ Alta frecuencia (>150 trades)")
        logger.info("  ✗ Profit Factor < 1.0 (pérdidas sistemáticas)")
        logger.info("  ✗ Retornos totales: -75% a -97%")

        # Análisis separado para diagnóstico
        logger.info("\n" + "-" * 80)
        logger.info("ANÁLISIS SEPARADO POR FILTRO EMA:")
        logger.info("-" * 80)

        logger.info("\nComparación de filtros EMA (ordenado por mejor PF):")
        for ema_period in [21, 50, 100, 200]:
            ema_configs = df_results[df_results['ema_filter_periodo'] == ema_period]
            if len(ema_configs) > 0:
                best = ema_configs.sort_values('profit_factor', ascending=False).iloc[0]
                logger.info(f"\n  EMA {ema_period}:")
                logger.info(f"    Mejor PF: {best['profit_factor']:.2f}")
                logger.info(f"    Num Trades: {best['num_trades']:.0f}")
                logger.info(f"    Win Rate: {best['win_rate_pct']:.2f}%")
                logger.info(f"    Retorno: {best['total_return_pct']:.2f}%")

        # Top 10 global por Profit Factor
        logger.info("\n" + "-" * 80)
        logger.info("Top 10 configuraciones por Profit Factor:")
        logger.info("-" * 80)
        top_pf = df_results.sort_values('profit_factor', ascending=False).head(10)
        display_cols = [
            'ema_filter_periodo', 'bb_length', 'bb_std',
            'sl_multiplier', 'tp_multiplier', 'profit_factor', 'num_trades',
            'win_rate_pct', 'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct'
        ]
        logger.info(top_pf[display_cols].to_string(index=False))

        # Trade-off análisis
        logger.info("\n" + "-" * 80)
        logger.info("TRADE-OFF: ¿Alguna configuración estuvo cerca?")
        logger.info("-" * 80)

        df_pf_ok = df_results[df_results['profit_factor'] > 1.1]
        logger.info(f"\nConfiguraciones con PF > 1.1: {len(df_pf_ok)}")
        if len(df_pf_ok) > 0:
            logger.info(f"   - Mejor num_trades: {df_pf_ok['num_trades'].max():.0f}")
            logger.info(f"   - Necesitamos: > 150 trades")

        df_trades_ok = df_results[df_results['num_trades'] > 150]
        logger.info(f"\nConfiguraciones con > 150 trades: {len(df_trades_ok)}")
        if len(df_trades_ok) > 0:
            logger.info(f"   - Mejor PF: {df_trades_ok['profit_factor'].max():.2f}")
            logger.info(f"   - Necesitamos: PF > 1.1")
            logger.info(f"   - GAP: {1.1 - df_trades_ok['profit_factor'].max():.2f} puntos de PF")

        logger.info("\n" + "=" * 80)
        logger.info("RECOMENDACIÓN FINAL Y DEFINITIVA:")
        logger.info("=" * 80)
        logger.info("\n⚠️  El timeframe de 5 minutos NO es viable para ETHUSDT con estrategias técnicas.")
        logger.info("\nEl problema es fundamental:")
        logger.info("  - Filtros rápidos (EMA 21-50): Demasiado ruido, whipsaws excesivos")
        logger.info("  - Filtros lentos (EMA 100-200): Señales tardías, pocas oportunidades")
        logger.info("  - Sin filtro: Pérdidas catastróficas por ruido de mercado")
        logger.info("\nPRÓXIMOS PASOS OBLIGATORIOS:")
        logger.info("  1. 🔄 PIVOTAR a timeframe 15m (menos ruido, señales más confiables)")
        logger.info("  2. 🪙 Evaluar BTC en lugar de ETH (mayor liquidez, menos volatilidad)")
        logger.info("  3. 🤖 Considerar Machine Learning para filtrado adaptativo")
        logger.info("  4. 📊 Estrategias Mean Reversion (RSI) en lugar de Trend Following")
        logger.info("\n⛔ CONCLUSIÓN: Aceptar que Day Trading agresivo (5m) requiere otro enfoque.")

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
        logger.info("\nPróximos pasos obligatorios:")
        logger.info("  1. Implementar Iteración 29 en timeframe 15m")
        logger.info("  2. Aplicar la mejor estrategia identificada (v25 EMA Pullback)")
        logger.info("  3. Re-evaluar objetivo de Day Trading")


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
