#!/usr/bin/env python3
"""
FASE 2: Backtesting y Optimización - ITERACIÓN 13 (MÓDULO BAJISTA v1)
Ejecuta backtests con ESTRATEGIA BAJISTA DE 4 CAPAS (SHORT-ONLY).

PIVOTE ESTRATÉGICO CRÍTICO:
- Iteración 10.1 RECHAZADA: Estocástico Long → Win Rate 0%
- Iteración 11.1 RECHAZADA: Donchian Long → Win Rate 5%, baja frecuencia
- Iteración 12 RECHAZADA: Híbrido Long → Win Rate 27.51%, Sharpe -0.12, Return -33.30%
- Iteración 13 NUEVA: INVERSIÓN TOTAL DE LÓGICA (SHORT-ONLY)

Hipótesis: Las señales de compra fallidas eran, de hecho, señales de venta.
Probaremos si operar en corto (ventas en corto) produce rentabilidad positiva.

Estrategia de 4 Capas (SHORT-ONLY):
- CAPA 1: Filtro de régimen con EMA(200) - Solo opera en tendencia BAJISTA (precio < EMA_200) ⚠️ INVERTIDO
- CAPA 2: Filtro de momentum con RSI(14) - Solo entra con momentum BAJISTA (RSI < nivel) ⚠️ INVERTIDO
- CAPA 3: Señal de entrada (VENTA EN CORTO) con cruce BAJISTA MACD ⚠️ INVERTIDO
- CAPA 3: Señal de salida (CUBRIR CORTO) con cruce ALCISTA MACD ⚠️ INVERTIDO
- CAPA 4: Gestión de riesgo con Stop Loss ATR POR ENCIMA del entry_price ⚠️ INVERTIDO

Stop Loss Dinámico ATR INVERTIDO:
- SL = entry_price + (ATR × atr_multiplier) ⚠️ INVERTIDO (antes era entry - ATR)
- Verificación: Se cierra si df['high'] toca/cruza SL ⚠️ INVERTIDO (antes era df['low'])
- Motor actualizado para soportar posiciones LONG y SHORT simultáneamente

Activo: ETH/USDT
Timeframe: 15m
Dataset: 1 año completo
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.binance_client import BinanceClientManager
from src.backtest.optimizer import optimize_parameters, save_optimal_params
from src.utils.metrics import print_metrics


def main():
    print("=" * 80)
    print("FASE 2: BACKTESTING Y OPTIMIZACIÓN - ITERACIÓN 13 (BAJISTA v1)")
    print("ESTRATEGIA: 4 Capas SHORT-ONLY (Régimen Bajista + RSI + MACD + ATR)")
    print("=" * 80)

    # Crear cliente
    print("\n1. Conectando a Binance...")
    manager = BinanceClientManager()
    client = manager.get_public_client()
    print("   ✓ Cliente creado")

    # Definir grid de parámetros para ESTRATEGIA BAJISTA
    # ITERACIÓN 13 - Módulo Bajista v1 (SHORT-ONLY)
    print("\n2. Definiendo grid de parámetros a optimizar...")
    param_grid = {
        # CAPA 1: Filtro de Régimen BAJISTA (EMA de Tendencia) ⚠️ INVERTIDO
        'ema_trend': [200],  # EMA 200 como filtro de tendencia BAJISTA (precio < EMA)

        # CAPA 2: Filtro de Momentum BAJISTA (RSI) - PARÁMETRO A OPTIMIZAR ⚠️ INVERTIDO
        'rsi_period': [14],                         # RSI estándar
        'rsi_momentum_level': [55, 50, 45],        # Nivel MÁXIMO de RSI para entrada SHORT (3 valores) ⚠️ INVERTIDO

        # CAPA 3: Señal de Entrada/Salida (MACD) ⚠️ INVERTIDO
        'macd_fast': [12],      # MACD rápido (estándar)
        'macd_slow': [26],      # MACD lento (estándar)
        'macd_signal': [9],     # MACD señal (estándar)

        # CAPA 4: Stop Loss ATR INVERTIDO (Gestión de Riesgo) - PARÁMETRO A OPTIMIZAR
        'atr_length': [14],                        # ATR estándar
        'atr_multiplier': [1.5, 2.0, 2.5],        # Multiplicador para SL a optimizar (3 valores)

        # Parámetros para compatibilidad con cálculo de indicadores
        # (no se usan en la estrategia bajista pero son necesarios para agregar_indicadores)
        'bb_length': [20],
        'bb_std': [2],
        'stoch_k': [14],
        'stoch_d': [3],
        'stoch_smooth': [3],
        'donchian_period': [20]  # No se usa en bajista, pero necesario para agregar_indicadores
    }

    from sklearn.model_selection import ParameterGrid
    total_combinations = len(list(ParameterGrid(param_grid)))
    print(f"   ✓ Total de combinaciones: {total_combinations}")
    print(f"   ✓ Parámetros a optimizar:")
    print(f"      - rsi_momentum_level: {param_grid['rsi_momentum_level']} (⚠️ INVERTIDO: RSI < nivel)")
    print(f"      - atr_multiplier: {param_grid['atr_multiplier']}")

    # Ejecutar optimización con ESTRATEGIA BAJISTA + STOP LOSS ATR INVERTIDO
    print("\n3. Ejecutando optimización (esto puede tardar varios minutos)...")
    print("   ACTIVO: ETH/USDT")
    print("   TIMEFRAME: 15 minutos")
    print("   DATASET: 1 año completo (365 días)")
    print("   ESTRATEGIA: Bajista 4 Capas SHORT-ONLY (precio < EMA_200 + RSI < nivel + MACD bajista)")
    print("   STOP LOSS: ACTIVADO (dinámico basado en ATR, INVERTIDO: entry + ATR, verifica df['high'])")
    print("=" * 80)

    results = optimize_parameters(
        client=client,
        param_grid=param_grid,
        symbol='ETHUSDT',  # ETH/USDT según especificación Iteración 13
        interval='15m',     # Timeframe 15 minutos para day trading
        start_date='365 days ago UTC',  # 1 año completo de datos históricos
        signal_function='generar_senales_bajista_v1',  # NUEVA FUNCIÓN BAJISTA SHORT-ONLY (ITERACIÓN 13)
        use_stop_loss=True  # MANTENER STOP LOSS DINÁMICO BASADO EN ATR (ahora con soporte SHORT)
    )

    # Guardar TODOS los resultados en CSV para auditoría
    print("\n4. Guardando resultados completos del Grid Search...")
    results_file = 'backtest_results_eth_v13_bajista.csv'  # ITERACIÓN 13 (BAJISTA SHORT-ONLY)
    results.to_csv(results_file, index=False)
    print(f"   ✓ Resultados guardados en: {results_file}")
    print(f"   ✓ Total de combinaciones evaluadas: {len(results)}")

    # Mostrar resultados
    print("\n" + "=" * 80)
    print("TOP 5 MEJORES CONFIGURACIONES")
    print("=" * 80)

    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    top5 = results.head(5)
    print("\nResumen:")
    print(top5[['ema_trend', 'rsi_momentum_level', 'atr_multiplier',
                'sharpe_ratio', 'total_return_pct', 'max_drawdown_pct', 'win_rate_pct', 'profit_factor', 'num_trades']].to_string(index=False))

    # Mostrar métricas detalladas del mejor
    print("\n" + "=" * 80)
    print("MÉTRICAS DETALLADAS DE LA MEJOR CONFIGURACIÓN")
    print("=" * 80)

    best_result = results.iloc[0].to_dict()
    print(f"\nParámetros Optimizados - Estrategia Bajista (SHORT-ONLY):")
    print(f"  CAPA 1 (Régimen BAJISTA) ⚠️ INVERTIDO:")
    print(f"    - EMA Tendencia: {int(best_result['ema_trend'])} períodos (Solo opera si precio < EMA)")
    print(f"  CAPA 2 (Momentum BAJISTA) ⚠️ INVERTIDO:")
    print(f"    - RSI: {int(best_result['rsi_period'])} períodos")
    print(f"    - RSI Nivel MÁXIMO: {int(best_result['rsi_momentum_level'])} (Solo opera si RSI < nivel)")
    print(f"  CAPA 3 (Señal VENTA EN CORTO) ⚠️ INVERTIDO:")
    print(f"    - MACD: {int(best_result['macd_fast'])}/{int(best_result['macd_slow'])}/{int(best_result['macd_signal'])}")
    print(f"    - Entrada: Cruce BAJISTA (MACD < Signal)")
    print(f"    - Salida: Cruce ALCISTA (MACD > Signal)")
    print(f"  CAPA 4 (Riesgo SHORT) ⚠️ INVERTIDO:")
    print(f"    - Stop Loss ATR: entry + ({int(best_result['atr_length'])} períodos × {best_result['atr_multiplier']}x)")
    print(f"    - Verificación: df['high'] >= SL")

    # Convertir a formato de métricas
    metrics = {k: v for k, v in best_result.items()
               if k in ['initial_capital', 'final_value', 'net_profit', 'total_return_pct',
                       'annual_return_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                       'max_drawdown_pct', 'num_trades', 'win_rate_pct', 'profit_factor',
                       'avg_trade', 'best_trade', 'worst_trade']}

    print_metrics(metrics)

    # Guardar parámetros óptimos para ESTRATEGIA HÍBRIDA
    print("\n5. Guardando parámetros óptimos...")
    best_params = results.iloc[0][['ema_trend', 'rsi_period', 'rsi_momentum_level',
                                     'macd_fast', 'macd_slow', 'macd_signal',
                                     'atr_length', 'atr_multiplier',
                                     'bb_length', 'bb_std',
                                     'stoch_k', 'stoch_d', 'stoch_smooth',
                                     'donchian_period']].to_dict()

    # Convertir a int/float según corresponda
    for key in best_params:
        if key in ['bb_std', 'atr_multiplier']:
            best_params[key] = float(best_params[key])
        elif isinstance(best_params[key], (float, int)):
            best_params[key] = int(best_params[key])

    save_optimal_params(best_params)

    print("\n" + "=" * 80)
    print("✓ FASE 2 COMPLETADA EXITOSAMENTE - ITERACIÓN 13 (BAJISTA v1 - SHORT-ONLY)")
    print("=" * 80)
    print("\nResultados:")
    print(f"  - CSV completo: {results_file} ({len(results)} combinaciones)")
    print(f"  - Parámetros óptimos: config/optimal_params.json")
    print(f"  - Estrategia: Bajista 4 Capas SHORT-ONLY (precio < EMA_200 + RSI < nivel + MACD bajista)")
    print(f"  - Stop Loss ATR: ACTIVADO (INVERTIDO: entry + ATR, verifica df['high'])")
    print(f"  - Motor de Backtest: ACTUALIZADO para soportar posiciones LONG y SHORT")
    print("\nComparación con Iteraciones Anteriores (TODAS LONG-ONLY):")
    print("  - Iteración 10.1 (Estocástico Long): Win Rate 0%, Sharpe -0.14 a -0.26")
    print("  - Iteración 11.1 (Donchian Long): Win Rate 5%, Sharpe 0.03")
    print("  - Iteración 12 (Híbrido Long): Win Rate 27.51%, Sharpe -0.12, Return -33.30%")
    print("  - Iteración 13 (Bajista SHORT): ¿Hipótesis correcta? ¿Señales invertidas rentables?")
    print("\nPróximos pasos:")
    print("  - Auditoría del Quant-Auditor: N° Trades, Win Rate >35%, Profit Factor >1.0")
    print("  - Verificar correcta implementación de lógica SHORT (SL invertido, df['high'])")
    print("  - Si aprueba auditoría → Fase 3: python scripts/phase3_paper.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Operación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
