#!/usr/bin/env python3
"""
ITERACIÓN 22: Optimización Estrategia v18 Donchian para Day Trading (15m)

CONTEXTO:
- La Iteración 21 demostró que las estrategias con lag (v19, v21) NO son rentables
- La estrategia v18 (Donchian Breakout + Filtro EMA_200) tiene un PF de 1.13
- PROBLEMA v18: Baja frecuencia (24 trades/año) - No es Day Trading
- OBJETIVO v22: Aumentar frecuencia a >150 trades/año manteniendo PF > 1.1

HIPÓTESIS v22:
Podemos mantener un Profit Factor > 1.1 mientras aumentamos el número de operaciones
a > 150/año, optimizando (reduciendo) los períodos de los indicadores de la estrategia
v18 en el timeframe de 15m.

ESTRATEGIA BASE (v18 - Donchian Breakout + Filtro EMA):
- COMPRA: Breakout Donchian Superior Y Precio > EMA_Filtro
- VENTA: Breakout Donchian Inferior Y Precio < EMA_Filtro
- Gestión de Riesgo: SL/TP basados en ATR con multiplicadores separados

GRID SEARCH A OPTIMIZAR:
- donchian_period: [10, 15, 20] (Períodos más cortos para breakouts más rápidos)
- ema_filter_period: [50, 100, 150, 200] (Filtros de tendencia más cortos + original)
- sl_multiplier: [2.0, 3.0, 4.0] (Stop Loss ATR)
- tp_multiplier: [3.0, 4.0, 5.0, 6.0] (Take Profit ATR - Ratios R:R altos)

TOTAL COMBINACIONES: 3 × 4 × 3 × 4 = 144 combinaciones

CRITERIOS DE FILTRADO:
- Profit Factor > 1.05 (Mínimo aceptable)
- Num Trades > 100 (Objetivo de frecuencia)
- Ordenar por Profit Factor (descendente)
- Mostrar Top 10

CRITERIO DE ÉXITO:
Encontrar una combinación con PF > 1.1 y Num Trades > 150

Activo: ETH/USDT
Timeframe: 15 minutos
Dataset: 1 año completo (datos pre-descargados en ETHUSDT_15m_OHLCV_2025-11-05.csv)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.model_selection import ParameterGrid

# Usar versión simple de indicadores (sin pandas-ta)
from src.indicators.technical_simple import agregar_indicadores_simple as agregar_indicadores
from src.strategy.signal_generator import generar_senales_momentum_v1
from src.backtest.engine import VectorizedBacktester
from src.utils.metrics import print_metrics


def run_single_backtest(df_base, donchian_period, ema_filter_period, sl_multiplier, tp_multiplier):
    """
    Ejecuta un backtest individual con los parámetros especificados.

    Args:
        df_base: DataFrame con datos OHLCV
        donchian_period: Período del Canal de Donchian
        ema_filter_period: Período de la EMA de filtro de tendencia
        sl_multiplier: Multiplicador ATR para Stop Loss
        tp_multiplier: Multiplicador ATR para Take Profit

    Returns:
        Diccionario con parámetros y métricas
    """
    # Copiar DataFrame
    df = df_base.copy()

    # Configuración de indicadores
    config = {
        'ema_trend': ema_filter_period,      # EMA de filtro de tendencia
        'donchian_period': donchian_period,  # Canal de Donchian
        'atr_length': 14                      # ATR estándar
    }

    # Agregar indicadores
    df = agregar_indicadores(df, config=config)

    # Generar señales usando estrategia Donchian Breakout (generar_senales_momentum_v1)
    df = generar_senales_momentum_v1(df, config=config)

    # Ejecutar backtest con SL/TP
    backtester = VectorizedBacktester(df, initial_capital=10000)
    backtester.run_backtest_with_sl_tp(
        atr_column='ATRr_14',
        sl_multiplier=sl_multiplier,
        tp_multiplier=tp_multiplier
    )

    # Calcular métricas
    metrics = backtester.calculate_metrics()

    # Combinar parámetros con métricas
    result = {
        'donchian_period': donchian_period,
        'ema_filter_period': ema_filter_period,
        'sl_multiplier': sl_multiplier,
        'tp_multiplier': tp_multiplier,
        **metrics
    }

    return result


def main():
    print("=" * 80)
    print("ITERACIÓN 22: OPTIMIZACIÓN DONCHIAN v18 PARA DAY TRADING (15m)")
    print("Estrategia: Donchian Breakout + Filtro EMA + SL/TP ATR")
    print("=" * 80)

    # 1. Cargar datos de 15m
    print("\n1. Cargando datos de ETHUSDT 15m...")
    data_file = 'ETHUSDT_15m_OHLCV_2025-11-05.csv'

    if not os.path.exists(data_file):
        print(f"   ❌ Error: Archivo {data_file} no encontrado")
        print("   Por favor, ejecuta el script de descarga de datos primero")
        sys.exit(1)

    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   ✓ Datos cargados: {len(df)} velas (aprox. 1 año)")
    print(f"   ✓ Período: {df['timestamp'].min()} a {df['timestamp'].max()}")

    # 2. Definir grid de parámetros
    print("\n2. Definiendo grid de parámetros a optimizar...")
    param_grid = {
        'donchian_period': [10, 15, 20],           # Períodos más cortos para mayor frecuencia
        'ema_filter_period': [50, 100, 150, 200],  # Filtros de tendencia (más cortos + original)
        'sl_multiplier': [2.0, 3.0, 4.0],          # Stop Loss ATR
        'tp_multiplier': [3.0, 4.0, 5.0, 6.0]      # Take Profit ATR (ratios R:R altos)
    }

    total_combinations = len(list(ParameterGrid(param_grid)))
    print(f"   ✓ Total de combinaciones: {total_combinations}")
    print(f"   ✓ Parámetros a optimizar:")
    print(f"      - donchian_period: {param_grid['donchian_period']}")
    print(f"      - ema_filter_period: {param_grid['ema_filter_period']}")
    print(f"      - sl_multiplier: {param_grid['sl_multiplier']}")
    print(f"      - tp_multiplier: {param_grid['tp_multiplier']}")

    # 3. Ejecutar Grid Search
    print("\n3. Ejecutando Grid Search (esto puede tardar varios minutos)...")
    print("=" * 80)

    results = []

    for i, params in enumerate(ParameterGrid(param_grid), 1):
        print(f"[{i}/{total_combinations}] Probando: Donchian={params['donchian_period']}, "
              f"EMA={params['ema_filter_period']}, SL={params['sl_multiplier']}x, TP={params['tp_multiplier']}x")

        try:
            result = run_single_backtest(
                df_base=df,
                donchian_period=params['donchian_period'],
                ema_filter_period=params['ema_filter_period'],
                sl_multiplier=params['sl_multiplier'],
                tp_multiplier=params['tp_multiplier']
            )
            results.append(result)

            # Mostrar métricas clave
            pf = result['profit_factor']
            nt = result['num_trades']
            wr = result['win_rate_pct']
            ret = result['total_return_pct']
            print(f"   PF: {pf:.2f}, Trades: {nt}, WinRate: {wr:.1f}%, Return: {ret:.2f}%\n")

        except Exception as e:
            print(f"   ❌ Error: {e}\n")

    # 4. Convertir a DataFrame
    print("=" * 80)
    print("\n4. Procesando resultados del Grid Search...")
    results_df = pd.DataFrame(results)

    # Guardar todos los resultados
    results_file = 'results/backtest_results_v22_donchian_15m_full.csv'
    os.makedirs('results', exist_ok=True)
    results_df.to_csv(results_file, index=False)
    print(f"   ✓ Resultados completos guardados en: {results_file}")
    print(f"   ✓ Total de combinaciones evaluadas: {len(results_df)}")

    # 5. Filtrar resultados
    print("\n5. Filtrando resultados según criterios...")
    print(f"   CRITERIOS:")
    print(f"   - Profit Factor > 1.05")
    print(f"   - Num Trades > 100")

    filtered_df = results_df[
        (results_df['profit_factor'] > 1.05) &
        (results_df['num_trades'] > 100)
    ].copy()

    print(f"   ✓ Combinaciones que cumplen criterios: {len(filtered_df)}")

    if len(filtered_df) == 0:
        print("\n   ⚠️  No se encontraron combinaciones que cumplan los criterios mínimos")
        print("   Mostrando Top 10 sin filtrar (ordenado por Profit Factor)...")
        top10_df = results_df.sort_values('profit_factor', ascending=False).head(10)
    else:
        # Ordenar por Profit Factor (descendente)
        top10_df = filtered_df.sort_values('profit_factor', ascending=False).head(10)

    # Guardar resultados filtrados
    filtered_file = 'results/backtest_results_v22_donchian_15m_filtered.csv'
    if len(filtered_df) > 0:
        filtered_df.to_csv(filtered_file, index=False)
        print(f"   ✓ Resultados filtrados guardados en: {filtered_file}")

    # 6. Mostrar Top 10
    print("\n" + "=" * 80)
    print("TOP 10 MEJORES CONFIGURACIONES")
    print("=" * 80)

    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)

    # Seleccionar columnas clave para mostrar
    display_cols = [
        'donchian_period', 'ema_filter_period', 'sl_multiplier', 'tp_multiplier',
        'profit_factor', 'num_trades', 'win_rate_pct', 'total_return_pct',
        'sharpe_ratio', 'max_drawdown_pct'
    ]

    print("\nResumen:")
    print(top10_df[display_cols].to_string(index=False))

    # 7. Análisis detallado del mejor resultado
    print("\n" + "=" * 80)
    print("ANÁLISIS DETALLADO - MEJOR CONFIGURACIÓN")
    print("=" * 80)

    best_result = top10_df.iloc[0].to_dict()

    print(f"\nParámetros Optimizados:")
    print(f"  Estrategia: Donchian Breakout + Filtro EMA")
    print(f"  - Período Donchian: {int(best_result['donchian_period'])} velas")
    print(f"  - EMA Filtro de Tendencia: {int(best_result['ema_filter_period'])} velas")
    print(f"  - Stop Loss: {best_result['sl_multiplier']}x ATR")
    print(f"  - Take Profit: {best_result['tp_multiplier']}x ATR")
    print(f"  - Ratio Riesgo:Recompensa: 1:{best_result['tp_multiplier']/best_result['sl_multiplier']:.2f}")

    # Métricas de rendimiento
    metrics_dict = {
        'initial_capital': best_result['initial_capital'],
        'final_value': best_result['final_value'],
        'net_profit': best_result['net_profit'],
        'total_return_pct': best_result['total_return_pct'],
        'annual_return_pct': best_result['annual_return_pct'],
        'sharpe_ratio': best_result['sharpe_ratio'],
        'sortino_ratio': best_result['sortino_ratio'],
        'calmar_ratio': best_result['calmar_ratio'],
        'max_drawdown_pct': best_result['max_drawdown_pct'],
        'num_trades': best_result['num_trades'],
        'win_rate_pct': best_result['win_rate_pct'],
        'profit_factor': best_result['profit_factor'],
        'avg_trade': best_result['avg_trade'],
        'best_trade': best_result['best_trade'],
        'worst_trade': best_result['worst_trade']
    }

    print_metrics(metrics_dict)

    # 8. Evaluación de criterios de éxito
    print("\n" + "=" * 80)
    print("EVALUACIÓN DE CRITERIOS DE ÉXITO")
    print("=" * 80)

    pf = best_result['profit_factor']
    nt = best_result['num_trades']

    print(f"\nCRITERIO DE ÉXITO: PF > 1.1 Y Num Trades > 150")
    print(f"  Profit Factor: {pf:.2f} {'✓ CUMPLE' if pf > 1.1 else '✗ NO CUMPLE'} (objetivo: > 1.1)")
    print(f"  Num Trades: {int(nt)} {'✓ CUMPLE' if nt > 150 else '✗ NO CUMPLE'} (objetivo: > 150)")

    if pf > 1.1 and nt > 150:
        print("\n🎉 ¡ÉXITO! La hipótesis v22 es CORRECTA:")
        print("   Hemos logrado aumentar la frecuencia a >150 trades/año")
        print("   manteniendo un Profit Factor >1.1")
    elif pf > 1.05 and nt > 100:
        print("\n⚠️  ÉXITO PARCIAL:")
        print("   Se cumplieron los criterios mínimos (PF > 1.05, Trades > 100)")
        print("   pero no se alcanzó el objetivo ideal (PF > 1.1, Trades > 150)")
    else:
        print("\n❌ La hipótesis v22 NO se cumplió con los parámetros probados")
        print("   Se requiere explorar otros rangos de parámetros")

    # 9. Comparación con v18
    print("\n" + "=" * 80)
    print("COMPARACIÓN CON ESTRATEGIA v18 (BASELINE)")
    print("=" * 80)

    print(f"\nv18 (Donchian 20 + EMA 200 - Timeframe original):")
    print(f"  - Profit Factor: 1.13")
    print(f"  - Num Trades: 24/año")
    print(f"  - Problema: Baja frecuencia (NO es Day Trading)")

    print(f"\nv22 (Mejor configuración - 15m optimizado):")
    print(f"  - Profit Factor: {pf:.2f}")
    print(f"  - Num Trades: {int(nt)}/año")
    print(f"  - Mejora en frecuencia: {((nt/24)-1)*100:.1f}%")
    print(f"  - Cambio en PF: {((pf/1.13)-1)*100:.1f}%")

    # 10. Próximos pasos
    print("\n" + "=" * 80)
    print("PRÓXIMOS PASOS")
    print("=" * 80)

    if pf > 1.1 and nt > 150:
        print("\n1. ✓ Validar resultados con walk-forward analysis")
        print("2. ✓ Ejecutar paper trading (Fase 3) con parámetros optimizados")
        print("3. ✓ Monitorear performance en datos out-of-sample")
    else:
        print("\n1. Revisar Top 10 para identificar rangos prometedores")
        print("2. Considerar refinar grid search con rangos más finos")
        print("3. Evaluar si la estrategia base requiere modificaciones")

    print("\n" + "=" * 80)
    print("✓ OPTIMIZACIÓN COMPLETADA")
    print("=" * 80)
    print(f"\nArchivos generados:")
    print(f"  - {results_file} (todos los resultados)")
    if len(filtered_df) > 0:
        print(f"  - {filtered_file} (resultados filtrados)")
    print(f"\nPara ejecutar backtest con mejor configuración:")
    print(f"  Donchian: {int(best_result['donchian_period'])}, "
          f"EMA: {int(best_result['ema_filter_period'])}, "
          f"SL: {best_result['sl_multiplier']}x, "
          f"TP: {best_result['tp_multiplier']}x")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Operación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
