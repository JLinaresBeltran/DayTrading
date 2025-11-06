"""
Script para analizar el archivo mejor_resultado.csv y encontrar las mejores estrategias.
Filtra por criterios de éxito y genera un reporte detallado.
"""

import pandas as pd
import numpy as np

# Cargar los resultados
df = pd.read_csv('results/mejor_resultado.csv')

print("="*80)
print("ANÁLISIS DE RESULTADOS - BÚSQUEDA DE ESTRATEGIA ÓPTIMA")
print("="*80)
print(f"\nTotal de combinaciones probadas: {len(df)}")
print(f"Columnas disponibles: {list(df.columns)}\n")

# Estadísticas generales
print("="*80)
print("ESTADÍSTICAS GENERALES")
print("="*80)
print(f"Profit Factor:")
print(f"  - Promedio: {df['profit_factor'].mean():.4f}")
print(f"  - Máximo: {df['profit_factor'].max():.4f}")
print(f"  - Mínimo: {df['profit_factor'].min():.4f}")
print(f"  - Estrategias con PF > 1.0: {(df['profit_factor'] > 1.0).sum()} ({(df['profit_factor'] > 1.0).sum() / len(df) * 100:.1f}%)")
print(f"  - Estrategias con PF >= 1.5: {(df['profit_factor'] >= 1.5).sum()} ({(df['profit_factor'] >= 1.5).sum() / len(df) * 100:.1f}%)")
print(f"  - Estrategias con PF >= 2.0: {(df['profit_factor'] >= 2.0).sum()} ({(df['profit_factor'] >= 2.0).sum() / len(df) * 100:.1f}%)")

print(f"\nWin Rate:")
print(f"  - Promedio: {df['win_rate_pct'].mean():.2f}%")
print(f"  - Máximo: {df['win_rate_pct'].max():.2f}%")
print(f"  - Mínimo: {df['win_rate_pct'].min():.2f}%")
print(f"  - Estrategias con WR >= 40%: {(df['win_rate_pct'] >= 40).sum()} ({(df['win_rate_pct'] >= 40).sum() / len(df) * 100:.1f}%)")

print(f"\nRetorno Total:")
print(f"  - Promedio: {df['total_return_pct'].mean():.2f}%")
print(f"  - Máximo: {df['total_return_pct'].max():.2f}%")
print(f"  - Mínimo: {df['total_return_pct'].min():.2f}%")
print(f"  - Estrategias rentables (>0%): {(df['total_return_pct'] > 0).sum()} ({(df['total_return_pct'] > 0).sum() / len(df) * 100:.1f}%)")
print(f"  - Estrategias con >50% retorno: {(df['total_return_pct'] > 50).sum()} ({(df['total_return_pct'] > 50).sum() / len(df) * 100:.1f}%)")
print(f"  - Estrategias con >100% retorno: {(df['total_return_pct'] > 100).sum()} ({(df['total_return_pct'] > 100).sum() / len(df) * 100:.1f}%)")

print(f"\nMax Drawdown:")
print(f"  - Promedio: {df['max_drawdown_pct'].mean():.2f}%")
print(f"  - Mejor (menor): {df['max_drawdown_pct'].min():.2f}%")
print(f"  - Peor (mayor): {df['max_drawdown_pct'].max():.2f}%")
print(f"  - Estrategias con DD < 15%: {(df['max_drawdown_pct'] < 15).sum()} ({(df['max_drawdown_pct'] < 15).sum() / len(df) * 100:.1f}%)")

print(f"\nNúmero de Trades:")
print(f"  - Promedio: {df['num_trades'].mean():.1f}")
print(f"  - Máximo: {df['num_trades'].max()}")
print(f"  - Mínimo: {df['num_trades'].min()}")
print(f"  - Estrategias con >= 50 trades: {(df['num_trades'] >= 50).sum()} ({(df['num_trades'] >= 50).sum() / len(df) * 100:.1f}%)")

# Filtrar estrategias según criterios de éxito
print("\n" + "="*80)
print("FILTRADO POR CRITERIOS DE ÉXITO")
print("="*80)

# NIVEL 1: Criterios mínimos aceptables
print("\n🥉 NIVEL 1: CRITERIOS MÍNIMOS ACEPTABLES")
print("-" * 80)
print("  - Profit Factor >= 1.3")
print("  - Win Rate >= 30%")
print("  - Total Return > 0%")
print("  - Max Drawdown < 25%")
print("  - Num Trades >= 30")

nivel1 = df[
    (df['profit_factor'] >= 1.3) &
    (df['win_rate_pct'] >= 30) &
    (df['total_return_pct'] > 0) &
    (df['max_drawdown_pct'] < 25) &
    (df['num_trades'] >= 30)
].sort_values('profit_factor', ascending=False)

print(f"\n✓ Estrategias que cumplen Nivel 1: {len(nivel1)} ({len(nivel1) / len(df) * 100:.1f}%)")

if len(nivel1) > 0:
    print("\nTOP 5 NIVEL 1:")
    for idx, row in nivel1.head(5).iterrows():
        print(f"\n  #{row['id']} - EMA({row['ema_fast_m15']}, {row['ema_slow_m15']}, {row['ema_trend_h1']}) ATR({row['atr_period']}, {row['atr_lookback']}, {row['atr_multiplier']})")
        print(f"    Profit Factor: {row['profit_factor']:.2f}")
        print(f"    Win Rate: {row['win_rate_pct']:.2f}%")
        print(f"    Return: {row['total_return_pct']:.2f}%")
        print(f"    Max DD: {row['max_drawdown_pct']:.2f}%")
        print(f"    Trades: {int(row['num_trades'])}")
        print(f"    Sharpe: {row['sharpe_ratio']:.2f}")

# NIVEL 2: Criterios objetivo
print("\n🥈 NIVEL 2: CRITERIOS OBJETIVO")
print("-" * 80)
print("  - Profit Factor >= 1.5")
print("  - Win Rate >= 35%")
print("  - Total Return >= 25%")
print("  - Max Drawdown < 20%")
print("  - Num Trades >= 40")
print("  - Sharpe Ratio > 0")

nivel2 = df[
    (df['profit_factor'] >= 1.5) &
    (df['win_rate_pct'] >= 35) &
    (df['total_return_pct'] >= 25) &
    (df['max_drawdown_pct'] < 20) &
    (df['num_trades'] >= 40) &
    (df['sharpe_ratio'] > 0)
].sort_values('profit_factor', ascending=False)

print(f"\n✓ Estrategias que cumplen Nivel 2: {len(nivel2)} ({len(nivel2) / len(df) * 100:.1f}%)")

if len(nivel2) > 0:
    print("\nTOP 5 NIVEL 2:")
    for idx, row in nivel2.head(5).iterrows():
        print(f"\n  #{row['id']} - EMA({row['ema_fast_m15']}, {row['ema_slow_m15']}, {row['ema_trend_h1']}) ATR({row['atr_period']}, {row['atr_lookback']}, {row['atr_multiplier']})")
        print(f"    Profit Factor: {row['profit_factor']:.2f}")
        print(f"    Win Rate: {row['win_rate_pct']:.2f}%")
        print(f"    Return: {row['total_return_pct']:.2f}%")
        print(f"    Max DD: {row['max_drawdown_pct']:.2f}%")
        print(f"    Trades: {int(row['num_trades'])}")
        print(f"    Sharpe: {row['sharpe_ratio']:.2f}")

# NIVEL 3: Criterios ideales (tu objetivo)
print("\n🥇 NIVEL 3: CRITERIOS IDEALES (TU OBJETIVO)")
print("-" * 80)
print("  - Profit Factor >= 2.0")
print("  - Win Rate >= 40%")
print("  - Total Return >= 100%")
print("  - Max Drawdown < 15%")
print("  - Num Trades >= 50")
print("  - Sharpe Ratio >= 0.5")

nivel3 = df[
    (df['profit_factor'] >= 2.0) &
    (df['win_rate_pct'] >= 40) &
    (df['total_return_pct'] >= 100) &
    (df['max_drawdown_pct'] < 15) &
    (df['num_trades'] >= 50) &
    (df['sharpe_ratio'] >= 0.5)
].sort_values('profit_factor', ascending=False)

print(f"\n✓ Estrategias que cumplen Nivel 3: {len(nivel3)} ({len(nivel3) / len(df) * 100:.1f}%)")

if len(nivel3) > 0:
    print("\nTOP 5 NIVEL 3:")
    for idx, row in nivel3.head(5).iterrows():
        print(f"\n  #{row['id']} - EMA({row['ema_fast_m15']}, {row['ema_slow_m15']}, {row['ema_trend_h1']}) ATR({row['atr_period']}, {row['atr_lookback']}, {row['atr_multiplier']})")
        print(f"    Profit Factor: {row['profit_factor']:.2f}")
        print(f"    Win Rate: {row['win_rate_pct']:.2f}%")
        print(f"    Return: {row['total_return_pct']:.2f}%")
        print(f"    Max DD: {row['max_drawdown_pct']:.2f}%")
        print(f"    Trades: {int(row['num_trades'])}")
        print(f"    Sharpe: {row['sharpe_ratio']:.2f}")

# TOP 10 GENERAL (ordenadas por score compuesto)
print("\n" + "="*80)
print("TOP 10 ESTRATEGIAS GENERALES (Score Compuesto)")
print("="*80)
print("Score = (PF × 0.4) + (WR% × 0.3) + (Return% / 100 × 0.2) - (MaxDD% / 100 × 0.1)")

df['score'] = (
    df['profit_factor'] * 0.4 +
    df['win_rate_pct'] * 0.3 +
    (df['total_return_pct'] / 100) * 0.2 -
    (df['max_drawdown_pct'] / 100) * 0.1
)

top10 = df.sort_values('score', ascending=False).head(10)

for idx, row in top10.iterrows():
    print(f"\n#{int(row['id'])} - Score: {row['score']:.2f}")
    print(f"  Parámetros: EMA({int(row['ema_fast_m15'])}, {int(row['ema_slow_m15'])}, {int(row['ema_trend_h1'])}) + ATR({int(row['atr_period'])}, {int(row['atr_lookback'])}, {row['atr_multiplier']:.1f})")
    print(f"  Métricas:")
    print(f"    • Profit Factor: {row['profit_factor']:.2f}")
    print(f"    • Win Rate: {row['win_rate_pct']:.2f}%")
    print(f"    • Return: {row['total_return_pct']:.2f}%")
    print(f"    • Max DD: {row['max_drawdown_pct']:.2f}%")
    print(f"    • Sharpe: {row['sharpe_ratio']:.2f}")
    print(f"    • Trades: {int(row['num_trades'])}")
    print(f"    • Avg Trade: ${row['avg_trade']:.2f}")

# Análisis de parámetros óptimos
print("\n" + "="*80)
print("ANÁLISIS DE PARÁMETROS ÓPTIMOS")
print("="*80)

# Filtrar solo estrategias rentables para el análisis
rentables = df[df['profit_factor'] >= 1.0]

print(f"\nEstadísticas de estrategias rentables (PF >= 1.0): {len(rentables)} estrategias\n")

print("Parámetros más frecuentes en estrategias rentables:")
print(f"  EMA Fast (15m): {rentables['ema_fast_m15'].mode().values}")
print(f"  EMA Slow (15m): {rentables['ema_slow_m15'].mode().values}")
print(f"  EMA Trend (1h): {rentables['ema_trend_h1'].mode().values}")
print(f"  ATR Period: {rentables['atr_period'].mode().values}")
print(f"  ATR Lookback: {rentables['atr_lookback'].mode().values}")
print(f"  ATR Multiplier: {rentables['atr_multiplier'].mode().values}")

print("\nRangos óptimos (promedio de mejores 20 estrategias):")
mejores20 = df.sort_values('profit_factor', ascending=False).head(20)
print(f"  EMA Fast (15m): {mejores20['ema_fast_m15'].mean():.1f} ± {mejores20['ema_fast_m15'].std():.1f}")
print(f"  EMA Slow (15m): {mejores20['ema_slow_m15'].mean():.1f} ± {mejores20['ema_slow_m15'].std():.1f}")
print(f"  EMA Trend (1h): {mejores20['ema_trend_h1'].mean():.1f} ± {mejores20['ema_trend_h1'].std():.1f}")
print(f"  ATR Period: {mejores20['atr_period'].mean():.1f} ± {mejores20['atr_period'].std():.1f}")
print(f"  ATR Lookback: {mejores20['atr_lookback'].mean():.1f} ± {mejores20['atr_lookback'].std():.1f}")
print(f"  ATR Multiplier: {mejores20['atr_multiplier'].mean():.2f} ± {mejores20['atr_multiplier'].std():.2f}")

# Guardar resultados filtrados
print("\n" + "="*80)
print("GUARDANDO RESULTADOS FILTRADOS")
print("="*80)

# Guardar top 10
top10.to_csv('results/top10_estrategias.csv', index=False)
print("✓ Guardado: results/top10_estrategias.csv")

# Guardar nivel 1
if len(nivel1) > 0:
    nivel1.to_csv('results/estrategias_nivel1.csv', index=False)
    print(f"✓ Guardado: results/estrategias_nivel1.csv ({len(nivel1)} estrategias)")

# Guardar nivel 2
if len(nivel2) > 0:
    nivel2.to_csv('results/estrategias_nivel2.csv', index=False)
    print(f"✓ Guardado: results/estrategias_nivel2.csv ({len(nivel2)} estrategias)")

# Guardar nivel 3
if len(nivel3) > 0:
    nivel3.to_csv('results/estrategias_nivel3.csv', index=False)
    print(f"✓ Guardado: results/estrategias_nivel3.csv ({len(nivel3)} estrategias)")

print("\n" + "="*80)
print("FIN DEL ANÁLISIS")
print("="*80)
