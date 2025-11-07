"""
Análisis del archivo mejor_resultado.csv
Este script analiza los resultados de optimización de estrategias
"""

import pandas as pd
import numpy as np

# Cargar datos
df = pd.read_csv('results/mejor_resultado.csv')

print("=" * 80)
print("ANÁLISIS DEL MEJOR RESULTADO - OPTIMIZACIÓN DE ESTRATEGIAS")
print("=" * 80)
print()

# Información general
print(f"📊 INFORMACIÓN GENERAL")
print(f"   Total de combinaciones probadas: {len(df):,}")
print(f"   Líneas vacías eliminadas: {df.isna().any(axis=1).sum()}")
print()

# Eliminar filas vacías si existen
df = df.dropna()

# Estadísticas básicas de rendimiento
print("=" * 80)
print("💰 RENDIMIENTO GENERAL")
print("=" * 80)
print(f"   Retorno Total Promedio: {df['total_return_pct'].mean():.2f}%")
print(f"   Retorno Total Mediano: {df['total_return_pct'].median():.2f}%")
print(f"   Mejor Retorno: {df['total_return_pct'].max():.2f}%")
print(f"   Peor Retorno: {df['total_return_pct'].min():.2f}%")
print()
print(f"   Estrategias Rentables: {(df['total_return_pct'] > 0).sum()} ({(df['total_return_pct'] > 0).sum()/len(df)*100:.1f}%)")
print(f"   Estrategias con Pérdidas: {(df['total_return_pct'] <= 0).sum()} ({(df['total_return_pct'] <= 0).sum()/len(df)*100:.1f}%)")
print()

# Top 10 mejores estrategias
print("=" * 80)
print("🏆 TOP 10 MEJORES ESTRATEGIAS")
print("=" * 80)
top10 = df.nlargest(10, 'total_return_pct')
for idx, row in top10.iterrows():
    print(f"\n#{row['id']}: Retorno {row['total_return_pct']:.2f}%")
    print(f"   📈 Parámetros: EMA_fast={int(row['ema_fast_m15'])}, EMA_slow={int(row['ema_slow_m15'])}, EMA_trend={int(row['ema_trend_h1'])}")
    print(f"   📊 ATR: period={int(row['atr_period'])}, lookback={int(row['atr_lookback'])}, multiplier={row['atr_multiplier']:.1f}")
    print(f"   💵 Capital Final: ${row['final_value']:,.2f}")
    print(f"   📉 Max Drawdown: {row['max_drawdown_pct']:.2f}%")
    print(f"   🎯 Win Rate: {row['win_rate_pct']:.2f}%")
    print(f"   📊 Sharpe Ratio: {row['sharpe_ratio']:.2f}")
    print(f"   🔢 Trades: {int(row['num_trades'])}")
    print(f"   💰 Profit Factor: {row['profit_factor']:.2f}")

# LA MEJOR estrategia
print("\n" + "=" * 80)
print("🥇 LA MEJOR ESTRATEGIA")
print("=" * 80)
best = df.loc[df['total_return_pct'].idxmax()]
print(f"\nID: {int(best['id'])}")
print(f"{'─' * 80}")
print(f"\n📈 RENDIMIENTO:")
print(f"   Retorno Total: {best['total_return_pct']:.2f}%")
print(f"   Retorno Anual: {best['annual_return_pct']:.2f}%")
print(f"   Capital Inicial: ${best['initial_capital']:,.2f}")
print(f"   Capital Final: ${best['final_value']:,.2f}")
print(f"   Ganancia Neta: ${best['net_profit']:,.2f}")
print()
print(f"📊 PARÁMETROS DE LA ESTRATEGIA:")
print(f"   EMA Rápida (M15): {int(best['ema_fast_m15'])}")
print(f"   EMA Lenta (M15): {int(best['ema_slow_m15'])}")
print(f"   EMA Tendencia (H1): {int(best['ema_trend_h1'])}")
print(f"   ATR Period: {int(best['atr_period'])}")
print(f"   ATR Lookback: {int(best['atr_lookback'])}")
print(f"   ATR Multiplier: {best['atr_multiplier']:.1f}")
print()
print(f"📉 MÉTRICAS DE RIESGO:")
print(f"   Sharpe Ratio: {best['sharpe_ratio']:.2f}")
print(f"   Sortino Ratio: {best['sortino_ratio']:.2f}")
print(f"   Calmar Ratio: {best['calmar_ratio']:.2f}")
print(f"   Max Drawdown: {best['max_drawdown_pct']:.2f}%")
print()
print(f"🎯 ESTADÍSTICAS DE TRADING:")
print(f"   Número de Trades: {int(best['num_trades'])}")
print(f"   Win Rate: {best['win_rate_pct']:.2f}%")
print(f"   Profit Factor: {best['profit_factor']:.2f}")
print(f"   Trade Promedio: ${best['avg_trade']:.2f}")
print(f"   Mejor Trade: ${best['best_trade']:.2f}")
print(f"   Peor Trade: ${best['worst_trade']:.2f}")
print()
print(f"💼 COMPARACIÓN CON BUY & HOLD:")
print(f"   Buy & Hold Return: {best['buy_hold_return_pct']:.2f}%")
print(f"   Exceso de Retorno: {best['excess_return_pct']:.2f}%")
print()

# Análisis de parámetros
print("=" * 80)
print("🔍 ANÁLISIS DE PARÁMETROS")
print("=" * 80)

# Correlaciones con el retorno
correlations = df[[
    'ema_fast_m15', 'ema_slow_m15', 'ema_trend_h1',
    'atr_period', 'atr_lookback', 'atr_multiplier',
    'total_return_pct'
]].corr()['total_return_pct'].sort_values(ascending=False)

print("\n📊 Correlación de parámetros con el retorno:")
for param, corr in correlations.items():
    if param != 'total_return_pct':
        print(f"   {param}: {corr:.4f}")

# Mejores valores por parámetro
print("\n🎯 MEJORES VALORES POR PARÁMETRO (promedio del top 10%):")
top_10_pct = df.nlargest(int(len(df) * 0.1), 'total_return_pct')
print(f"   EMA Fast (M15): {top_10_pct['ema_fast_m15'].mode().values[0]:.0f} (más común)")
print(f"   EMA Slow (M15): {top_10_pct['ema_slow_m15'].mode().values[0]:.0f} (más común)")
print(f"   EMA Trend (H1): {top_10_pct['ema_trend_h1'].mode().values[0]:.0f} (más común)")
print(f"   ATR Period: {top_10_pct['atr_period'].mode().values[0]:.0f} (más común)")
print(f"   ATR Lookback: {top_10_pct['atr_lookback'].mode().values[0]:.0f} (más común)")
print(f"   ATR Multiplier: {top_10_pct['atr_multiplier'].mode().values[0]:.1f} (más común)")

# Distribución de métricas clave
print("\n" + "=" * 80)
print("📊 DISTRIBUCIÓN DE MÉTRICAS CLAVE")
print("=" * 80)
print(f"\nSharpe Ratio:")
print(f"   Promedio: {df['sharpe_ratio'].mean():.2f}")
print(f"   Máximo: {df['sharpe_ratio'].max():.2f}")
print(f"   Estrategias con Sharpe > 0: {(df['sharpe_ratio'] > 0).sum()} ({(df['sharpe_ratio'] > 0).sum()/len(df)*100:.1f}%)")

print(f"\nWin Rate:")
print(f"   Promedio: {df['win_rate_pct'].mean():.2f}%")
print(f"   Máximo: {df['win_rate_pct'].max():.2f}%")
print(f"   Estrategias con Win Rate > 50%: {(df['win_rate_pct'] > 50).sum()} ({(df['win_rate_pct'] > 50).sum()/len(df)*100:.1f}%)")

print(f"\nProfit Factor:")
print(f"   Promedio: {df['profit_factor'].mean():.2f}")
print(f"   Máximo: {df['profit_factor'].max():.2f}")
print(f"   Estrategias con Profit Factor > 1: {(df['profit_factor'] > 1).sum()} ({(df['profit_factor'] > 1).sum()/len(df)*100:.1f}%)")

print(f"\nMax Drawdown:")
print(f"   Promedio: {df['max_drawdown_pct'].mean():.2f}%")
print(f"   Mínimo (mejor): {df['max_drawdown_pct'].min():.2f}%")
print(f"   Máximo (peor): {df['max_drawdown_pct'].max():.2f}%")

print("\n" + "=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
