"""
Comparación de las dos mejores estrategias encontradas
para diseñar una estrategia híbrida optimizada para day trading de alta frecuencia
"""

print("=" * 80)
print("COMPARACIÓN DE ESTRATEGIAS - ANÁLISIS PARA DAY TRADING")
print("=" * 80)
print()

# ESTRATEGIA 1: mejor_resultado.csv #345 (MTF)
estrategia_mtf = {
    "nombre": "Multi-Timeframe EMA (MTF)",
    "id": 345,
    "archivo": "results/mejor_resultado.csv",

    # Parámetros
    "ema_fast_m15": 15,
    "ema_slow_m15": 21,
    "ema_trend_h1": 150,
    "atr_period": 14,
    "atr_lookback": 3,
    "atr_multiplier": 3.0,

    # Métricas
    "retorno_total": 55.13,
    "retorno_anual": 144.82,
    "profit_factor": 1.46,
    "max_drawdown": 11.85,
    "sharpe_ratio": 0.21,
    "calmar_ratio": 12.22,

    # Trading
    "num_trades": 153,
    "trades_por_dia": 153 / 365,  # ~0.42 trades/día
    "win_rate": 36.60,
    "mejor_trade": 1386.77,
    "peor_trade": -336.53,

    # Comparación
    "buy_hold": 38.24,
    "excess_return": 16.89,

    # Temporalidades
    "timeframes": ["M15", "H1"],
    "estilo": "Swing Trading (posiciones más largas)",
}

# ESTRATEGIA 2: OPTIMAL_STRATEGY_REPORT.md #254 (Supertrend + RSI)
estrategia_supertrend = {
    "nombre": "Supertrend + RSI (Hybrid)",
    "id": 254,
    "archivo": "OPTIMAL_STRATEGY_REPORT.md",

    # Parámetros
    "supertrend_length": 7,
    "supertrend_multiplier": 1.5,
    "rsi_period": 14,
    "rsi_oversold": 30,
    "rsi_overbought": 65,
    "atr_period": 14,
    "sl_atr_multiplier": 2.5,
    "tp_atr_multiplier": 5.0,
    "direccion": "hybrid",  # long + short

    # Métricas
    "retorno_total": 107.15,
    "retorno_anual": 107.15,  # Ya es anualizado
    "profit_factor": 2.38,
    "max_drawdown": 18.64,
    "sharpe_ratio": 0.14,
    "calmar_ratio": 5.77,

    # Trading
    "num_trades": 50,
    "trades_por_dia": 50 / 365,  # ~0.14 trades/día
    "win_rate": 24.0,
    "mejor_trade": 3681.15,
    "peor_trade": -396.89,

    # Comparación
    "buy_hold": 23.45,
    "excess_return": 83.70,

    # Temporalidades
    "timeframes": ["No especificado"],
    "estilo": "Trend Following (50 trades/año)",
}

print("ESTRATEGIA 1: Multi-Timeframe EMA (MTF)")
print("─" * 80)
print(f"📊 Enfoque: {estrategia_mtf['estilo']}")
print(f"⏰ Timeframes: {', '.join(estrategia_mtf['timeframes'])}")
print(f"📈 Retorno Total: {estrategia_mtf['retorno_total']:.2f}%")
print(f"💰 Profit Factor: {estrategia_mtf['profit_factor']:.2f}")
print(f"📉 Max Drawdown: {estrategia_mtf['max_drawdown']:.2f}%")
print(f"🎯 Win Rate: {estrategia_mtf['win_rate']:.2f}%")
print(f"📊 Trades/día: {estrategia_mtf['trades_por_dia']:.2f}")
print(f"🔢 Total Trades: {estrategia_mtf['num_trades']}")
print()

print("ESTRATEGIA 2: Supertrend + RSI (Hybrid)")
print("─" * 80)
print(f"📊 Enfoque: {estrategia_supertrend['estilo']}")
print(f"⏰ Timeframes: {', '.join(estrategia_supertrend['timeframes'])}")
print(f"📈 Retorno Total: {estrategia_supertrend['retorno_total']:.2f}%")
print(f"💰 Profit Factor: {estrategia_supertrend['profit_factor']:.2f}")
print(f"📉 Max Drawdown: {estrategia_supertrend['max_drawdown']:.2f}%")
print(f"🎯 Win Rate: {estrategia_supertrend['win_rate']:.2f}%")
print(f"📊 Trades/día: {estrategia_supertrend['trades_por_dia']:.2f}")
print(f"🔢 Total Trades: {estrategia_supertrend['num_trades']}")
print()

print("=" * 80)
print("ANÁLISIS COMPARATIVO")
print("=" * 80)
print()

print("📊 FORTALEZAS DE CADA ESTRATEGIA:")
print()
print("MTF (Estrategia 1):")
print("  ✅ Mejor control de riesgo (DD: 11.85% vs 18.64%)")
print("  ✅ Mayor frecuencia de trading (153 vs 50 trades)")
print("  ✅ Mejor Calmar Ratio (12.22 vs 5.77)")
print("  ✅ Mayor Win Rate (36.6% vs 24%)")
print("  ✅ Menor pérdida máxima (-$336 vs -$396)")
print()
print("Supertrend + RSI (Estrategia 2):")
print("  ✅ Mayor retorno total (107% vs 55%)")
print("  ✅ Mejor Profit Factor (2.38 vs 1.46)")
print("  ✅ Mayor ganancia por trade ($214 vs $36)")
print("  ✅ Mejor trade más grande ($3,681 vs $1,386)")
print("  ✅ Opera en ambas direcciones (long + short)")
print()

print("=" * 80)
print("OBJETIVO: DAY TRADING DE ALTA FRECUENCIA")
print("=" * 80)
print()
print("Meta: 2-3 operaciones diarias (730-1,095 trades/año)")
print()
print("Frecuencia actual:")
print(f"  • MTF: {estrategia_mtf['trades_por_dia']:.2f} trades/día (~3 por semana)")
print(f"  • Supertrend: {estrategia_supertrend['trades_por_dia']:.2f} trades/día (~1 por semana)")
print()
print("⚠️ AMBAS ESTRATEGIAS SON DE BAJA FRECUENCIA PARA DAY TRADING")
print()
print("Para lograr 2-3 trades/día necesitamos:")
print("  • Temporalidad más corta (M5 o M15)")
print("  • Indicadores más sensibles")
print("  • Múltiples timeframes para filtrar ruido")
print("  • Stop loss más ajustado (trades más cortos)")
print("  • Take profit más cercano (capitalizar movimientos pequeños)")
print()

print("=" * 80)
print("PROPUESTA: ESTRATEGIA HÍBRIDA PARA DAY TRADING")
print("=" * 80)
print()
print("📋 CONCEPTO:")
print("Combinar lo mejor de ambas estrategias adaptándolas para alta frecuencia")
print()
print("🎯 ELEMENTOS A TOMAR:")
print()
print("De MTF (Estrategia 1):")
print("  ✅ Multi-timeframe approach (contexto + ejecución)")
print("  ✅ Control de riesgo estricto (bajo DD)")
print("  ✅ EMAs para tendencia clara")
print("  ✅ ATR lookback de 3 (rápido)")
print()
print("De Supertrend + RSI (Estrategia 2):")
print("  ✅ Supertrend para señales claras")
print("  ✅ RSI para filtrar momentum")
print("  ✅ Dirección híbrida (long + short)")
print("  ✅ Profit Factor superior (2.38)")
print("  ✅ RSI overbought=65 (más permisivo)")
print()

print("=" * 80)
print("PARÁMETROS PROPUESTOS PARA ESTRATEGIA HÍBRIDA")
print("=" * 80)
print()

estrategia_hibrida = {
    # Temporalidades
    "timeframe_ejecucion": "M5",  # Para 2-3 trades/día
    "timeframe_filtro": "M15",    # Para tendencia
    "timeframe_contexto": "H1",   # Para sesgo general

    # Indicadores de Entrada (combinados)
    "supertrend_length": 7,       # De Estrategia 2 (sensible)
    "supertrend_multiplier": 1.5, # De Estrategia 2 (sensible)
    "ema_fast": 9,                # Más rápido que 15 para M5
    "ema_slow": 21,               # De Estrategia 1
    "rsi_period": 14,             # Estándar
    "rsi_oversold": 30,           # De Estrategia 2
    "rsi_overbought": 65,         # De Estrategia 2 (más permisivo)

    # Filtros de Tendencia
    "ema_trend_m15": 50,          # Filtro en M15
    "ema_trend_h1": 100,          # Filtro en H1 (más corto que 150)

    # Gestión de Riesgo (ajustado para day trading)
    "atr_period": 14,             # Estándar
    "atr_lookback": 3,            # De Estrategia 1 (rápido)
    "sl_atr_multiplier": 2.0,     # Más ajustado para trades cortos
    "tp_atr_multiplier": 3.0,     # Ratio 1:1.5 (más realista para day trading)

    # Dirección
    "direccion": "hybrid",        # Long + short

    # Características
    "ventaja_1": "Multi-timeframe para filtrar ruido",
    "ventaja_2": "Supertrend + RSI para señales claras",
    "ventaja_3": "Stops ajustados para day trading",
    "ventaja_4": "Opera en ambas direcciones",
    "ventaja_5": "M5 permite alta frecuencia",
}

print("🎯 ESTRATEGIA HÍBRIDA - Day Trading Alta Frecuencia")
print()
print("📊 TEMPORALIDADES:")
print(f"  • Ejecución: {estrategia_hibrida['timeframe_ejecucion']} (señales de entrada/salida)")
print(f"  • Filtro: {estrategia_hibrida['timeframe_filtro']} (tendencia intermedia)")
print(f"  • Contexto: {estrategia_hibrida['timeframe_contexto']} (sesgo direccional)")
print()
print("📈 INDICADORES:")
print(f"  • Supertrend: length={estrategia_hibrida['supertrend_length']}, multiplier={estrategia_hibrida['supertrend_multiplier']}")
print(f"  • EMA Fast (M5): {estrategia_hibrida['ema_fast']}")
print(f"  • EMA Slow (M5): {estrategia_hibrida['ema_slow']}")
print(f"  • RSI: period={estrategia_hibrida['rsi_period']}, OS={estrategia_hibrida['rsi_oversold']}, OB={estrategia_hibrida['rsi_overbought']}")
print()
print("🎯 FILTROS DE TENDENCIA:")
print(f"  • EMA Trend M15: {estrategia_hibrida['ema_trend_m15']}")
print(f"  • EMA Trend H1: {estrategia_hibrida['ema_trend_h1']}")
print()
print("⚖️ GESTIÓN DE RIESGO:")
print(f"  • ATR Period: {estrategia_hibrida['atr_period']}")
print(f"  • ATR Lookback: {estrategia_hibrida['atr_lookback']}")
print(f"  • Stop Loss: {estrategia_hibrida['sl_atr_multiplier']}x ATR")
print(f"  • Take Profit: {estrategia_hibrida['tp_atr_multiplier']}x ATR")
print(f"  • Ratio SL:TP = 1:1.5")
print()
print("🔄 DIRECCIÓN:")
print(f"  • {estrategia_hibrida['direccion'].upper()} (Long + Short)")
print()

print("=" * 80)
print("LÓGICA DE ENTRADA - ESTRATEGIA HÍBRIDA")
print("=" * 80)
print()
print("🟢 SEÑAL DE COMPRA (LONG):")
print("  1. [H1] Precio > EMA_100 (tendencia alcista de fondo)")
print("  2. [M15] Precio > EMA_50 (confirmación de tendencia)")
print("  3. [M5] EMA_9 cruza por encima de EMA_21")
print("  4. [M5] Supertrend cambia a alcista")
print("  5. [M5] RSI > 30 y RSI < 65 (momentum pero no sobrecompra)")
print("  → ENTRAR EN LARGO")
print()
print("🔴 SEÑAL DE VENTA (SHORT):")
print("  1. [H1] Precio < EMA_100 (tendencia bajista de fondo)")
print("  2. [M15] Precio < EMA_50 (confirmación de tendencia)")
print("  3. [M5] EMA_9 cruza por debajo de EMA_21")
print("  4. [M5] Supertrend cambia a bajista")
print("  5. [M5] RSI < 70 y RSI > 35 (momentum pero no sobreventa)")
print("  → ENTRAR EN CORTO")
print()
print("🛑 SALIDA:")
print("  • Stop Loss: Precio - (2.0 * ATR_14)")
print("  • Take Profit: Precio + (3.0 * ATR_14)")
print("  • O cuando Supertrend cambia de dirección")
print()

print("=" * 80)
print("EXPECTATIVAS DE RENDIMIENTO")
print("=" * 80)
print()
print("📊 PROYECCIONES (basadas en combinación de estrategias):")
print()
print("Frecuencia:")
print(f"  • Objetivo: 2-3 trades/día = ~730-1,095 trades/año")
print(f"  • Esperado con M5: ~500-800 trades/año (ajuste realista)")
print()
print("Métricas esperadas:")
print(f"  • Retorno anual: 70-90% (entre 55% y 107%)")
print(f"  • Profit Factor: 1.8-2.0 (balance entre 1.46 y 2.38)")
print(f"  • Max Drawdown: 12-15% (mejor que Supertrend, ajustado de MTF)")
print(f"  • Win Rate: 28-32% (entre 24% y 36.6%)")
print(f"  • Sharpe Ratio: 0.15-0.20")
print()
print("⚠️ ADVERTENCIA:")
print("Estas son PROYECCIONES. La estrategia debe ser:")
print("  1. Backesteada en M5 con datos históricos")
print("  2. Optimizada con grid search específico")
print("  3. Validada en paper trading por 30 días")
print("  4. Probada en diferentes condiciones de mercado")
print()

print("=" * 80)
print("VENTAJAS DE LA ESTRATEGIA HÍBRIDA")
print("=" * 80)
print()
for i in range(1, 6):
    print(f"  {i}. {estrategia_hibrida[f'ventaja_{i}']}")
print()

print("=" * 80)
print("PRÓXIMOS PASOS PARA IMPLEMENTACIÓN")
print("=" * 80)
print()
print("1. ✍️  Crear script de backtest con la estrategia híbrida")
print("2. 📊 Descargar datos OHLCV en M5, M15 y H1")
print("3. 🔍 Ejecutar grid search en parámetros ajustados")
print("4. 📈 Analizar resultados y optimizar")
print("5. 📝 Validar con walk-forward optimization")
print("6. 🧪 Paper trading por 30 días")
print("7. 💰 Live trading con capital pequeño")
print()

print("=" * 80)
print("✅ ANÁLISIS COMPLETADO")
print("=" * 80)
print()
print("La estrategia híbrida combina:")
print("  • Control de riesgo de MTF (DD bajo)")
print("  • Profit Factor de Supertrend + RSI")
print("  • Multi-timeframe para filtrar señales")
print("  • Alta frecuencia con M5")
print()
print("Siguiente paso: Implementar y backestear la estrategia híbrida.")
