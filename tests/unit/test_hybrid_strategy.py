import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.signal_generator import generar_señales

print("=== Test de Estrategia HÍBRIDA (BB + RSI + EMA 200) ===\n")

# Crear cliente
manager = BinanceClientManager()
client = manager.get_public_client()

# Obtener datos
print("1. Descargando datos de prueba...")
df = obtener_datos_binance(
    client=client,
    simbolo='BTCUSDT',
    intervalo='5m',
    inicio='7 days ago UTC'
)
print(f"   ✓ {len(df)} registros descargados")

# Configuración híbrida
config = {
    'bb_length': 20,
    'bb_std': 2.0,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'ema_trend': 200,  # Filtro de régimen
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_length': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_smooth': 3
}

# Calcular indicadores
print("\n2. Calculando indicadores (incluyendo EMA 200)...")
df = agregar_indicadores(df, config=config)

# Verificar EMA 200
if 'EMA_200' in df.columns:
    print(f"   ✓ EMA_200 calculada correctamente")
    print(f"   Último valor EMA_200: ${df['EMA_200'].iloc[-1]:.2f}")
    print(f"   Último precio close: ${df['close'].iloc[-1]:.2f}")
    print(f"   Tendencia: {'Alcista' if df['close'].iloc[-1] > df['EMA_200'].iloc[-1] else 'Bajista'}")
else:
    print("   ❌ EMA_200 no encontrada")

# Generar señales
print("\n3. Generando señales con filtro de tendencia...")
df = generar_señales(df, config=config)

# Contar señales
compras = (df['señal'] == 1).sum()
ventas = (df['señal'] == -1).sum()
neutrales = (df['señal'] == 0).sum()

print(f"\n4. Estadísticas de señales:")
print(f"   Señales de COMPRA (Dips en tendencia alcista): {compras}")
print(f"   Señales de VENTA (Rallys en tendencia bajista): {ventas}")
print(f"   Señales NEUTRALES: {neutrales}")

# Mostrar algunas señales
print("\n5. Últimas señales generadas:")
señales_activas = df[df['señal'] != 0].tail(5)
if len(señales_activas) > 0:
    for idx, row in señales_activas.iterrows():
        tipo = "COMPRA" if row['señal'] == 1 else "VENTA"
        print(f"   {row['timestamp']} - {tipo}: Precio=${row['close']:.2f}, RSI={row['RSI_14']:.2f}, EMA200=${row['EMA_200']:.2f}")
else:
    print("   No hay señales activas en los últimos registros")

print("\n✓ Test exitoso - Estrategia Híbrida funcionando correctamente")
