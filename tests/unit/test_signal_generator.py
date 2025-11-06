import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.signal_generator import generar_señales

print("=== Test de Generador de Señales con BB + RSI ===\n")

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

# Configuración de prueba
config = {
    'bb_length': 20,
    'bb_std': 2,
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_length': 14,
    'stoch_k': 14,
    'stoch_d': 3,
    'stoch_smooth': 3
}

# Calcular indicadores
print("\n2. Calculando indicadores...")
df = agregar_indicadores(df, config=config)

# Mostrar columnas de BB generadas
bb_cols = [col for col in df.columns if col.startswith('BB')]
print(f"   Columnas de Bollinger Bands: {bb_cols}")

# Generar señales
print("\n3. Generando señales...")
df = generar_señales(df, config=config)

# Contar señales
compras = (df['señal'] == 1).sum()
ventas = (df['señal'] == -1).sum()
neutrales = (df['señal'] == 0).sum()

print(f"\n4. Estadísticas de señales:")
print(f"   Señales de COMPRA: {compras}")
print(f"   Señales de VENTA: {ventas}")
print(f"   Señales NEUTRALES: {neutrales}")

print("\n✓ Test exitoso - Estrategia de Reversión a la Media funcionando")
