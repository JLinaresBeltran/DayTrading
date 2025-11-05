"""
Módulo para calcular indicadores técnicos usando solo pandas y numpy.
Versión simplificada sin pandas-ta para entornos con dependencias limitadas.
"""

import pandas as pd
import numpy as np


def agregar_indicadores_simple(df, config=None):
    """
    Añade indicadores técnicos al DataFrame usando solo pandas y numpy.

    Indicadores incluidos:
    - EMA (Exponential Moving Average)
    - ATR (Average True Range)
    - Canales de Donchian

    Args:
        df: DataFrame con columnas OHLCV (open, high, low, close, volume)
        config: Diccionario con parámetros personalizados (opcional)

    Returns:
        DataFrame con indicadores añadidos
    """
    # Parámetros por defecto
    if config is None:
        config = {
            'ema_trend': 200,
            'donchian_period': 20,
            'atr_length': 14
        }

    # Hacer una copia para evitar modificar el original
    df = df.copy()

    # EMA (Exponential Moving Average)
    if 'ema_trend' in config:
        df[f'EMA_{config["ema_trend"]}'] = df['close'].ewm(span=config['ema_trend'], adjust=False).mean()

    if 'ema_filter' in config:
        df[f'EMA_{config["ema_filter"]}'] = df['close'].ewm(span=config['ema_filter'], adjust=False).mean()

    # ATR (Average True Range)
    atr_length = config.get('atr_length', 14)
    df['h-l'] = df['high'] - df['low']
    df['h-pc'] = abs(df['high'] - df['close'].shift(1))
    df['l-pc'] = abs(df['low'] - df['close'].shift(1))
    df['tr'] = df[['h-l', 'h-pc', 'l-pc']].max(axis=1)
    df[f'ATRr_{atr_length}'] = df['tr'].rolling(window=atr_length).mean()

    # Limpiar columnas temporales
    df.drop(['h-l', 'h-pc', 'l-pc', 'tr'], axis=1, inplace=True)

    # Canales de Donchian
    if 'donchian_period' in config:
        periods = config['donchian_period'] if isinstance(config['donchian_period'], list) else [config['donchian_period']]

        for period in periods:
            # DONCHI_h = Donchian Channel Upper (máximo de N períodos)
            # DONCHI_l = Donchian Channel Lower (mínimo de N períodos)
            # DONCHIm = Donchian Channel Middle (promedio de upper y lower)
            df[f'DONCHI_h_{period}'] = df['high'].rolling(window=period).max()
            df[f'DONCHI_l_{period}'] = df['low'].rolling(window=period).min()
            df[f'DONCHIm_{period}'] = (df[f'DONCHI_h_{period}'] + df[f'DONCHI_l_{period}']) / 2

    # Rellenar valores NaN al principio
    df.fillna(method='bfill', inplace=True)

    return df


if __name__ == "__main__":
    # Test básico
    print("=== Test de Indicadores Técnicos Simples ===\n")

    # Crear datos de prueba
    dates = pd.date_range('2024-01-01', periods=100, freq='15min')
    np.random.seed(42)

    df = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 2500,
        'high': np.random.randn(100).cumsum() + 2510,
        'low': np.random.randn(100).cumsum() + 2490,
        'close': np.random.randn(100).cumsum() + 2500,
        'volume': np.random.uniform(1000, 10000, 100)
    })

    print("1. Calculando indicadores técnicos...")
    df = agregar_indicadores_simple(df, config={
        'ema_trend': 20,
        'donchian_period': 10,
        'atr_length': 14
    })

    print(f"   ✓ Datos con indicadores: {len(df)} filas, {len(df.columns)} columnas")
    print(f"   ✓ Columnas: {df.columns.tolist()}")

    print("\n2. Últimas 5 filas:")
    print(df.tail())

    print("\n✓ Test completado exitosamente")
