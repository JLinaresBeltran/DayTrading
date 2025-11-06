"""
Módulo simplificado para calcular indicadores técnicos usando solo pandas.
Sin dependencias de pandas-ta.
"""

import pandas as pd
import numpy as np


def calcular_ema(df, column='close', period=20):
    """Calcula EMA (Exponential Moving Average)"""
    return df[column].ewm(span=period, adjust=False).mean()


def calcular_atr(df, period=14):
    """Calcula ATR (Average True Range)"""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR es el promedio móvil del True Range
    atr = tr.ewm(span=period, adjust=False).mean()

    return atr


def calcular_donchian(df, period=20):
    """
    Calcula Canales de Donchian.

    Returns:
        tuple: (upper_channel, lower_channel, middle_channel)
    """
    upper = df['high'].rolling(window=period).max()
    lower = df['low'].rolling(window=period).min()
    middle = (upper + lower) / 2

    return upper, lower, middle


def calcular_rsi(df, column='close', period=14):
    """Calcula RSI (Relative Strength Index)"""
    delta = df[column].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calcular_bollinger_bands(df, column='close', period=20, std_dev=2):
    """
    Calcula Bandas de Bollinger.

    Returns:
        tuple: (upper_band, middle_band, lower_band)
    """
    middle = df[column].rolling(window=period).mean()
    std = df[column].rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def calcular_macd(df, column='close', fast=12, slow=26, signal=9):
    """
    Calcula MACD (Moving Average Convergence Divergence).

    Returns:
        tuple: (macd_line, signal_line, histogram)
    """
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calcular_stochastic(df, period=14, smooth_k=3, smooth_d=3):
    """
    Calcula Oscilador Estocástico.

    Returns:
        tuple: (k_line, d_line)
    """
    lowest_low = df['low'].rolling(window=period).min()
    highest_high = df['high'].rolling(window=period).max()

    # %K
    k_raw = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))
    k_line = k_raw.rolling(window=smooth_k).mean()

    # %D
    d_line = k_line.rolling(window=smooth_d).mean()

    return k_line, d_line


def calcular_adx(df, period=14):
    """
    Calcula ADX (Average Directional Index).

    El ADX mide la fuerza de la tendencia (no la dirección).
    Valores:
    - ADX < 20: Tendencia débil o mercado lateral
    - ADX 20-25: Tendencia emergente
    - ADX > 25: Tendencia fuerte
    - ADX > 40: Tendencia muy fuerte

    Args:
        df: DataFrame con columnas OHLCV
        period: Período para el cálculo (default 14)

    Returns:
        Series: ADX values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calcular +DM y -DM (Directional Movement)
    high_diff = high.diff()
    low_diff = -low.diff()

    plus_dm = high_diff.where((high_diff > low_diff) & (high_diff > 0), 0)
    minus_dm = low_diff.where((low_diff > high_diff) & (low_diff > 0), 0)

    # Calcular ATR (True Range)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Suavizar con EMA
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    # Calcular DX (Directional Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calcular ADX (suavizado de DX)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx


def calcular_vwap(df):
    """
    Calcula VWAP (Volume Weighted Average Price) diario.

    El VWAP se reinicia cada día (nueva sesión).

    Args:
        df: DataFrame con columnas OHLCV y timestamp

    Returns:
        Series: VWAP values
    """
    df = df.copy()

    # Asegurar que timestamp sea datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
    else:
        # Si no hay timestamp, calcular VWAP acumulado simple
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        cumulative_tpv = (typical_price * df['volume']).cumsum()
        cumulative_volume = df['volume'].cumsum()
        return cumulative_tpv / cumulative_volume

    # Calcular Typical Price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Calcular TPV (Typical Price × Volume)
    df['tpv'] = typical_price * df['volume']

    # Agrupar por día y calcular VWAP diario
    df['cumulative_tpv'] = df.groupby('date')['tpv'].cumsum()
    df['cumulative_volume'] = df.groupby('date')['volume'].cumsum()

    # VWAP = Cumulative TPV / Cumulative Volume
    vwap = df['cumulative_tpv'] / df['cumulative_volume']

    return vwap


def agregar_indicadores(df, config=None):
    """
    Añade indicadores técnicos al DataFrame.
    Versión simplificada que usa solo pandas.

    Args:
        df: DataFrame con columnas OHLCV
        config: Diccionario con parámetros

    Returns:
        DataFrame con indicadores añadidos
    """
    # Parámetros por defecto
    if config is None:
        config = {
            'ema_short': 21,
            'ema_long': 50,
            'ema_trend': 200,
            'rsi_period': 14,
            'bb_length': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_length': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'donchian_period': 20
        }

    df = df.copy()

    # EMA
    if 'ema_short' in config:
        df[f"EMA_{config['ema_short']}"] = calcular_ema(df, period=config['ema_short'])
    if 'ema_long' in config:
        df[f"EMA_{config['ema_long']}"] = calcular_ema(df, period=config['ema_long'])
    if 'ema_trend' in config:
        df[f"EMA_{config['ema_trend']}"] = calcular_ema(df, period=config['ema_trend'])
    if 'ema_filter' in config:
        df[f"EMA_{config['ema_filter']}"] = calcular_ema(df, period=config['ema_filter'])

    # ATR
    atr_period = config.get('atr_length', 14)
    df[f"ATRr_{atr_period}"] = calcular_atr(df, period=atr_period)

    # RSI
    rsi_period = config.get('rsi_period', 14)
    df[f"RSI_{rsi_period}"] = calcular_rsi(df, period=rsi_period)

    # Bollinger Bands
    bb_length = config.get('bb_length', 20)
    bb_std = config.get('bb_std', 2)
    upper, middle, lower = calcular_bollinger_bands(df, period=bb_length, std_dev=bb_std)
    df[f"BBU_{bb_length}_{bb_std}.0"] = upper
    df[f"BBM_{bb_length}_{bb_std}.0"] = middle
    df[f"BBL_{bb_length}_{bb_std}.0"] = lower

    # MACD
    macd_fast = config.get('macd_fast', 12)
    macd_slow = config.get('macd_slow', 26)
    macd_signal = config.get('macd_signal', 9)
    macd_line, signal_line, histogram = calcular_macd(df, fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df[f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"] = macd_line
    df[f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"] = signal_line
    df[f"MACDh_{macd_fast}_{macd_slow}_{macd_signal}"] = histogram

    # Stochastic
    stoch_k = config.get('stoch_k', 14)
    stoch_d = config.get('stoch_d', 3)
    stoch_smooth = config.get('stoch_smooth', 3)
    k_line, d_line = calcular_stochastic(df, period=stoch_k, smooth_k=stoch_smooth, smooth_d=stoch_d)
    df[f"STOCHk_{stoch_k}_{stoch_d}_{stoch_smooth}"] = k_line
    df[f"STOCHd_{stoch_k}_{stoch_d}_{stoch_smooth}"] = d_line

    # ADX (Average Directional Index)
    adx_period = config.get('adx_period', 14)
    df[f"ADX_{adx_period}"] = calcular_adx(df, period=adx_period)

    # Canales de Donchian
    if 'donchian_period' in config:
        periods = config['donchian_period'] if isinstance(config['donchian_period'], list) else [config['donchian_period']]

        for period in periods:
            upper, lower, middle = calcular_donchian(df, period=period)
            df[f'DONCHI_h_{period}'] = upper
            df[f'DONCHI_l_{period}'] = lower
            df[f'DONCHIm_{period}'] = middle

    # VWAP (Volume Weighted Average Price)
    if config.get('vwap', False):
        df['VWAP'] = calcular_vwap(df)

    return df


if __name__ == "__main__":
    # Test
    print("=== Test de Indicadores Técnicos ===")

    # Crear datos de prueba
    dates = pd.date_range('2024-01-01', periods=100, freq='1h')
    df_test = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100)
    })

    print("\n1. Calculando indicadores...")
    df_test = agregar_indicadores(df_test)

    print(f"\n2. Indicadores calculados: {len([c for c in df_test.columns if c not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])}")
    print("\nColumnas añadidas:")
    for col in df_test.columns:
        if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
            print(f"  - {col}")

    print("\n✓ Test completado exitosamente")
