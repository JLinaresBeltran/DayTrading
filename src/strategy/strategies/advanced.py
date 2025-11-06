"""
Estrategias avanzadas con múltiples indicadores.
"""

import pandas as pd


def generar_señales_avanzadas(df, config=None):
    """
    Genera señales de trading con lógica más avanzada (múltiples indicadores).

    Lógica:
    - COMPRA: EMA alcista + RSI favorable + MACD positivo + Precio cerca de banda inferior
    - VENTA: EMA bajista + RSI desfavorable + MACD negativo + Precio cerca de banda superior

    Args:
        df: DataFrame con indicadores
        config: Parámetros de estrategia

    Returns:
        DataFrame con columna 'señal' añadida
    """
    if config is None:
        config = {
            'ema_short': 21,
            'ema_long': 50,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_length': 20,
            'bb_std': 2
        }

    df = df.copy()

    # Nombres de columnas
    ema_short_col = f"EMA_{config['ema_short']}"
    ema_long_col = f"EMA_{config['ema_long']}"
    rsi_col = f"RSI_{config['rsi_period']}"
    macd_col = f"MACD_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"
    macd_signal_col = f"MACDs_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"
    bb_lower_col = f"BBL_{config['bb_length']}_{config['bb_std']}.0"
    bb_upper_col = f"BBU_{config['bb_length']}_{config['bb_std']}.0"

    # Inicializar señales
    df['señal'] = 0

    # COMPRA (múltiples confirmaciones)
    condicion_compra = (
        (df[ema_short_col] > df[ema_long_col]) &        # Tendencia alcista
        (df[rsi_col] > 30) & (df[rsi_col] < 70) &       # RSI en rango neutral-alcista
        (df[macd_col] > df[macd_signal_col]) &          # MACD alcista
        (df['close'] > df[bb_lower_col])                # Precio por encima de banda inferior
    )
    df.loc[condicion_compra, 'señal'] = 1

    # VENTA (múltiples confirmaciones)
    condicion_venta = (
        (df[ema_short_col] < df[ema_long_col]) &        # Tendencia bajista
        (df[rsi_col] > 30) & (df[rsi_col] < 70) &       # RSI en rango neutral-bajista
        (df[macd_col] < df[macd_signal_col]) &          # MACD bajista
        (df['close'] < df[bb_upper_col])                # Precio por debajo de banda superior
    )
    df.loc[condicion_venta, 'señal'] = -1

    return df


def generar_señales_con_filtro_tendencia(df, config=None):
    """
    Genera señales solo cuando hay una tendencia clara (filtro de mercado lateral).

    Args:
        df: DataFrame con indicadores
        config: Parámetros de estrategia

    Returns:
        DataFrame con columna 'señal' añadida
    """
    # Import local para evitar dependencias circulares
    from .ema_cross import generar_señales

    if config is None:
        config = {
            'ema_short': 21,
            'ema_long': 50,
            'rsi_period': 14,
            'atr_length': 14,
            'min_atr_threshold': 0.5  # ATR mínimo como % del precio
        }

    df = df.copy()

    # Calcular fuerza de tendencia usando ATR
    atr_col = f"ATR_{config['atr_length']}"
    df['atr_pct'] = (df[atr_col] / df['close']) * 100  # ATR como % del precio

    # Solo generar señales si hay volatilidad suficiente (mercado no lateral)
    hay_tendencia = df['atr_pct'] > config['min_atr_threshold']

    # Generar señales base
    df = generar_señales(df, config)

    # Filtrar señales en mercado lateral
    df.loc[~hay_tendencia, 'señal'] = 0

    return df
