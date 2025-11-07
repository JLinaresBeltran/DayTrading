"""
Detector de Patrones de Velas Japonesas y Patrones Chartistas

Incluye:
1. Patrones de velas japonesas (Doji, Hammer, Engulfing, etc.)
2. Patrones chartistas (Soportes/Resistencias, Triángulos, Canales)
3. Estructura de mercado (Higher Highs, Lower Lows, etc.)
"""

import pandas as pd
import numpy as np
from scipy.signal import argrelextrema


# ============================================================================
# PATRONES DE VELAS JAPONESAS
# ============================================================================

def detect_doji(df, threshold=0.1):
    """
    Detecta patrón Doji: Vela con cuerpo muy pequeño (indecisión).

    Args:
        df: DataFrame con OHLC
        threshold: Umbral del cuerpo como % del rango total (default 0.1 = 10%)

    Returns:
        Series booleana
    """
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']

    # Evitar división por cero
    total_range = total_range.replace(0, np.nan)

    body_pct = body / total_range

    return body_pct < threshold


def detect_hammer(df, body_ratio=0.3, shadow_ratio=2.0):
    """
    Detecta patrón Hammer: Señal alcista de reversión.

    Características:
    - Cuerpo pequeño en la parte superior
    - Sombra inferior larga (al menos 2x el cuerpo)
    - Sombra superior pequeña o inexistente
    - Aparece en tendencia bajista

    Args:
        df: DataFrame con OHLC
        body_ratio: Máximo tamaño del cuerpo como % del rango total
        shadow_ratio: Mínima longitud de sombra inferior vs cuerpo

    Returns:
        Series booleana
    """
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    lower_shadow = np.minimum(df['open'], df['close']) - df['low']
    upper_shadow = df['high'] - np.maximum(df['open'], df['close'])

    # Evitar división por cero
    body = body.replace(0, 1e-10)
    total_range = total_range.replace(0, np.nan)

    is_hammer = (
        (body / total_range < body_ratio) &  # Cuerpo pequeño
        (lower_shadow / body > shadow_ratio) &  # Sombra inferior larga
        (upper_shadow < body)  # Sombra superior pequeña
    )

    return is_hammer


def detect_inverted_hammer(df, body_ratio=0.3, shadow_ratio=2.0):
    """
    Detecta patrón Inverted Hammer (Martillo Invertido).

    Similar al Hammer pero con sombra superior larga.
    """
    body = abs(df['close'] - df['open'])
    total_range = df['high'] - df['low']
    lower_shadow = np.minimum(df['open'], df['close']) - df['low']
    upper_shadow = df['high'] - np.maximum(df['open'], df['close'])

    body = body.replace(0, 1e-10)
    total_range = total_range.replace(0, np.nan)

    is_inverted_hammer = (
        (body / total_range < body_ratio) &
        (upper_shadow / body > shadow_ratio) &
        (lower_shadow < body)
    )

    return is_inverted_hammer


def detect_shooting_star(df, body_ratio=0.3, shadow_ratio=2.0):
    """
    Detecta patrón Shooting Star: Señal bajista de reversión.

    Similar al Inverted Hammer pero aparece en tendencia alcista.
    """
    return detect_inverted_hammer(df, body_ratio, shadow_ratio)


def detect_engulfing_bullish(df):
    """
    Detecta patrón Bullish Engulfing: Vela alcista que envuelve a la anterior bajista.

    Señal alcista fuerte.
    """
    # Vela anterior bajista
    prev_bearish = df['close'].shift(1) < df['open'].shift(1)

    # Vela actual alcista
    curr_bullish = df['close'] > df['open']

    # Vela actual envuelve a la anterior
    engulfs = (
        (df['open'] < df['close'].shift(1)) &  # Abre por debajo del cierre anterior
        (df['close'] > df['open'].shift(1))    # Cierra por encima de la apertura anterior
    )

    return prev_bearish & curr_bullish & engulfs


def detect_engulfing_bearish(df):
    """
    Detecta patrón Bearish Engulfing: Vela bajista que envuelve a la anterior alcista.

    Señal bajista fuerte.
    """
    # Vela anterior alcista
    prev_bullish = df['close'].shift(1) > df['open'].shift(1)

    # Vela actual bajista
    curr_bearish = df['close'] < df['open']

    # Vela actual envuelve a la anterior
    engulfs = (
        (df['open'] > df['close'].shift(1)) &
        (df['close'] < df['open'].shift(1))
    )

    return prev_bullish & curr_bearish & engulfs


def detect_morning_star(df):
    """
    Detecta patrón Morning Star: Reversión alcista de 3 velas.

    1. Vela bajista grande
    2. Vela pequeña (gap down)
    3. Vela alcista grande que cierra en la mitad superior de la primera
    """
    # Vela 1: Bajista grande
    candle1_bearish = (df['close'].shift(2) < df['open'].shift(2))
    candle1_body = abs(df['close'].shift(2) - df['open'].shift(2))

    # Vela 2: Pequeña (doji o spinning top)
    candle2_small = abs(df['close'].shift(1) - df['open'].shift(1)) < candle1_body * 0.3

    # Vela 3: Alcista grande
    candle3_bullish = (df['close'] > df['open'])
    candle3_body = abs(df['close'] - df['open'])

    # Vela 3 cierra en la mitad superior de vela 1
    closes_high = df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2

    return candle1_bearish & candle2_small & candle3_bullish & closes_high


def detect_evening_star(df):
    """
    Detecta patrón Evening Star: Reversión bajista de 3 velas.

    Similar a Morning Star pero invertido.
    """
    # Vela 1: Alcista grande
    candle1_bullish = (df['close'].shift(2) > df['open'].shift(2))
    candle1_body = abs(df['close'].shift(2) - df['open'].shift(2))

    # Vela 2: Pequeña
    candle2_small = abs(df['close'].shift(1) - df['open'].shift(1)) < candle1_body * 0.3

    # Vela 3: Bajista grande
    candle3_bearish = (df['close'] < df['open'])

    # Vela 3 cierra en la mitad inferior de vela 1
    closes_low = df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2

    return candle1_bullish & candle2_small & candle3_bearish & closes_low


def detect_three_white_soldiers(df):
    """
    Detecta patrón Three White Soldiers: 3 velas alcistas consecutivas.

    Señal alcista muy fuerte.
    """
    # 3 velas alcistas consecutivas
    bull1 = df['close'].shift(2) > df['open'].shift(2)
    bull2 = df['close'].shift(1) > df['open'].shift(1)
    bull3 = df['close'] > df['open']

    # Cada vela cierra más alto que la anterior
    higher_closes = (
        (df['close'].shift(1) > df['close'].shift(2)) &
        (df['close'] > df['close'].shift(1))
    )

    # Cada vela abre dentro del cuerpo de la anterior
    opens_inside = (
        (df['open'].shift(1) > df['open'].shift(2)) &
        (df['open'].shift(1) < df['close'].shift(2)) &
        (df['open'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )

    return bull1 & bull2 & bull3 & higher_closes


def detect_three_black_crows(df):
    """
    Detecta patrón Three Black Crows: 3 velas bajistas consecutivas.

    Señal bajista muy fuerte.
    """
    # 3 velas bajistas consecutivas
    bear1 = df['close'].shift(2) < df['open'].shift(2)
    bear2 = df['close'].shift(1) < df['open'].shift(1)
    bear3 = df['close'] < df['open']

    # Cada vela cierra más bajo que la anterior
    lower_closes = (
        (df['close'].shift(1) < df['close'].shift(2)) &
        (df['close'] < df['close'].shift(1))
    )

    return bear1 & bear2 & bear3 & lower_closes


# ============================================================================
# PATRONES CHARTISTAS: SOPORTES Y RESISTENCIAS
# ============================================================================

def find_support_resistance(df, window=20, num_levels=3):
    """
    Encuentra niveles de soporte y resistencia usando pivots.

    Args:
        df: DataFrame con OHLC
        window: Ventana para detectar pivots
        num_levels: Número de niveles a retornar

    Returns:
        DataFrame con columnas support_X y resistance_X
    """
    df = df.copy()

    # Encontrar máximos y mínimos locales
    local_max_idx = argrelextrema(df['high'].values, np.greater, order=window)[0]
    local_min_idx = argrelextrema(df['low'].values, np.less, order=window)[0]

    # Niveles de resistencia (máximos locales)
    resistances = df.iloc[local_max_idx]['high'].values

    # Niveles de soporte (mínimos locales)
    supports = df.iloc[local_min_idx]['low'].values

    # Agrupar niveles cercanos
    def cluster_levels(levels, tolerance=0.02):
        """Agrupa niveles que están dentro del tolerance%"""
        if len(levels) == 0:
            return []

        levels = np.sort(levels)[::-1]  # Ordenar descendente
        clustered = []
        current_cluster = [levels[0]]

        for level in levels[1:]:
            if abs(level - current_cluster[0]) / current_cluster[0] < tolerance:
                current_cluster.append(level)
            else:
                clustered.append(np.mean(current_cluster))
                current_cluster = [level]

        clustered.append(np.mean(current_cluster))
        return clustered

    # Clusterizar
    resistance_levels = cluster_levels(resistances)[:num_levels]
    support_levels = cluster_levels(supports)[:num_levels]

    # Añadir al DataFrame
    for i in range(num_levels):
        if i < len(resistance_levels):
            df[f'resistance_{i+1}'] = resistance_levels[i]
        else:
            df[f'resistance_{i+1}'] = np.nan

        if i < len(support_levels):
            df[f'support_{i+1}'] = support_levels[i]
        else:
            df[f'support_{i+1}'] = np.nan

    return df


def detect_near_support_resistance(df, tolerance=0.01):
    """
    Detecta si el precio está cerca de soporte o resistencia.

    Args:
        df: DataFrame con columnas support_X y resistance_X
        tolerance: Distancia como % del precio (default 1%)

    Returns:
        DataFrame con columnas near_support y near_resistance
    """
    df = df.copy()

    # Buscar todas las columnas de soporte y resistencia
    support_cols = [col for col in df.columns if col.startswith('support_')]
    resistance_cols = [col for col in df.columns if col.startswith('resistance_')]

    # Inicializar
    df['near_support'] = False
    df['near_resistance'] = False
    df['dist_to_support'] = np.inf
    df['dist_to_resistance'] = np.inf

    # Calcular distancia al soporte más cercano
    for col in support_cols:
        dist = abs(df['close'] - df[col]) / df['close']
        df.loc[dist < tolerance, 'near_support'] = True
        df['dist_to_support'] = np.minimum(df['dist_to_support'], dist)

    # Calcular distancia a la resistencia más cercana
    for col in resistance_cols:
        dist = abs(df['close'] - df[col]) / df['close']
        df.loc[dist < tolerance, 'near_resistance'] = True
        df['dist_to_resistance'] = np.minimum(df['dist_to_resistance'], dist)

    return df


# ============================================================================
# ESTRUCTURA DE MERCADO
# ============================================================================

def detect_market_structure(df, window=10):
    """
    Detecta estructura de mercado: Higher Highs, Lower Lows, etc.

    Returns:
        DataFrame con columnas:
        - higher_high: Máximo más alto que el anterior
        - lower_low: Mínimo más bajo que el anterior
        - higher_low: Mínimo más alto (alcista)
        - lower_high: Máximo más bajo (bajista)
        - market_structure: 1 (alcista), -1 (bajista), 0 (lateral)
    """
    df = df.copy()

    # Rolling highs y lows
    rolling_high = df['high'].rolling(window).max()
    rolling_low = df['low'].rolling(window).min()

    # Higher High: nuevo máximo
    df['higher_high'] = df['high'] > rolling_high.shift(1)

    # Lower Low: nuevo mínimo
    df['lower_low'] = df['low'] < rolling_low.shift(1)

    # Higher Low: mínimo más alto (señal alcista)
    prev_low = df['low'].rolling(window).min().shift(window)
    df['higher_low'] = (df['low'] > prev_low) & (df['low'] == rolling_low)

    # Lower High: máximo más bajo (señal bajista)
    prev_high = df['high'].rolling(window).max().shift(window)
    df['lower_high'] = (df['high'] < prev_high) & (df['high'] == rolling_high)

    # Estructura general del mercado
    df['market_structure'] = 0  # Lateral por defecto
    df.loc[df['higher_high'] & df['higher_low'], 'market_structure'] = 1   # Alcista
    df.loc[df['lower_high'] & df['lower_low'], 'market_structure'] = -1    # Bajista

    return df


# ============================================================================
# FUNCIÓN PRINCIPAL: AGREGAR TODOS LOS PATRONES
# ============================================================================

def add_candlestick_patterns(df):
    """
    Agrega todos los patrones de velas y chartistas al DataFrame.

    Args:
        df: DataFrame con OHLC

    Returns:
        DataFrame con columnas adicionales de patrones
    """
    df = df.copy()

    # ========================================================================
    # PATRONES DE VELAS JAPONESAS
    # ========================================================================

    df['pattern_doji'] = detect_doji(df).astype(int)
    df['pattern_hammer'] = detect_hammer(df).astype(int)
    df['pattern_inverted_hammer'] = detect_inverted_hammer(df).astype(int)
    df['pattern_shooting_star'] = detect_shooting_star(df).astype(int)
    df['pattern_engulfing_bullish'] = detect_engulfing_bullish(df).astype(int)
    df['pattern_engulfing_bearish'] = detect_engulfing_bearish(df).astype(int)
    df['pattern_morning_star'] = detect_morning_star(df).astype(int)
    df['pattern_evening_star'] = detect_evening_star(df).astype(int)
    df['pattern_three_white_soldiers'] = detect_three_white_soldiers(df).astype(int)
    df['pattern_three_black_crows'] = detect_three_black_crows(df).astype(int)

    # Contador de patrones alcistas y bajistas
    bullish_patterns = [
        'pattern_hammer',
        'pattern_inverted_hammer',
        'pattern_engulfing_bullish',
        'pattern_morning_star',
        'pattern_three_white_soldiers'
    ]

    bearish_patterns = [
        'pattern_shooting_star',
        'pattern_engulfing_bearish',
        'pattern_evening_star',
        'pattern_three_black_crows'
    ]

    df['bullish_pattern_count'] = df[bullish_patterns].sum(axis=1)
    df['bearish_pattern_count'] = df[bearish_patterns].sum(axis=1)
    df['pattern_score'] = df['bullish_pattern_count'] - df['bearish_pattern_count']

    # ========================================================================
    # PATRONES CHARTISTAS
    # ========================================================================

    df = find_support_resistance(df, window=20, num_levels=3)
    df = detect_near_support_resistance(df, tolerance=0.01)
    df = detect_market_structure(df, window=10)

    return df


if __name__ == "__main__":
    # Test
    print("Módulo de patrones de velas cargado correctamente")
    print("\nPatrones disponibles:")
    print("- Patrones de velas: Doji, Hammer, Engulfing, Morning/Evening Star, etc.")
    print("- Soportes y resistencias")
    print("- Estructura de mercado (Higher Highs, Lower Lows)")
