"""
Estrategia básica de cruces EMA (Long-Only).
Golden Cross / Death Cross con filtro de tendencia.
"""

import pandas as pd


def generar_señales(df, config=None):
    """
    Genera señales de trading para ESTRATEGIA LONG-ONLY (Solo Alcista).

    Lógica de señales:
    - COMPRA (1): (EMA rápida cruza por ENCIMA de EMA lenta) Y (precio > EMA filtro de tendencia)
    - VENTA (-1): EMA rápida cruza por DEBAJO de EMA lenta → CIERRE DE POSICIÓN LARGA
    - NEUTRAL (0): Sin cruce O precio bajo EMA de tendencia

    Esta estrategia NUNCA abre posiciones cortas. Solo opera en tendencias alcistas:
    1. Entra LONG cuando detecta inicio de tendencia alcista (cruce EMA + confirmación precio > EMA filtro)
    2. Sale cuando detecta fin de tendencia alcista (cruce bajista EMA)
    3. Permanece fuera del mercado durante tendencias bajistas

    Diseñada para timeframes cortos (15m) con alta frecuencia de señales.

    Args:
        df: DataFrame con indicadores calculados
        config: Diccionario con parámetros de estrategia

    Returns:
        DataFrame con columna 'señal' añadida
    """
    # Parámetros por defecto para estrategia Long-Only
    if config is None:
        config = {
            'ema_short': 9,      # EMA rápida (corta para 15m)
            'ema_long': 21,      # EMA lenta (media para 15m)
            'ema_filter': 50     # EMA de filtro de tendencia (larga para 15m)
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_short_col = f"EMA_{config['ema_short']}"
    ema_long_col = f"EMA_{config['ema_long']}"
    ema_filter_col = f"EMA_{config['ema_filter']}"

    # Verificar que las columnas existan
    required_cols = [ema_short_col, ema_long_col, ema_filter_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing_cols}. "
                        f"Asegúrate de calcular indicadores primero. "
                        f"Columnas disponibles: {df.columns.tolist()}")

    # Inicializar columna de señales en 0 (NEUTRAL)
    df['señal'] = 0

    # FILTRO DE TENDENCIA ALCISTA: Precio debe estar por encima de EMA de filtro
    tendencia_alcista = df['close'] > df[ema_filter_col]

    # SEÑAL DE COMPRA: Golden Cross (EMA rápida cruza por ENCIMA de lenta) + Filtro de tendencia
    golden_cross = (
        (df[ema_short_col] > df[ema_long_col]) &
        (df[ema_short_col].shift(1) <= df[ema_long_col].shift(1))
    )

    # SEÑAL DE VENTA (CIERRE): Death Cross (EMA rápida cruza por DEBAJO de lenta)
    # Esto cierra la posición larga, NO abre posición corta
    death_cross = (
        (df[ema_short_col] < df[ema_long_col]) &
        (df[ema_short_col].shift(1) >= df[ema_long_col].shift(1))
    )

    # Asignar señales
    df.loc[golden_cross & tendencia_alcista, 'señal'] = 1   # COMPRA (abrir long)
    df.loc[death_cross, 'señal'] = -1                        # VENTA (cerrar long)

    # Crear columna de POSICIÓN para backtest vectorizado Long-Only
    # Esta columna indica si estamos DENTRO (1) o FUERA (0) de una posición larga
    df['position'] = 0
    in_position = False

    for i in range(len(df)):
        signal = df['señal'].iloc[i]

        if signal == 1 and not in_position:  # Abrir posición larga
            in_position = True
        elif signal == -1 and in_position:   # Cerrar posición larga
            in_position = False

        df.iloc[i, df.columns.get_loc('position')] = 1 if in_position else 0

    return df
