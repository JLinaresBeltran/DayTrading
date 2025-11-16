"""
Estrategia de Momentum con Canales de Donchian (Long-Only).
Breakout alcista como señal de entrada.
"""

import pandas as pd


def generar_senales_momentum_v1(df, config=None):
    """
    Genera señales de trading usando ESTRATEGIA DE MOMENTUM (Módulo de Momentum v1).

    ITERACIÓN 11: Pivote estratégico - Abandonamos Reversión a la Media (comprar debilidad)
    y adoptamos Momentum (comprar fuerza con breakouts de Donchian Channels).

    Estrategia de 3 Capas (LONG-ONLY):

    CAPA 1 (Filtro de Régimen):
    - Solo opera cuando price > EMA_200 (tendencia alcista establecida)
    - Evita operar en mercados bajistas o laterales

    CAPA 2 (Señal de Entrada - Breakout Alcista):
    - Compra cuando el precio rompe HACIA ARRIBA el Canal de Donchian Superior
    - Condición: close[-1] < DONCHI_h_{period} AND close >= DONCHI_h_{period}
    - Rationale: Breakout de nuevos máximos indica momentum alcista fuerte

    CAPA 3 (Señal de Salida - Trailing Stop):
    - Vende cuando el precio rompe HACIA ABAJO el Canal de Donchian Inferior
    - Condición: close[-1] > DONCHI_l_{period} AND close <= DONCHI_l_{period}
    - Rationale: Romper el canal inferior indica pérdida de momentum
    - NOTA: El Stop Loss ATR actúa como protección adicional (implementado en backtester)

    Diferencias clave vs Iteración 10.1 (Estocástico):
    - Iteración 10.1: Comprar en DEBILIDAD (Estocástico <20) → Win Rate 0%
    - Iteración 11: Comprar en FUERZA (Breakout de máximos) → Hipótesis: Win Rate >40%

    Args:
        df: DataFrame con indicadores calculados
        config: Diccionario con parámetros de estrategia
            - ema_trend: Período de EMA de régimen (default: 200)
            - donchian_period: Período del Canal de Donchian (default: 20)
            - atr_length: Período del ATR para SL (default: 14, usado por backtester)

    Returns:
        DataFrame con columnas 'señal' y 'position' añadidas
    """
    # Parámetros por defecto
    if config is None:
        config = {
            'ema_trend': 200,
            'donchian_period': 20,
            'atr_length': 14
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_trend_col = f"EMA_{config['ema_trend']}"
    donchian_period = config['donchian_period']
    donchian_upper_col = f"DONCHI_h_{donchian_period}"
    donchian_lower_col = f"DONCHI_l_{donchian_period}"

    # Verificar que las columnas requeridas existan
    required_cols = [ema_trend_col, donchian_upper_col, donchian_lower_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing_cols}. "
                        f"Asegúrate de calcular indicadores primero. "
                        f"Columnas disponibles: {df.columns.tolist()}")

    # Inicializar columna de señales en 0 (NEUTRAL)
    df['señal'] = 0

    # ==========================================
    # CAPA 1: FILTRO DE RÉGIMEN (EMA_200)
    # ==========================================
    # Solo opera cuando el precio está por encima de EMA_200 (tendencia alcista)
    condicion_regimen_alcista = df['close'] > df[ema_trend_col]

    # ==========================================
    # CAPA 2: ENTRADA - BREAKOUT ALCISTA
    # ==========================================
    # Señal de COMPRA: Precio cruza HACIA ARRIBA el canal superior de Donchian
    # Esto indica un nuevo máximo de N períodos (breakout alcista)
    condicion_entrada_breakout = (
        (df['close'].shift(1) < df[donchian_upper_col].shift(1)) &
        (df['close'] >= df[donchian_upper_col])
    )

    # ==========================================
    # CAPA 3: SALIDA - TRAILING STOP LÓGICO
    # ==========================================
    # Señal de VENTA (Take Profit/Trailing Stop): Precio cruza HACIA ABAJO el canal inferior
    # Esto indica pérdida de momentum (rompe mínimo de N períodos)
    condicion_salida_trailing = (
        (df['close'].shift(1) > df[donchian_lower_col].shift(1)) &
        (df['close'] <= df[donchian_lower_col])
    )

    # ==========================================
    # GENERACIÓN DE SEÑALES
    # ==========================================
    # COMPRA: Solo si hay régimen alcista Y breakout del canal superior
    df.loc[condicion_regimen_alcista & condicion_entrada_breakout, 'señal'] = 1

    # VENTA: Cuando se rompe el canal inferior (Trailing Stop)
    # NOTA: El Stop Loss ATR se maneja en el backtester (run_backtest_with_stop_loss)
    df.loc[condicion_salida_trailing, 'señal'] = -1

    # ==========================================
    # COLUMNA DE POSICIÓN (Long-Only)
    # ==========================================
    # Crear columna 'position' que indica si estamos DENTRO (1) o FUERA (0) de posición
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
