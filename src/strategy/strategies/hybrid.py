"""
Estrategia Híbrida de 4 capas (Long-Only).
Régimen + Momentum RSI + MACD + ATR.
"""

import pandas as pd


def generar_senales_hibrido_v1(df, config=None):
    """
    Genera señales de trading usando ESTRATEGIA HÍBRIDA DE 4 CAPAS (Módulo Híbrido v1).

    ITERACIÓN 12: Nueva estrategia híbrida que combina:
    - Filtro de Régimen (EMA_200)
    - Filtro de Momentum (RSI)
    - Señal de Entrada/Salida (MACD crossover)
    - Gestión de Riesgo (ATR Stop Loss dinámico)

    ARQUITECTURA DE 4 CAPAS (LONG-ONLY):

    CAPA 1 (Filtro de Régimen - Tendencia Macro):
    - Indicador: EMA(200) como proxy de tendencia principal
    - Lógica: Solo abre posiciones LARGAS si precio > EMA(200)
    - Objetivo: Evitar operar contra la tendencia principal
    - Régimen ALCISTA: precio > EMA_200 → Permite operar LONG
    - Régimen BAJISTA: precio < EMA_200 → Fuera del mercado

    CAPA 2 (Filtro de Momentum - RSI):
    - Indicador: RSI(14)
    - Lógica: Solo compra si RSI > rsi_momentum_level (default: 50)
    - Objetivo: Confirmar que hay momentum alcista (no comprar en debilidad)
    - RSI > 50: Fuerza alcista confirmada
    - RSI < 50: Sin suficiente momentum (evitar entrada)

    CAPA 3 (Señal de Entrada/Salida - MACD):
    - Indicador: MACD (12, 26, 9)
    - Lógica COMPRA: Cruce alcista del MACD (MACD cruza por ENCIMA de Signal)
      * Condición: (MACD[-1] < Signal[-1]) AND (MACD >= Signal)
    - Lógica VENTA/TP: Cruce bajista del MACD (MACD cruza por DEBAJO de Signal)
      * Condición: (MACD[-1] > Signal[-1]) AND (MACD <= Signal)
    - Objetivo: Timing preciso de entrada/salida en tendencia confirmada

    CAPA 4 (Gestión de Riesgo - ATR):
    - Indicador: ATR(14)
    - Lógica: Stop Loss dinámico basado en volatilidad actual del mercado
      * SL = Precio_Entrada - (ATR × atr_multiplier)
    - Objetivo: Adaptar el riesgo a las condiciones de mercado
    - NOTA: La lógica de SL se implementa en el motor de backtesting, NO aquí

    FILOSOFÍA:
    Este es un sistema "Long-Only" que solo opera cuando las 4 capas están alineadas:
    1. Régimen correcto (precio > EMA_200)
    2. Momentum alcista confirmado (RSI > nivel de momentum)
    3. Timing de entrada apropiado (cruce alcista MACD)
    4. Gestión de riesgo adaptativa (ATR Stop Loss)

    Comparación con Iteraciones Anteriores:
    - Iteración 10.1 (Estocástico): Comprar debilidad → Win Rate 0%
    - Iteración 11.1 (Momentum): Comprar breakouts Donchian → Win Rate 5%, baja frecuencia
    - Iteración 12 (Híbrido): Combinar régimen + momentum + MACD → Hipótesis: Win Rate >35%, mayor frecuencia

    Args:
        df: DataFrame con indicadores calculados (debe contener EMA_200, RSI_14, MACD, ATR)
        config: Diccionario con parámetros:
            - ema_trend: Período de EMA para filtro de régimen (default: 200)
            - rsi_period: Período del RSI (default: 14)
            - rsi_momentum_level: Nivel mínimo de RSI para entrar (default: 50)
            - macd_fast: Período rápido del MACD (default: 12)
            - macd_slow: Período lento del MACD (default: 26)
            - macd_signal: Período de señal del MACD (default: 9)
            - atr_length: Período del ATR (default: 14)

    Returns:
        DataFrame con columna 'señal' añadida:
            1 = COMPRA (abrir posición LONG)
           -1 = VENTA/TP (cerrar posición LONG)
            0 = NEUTRAL (sin acción)

        Y columna 'position' para tracking Long-Only en backtest.
    """
    # Parámetros por defecto - Módulo Híbrido v1
    if config is None:
        config = {
            'ema_trend': 200,           # EMA de tendencia (filtro macro)
            'rsi_period': 14,           # RSI para momentum
            'rsi_momentum_level': 50,   # Nivel mínimo de RSI para entrar
            'macd_fast': 12,            # MACD rápido
            'macd_slow': 26,            # MACD lento
            'macd_signal': 9,           # MACD señal
            'atr_length': 14            # ATR para gestión de riesgo
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_trend_col = f"EMA_{config['ema_trend']}"
    rsi_col = f"RSI_{config['rsi_period']}"
    macd_col = f"MACD_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"
    macd_signal_col = f"MACDs_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"
    atr_col = f"ATRr_{config['atr_length']}"  # pandas-ta usa 'ATRr' en lugar de 'ATR'

    # Verificar que las columnas existan
    required_cols = [ema_trend_col, rsi_col, macd_col, macd_signal_col, atr_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(
            f"Columnas faltantes en DataFrame: {missing_cols}\n"
            f"Asegúrate de calcular indicadores primero.\n"
            f"Columnas disponibles: {df.columns.tolist()}"
        )

    # Inicializar columna de señales en 0 (NEUTRAL)
    df['señal'] = 0

    # ==========================================
    # CAPA 1: FILTRO DE RÉGIMEN (EMA_200)
    # ==========================================
    # Solo operamos en LONG cuando estamos en régimen alcista
    condicion_regimen = df['close'] > df[ema_trend_col]

    # ==========================================
    # CAPA 2: FILTRO DE MOMENTUM (RSI)
    # ==========================================
    # Solo entramos si RSI confirma momentum alcista
    condicion_momentum = df[rsi_col] > config['rsi_momentum_level']

    # ==========================================
    # CAPA 3: SEÑAL DE ENTRADA (CRUCE MACD ALCISTA)
    # ==========================================
    # COMPRA: MACD cruza por ENCIMA de su línea de señal
    condicion_entrada = (
        (df[macd_col].shift(1) < df[macd_signal_col].shift(1)) &  # MACD anterior estaba por debajo
        (df[macd_col] >= df[macd_signal_col])                      # MACD actual cruza hacia arriba
    )

    # ==========================================
    # CAPA 3: SEÑAL DE SALIDA (CRUCE MACD BAJISTA)
    # ==========================================
    # VENTA/TP: MACD cruza por DEBAJO de su línea de señal
    condicion_salida_tp = (
        (df[macd_col].shift(1) > df[macd_signal_col].shift(1)) &  # MACD anterior estaba por encima
        (df[macd_col] <= df[macd_signal_col])                      # MACD actual cruza hacia abajo
    )

    # ==========================================
    # GENERACIÓN DE SEÑALES (CONFLUENCIA DE 4 CAPAS)
    # ==========================================

    # COMPRA: Régimen alcista (Capa 1) + Momentum (Capa 2) + Cruce alcista MACD (Capa 3)
    df.loc[condicion_regimen & condicion_momentum & condicion_entrada, 'señal'] = 1

    # VENTA/TP: Cruce bajista MACD (Capa 3)
    # No necesitamos filtros para salir - queremos proteger ganancias
    df.loc[condicion_salida_tp, 'señal'] = -1

    # ==========================================
    # COLUMNA DE POSICIÓN PARA BACKTEST LONG-ONLY
    # ==========================================
    # Esta columna indica si estamos DENTRO (1) o FUERA (0) de una posición larga
    # El motor de backtesting usa esta columna para simular la estrategia
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
