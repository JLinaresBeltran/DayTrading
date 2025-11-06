"""
Estrategia Bajista de 4 capas (Short-Only).
Inversión de lógica Long para operar en tendencias bajistas.
"""

import pandas as pd


def generar_senales_bajista_v1(df, config=None):
    """
    Genera señales de trading usando ESTRATEGIA BAJISTA DE 4 CAPAS (Módulo Bajista v1).

    ITERACIÓN 13: Pivote estratégico - Invertir la lógica de las estrategias Long-Only fallidas.
    Todas las iteraciones Long (10, 11, 12) resultaron en pérdidas. Esta estrategia prueba
    la hipótesis de que las señales de compra fallidas eran, de hecho, señales de venta.

    ARQUITECTURA DE 4 CAPAS (SHORT-ONLY):

    CAPA 1 (Filtro de Régimen - Tendencia Bajista):
    - Indicador: EMA(200) como proxy de tendencia principal
    - Lógica: Solo abre posiciones CORTAS si precio < EMA(200) ⚠️ INVERTIDO
    - Objetivo: Operar solo en tendencias bajistas
    - Régimen BAJISTA: precio < EMA_200 → Permite operar SHORT
    - Régimen ALCISTA: precio > EMA_200 → Fuera del mercado

    CAPA 2 (Filtro de Momentum - RSI Bajista):
    - Indicador: RSI(14)
    - Lógica: Solo entra si RSI < rsi_momentum_level (default: 50) ⚠️ INVERTIDO
    - Objetivo: Confirmar que hay momentum bajista (no vender en fuerza)
    - RSI < 50: Debilidad bajista confirmada
    - RSI > 50: Sin suficiente momentum bajista (evitar entrada)

    CAPA 3 (Señal de Entrada - VENTA EN CORTO):
    - Indicador: MACD (12, 26, 9)
    - Lógica VENTA EN CORTO: Cruce bajista del MACD (MACD cruza por DEBAJO de Signal) ⚠️ INVERTIDO
      * Condición: (MACD[-1] > Signal[-1]) AND (MACD <= Signal)
    - Lógica CUBRIR CORTO (TP): Cruce alcista del MACD (MACD cruza por ENCIMA de Signal) ⚠️ INVERTIDO
      * Condición: (MACD[-1] < Signal[-1]) AND (MACD >= Signal)
    - Objetivo: Timing preciso de entrada/salida en tendencia bajista confirmada

    CAPA 4 (Gestión de Riesgo - ATR Stop Loss INVERTIDO):
    - Indicador: ATR(14)
    - Lógica: Stop Loss dinámico POR ENCIMA del precio de entrada ⚠️ INVERTIDO
      * SL = Precio_Entrada + (ATR × atr_multiplier)
    - Verificación: Cierra si df['high'] toca o cruza el SL ⚠️ INVERTIDO (no 'low')
    - Objetivo: Proteger contra reversiones alcistas inesperadas
    - NOTA: La lógica de SL se implementa en el motor de backtesting

    FILOSOFÍA:
    Este es un sistema "Short-Only" que solo opera cuando las 4 capas están alineadas:
    1. Régimen correcto (precio < EMA_200) - tendencia bajista
    2. Momentum bajista confirmado (RSI < nivel de momentum)
    3. Timing de entrada apropiado (cruce bajista MACD)
    4. Gestión de riesgo adaptativa (ATR Stop Loss invertido)

    DIFERENCIAS CLAVE vs Iteración 12 (Híbrido Long):
    - Régimen: precio > EMA_200 → precio < EMA_200
    - Momentum: RSI > nivel → RSI < nivel
    - Entrada: Cruce alcista MACD → Cruce bajista MACD
    - Salida: Cruce bajista MACD → Cruce alcista MACD
    - Señal entrada: +1 → -1 (VENTA EN CORTO)
    - Señal salida: -1 → +1 (CUBRIR CORTO)
    - Stop Loss: entry - (ATR × mult) → entry + (ATR × mult)
    - Verificación SL: df['low'] → df['high']

    Args:
        df: DataFrame con indicadores calculados (debe contener EMA_200, RSI_14, MACD, ATR)
        config: Diccionario con parámetros:
            - ema_trend: Período de EMA para filtro de régimen (default: 200)
            - rsi_period: Período del RSI (default: 14)
            - rsi_momentum_level: Nivel máximo de RSI para entrar (default: 50) ⚠️ INVERTIDO
            - macd_fast: Período rápido del MACD (default: 12)
            - macd_slow: Período lento del MACD (default: 26)
            - macd_signal: Período de señal del MACD (default: 9)
            - atr_length: Período del ATR (default: 14)

    Returns:
        DataFrame con columna 'señal' añadida:
           -1 = VENTA EN CORTO (abrir posición SHORT)
            1 = CUBRIR CORTO (cerrar posición SHORT / Take Profit)
            0 = NEUTRAL (sin acción)

        Y columna 'position' para tracking Short-Only en backtest.
        NOTA: position = -1 indica DENTRO de posición corta, 0 = FUERA
    """
    # Parámetros por defecto - Módulo Bajista v1
    if config is None:
        config = {
            'ema_trend': 200,           # EMA de tendencia (filtro macro)
            'rsi_period': 14,           # RSI para momentum
            'rsi_momentum_level': 50,   # Nivel máximo de RSI para entrar SHORT
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
    # CAPA 1: FILTRO DE RÉGIMEN (EMA_200) ⚠️ INVERTIDO
    # ==========================================
    # Solo operamos en SHORT cuando estamos en régimen bajista
    condicion_regimen = df['close'] < df[ema_trend_col]  # ⚠️ INVERTIDO: < en vez de >

    # ==========================================
    # CAPA 2: FILTRO DE MOMENTUM (RSI) ⚠️ INVERTIDO
    # ==========================================
    # Solo entramos si RSI confirma momentum bajista
    condicion_momentum = df[rsi_col] < config['rsi_momentum_level']  # ⚠️ INVERTIDO: < en vez de >

    # ==========================================
    # CAPA 3: SEÑAL DE ENTRADA (CRUCE MACD BAJISTA) ⚠️ INVERTIDO
    # ==========================================
    # VENTA EN CORTO: MACD cruza por DEBAJO de su línea de señal
    condicion_entrada = (
        (df[macd_col].shift(1) > df[macd_signal_col].shift(1)) &  # ⚠️ INVERTIDO: > en vez de <
        (df[macd_col] <= df[macd_signal_col])                      # ⚠️ INVERTIDO: <= en vez de >=
    )

    # ==========================================
    # CAPA 3: SEÑAL DE SALIDA (CRUCE MACD ALCISTA) ⚠️ INVERTIDO
    # ==========================================
    # CUBRIR CORTO (TP): MACD cruza por ENCIMA de su línea de señal
    condicion_salida_tp = (
        (df[macd_col].shift(1) < df[macd_signal_col].shift(1)) &  # ⚠️ INVERTIDO: < en vez de >
        (df[macd_col] >= df[macd_signal_col])                      # ⚠️ INVERTIDO: >= en vez de <=
    )

    # ==========================================
    # GENERACIÓN DE SEÑALES (CONFLUENCIA DE 4 CAPAS) ⚠️ INVERTIDO
    # ==========================================

    # VENTA EN CORTO: Régimen bajista (Capa 1) + Momentum bajista (Capa 2) + Cruce bajista MACD (Capa 3)
    df.loc[condicion_regimen & condicion_momentum & condicion_entrada, 'señal'] = -1  # ⚠️ INVERTIDO: -1 en vez de 1

    # CUBRIR CORTO (TP): Cruce alcista MACD (Capa 3)
    # Queremos proteger ganancias cerrando la posición corta
    df.loc[condicion_salida_tp, 'señal'] = 1  # ⚠️ INVERTIDO: 1 en vez de -1

    # ==========================================
    # COLUMNA DE POSICIÓN PARA BACKTEST SHORT-ONLY
    # ==========================================
    # Esta columna indica si estamos DENTRO (-1) o FUERA (0) de una posición corta
    # El motor de backtesting usa esta columna para simular la estrategia
    df['position'] = 0
    in_position = False

    for i in range(len(df)):
        signal = df['señal'].iloc[i]

        if signal == -1 and not in_position:  # Abrir posición corta
            in_position = True
        elif signal == 1 and in_position:     # Cerrar posición corta
            in_position = False

        # NOTA: position = -1 indica posición SHORT activa, 0 = sin posición
        df.iloc[i, df.columns.get_loc('position')] = -1 if in_position else 0

    return df
