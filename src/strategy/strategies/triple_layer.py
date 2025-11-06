"""
Estrategia de confluencia de 3 capas (Long-Only).
Régimen + Estocástico + ATR.
"""

import pandas as pd


def generar_senales_triple_capa(df, config=None):
    """
    Genera señales de trading usando ESTRATEGIA DE CONFLUENCIA DE 3 CAPAS (Módulo Alcista v1).

    *** ITERACIÓN 10 - MÓDULO ALCISTA v1 ***
    Esta estrategia abandona los indicadores únicos y adopta un modelo profesional
    de confluencia multi-capa para day trading en ETH/USDT (15m).

    ARQUITECTURA DE 3 CAPAS:

    CAPA 1 - FILTRO DE RÉGIMEN (Tendencia Macro):
        - Indicador: EMA(200) como proxy de tendencia 1h-4h (Análisis Multi-Timeframe)
        - Lógica: Solo abrimos posiciones LARGAS si precio > EMA(200)
        - Objetivo: Evitar operar contra la tendencia principal
        - Régimen ALCISTA: precio > EMA_200 → Permite operar LONG
        - Régimen BAJISTA: precio < EMA_200 → Fuera del mercado

    CAPA 2 - SEÑAL DE ENTRADA (Impulso):
        - Indicador: Oscilador Estocástico (K, D)
        - Lógica COMPRA: Estocástico cruza HACIA ARRIBA desde zona sobreventa
          * Condición: (K anterior < oversold) AND (K actual >= oversold)
          * Confirmación: K cruza por encima de D (opcional pero recomendado)
        - Lógica VENTA/TP: Estocástico cruza HACIA ABAJO desde zona sobrecompra
          * Condición: (K anterior > overbought) AND (K actual <= overbought)
        - Objetivo: Capturar retrocesos en tendencia alcista confirmada

    CAPA 3 - GESTIÓN DE RIESGO (Volatilidad):
        - Indicador: ATR (Average True Range)
        - Lógica: Stop Loss dinámico basado en volatilidad actual del mercado
          * SL = Precio_Entrada - (ATR * atr_multiplier)
          * TP = Precio_Entrada + (ATR * atr_multiplier * 1.5)  [ratio 1:1.5]
        - Objetivo: Adaptar el riesgo a las condiciones de mercado
        - NOTA: La lógica de SL se implementa en el motor de backtesting, NO aquí

    FILOSOFÍA:
    Este es un sistema "Long-Only" que solo opera cuando las 3 capas están alineadas:
    1. Régimen correcto (alcista)
    2. Momento de entrada apropiado (retroceso técnico)
    3. Gestión de riesgo adaptativa (volatilidad medida)

    Args:
        df: DataFrame con indicadores calculados (debe contener EMA_200, STOCH, ATR)
        config: Diccionario con parámetros:
            - ema_trend: Período de EMA para filtro de régimen (default: 200)
            - stoch_k: Período K del estocástico (default: 14)
            - stoch_d: Período D del estocástico (default: 3)
            - stoch_smooth: Smooth del estocástico (default: 3)
            - stoch_oversold: Nivel de sobreventa (default: 20)
            - stoch_overbought: Nivel de sobrecompra (default: 80)
            - atr_length: Período del ATR (default: 14)

    Returns:
        DataFrame con columna 'señal' añadida:
            1 = COMPRA (abrir posición LONG)
           -1 = VENTA/TP (cerrar posición LONG)
            0 = NEUTRAL (sin acción)

        Y columna 'position' para tracking Long-Only en backtest vectorizado.
    """
    # Parámetros por defecto - Módulo Alcista v1
    if config is None:
        config = {
            'ema_trend': 200,          # EMA de tendencia (filtro macro)
            'stoch_k': 14,             # Estocástico K
            'stoch_d': 3,              # Estocástico D
            'stoch_smooth': 3,         # Estocástico Smooth
            'stoch_oversold': 20,      # Nivel de sobreventa
            'stoch_overbought': 80,    # Nivel de sobrecompra
            'atr_length': 14           # ATR para gestión de riesgo
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_trend_col = f"EMA_{config['ema_trend']}"
    stoch_k_col = f"STOCHk_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}"
    stoch_d_col = f"STOCHd_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}"
    atr_col = f"ATRr_{config['atr_length']}"  # pandas-ta usa 'ATRr' en lugar de 'ATR'

    # Verificar que las columnas existan
    required_cols = [ema_trend_col, stoch_k_col, stoch_d_col, atr_col]
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
    # CAPA 1: FILTRO DE RÉGIMEN (EMA 200)
    # ==========================================
    # Solo operamos en LONG cuando estamos en régimen alcista
    condicion_regimen_alcista = df['close'] > df[ema_trend_col]

    # ==========================================
    # CAPA 2: SEÑAL DE ENTRADA (ESTOCÁSTICO)
    # ==========================================

    # SEÑAL DE COMPRA: Cruce del estocástico HACIA ARRIBA desde zona sobreventa
    # Detectamos cuando K estaba por debajo del nivel y ahora cruza hacia arriba
    condicion_entrada = (
        (df[stoch_k_col].shift(1) < config['stoch_oversold']) &  # K anterior estaba en sobreventa
        (df[stoch_k_col] >= config['stoch_oversold'])            # K actual cruzó hacia arriba
    )

    # Confirmación adicional (opcional): K > D (impulso alcista confirmado)
    # Esto reduce false signals pero también reduce frecuencia
    # confirmacion_k_mayor_d = df[stoch_k_col] > df[stoch_d_col]

    # SEÑAL DE VENTA/TP: Cruce del estocástico HACIA ABAJO desde zona sobrecompra
    # Detectamos cuando K estaba por encima del nivel y ahora cruza hacia abajo
    condicion_salida_tp = (
        (df[stoch_k_col].shift(1) > config['stoch_overbought']) &  # K anterior estaba en sobrecompra
        (df[stoch_k_col] <= config['stoch_overbought'])            # K actual cruzó hacia abajo
    )

    # ==========================================
    # GENERACIÓN DE SEÑALES (CONFLUENCIA)
    # ==========================================

    # COMPRA: Régimen alcista (Capa 1) + Entrada por estocástico (Capa 2)
    df.loc[condicion_regimen_alcista & condicion_entrada, 'señal'] = 1

    # VENTA/TP: Salida por estocástico en sobrecompra (Capa 2)
    # No necesitamos filtro de régimen para salir - queremos proteger ganancias
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
