"""
Módulo para generar señales de trading basadas en indicadores técnicos.
Señales: 1 (COMPRA), -1 (VENTA), 0 (NADA/NEUTRAL)
"""

import pandas as pd
import numpy as np


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


def obtener_señales_recientes(df, n=10):
    """
    Obtiene las últimas N señales generadas.

    Args:
        df: DataFrame con señales
        n: Número de registros recientes a mostrar

    Returns:
        DataFrame con las últimas N filas y columnas relevantes
    """
    columnas_relevantes = [
        'timestamp', 'close', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
        'RSI_14', 'ATR_14', 'señal'
    ]

    # Filtrar solo las columnas que existan
    columnas_existentes = [col for col in columnas_relevantes if col in df.columns]

    return df[columnas_existentes].tail(n)


def contar_señales(df):
    """
    Cuenta las señales generadas en el DataFrame.

    Args:
        df: DataFrame con señales

    Returns:
        Diccionario con conteo de señales
    """
    if 'señal' not in df.columns:
        raise ValueError("DataFrame no contiene columna 'señal'")

    total = len(df)
    compras = (df['señal'] == 1).sum()
    ventas = (df['señal'] == -1).sum()
    neutrales = (df['señal'] == 0).sum()

    return {
        'total': total,
        'compras': compras,
        'ventas': ventas,
        'neutrales': neutrales,
        'pct_compras': (compras / total) * 100,
        'pct_ventas': (ventas / total) * 100,
        'pct_neutrales': (neutrales / total) * 100
    }


def generar_senales_donchian_ema(df, config=None):
    """
    Genera señales de trading usando ESTRATEGIA DONCHIAN + EMA (v18/v22/v23).

    ITERACIÓN 23: Parámetros ultra-agresivos para maximizar frecuencia en 15m.

    Estrategia de 2 Capas (LONG-ONLY):

    CAPA 1 (Filtro de Tendencia):
    - Solo opera cuando close > EMA (tendencia alcista)
    - Evita operar contra la tendencia principal

    CAPA 2 (Señal de Entrada - Breakout Donchian):
    - COMPRA: Precio rompe HACIA ARRIBA el canal superior de Donchian
    - VENTA: Precio rompe HACIA ABAJO el canal inferior de Donchian
    - Rationale: Breakout de máximos/mínimos indica momentum

    Args:
        df: DataFrame con indicadores calculados
        config: Diccionario con parámetros:
            - donchian_period: Período del Canal de Donchian (default: 20)
            - ema_filter_period: Período de EMA de filtro (default: 50)

    Returns:
        DataFrame con columnas 'señal' y 'position' añadidas
    """
    # Parámetros por defecto
    if config is None:
        config = {
            'donchian_period': 20,
            'ema_filter_period': 50
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    donchian_period = config['donchian_period']
    ema_filter_col = f"EMA_{config['ema_filter_period']}"
    donchian_upper_col = f"DONCHI_h_{donchian_period}"
    donchian_lower_col = f"DONCHI_l_{donchian_period}"

    # Verificar que las columnas requeridas existan
    required_cols = [ema_filter_col, donchian_upper_col, donchian_lower_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Columnas faltantes en DataFrame: {missing_cols}. "
                        f"Asegúrate de calcular indicadores primero. "
                        f"Columnas disponibles: {df.columns.tolist()}")

    # Inicializar columna de señales en 0 (NEUTRAL)
    df['señal'] = 0

    # ==========================================
    # CAPA 1: FILTRO DE TENDENCIA (EMA)
    # ==========================================
    # Solo opera cuando el precio está por encima de EMA (tendencia alcista)
    condicion_tendencia_alcista = df['close'] > df[ema_filter_col]

    # ==========================================
    # CAPA 2: ENTRADA - BREAKOUT ALCISTA
    # ==========================================
    # Señal de COMPRA: Precio cruza HACIA ARRIBA el canal superior de Donchian
    condicion_entrada_breakout = (
        (df['close'].shift(1) < df[donchian_upper_col].shift(1)) &
        (df['close'] >= df[donchian_upper_col])
    )

    # ==========================================
    # CAPA 2: SALIDA - BREAKOUT BAJISTA
    # ==========================================
    # Señal de VENTA: Precio cruza HACIA ABAJO el canal inferior
    condicion_salida_breakout = (
        (df['close'].shift(1) > df[donchian_lower_col].shift(1)) &
        (df['close'] <= df[donchian_lower_col])
    )

    # ==========================================
    # GENERACIÓN DE SEÑALES
    # ==========================================
    # COMPRA: Solo si hay tendencia alcista Y breakout del canal superior
    df.loc[condicion_tendencia_alcista & condicion_entrada_breakout, 'señal'] = 1

    # VENTA: Cuando se rompe el canal inferior
    df.loc[condicion_salida_breakout, 'señal'] = -1

    # ==========================================
    # COLUMNA DE POSICIÓN (Long-Only)
    # ==========================================
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


if __name__ == "__main__":
    # Test básico del módulo
    from src.data.binance_client import BinanceClientManager
    from src.data.data_fetcher import obtener_datos_binance
    from src.indicators.technical import agregar_indicadores

    print("=== Test de Generación de Señales ===\n")

    # Obtener y preparar datos
    manager = BinanceClientManager()
    client = manager.get_public_client()

    print("1. Descargando datos...")
    df = obtener_datos_binance(
        client=client,
        simbolo='BTCUSDT',
        intervalo='5m',
        inicio='7 days ago UTC'
    )

    print("2. Calculando indicadores...")
    df = agregar_indicadores(df)

    print("3. Generando señales (estrategia básica)...")
    df = generar_señales(df)

    # Estadísticas de señales
    stats = contar_señales(df)
    print(f"\n4. Estadísticas de señales:")
    print(f"   Total de registros: {stats['total']}")
    print(f"   Señales de COMPRA: {stats['compras']} ({stats['pct_compras']:.2f}%)")
    print(f"   Señales de VENTA: {stats['ventas']} ({stats['pct_ventas']:.2f}%)")
    print(f"   Señales NEUTRALES: {stats['neutrales']} ({stats['pct_neutrales']:.2f}%)")

    # Mostrar últimas señales
    print("\n5. Últimas 10 señales generadas:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(obtener_señales_recientes(df, n=10))

    # Test de señales avanzadas
    print("\n6. Generando señales (estrategia avanzada)...")
    df_avanzado = generar_señales_avanzadas(df)
    stats_avanzado = contar_señales(df_avanzado)
    print(f"   Señales COMPRA (avanzada): {stats_avanzado['compras']}")
    print(f"   Señales VENTA (avanzada): {stats_avanzado['ventas']}")

    print("\n✓ Test completado exitosamente")


def generar_senales_pullback_ema_v25(df, config=None):
    """
    ITERACIÓN 25: ESTRATEGIA EMA PULLBACK (Retrocesos a Media Móvil)

    Hipótesis: Podemos obtener alta frecuencia y rentabilidad comprando los retrocesos
    (pullbacks) a una EMA corta, solo en la dirección de la tendencia principal.

    Esta estrategia busca "comprar la caída" cuando el precio retrocede hacia el
    soporte dinámico (EMA gatillo), pero solo si la tendencia general es alcista.

    LÓGICA DE SEÑALES:

    COMPRA (señal = 1):
        1. Filtro de Tendencia: Precio[t] > EMA_Filtro[t] (Solo tendencia alcista)
        2. Setup: Precio[t-1] > EMA_Gatillo[t-1] (Vela anterior por encima del soporte)
        3. Trigger: Low[t] <= EMA_Gatillo[t] (Vela actual tocó o perforó el soporte)

        Interpretación: El precio está en tendencia alcista, retrocedió hasta el soporte
        dinámico (EMA gatillo), y rebotó. Este es un pullback en tendencia alcista.

    VENTA (señal = -1):
        1. Filtro de Tendencia: Precio[t] < EMA_Filtro[t] (Solo tendencia bajista)
        2. Setup: Precio[t-1] < EMA_Gatillo[t-1] (Vela anterior por debajo de resistencia)
        3. Trigger: High[t] >= EMA_Gatillo[t] (Vela actual tocó o perforó la resistencia)

        Interpretación: El precio está en tendencia bajista, rebotó hasta la resistencia
        dinámica (EMA gatillo), y cayó. Este es un pullback en tendencia bajista.

    PARÁMETROS:
        - ema_gatillo_periodo: EMA corta para el punto de entrada (default: 21)
        - ema_filtro_periodo: EMA larga para determinar la tendencia (default: 100)

    Args:
        df: DataFrame con columnas OHLCV y EMAs calculadas
        config: Diccionario con parámetros de estrategia

    Returns:
        DataFrame con columna 'señal' añadida (1=COMPRA, -1=VENTA, 0=NEUTRAL)
    """
    # Parámetros por defecto
    if config is None:
        config = {
            'ema_gatillo_periodo': 21,   # EMA corta (soporte/resistencia dinámica)
            'ema_filtro_periodo': 100    # EMA larga (filtro de tendencia)
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_gatillo_col = f"EMA_{config['ema_gatillo_periodo']}"
    ema_filtro_col = f"EMA_{config['ema_filtro_periodo']}"

    # Verificar que las columnas existan
    required_cols = [ema_gatillo_col, ema_filtro_col]
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
    # SEÑAL DE COMPRA (Pullback Alcista)
    # ==========================================
    # 1. Filtro de tendencia: Precio actual por encima de EMA filtro
    tendencia_alcista = df['close'] > df[ema_filtro_col]

    # 2. Setup: Vela anterior estaba por encima de EMA gatillo
    precio_anterior_arriba = df['close'].shift(1) > df[ema_gatillo_col].shift(1)

    # 3. Trigger: Low actual tocó o perforó EMA gatillo (pullback)
    pullback_alcista = df['low'] <= df[ema_gatillo_col]

    # Condición completa de COMPRA
    condicion_compra = tendencia_alcista & precio_anterior_arriba & pullback_alcista

    # ==========================================
    # SEÑAL DE VENTA (Pullback Bajista)
    # ==========================================
    # 1. Filtro de tendencia: Precio actual por debajo de EMA filtro
    tendencia_bajista = df['close'] < df[ema_filtro_col]

    # 2. Setup: Vela anterior estaba por debajo de EMA gatillo
    precio_anterior_abajo = df['close'].shift(1) < df[ema_gatillo_col].shift(1)

    # 3. Trigger: High actual tocó o perforó EMA gatillo (pullback)
    pullback_bajista = df['high'] >= df[ema_gatillo_col]

    # Condición completa de VENTA
    condicion_venta = tendencia_bajista & precio_anterior_abajo & pullback_bajista

    # Asignar señales
    df.loc[condicion_compra, 'señal'] = 1   # COMPRA
    df.loc[condicion_venta, 'señal'] = -1   # VENTA

    return df


def generar_senales_macd_crossover_v26(df, config=None):
    """
    ITERACIÓN 26: ESTRATEGIA MACD CROSSOVER CON FILTRO DE TENDENCIA EMA

    Hipótesis: El cruce de MACD dentro de una tendencia EMA proporciona señales
    de entrada más rápidas y fiables que breakouts o pullbacks en 5m.

    COMPRA (señal = 1):
        1. Filtro de Tendencia: Precio[t] > EMA_Filtro[t]
        2. Cruce Alcista: MACD[t] cruza por encima de Señal_MACD[t]
           (MACD[t-1] <= Señal[t-1] AND MACD[t] > Señal[t])

    VENTA (señal = -1):
        1. Filtro de Tendencia: Precio[t] < EMA_Filtro[t]
        2. Cruce Bajista: MACD[t] cruza por debajo de Señal_MACD[t]
           (MACD[t-1] >= Señal[t-1] AND MACD[t] < Señal[t])

    Args:
        df: DataFrame con OHLCV e indicadores calculados
        config: Diccionario con parámetros:
            - ema_filter_periodo: Período de EMA para filtro de tendencia
            - macd_fast: Período rápido del MACD (default 12)
            - macd_slow: Período lento del MACD (default 26)
            - macd_signal: Período de la señal MACD (default 9)

    Returns:
        DataFrame con columna 'señal' (1=COMPRA, -1=VENTA, 0=NEUTRAL)
    """
    if config is None:
        config = {
            'ema_filter_periodo': 100,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_filter_col = f"EMA_{config['ema_filter_periodo']}"
    macd_line_col = f"MACD_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"
    macd_signal_col = f"MACDs_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"

    # Verificar que las columnas existan
    required_cols = [ema_filter_col, macd_line_col, macd_signal_col]
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
    # SEÑAL DE COMPRA (Cruce Alcista MACD)
    # ==========================================
    # 1. Filtro de tendencia: Precio por encima de EMA filtro
    tendencia_alcista = df['close'] > df[ema_filter_col]

    # 2. Cruce alcista: MACD cruza por encima de su señal
    macd_anterior_abajo = df[macd_line_col].shift(1) <= df[macd_signal_col].shift(1)
    macd_actual_arriba = df[macd_line_col] > df[macd_signal_col]
    cruce_alcista = macd_anterior_abajo & macd_actual_arriba

    # Condición completa de COMPRA
    condicion_compra = tendencia_alcista & cruce_alcista

    # ==========================================
    # SEÑAL DE VENTA (Cruce Bajista MACD)
    # ==========================================
    # 1. Filtro de tendencia: Precio por debajo de EMA filtro
    tendencia_bajista = df['close'] < df[ema_filter_col]

    # 2. Cruce bajista: MACD cruza por debajo de su señal
    macd_anterior_arriba = df[macd_line_col].shift(1) >= df[macd_signal_col].shift(1)
    macd_actual_abajo = df[macd_line_col] < df[macd_signal_col]
    cruce_bajista = macd_anterior_arriba & macd_actual_abajo

    # Condición completa de VENTA
    condicion_venta = tendencia_bajista & cruce_bajista

    # Asignar señales
    df.loc[condicion_compra, 'señal'] = 1   # COMPRA
    df.loc[condicion_venta, 'señal'] = -1   # VENTA

    return df


def generar_senales_stoch_crossover_v27(df, config=None):
    """
    ITERACIÓN 27: ESTRATEGIA STOCHASTIC CROSSOVER CON FILTRO DE TENDENCIA EMA

    Hipótesis: El Oscilador Estocástico puede identificar reversiones rápidas en zonas
    de sobreventa/sobrecompra, proporcionando un "edge" en el timeframe de 5m cuando
    se opera a favor de la tendencia principal (EMA).

    ÚLTIMA ESTRATEGIA DE DAY TRADING EN 5M:
    Si esta iteración falla (PF < 1.1), habremos agotado las cuatro familias principales
    de indicadores técnicos (Precio, Momentum MACD, Oscilador Stochastic, Breakout).

    COMPRA (señal = 1):
        1. Filtro de Tendencia: Precio[t] > EMA_Filtro[t] (Solo en tendencia alcista)
        2. Zona de Sobreventa: Stoch_K[t] < 20 (Precio oversold)
        3. Cruce Alcista: Stoch_K[t] cruza por encima de Stoch_D[t]
           (Stoch_K[t-1] <= Stoch_D[t-1] AND Stoch_K[t] > Stoch_D[t])

    VENTA (señal = -1):
        1. Filtro de Tendencia: Precio[t] < EMA_Filtro[t] (Solo en tendencia bajista)
        2. Zona de Sobrecompra: Stoch_K[t] > 80 (Precio overbought)
        3. Cruce Bajista: Stoch_K[t] cruza por debajo de Stoch_D[t]
           (Stoch_K[t-1] >= Stoch_D[t-1] AND Stoch_K[t] < Stoch_D[t])

    Args:
        df: DataFrame con OHLCV e indicadores calculados
        config: Diccionario con parámetros:
            - ema_filter_periodo: Período de EMA para filtro de tendencia
            - stoch_k: Período del %K (default 14)
            - stoch_d: Período del %D (default 3)
            - stoch_smooth: Suavizado del %K (default 3)

    Returns:
        DataFrame con columna 'señal' (1=COMPRA, -1=VENTA, 0=NEUTRAL)
    """
    if config is None:
        config = {
            'ema_filter_periodo': 200,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_filter_col = f"EMA_{config['ema_filter_periodo']}"
    stoch_k_col = f"STOCHk_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}"
    stoch_d_col = f"STOCHd_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}"

    # Verificar que las columnas existan
    required_cols = [ema_filter_col, stoch_k_col, stoch_d_col]
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
    # SEÑAL DE COMPRA (Cruce Alcista en Sobreventa)
    # ==========================================
    # 1. Filtro de tendencia: Precio por encima de EMA filtro
    tendencia_alcista = df['close'] > df[ema_filter_col]

    # 2. Zona de sobreventa: Stoch_K por debajo de 20
    zona_sobreventa = df[stoch_k_col] < 20

    # 3. Cruce alcista: Stoch_K cruza por encima de Stoch_D
    stoch_k_anterior_abajo = df[stoch_k_col].shift(1) <= df[stoch_d_col].shift(1)
    stoch_k_actual_arriba = df[stoch_k_col] > df[stoch_d_col]
    cruce_alcista = stoch_k_anterior_abajo & stoch_k_actual_arriba

    # Condición completa de COMPRA
    condicion_compra = tendencia_alcista & zona_sobreventa & cruce_alcista

    # ==========================================
    # SEÑAL DE VENTA (Cruce Bajista en Sobrecompra)
    # ==========================================
    # 1. Filtro de tendencia: Precio por debajo de EMA filtro
    tendencia_bajista = df['close'] < df[ema_filter_col]

    # 2. Zona de sobrecompra: Stoch_K por encima de 80
    zona_sobrecompra = df[stoch_k_col] > 80

    # 3. Cruce bajista: Stoch_K cruza por debajo de Stoch_D
    stoch_k_anterior_arriba = df[stoch_k_col].shift(1) >= df[stoch_d_col].shift(1)
    stoch_k_actual_abajo = df[stoch_k_col] < df[stoch_d_col]
    cruce_bajista = stoch_k_anterior_arriba & stoch_k_actual_abajo

    # Condición completa de VENTA
    condicion_venta = tendencia_bajista & zona_sobrecompra & cruce_bajista

    # Asignar señales
    df.loc[condicion_compra, 'señal'] = 1   # COMPRA
    df.loc[condicion_venta, 'señal'] = -1   # VENTA

    return df


def generar_senales_bb_breakout_v28(df, config=None):
    """
    ITERACIÓN 28: ESTRATEGIA BOLLINGER BAND BREAKOUT CON MÚLTIPLES FILTROS EMA

    CONTEXTO CRÍTICO:
    Las iteraciones v24-v27 fallaron en 5m debido a un conflicto entre velocidad y filtro:
    - Filtros lentos (EMA 100-200) → Muy pocas señales o señales tardías
    - Estrategias sin filtro → Demasiado ruido y whipsaws

    HIPÓTESIS V28 (ÚLTIMA PRUEBA TÉCNICA):
    Una EMA de tendencia más rápida (21 o 50) puede capturar micro-tendencias en 5m,
    haciendo rentable la ruptura de volatilidad (Bollinger Bands).

    INNOVACIÓN:
    Probaremos un rango amplio de filtros EMA [21, 50, 100, 200] para encontrar el
    equilibrio óptimo entre velocidad de señal y calidad de filtrado.

    ESTRATEGIA: BOLLINGER BAND BREAKOUT
    - Detecta expansiones de volatilidad cuando el precio rompe las bandas
    - Filtra con EMA para operar solo a favor de micro-tendencias
    - BB detecta momentum explosivo, EMA valida dirección

    COMPRA (señal = 1):
        1. Filtro de Tendencia: Precio[t] > EMA_Filtro[t] (Micro-tendencia alcista)
        2. Breakout Alcista: Precio[t] cruza por encima de BB_Upper
           (Close[t-1] <= BB_Upper[t-1] AND Close[t] > BB_Upper[t])

    VENTA (señal = -1):
        1. Filtro de Tendencia: Precio[t] < EMA_Filtro[t] (Micro-tendencia bajista)
        2. Breakout Bajista: Precio[t] cruza por debajo de BB_Lower
           (Close[t-1] >= BB_Lower[t-1] AND Close[t] < BB_Lower[t])

    Args:
        df: DataFrame con OHLCV e indicadores calculados
        config: Diccionario con parámetros:
            - ema_filter_periodo: Período de EMA para filtro (21, 50, 100, 200)
            - bb_length: Período de las Bollinger Bands (default 20)
            - bb_std: Desviaciones estándar de las BB (default 2.0)

    Returns:
        DataFrame con columna 'señal' (1=COMPRA, -1=VENTA, 0=NEUTRAL)
    """
    if config is None:
        config = {
            'ema_filter_periodo': 50,
            'bb_length': 20,
            'bb_std': 2.0
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_filter_col = f"EMA_{config['ema_filter_periodo']}"
    bb_upper_col = f"BBU_{config['bb_length']}_{config['bb_std']}.0"
    bb_lower_col = f"BBL_{config['bb_length']}_{config['bb_std']}.0"

    # Verificar que las columnas existan
    required_cols = [ema_filter_col, bb_upper_col, bb_lower_col]
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
    # SEÑAL DE COMPRA (Breakout Alcista BB)
    # ==========================================
    # 1. Filtro de tendencia: Precio por encima de EMA filtro
    tendencia_alcista = df['close'] > df[ema_filter_col]

    # 2. Breakout alcista: Precio cruza por encima de BB_Upper
    precio_anterior_dentro = df['close'].shift(1) <= df[bb_upper_col].shift(1)
    precio_actual_fuera = df['close'] > df[bb_upper_col]
    breakout_alcista = precio_anterior_dentro & precio_actual_fuera

    # Condición completa de COMPRA
    condicion_compra = tendencia_alcista & breakout_alcista

    # ==========================================
    # SEÑAL DE VENTA (Breakout Bajista BB)
    # ==========================================
    # 1. Filtro de tendencia: Precio por debajo de EMA filtro
    tendencia_bajista = df['close'] < df[ema_filter_col]

    # 2. Breakout bajista: Precio cruza por debajo de BB_Lower
    precio_anterior_dentro_lower = df['close'].shift(1) >= df[bb_lower_col].shift(1)
    precio_actual_fuera_lower = df['close'] < df[bb_lower_col]
    breakout_bajista = precio_anterior_dentro_lower & precio_actual_fuera_lower

    # Condición completa de VENTA
    condicion_venta = tendencia_bajista & breakout_bajista

    # Asignar señales
    df.loc[condicion_compra, 'señal'] = 1   # COMPRA
    df.loc[condicion_venta, 'señal'] = -1   # VENTA

    return df


def generar_senales_bb_adx_filter_v30(df, config=None):
    """
    ITERACIÓN 30: BB BREAKOUT CON TRIPLE FILTRO (ADX + EMA + ESPECTRO COMPLETO)

    CONTEXTO DEFINITIVO:
    Las iteraciones v24-v28 fracasaron en 5m porque el filtro de tendencia EMA no era
    suficiente para eliminar el ruido. Los resultados mostraron:
    - v24-v28: Todas PF 0.74-0.85 (pérdidas del 75-97%)
    - Problema: Demasiados whipsaws en consolidaciones laterales

    HIPÓTESIS V30 (PRUEBA DEFINITIVA DEL EDGE):
    El problema NO es el filtro EMA, sino la falta de filtro de MOMENTUM.
    ADX > 15 filtra consolidaciones laterales, permitiendo operar SOLO cuando hay
    momentum real, reduciendo whipsaws y mejorando el Profit Factor.

    INNOVACIÓN TRIPLE FILTRO:
    1. **ADX > 15:** Filtra mercado lateral (solo opera con momentum)
    2. **EMA Trend:** Valida dirección de micro-tendencia [21, 50, 100, 150, 200]
    3. **BB Breakout:** Detecta expansión de volatilidad

    ESTRATEGIA: BB BREAKOUT + ADX FILTER + ESPECTRO EMA
    - ADX elimina el ruido lateral (principal causa de whipsaws)
    - BB detecta momentum explosivo
    - EMA valida dirección

    COMPRA (señal = 1):
        1. Filtro de Momentum: ADX[t] > adx_threshold (Tendencia confirmada)
        2. Filtro de Dirección: Precio[t] > EMA_Filtro[t] (Micro-tendencia alcista)
        3. Breakout Alcista: Precio[t] cruza por encima de BB_Upper
           (Close[t-1] <= BB_Upper[t-1] AND Close[t] > BB_Upper[t])

    VENTA (señal = -1):
        1. Filtro de Momentum: ADX[t] > adx_threshold (Tendencia confirmada)
        2. Filtro de Dirección: Precio[t] < EMA_Filtro[t] (Micro-tendencia bajista)
        3. Breakout Bajista: Precio[t] cruza por debajo de BB_Lower
           (Close[t-1] >= BB_Lower[t-1] AND Close[t] < BB_Lower[t])

    Args:
        df: DataFrame con OHLCV e indicadores calculados
        config: Diccionario con parámetros:
            - ema_filter_periodo: Período de EMA para filtro (21, 50, 100, 150, 200)
            - bb_length: Período de las Bollinger Bands (default 20)
            - bb_std: Desviaciones estándar de las BB (default 2.0)
            - adx_period: Período del ADX (default 14)
            - adx_threshold: Umbral mínimo de ADX (default 15)

    Returns:
        DataFrame con columna 'señal' (1=COMPRA, -1=VENTA, 0=NEUTRAL)
    """
    if config is None:
        config = {
            'ema_filter_periodo': 50,
            'bb_length': 20,
            'bb_std': 2.0,
            'adx_period': 14,
            'adx_threshold': 15
        }

    df = df.copy()

    # Nombres de columnas de indicadores
    ema_filter_col = f"EMA_{config['ema_filter_periodo']}"
    bb_upper_col = f"BBU_{config['bb_length']}_{config['bb_std']}.0"
    bb_lower_col = f"BBL_{config['bb_length']}_{config['bb_std']}.0"
    adx_col = f"ADX_{config['adx_period']}"

    # Verificar que las columnas existan
    required_cols = [ema_filter_col, bb_upper_col, bb_lower_col, adx_col]
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
    # SEÑAL DE COMPRA (Triple Filtro Alcista)
    # ==========================================
    # 1. Filtro de momentum: ADX indica tendencia (no lateral)
    momentum_confirmed = df[adx_col] > config['adx_threshold']

    # 2. Filtro de dirección: Precio por encima de EMA filtro
    tendencia_alcista = df['close'] > df[ema_filter_col]

    # 3. Breakout alcista: Precio cruza por encima de BB_Upper
    precio_anterior_dentro = df['close'].shift(1) <= df[bb_upper_col].shift(1)
    precio_actual_fuera = df['close'] > df[bb_upper_col]
    breakout_alcista = precio_anterior_dentro & precio_actual_fuera

    # Condición completa de COMPRA (TRIPLE FILTRO)
    condicion_compra = momentum_confirmed & tendencia_alcista & breakout_alcista

    # ==========================================
    # SEÑAL DE VENTA (Triple Filtro Bajista)
    # ==========================================
    # 1. Filtro de momentum (mismo que compra)
    # momentum_confirmed ya calculado arriba

    # 2. Filtro de dirección: Precio por debajo de EMA filtro
    tendencia_bajista = df['close'] < df[ema_filter_col]

    # 3. Breakout bajista: Precio cruza por debajo de BB_Lower
    precio_anterior_dentro_lower = df['close'].shift(1) >= df[bb_lower_col].shift(1)
    precio_actual_fuera_lower = df['close'] < df[bb_lower_col]
    breakout_bajista = precio_anterior_dentro_lower & precio_actual_fuera_lower

    # Condición completa de VENTA (TRIPLE FILTRO)
    condicion_venta = momentum_confirmed & tendencia_bajista & breakout_bajista

    # Asignar señales
    df.loc[condicion_compra, 'señal'] = 1   # COMPRA
    df.loc[condicion_venta, 'señal'] = -1   # VENTA

    return df


def generar_senales_mtf_v001(df_m15, df_h1, config=None):
    """
    ITERACIÓN 001: ESTRATEGIA MULTI-TIMEFRAME BIDIRECCIONAL

    Primera estrategia completamente diferente a las anteriores:
    - Multi-Timeframe: Combina señales de 15m (ejecución) + 1h (régimen)
    - Bidireccional: Opera LONG y SHORT de forma dinámica
    - Sistema de 4 señales: 1, -1, 2, -2

    ARQUITECTURA DE 5 CAPAS:

    CAPA 1 - Régimen de Mercado (H1):
        - Alcista: Precio > EMA_200 en 1h
        - Bajista: Precio < EMA_200 en 1h
        - Propósito: Filtro macro de dirección

    CAPA 2 - Filtro de Volatilidad (M15):
        - ATR actual > ATR hace 10 períodos
        - Propósito: Solo operar con momentum creciente

    CAPA 3 - Entrada LONG (M15):
        - Régimen alcista (Capa 1) ✓
        - Volatilidad confirmada (Capa 2) ✓
        - EMA_9 cruza por encima de EMA_21
        - in_position == 0 (no hay posición abierta)
        → Señal = 1 (Abrir Long)

    CAPA 4 - Entrada SHORT (M15):
        - Régimen bajista (Capa 1) ✓
        - Volatilidad confirmada (Capa 2) ✓
        - EMA_9 cruza por debajo de EMA_21
        - in_position == 0 (no hay posición abierta)
        → Señal = -1 (Abrir Short)

    CAPA 5 - Salidas por Cruce Inverso (M15):
        - Si in_position == 1 (Long activo):
            - EMA_9 cruza por debajo de EMA_21
            → Señal = 2 (Cerrar Long)
        - Si in_position == -1 (Short activo):
            - EMA_9 cruza por encima de EMA_21
            → Señal = -2 (Cerrar Short)

    SISTEMA DE SEÑALES (4 valores):
        1 = Abrir Long
       -1 = Abrir Short
        2 = Cerrar Long (por cruce inverso)
       -2 = Cerrar Short (por cruce inverso)
        0 = Neutral (mantener posición o esperar)

    PRIORIDAD: Lógica de salida (Capa 5) se evalúa ANTES que lógica de entrada

    Args:
        df_m15: DataFrame con datos de 15 minutos (ejecución)
                Debe tener: timestamp, open, high, low, close, volume, EMA_9, EMA_21, ATR_14
        df_h1: DataFrame con datos de 1 hora (régimen macro)
               Debe tener: timestamp, EMA_200
        config: Diccionario con parámetros:
            - ema_fast_m15: Período de EMA rápida en 15m (default 9)
            - ema_slow_m15: Período de EMA lenta en 15m (default 21)
            - ema_trend_h1: Período de EMA de tendencia en 1h (default 200)
            - atr_period: Período del ATR (default 14)
            - atr_lookback: Períodos hacia atrás para comparar ATR (default 10)

    Returns:
        DataFrame (15m) con columna 'señal' conteniendo: 1, -1, 2, -2, 0
    """
    if config is None:
        config = {
            'ema_fast_m15': 9,
            'ema_slow_m15': 21,
            'ema_trend_h1': 200,
            'atr_period': 14,
            'atr_lookback': 10
        }

    # ==========================================
    # 1. PREPARACIÓN DE DATOS
    # ==========================================
    df_m15 = df_m15.copy()
    df_h1 = df_h1.copy()

    # Nombres de columnas de indicadores
    ema_fast_col = f"EMA_{config['ema_fast_m15']}"
    ema_slow_col = f"EMA_{config['ema_slow_m15']}"
    ema_trend_col_h1 = f"EMA_{config['ema_trend_h1']}"
    atr_col = f"ATRr_{config['atr_period']}"

    # Verificar columnas en df_m15
    required_m15 = ['timestamp', 'close', ema_fast_col, ema_slow_col, atr_col]
    missing_m15 = [col for col in required_m15 if col not in df_m15.columns]
    if missing_m15:
        raise ValueError(
            f"Columnas faltantes en df_m15 (15m): {missing_m15}\n"
            f"Columnas disponibles: {df_m15.columns.tolist()}"
        )

    # Verificar columnas en df_h1
    required_h1 = ['timestamp', ema_trend_col_h1]
    missing_h1 = [col for col in required_h1 if col not in df_h1.columns]
    if missing_h1:
        raise ValueError(
            f"Columnas faltantes en df_h1 (1h): {missing_h1}\n"
            f"Columnas disponibles: {df_h1.columns.tolist()}"
        )

    # ==========================================
    # 2. ALINEAMIENTO MULTI-TIMEFRAME
    # ==========================================
    # Resamplear EMA_200 de 1h a 15m usando forward-fill
    # Esto asegura que cada vela de 15m tiene el valor de EMA_200 correspondiente

    # Asegurar que timestamps sean datetime
    df_m15['timestamp'] = pd.to_datetime(df_m15['timestamp'])
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'])

    # Crear DataFrame de 1h solo con timestamp y EMA_200
    df_h1_ema = df_h1[['timestamp', ema_trend_col_h1]].copy()
    df_h1_ema = df_h1_ema.set_index('timestamp')

    # Resamplear a 15m y forward-fill
    df_h1_resampled = df_h1_ema.resample('15min').ffill()
    df_h1_resampled = df_h1_resampled.reset_index()
    df_h1_resampled.columns = ['timestamp', 'EMA_200_h1']

    # Hacer merge con df_m15
    df_m15 = pd.merge(df_m15, df_h1_resampled, on='timestamp', how='left')

    # Forward-fill para llenar cualquier NaN residual
    df_m15['EMA_200_h1'] = df_m15['EMA_200_h1'].ffill()

    # ==========================================
    # 3. CALCULAR CONDICIONES DE LAS 5 CAPAS
    # ==========================================

    # CAPA 1: Régimen de Mercado (H1)
    cond_regimen_alcista = df_m15['close'] > df_m15['EMA_200_h1']
    cond_regimen_bajista = df_m15['close'] < df_m15['EMA_200_h1']

    # CAPA 2: Filtro de Volatilidad (M15)
    atr_lookback = config['atr_lookback']
    cond_volatilidad = df_m15[atr_col] > df_m15[atr_col].shift(atr_lookback)

    # CAPA 3 y 4: Cruces de EMAs (M15)
    # Cruce alcista: EMA rápida cruza por encima de EMA lenta
    ema_fast_anterior_abajo = df_m15[ema_fast_col].shift(1) <= df_m15[ema_slow_col].shift(1)
    ema_fast_actual_arriba = df_m15[ema_fast_col] > df_m15[ema_slow_col]
    cruce_alcista = ema_fast_anterior_abajo & ema_fast_actual_arriba

    # Cruce bajista: EMA rápida cruza por debajo de EMA lenta
    ema_fast_anterior_arriba = df_m15[ema_fast_col].shift(1) >= df_m15[ema_slow_col].shift(1)
    ema_fast_actual_abajo = df_m15[ema_fast_col] < df_m15[ema_slow_col]
    cruce_bajista = ema_fast_anterior_arriba & ema_fast_actual_abajo

    # ==========================================
    # 4. GENERACIÓN DE SEÑALES CON BUCLE
    # ==========================================
    # Inicializar columna de señales
    df_m15['señal'] = 0

    # Estado de posición: 0 (plano), 1 (long), -1 (short)
    in_position = 0

    # Iterar sobre cada fila para tracking de estado
    for i in range(len(df_m15)):
        # Saltar primeras filas con NaN
        if pd.isna(df_m15[atr_col].iloc[i]) or pd.isna(df_m15['EMA_200_h1'].iloc[i]):
            continue

        # PRIORIDAD 1: SALIDAS (Capa 5)
        # Estas se evalúan ANTES que las entradas

        if in_position == 1:  # Long activo
            # Salida Long por cruce bajista
            if cruce_bajista.iloc[i]:
                df_m15.loc[df_m15.index[i], 'señal'] = 2  # Cerrar Long
                in_position = 0
                continue

        elif in_position == -1:  # Short activo
            # Salida Short por cruce alcista
            if cruce_alcista.iloc[i]:
                df_m15.loc[df_m15.index[i], 'señal'] = -2  # Cerrar Short
                in_position = 0
                continue

        # PRIORIDAD 2: ENTRADAS (Capas 3 y 4)
        # Solo se evalúan si in_position == 0

        if in_position == 0:  # No hay posición abierta
            # ENTRADA LONG (Capa 3)
            if (cond_regimen_alcista.iloc[i] and
                cond_volatilidad.iloc[i] and
                cruce_alcista.iloc[i]):
                df_m15.loc[df_m15.index[i], 'señal'] = 1  # Abrir Long
                in_position = 1

            # ENTRADA SHORT (Capa 4)
            elif (cond_regimen_bajista.iloc[i] and
                  cond_volatilidad.iloc[i] and
                  cruce_bajista.iloc[i]):
                df_m15.loc[df_m15.index[i], 'señal'] = -1  # Abrir Short
                in_position = -1

    return df_m15
