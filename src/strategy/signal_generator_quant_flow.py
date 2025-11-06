"""
Generador de señales para estrategia Quant-Flow.

Estrategia Multi-Timeframe (M15 + H1) con pullbacks a EMA y gestión de riesgo avanzada.
"""

import pandas as pd
import numpy as np


def generar_senales_quant_flow(df_m15, df_h1, config=None):
    """
    ESTRATEGIA QUANT-FLOW: Multi-Timeframe con Pullbacks

    Estrategia de trading que opera pullbacks a la EMA(21) en M15,
    con filtros de tendencia en H1 y confirmación de momentum con RSI.

    TIMEFRAMES:
        - M15 (ejecución): EMA(21), VWAP, RSI(14), ATR(14)
        - H1 (contexto): EMA(200) para filtro de tendencia

    LÓGICA DE ENTRADA LONG:
        1. Filtro ADX H1: ADX(14) > 20 (fuerza de tendencia)
        2. Filtro Horario: NO operar Vie 23:01 - Dom 12:00 UTC
        3. Filtro de Tendencia H1: Close H1 > EMA(200) H1
        4. Filtro de Contexto M15: Close M15 > VWAP M15
        5. Gatillo Pullback M15: Low <= EMA(21) AND Close > EMA(21)
        6. Filtro Momentum M15: 45 < RSI(14) < 65

    LÓGICA DE ENTRADA SHORT:
        1. Filtro ADX H1: ADX(14) > 20 (fuerza de tendencia)
        2. Filtro Horario: NO operar Vie 23:01 - Dom 12:00 UTC
        3. Filtro de Tendencia H1: Close H1 < EMA(200) H1
        4. Filtro de Contexto M15: Close M15 < VWAP M15
        5. Gatillo Pullback M15: High >= EMA(21) AND Close < EMA(21)
        6. Filtro Momentum M15: 35 < RSI(14) < 55

    SISTEMA DE SEÑALES:
        1 = Abrir Long
       -1 = Abrir Short
        0 = Neutral (mantener o esperar)

    GESTIÓN DE RIESGO (manejada por motor de backtest):
        - Stop Loss: 2.0 × ATR(14) desde precio de entrada
        - TP1 (1.5R): Mover SL a breakeven
        - TP2: Trailing Stop con mínimos/máximos de últimas 3 velas
        - Weekend Exit: Cierre forzoso Viernes 23:00 UTC

    Args:
        df_m15: DataFrame con datos de 15 minutos
                Debe tener: timestamp, open, high, low, close, volume, EMA_21, VWAP, RSI_14, ATR_14
        df_h1: DataFrame con datos de 1 hora
               Debe tener: timestamp, EMA_200, ADX_14
        config: Diccionario con parámetros (opcional):
            - ema_pullback: Período de EMA para pullback (default 21)
            - ema_trend_h1: Período de EMA de tendencia H1 (default 200)
            - adx_period: Período del ADX (default 14)
            - adx_threshold: Umbral mínimo de ADX (default 20)
            - rsi_period: Período del RSI (default 14)
            - atr_period: Período del ATR (default 14)
            - rsi_long_min: RSI mínimo para long (default 45)
            - rsi_long_max: RSI máximo para long (default 65)
            - rsi_short_min: RSI mínimo para short (default 35)
            - rsi_short_max: RSI máximo para short (default 55)

    Returns:
        DataFrame M15 con columna 'señal': 1 (Long), -1 (Short), 0 (Neutral)
    """
    # Parámetros por defecto
    if config is None:
        config = {
            'ema_pullback': 21,
            'ema_trend_h1': 200,
            'adx_period': 14,
            'adx_threshold': 20,
            'rsi_period': 14,
            'atr_period': 14,
            'rsi_long_min': 45,
            'rsi_long_max': 65,
            'rsi_short_min': 35,
            'rsi_short_max': 55
        }

    # ==========================================
    # 1. PREPARACIÓN DE DATOS
    # ==========================================
    df_m15 = df_m15.copy()
    df_h1 = df_h1.copy()

    # Nombres de columnas esperadas
    ema_pullback_col = f"EMA_{config['ema_pullback']}"
    ema_trend_h1_col = f"EMA_{config['ema_trend_h1']}"
    adx_h1_col = f"ADX_{config['adx_period']}"
    rsi_col = f"RSI_{config['rsi_period']}"
    atr_col = f"ATRr_{config['atr_period']}"

    # Verificar columnas en df_m15
    required_m15 = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    ema_pullback_col, 'VWAP', rsi_col, atr_col]
    missing_m15 = [col for col in required_m15 if col not in df_m15.columns]
    if missing_m15:
        raise ValueError(
            f"Columnas faltantes en df_m15 (15m): {missing_m15}\n"
            f"Columnas disponibles: {df_m15.columns.tolist()}"
        )

    # Verificar columnas en df_h1
    required_h1 = ['timestamp', 'close', ema_trend_h1_col, adx_h1_col]
    missing_h1 = [col for col in required_h1 if col not in df_h1.columns]
    if missing_h1:
        raise ValueError(
            f"Columnas faltantes en df_h1 (1h): {missing_h1}\n"
            f"Columnas disponibles: {df_h1.columns.tolist()}"
        )

    # ==========================================
    # 2. ALINEAMIENTO MULTI-TIMEFRAME
    # ==========================================
    # Asegurar que timestamps sean datetime
    df_m15['timestamp'] = pd.to_datetime(df_m15['timestamp'])
    df_h1['timestamp'] = pd.to_datetime(df_h1['timestamp'])

    # Crear DataFrame H1 con datos necesarios
    df_h1_data = df_h1[['timestamp', 'close', ema_trend_h1_col, adx_h1_col]].copy()
    df_h1_data = df_h1_data.set_index('timestamp')

    # Resamplear H1 a 15m usando forward-fill
    df_h1_resampled = df_h1_data.resample('15min').ffill()
    df_h1_resampled = df_h1_resampled.reset_index()
    df_h1_resampled.columns = ['timestamp', 'close_h1', f'{ema_trend_h1_col}_h1', f'{adx_h1_col}_h1']

    # Merge con df_m15
    df_m15 = pd.merge(df_m15, df_h1_resampled, on='timestamp', how='left')

    # Forward-fill para llenar cualquier NaN residual
    df_m15[f'{ema_trend_h1_col}_h1'] = df_m15[f'{ema_trend_h1_col}_h1'].ffill()
    df_m15[f'{adx_h1_col}_h1'] = df_m15[f'{adx_h1_col}_h1'].ffill()
    df_m15['close_h1'] = df_m15['close_h1'].ffill()

    # ==========================================
    # 3. CALCULAR CONDICIONES DE ENTRADA
    # ==========================================

    # --- FILTROS GLOBALES (para LONG y SHORT) ---
    # 1. Filtro ADX H1: ADX(14) > threshold
    filtro_adx = df_m15[f'{adx_h1_col}_h1'] > config['adx_threshold']

    # 2. Filtro Horario: NO operar Vie 23:01 - Dom 12:00 UTC
    df_m15['dayofweek'] = df_m15['timestamp'].dt.dayofweek  # 0=Lun, 4=Vie, 6=Dom
    df_m15['hour'] = df_m15['timestamp'].dt.hour

    # Bloquear operaciones desde Viernes 23:01 hasta Domingo 12:00
    filtro_horario = ~(
        # Viernes después de las 23:00
        ((df_m15['dayofweek'] == 4) & (df_m15['hour'] >= 23)) |
        # Todo el sábado
        (df_m15['dayofweek'] == 5) |
        # Domingo antes de las 12:00
        ((df_m15['dayofweek'] == 6) & (df_m15['hour'] < 12))
    )

    # --- FILTROS LONG ---
    # 3. Filtro de Tendencia H1: Close H1 > EMA(200) H1
    filtro_tendencia_alcista = df_m15['close_h1'] > df_m15[f'{ema_trend_h1_col}_h1']

    # 2. Filtro de Contexto M15: Close M15 > VWAP M15
    filtro_contexto_long = df_m15['close'] > df_m15['VWAP']

    # 3. Gatillo Pullback M15: Low <= EMA(21) AND Close > EMA(21)
    gatillo_pullback_long = (
        (df_m15['low'] <= df_m15[ema_pullback_col]) &
        (df_m15['close'] > df_m15[ema_pullback_col])
    )

    # 4. Filtro Momentum M15: 45 < RSI(14) < 65
    filtro_momentum_long = (
        (df_m15[rsi_col] > config['rsi_long_min']) &
        (df_m15[rsi_col] < config['rsi_long_max'])
    )

    # Condición completa LONG (con filtros globales)
    condicion_long = (
        filtro_adx &                  # NUEVO: Fuerza de tendencia
        filtro_horario &              # NUEVO: No operar fin de semana
        filtro_tendencia_alcista &
        filtro_contexto_long &
        gatillo_pullback_long &
        filtro_momentum_long
    )

    # --- FILTROS SHORT ---
    # 3. Filtro de Tendencia H1: Close H1 < EMA(200) H1
    filtro_tendencia_bajista = df_m15['close_h1'] < df_m15[f'{ema_trend_h1_col}_h1']

    # 2. Filtro de Contexto M15: Close M15 < VWAP M15
    filtro_contexto_short = df_m15['close'] < df_m15['VWAP']

    # 3. Gatillo Pullback M15: High >= EMA(21) AND Close < EMA(21)
    gatillo_pullback_short = (
        (df_m15['high'] >= df_m15[ema_pullback_col]) &
        (df_m15['close'] < df_m15[ema_pullback_col])
    )

    # 4. Filtro Momentum M15: 35 < RSI(14) < 55
    filtro_momentum_short = (
        (df_m15[rsi_col] > config['rsi_short_min']) &
        (df_m15[rsi_col] < config['rsi_short_max'])
    )

    # Condición completa SHORT (con filtros globales)
    condicion_short = (
        filtro_adx &                   # NUEVO: Fuerza de tendencia
        filtro_horario &               # NUEVO: No operar fin de semana
        filtro_tendencia_bajista &
        filtro_contexto_short &
        gatillo_pullback_short &
        filtro_momentum_short
    )

    # ==========================================
    # 4. GENERAR SEÑALES
    # ==========================================
    df_m15['señal'] = 0

    # Aplicar señales (sin necesidad de tracking de posición - solo señales de entrada)
    df_m15.loc[condicion_long, 'señal'] = 1   # Long
    df_m15.loc[condicion_short, 'señal'] = -1  # Short

    # ==========================================
    # 5. METADATOS ADICIONALES (para debug)
    # ==========================================
    # Agregar columnas de filtros individuales para análisis
    df_m15['filtro_adx'] = filtro_adx
    df_m15['filtro_horario'] = filtro_horario
    df_m15['filtro_tendencia_long'] = filtro_tendencia_alcista
    df_m15['filtro_contexto_long'] = filtro_contexto_long
    df_m15['gatillo_pullback_long'] = gatillo_pullback_long
    df_m15['filtro_momentum_long'] = filtro_momentum_long

    df_m15['filtro_tendencia_short'] = filtro_tendencia_bajista
    df_m15['filtro_contexto_short'] = filtro_contexto_short
    df_m15['gatillo_pullback_short'] = gatillo_pullback_short
    df_m15['filtro_momentum_short'] = filtro_momentum_short

    return df_m15


if __name__ == "__main__":
    print("=== Test de Quant-Flow Signal Generator ===")
    print("\nEste módulo genera señales para la estrategia Quant-Flow.")
    print("Requiere datos de M15 y H1 para funcionar.")
    print("\nUso:")
    print("  from src.strategy.signal_generator_quant_flow import generar_senales_quant_flow")
    print("  df_signals = generar_senales_quant_flow(df_m15, df_h1)")
