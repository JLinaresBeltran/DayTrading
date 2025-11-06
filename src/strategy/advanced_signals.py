"""
Generadores de señales avanzados con múltiples indicadores y filtros.
Sistema flexible para probar diferentes combinaciones de estrategias.
"""

import pandas as pd
import numpy as np


def generate_signals_multi_indicator(df, config):
    """
    Genera señales usando una combinación configurable de indicadores y filtros.

    Arquitectura:
    1. Filtro de Régimen (tendencia general)
    2. Señal de Entrada (combinación de indicadores)
    3. Filtros Adicionales (volumen, volatilidad, momentum)
    4. Tipo de posición (Long-Only, Short-Only, Hybrid)

    Args:
        df: DataFrame con indicadores calculados
        config: Dict con configuración de la estrategia:
            {
                # FILTRO DE RÉGIMEN
                'regime_type': 'ema' | 'sma' | 'adx' | 'none',
                'regime_period': 200,
                'regime_direction': 'long_only' | 'short_only' | 'hybrid',

                # SEÑALES DE ENTRADA (se pueden combinar)
                'entry_indicators': ['ema_cross', 'rsi', 'macd', 'bb', 'donchian'],
                'ema_fast': 9,
                'ema_slow': 21,
                'rsi_period': 14,
                'rsi_overbought': 70,
                'rsi_oversold': 30,
                'macd_fast': 12,
                'macd_slow': 26,
                'macd_signal': 9,

                # FILTROS ADICIONALES
                'use_volume_filter': True,
                'volume_ma_period': 20,
                'use_atr_filter': True,
                'atr_min_threshold': 0.5,  # % del precio
                'use_momentum_filter': True,
                'momentum_indicator': 'rsi' | 'macd',

                # GESTIÓN DE RIESGO
                'atr_period': 14,
                'sl_atr_multiplier': 2.0,
                'tp_atr_multiplier': 4.0,  # Ratio 1:2
            }

    Returns:
        DataFrame con columna 'señal' y metadatos de la estrategia
    """
    df = df.copy()
    df['señal'] = 0

    # ==========================================
    # CAPA 1: FILTRO DE RÉGIMEN
    # ==========================================
    regime_filter = _apply_regime_filter(df, config)

    # ==========================================
    # CAPA 2: SEÑALES DE ENTRADA
    # ==========================================
    entry_long, entry_short = _generate_entry_signals(df, config)

    # ==========================================
    # CAPA 3: FILTROS ADICIONALES
    # ==========================================
    additional_filters = _apply_additional_filters(df, config)

    # ==========================================
    # GENERAR SEÑALES FINALES
    # ==========================================
    regime_direction = config.get('regime_direction', 'long_only')

    if regime_direction == 'long_only':
        # Solo señales LONG
        df.loc[regime_filter & entry_long & additional_filters, 'señal'] = 1
        df.loc[~regime_filter | ~additional_filters, 'señal'] = -1  # Salir si pierde régimen

    elif regime_direction == 'short_only':
        # Solo señales SHORT
        df.loc[regime_filter & entry_short & additional_filters, 'señal'] = -1
        df.loc[~regime_filter | ~additional_filters, 'señal'] = 1  # Salir si pierde régimen

    elif regime_direction == 'hybrid':
        # Señales LONG y SHORT
        df.loc[regime_filter & entry_long & additional_filters, 'señal'] = 1
        df.loc[regime_filter & entry_short & additional_filters, 'señal'] = -1

    return df


def _apply_regime_filter(df, config):
    """Aplica filtro de régimen (tendencia general)."""
    regime_type = config.get('regime_type', 'ema')
    regime_period = config.get('regime_period', 200)

    if regime_type == 'none':
        return pd.Series(True, index=df.index)

    elif regime_type == 'ema':
        ema_col = f"EMA_{regime_period}"
        if ema_col not in df.columns:
            return pd.Series(True, index=df.index)
        return df['close'] > df[ema_col]

    elif regime_type == 'sma':
        sma_col = f"SMA_{regime_period}"
        if sma_col not in df.columns:
            return pd.Series(True, index=df.index)
        return df['close'] > df[sma_col]

    elif regime_type == 'adx':
        # ADX solo: Filtra por fuerza de tendencia (sin dirección)
        adx_col = f"ADX_{config.get('adx_period', 14)}"
        if adx_col not in df.columns:
            return pd.Series(True, index=df.index)
        return df[adx_col] > config.get('adx_threshold', 25)

    elif regime_type == 'ema_adx':
        # Combinación: EMA para dirección + ADX para fuerza
        ema_col = f"EMA_{regime_period}"
        adx_col = f"ADX_{config.get('adx_period', 14)}"
        if ema_col not in df.columns or adx_col not in df.columns:
            return pd.Series(True, index=df.index)
        return (df['close'] > df[ema_col]) & (df[adx_col] > config.get('adx_threshold', 25))

    elif regime_type == 'sma_adx':
        # Combinación: SMA para dirección + ADX para fuerza
        sma_col = f"SMA_{regime_period}"
        adx_col = f"ADX_{config.get('adx_period', 14)}"
        if sma_col not in df.columns or adx_col not in df.columns:
            return pd.Series(True, index=df.index)
        return (df['close'] > df[sma_col]) & (df[adx_col] > config.get('adx_threshold', 25))

    return pd.Series(True, index=df.index)


def _generate_entry_signals(df, config):
    """Genera señales de entrada combinando múltiples indicadores."""
    entry_indicators = config.get('entry_indicators', ['ema_cross'])

    # Inicializar como True (todas las señales empiezan activas)
    long_signals = pd.Series(True, index=df.index)
    short_signals = pd.Series(True, index=df.index)

    # Combinar indicadores con lógica AND (todos deben confirmar)
    for indicator in entry_indicators:
        if indicator == 'ema_cross':
            long, short = _signal_ema_cross(df, config)
            long_signals = long_signals & long
            short_signals = short_signals & short

        elif indicator == 'rsi':
            long, short = _signal_rsi(df, config)
            long_signals = long_signals & long
            short_signals = short_signals & short

        elif indicator == 'macd':
            long, short = _signal_macd(df, config)
            long_signals = long_signals & long
            short_signals = short_signals & short

        elif indicator == 'bb':
            long, short = _signal_bollinger(df, config)
            long_signals = long_signals & long
            short_signals = short_signals & short

        elif indicator == 'donchian':
            long, short = _signal_donchian(df, config)
            long_signals = long_signals & long
            short_signals = short_signals & short

        elif indicator == 'supertrend':
            long, short = _signal_supertrend(df, config)
            long_signals = long_signals & long
            short_signals = short_signals & short

        elif indicator == 'vwma_cross':
            long, short = _signal_vwma_cross(df, config)
            long_signals = long_signals & long
            short_signals = short_signals & short

    return long_signals, short_signals


def _signal_ema_cross(df, config):
    """Señal: Cruce de EMAs."""
    ema_fast = config.get('ema_fast', 9)
    ema_slow = config.get('ema_slow', 21)

    ema_fast_col = f"EMA_{ema_fast}"
    ema_slow_col = f"EMA_{ema_slow}"

    if ema_fast_col not in df.columns or ema_slow_col not in df.columns:
        return pd.Series(False, index=df.index), pd.Series(False, index=df.index)

    # Golden Cross (compra)
    long = (
        (df[ema_fast_col] > df[ema_slow_col]) &
        (df[ema_fast_col].shift(1) <= df[ema_slow_col].shift(1))
    )

    # Death Cross (venta)
    short = (
        (df[ema_fast_col] < df[ema_slow_col]) &
        (df[ema_fast_col].shift(1) >= df[ema_slow_col].shift(1))
    )

    return long, short


def _signal_rsi(df, config):
    """Señal: RSI en zonas de sobrecompra/sobreventa."""
    rsi_period = config.get('rsi_period', 14)
    rsi_oversold = config.get('rsi_oversold', 30)
    rsi_overbought = config.get('rsi_overbought', 70)

    rsi_col = f"RSI_{rsi_period}"

    if rsi_col not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)

    # Compra: RSI sale de sobreventa
    long = (df[rsi_col].shift(1) < rsi_oversold) & (df[rsi_col] >= rsi_oversold)

    # Venta: RSI sale de sobrecompra
    short = (df[rsi_col].shift(1) > rsi_overbought) & (df[rsi_col] <= rsi_overbought)

    return long, short


def _signal_macd(df, config):
    """Señal: Cruce MACD."""
    macd_fast = config.get('macd_fast', 12)
    macd_slow = config.get('macd_slow', 26)
    macd_signal = config.get('macd_signal', 9)

    macd_col = f"MACD_{macd_fast}_{macd_slow}_{macd_signal}"
    signal_col = f"MACDs_{macd_fast}_{macd_slow}_{macd_signal}"

    if macd_col not in df.columns or signal_col not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)

    # Compra: MACD cruza por encima de señal
    long = (
        (df[macd_col] > df[signal_col]) &
        (df[macd_col].shift(1) <= df[signal_col].shift(1))
    )

    # Venta: MACD cruza por debajo de señal
    short = (
        (df[macd_col] < df[signal_col]) &
        (df[macd_col].shift(1) >= df[signal_col].shift(1))
    )

    return long, short


def _signal_bollinger(df, config):
    """Señal: Precio toca bandas de Bollinger."""
    bb_length = config.get('bb_length', 20)
    bb_std = config.get('bb_std', 2.0)

    bbl_col = f"BBL_{bb_length}_{bb_std}"
    bbu_col = f"BBU_{bb_length}_{bb_std}"

    if bbl_col not in df.columns or bbu_col not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)

    # Compra: precio toca banda inferior
    long = df['close'] <= df[bbl_col]

    # Venta: precio toca banda superior
    short = df['close'] >= df[bbu_col]

    return long, short


def _signal_donchian(df, config):
    """Señal: Breakout de canales de Donchian."""
    donchian_period = config.get('donchian_period', 20)

    dch_col = f"DONCHI_h_{donchian_period}"
    dcl_col = f"DONCHI_l_{donchian_period}"

    if dch_col not in df.columns or dcl_col not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)

    # Compra: breakout al alza
    long = (
        (df['close'] >= df[dch_col]) &
        (df['close'].shift(1) < df[dch_col].shift(1))
    )

    # Venta: breakout a la baja
    short = (
        (df['close'] <= df[dcl_col]) &
        (df['close'].shift(1) > df[dcl_col].shift(1))
    )

    return long, short


def _signal_supertrend(df, config):
    """Señal: Supertrend indicator."""
    supertrend_length = config.get('supertrend_length', 10)
    supertrend_multiplier = config.get('supertrend_multiplier', 3.0)

    # Nombres de columnas de Supertrend
    st_col = f"SUPERT_{supertrend_length}_{supertrend_multiplier}"
    std_col = f"SUPERTd_{supertrend_length}_{supertrend_multiplier}"

    # Si no existen las columnas, retornar True (permitir todas las señales)
    if st_col not in df.columns or std_col not in df.columns:
        return pd.Series(True, index=df.index), pd.Series(True, index=df.index)

    # Compra: Supertrend cambia a alcista (dirección = 1)
    # Supertrend alcista cuando close > Supertrend line
    long = (
        (df[std_col] == 1) &
        (df[std_col].shift(1) == -1)
    )

    # Venta: Supertrend cambia a bajista (dirección = -1)
    short = (
        (df[std_col] == -1) &
        (df[std_col].shift(1) == 1)
    )

    return long, short


def _signal_vwma_cross(df, config):
    """Señal: Cruce de VWMAs (Volume Weighted Moving Averages)."""
    vwma_fast = config.get('vwma_fast', 9)
    vwma_slow = config.get('vwma_slow', 21)

    # Calcular VWMAs si no existen
    if 'volume' not in df.columns:
        # Si no hay volumen, usar EMA regular
        return _signal_ema_cross(df, {'ema_fast': vwma_fast, 'ema_slow': vwma_slow})

    # VWMA = Sum(close * volume) / Sum(volume)
    vwma_fast_col = f"VWMA_{vwma_fast}"
    vwma_slow_col = f"VWMA_{vwma_slow}"

    # Si no existen, calcularlos
    if vwma_fast_col not in df.columns:
        df[vwma_fast_col] = (df['close'] * df['volume']).rolling(vwma_fast).sum() / df['volume'].rolling(vwma_fast).sum()
    if vwma_slow_col not in df.columns:
        df[vwma_slow_col] = (df['close'] * df['volume']).rolling(vwma_slow).sum() / df['volume'].rolling(vwma_slow).sum()

    # Golden Cross (compra)
    long = (
        (df[vwma_fast_col] > df[vwma_slow_col]) &
        (df[vwma_fast_col].shift(1) <= df[vwma_slow_col].shift(1))
    )

    # Death Cross (venta)
    short = (
        (df[vwma_fast_col] < df[vwma_slow_col]) &
        (df[vwma_fast_col].shift(1) >= df[vwma_slow_col].shift(1))
    )

    return long, short


def _apply_additional_filters(df, config):
    """Aplica filtros adicionales (volumen, ATR, momentum)."""
    filters = pd.Series(True, index=df.index)

    # Filtro de volumen
    if config.get('use_volume_filter', False):
        volume_ma_period = config.get('volume_ma_period', 20)
        if 'volume' in df.columns:
            volume_ma = df['volume'].rolling(volume_ma_period).mean()
            filters = filters & (df['volume'] > volume_ma)

    # Filtro de ATR (volatilidad mínima)
    if config.get('use_atr_filter', False):
        atr_period = config.get('atr_period', 14)
        atr_col = f"ATRr_{atr_period}"
        if atr_col in df.columns:
            atr_pct = (df[atr_col] / df['close']) * 100
            atr_min = config.get('atr_min_threshold', 0.5)
            filters = filters & (atr_pct > atr_min)

    return filters


def generate_signals_multi_timeframe(df_trade, df_higher, config):
    """
    Genera señales usando estrategia Multi-Timeframe (MTF).

    Arquitectura:
    1. TIMEFRAME SUPERIOR (df_higher): Filtra TENDENCIA y CONDICIONES del mercado
       - Solo permite operar cuando hay tendencia clara
       - Reduce drawdown eliminando mercados laterales

    2. TIMEFRAME OPERACIÓN (df_trade): Busca ENTRADAS específicas
       - Genera señales con indicadores técnicos
       - Solo ejecuta si timeframe superior lo permite

    Resultado: Más frecuencia (TF bajo) + Menos DD (filtro TF alto)

    Args:
        df_trade: DataFrame del timeframe de operación (ej: 5m, 15m)
        df_higher: DataFrame del timeframe superior (ej: 1h, 4h)
        config: Configuración con parámetros de ambos timeframes:
            {
                # FILTRO TIMEFRAME SUPERIOR
                'htf_ema_fast': 50,
                'htf_ema_slow': 200,
                'htf_adx_period': 14,
                'htf_adx_threshold': 25,
                'htf_use_rsi_filter': False,  # RSI no extremo en TF alto
                'htf_rsi_min': 40,
                'htf_rsi_max': 60,

                # SEÑALES TIMEFRAME OPERACIÓN (igual que single TF)
                'entry_indicators': ['supertrend', 'rsi'],
                'supertrend_length': 7,
                'supertrend_multiplier': 1.5,
                'rsi_period': 14,
                'rsi_oversold': 30,
                'rsi_overbought': 65,

                # GESTIÓN DE RIESGO
                'atr_period': 14,
                'sl_atr_multiplier': 2.5,
                'tp_atr_multiplier': 5.0,
            }

    Returns:
        DataFrame del timeframe de operación con columna 'señal'
    """

    # 1. Calcular filtro de tendencia en timeframe superior
    df_higher = df_higher.copy()

    ema_fast_col = f"EMA_{config.get('htf_ema_fast', 50)}"
    ema_slow_col = f"EMA_{config.get('htf_ema_slow', 200)}"
    adx_col = f"ADX_{config.get('htf_adx_period', 14)}"
    rsi_col = f"RSI_{config.get('rsi_period', 14)}"

    # Asegurar que las columnas existen
    if ema_fast_col not in df_higher.columns:
        df_higher.ta.ema(length=config.get('htf_ema_fast', 50), append=True)
    if ema_slow_col not in df_higher.columns:
        df_higher.ta.ema(length=config.get('htf_ema_slow', 200), append=True)
    if adx_col not in df_higher.columns:
        df_higher.ta.adx(length=config.get('htf_adx_period', 14), append=True)

    adx_threshold = config.get('htf_adx_threshold', 25)

    # Determinar dirección permitida en cada vela del TF superior
    df_higher['can_long'] = (
        (df_higher[ema_fast_col] > df_higher[ema_slow_col]) &  # Tendencia alcista
        (df_higher[adx_col] > adx_threshold)  # Tendencia fuerte
    )

    df_higher['can_short'] = (
        (df_higher[ema_fast_col] < df_higher[ema_slow_col]) &  # Tendencia bajista
        (df_higher[adx_col] > adx_threshold)  # Tendencia fuerte
    )

    # Filtro RSI opcional en TF superior (evitar extremos)
    if config.get('htf_use_rsi_filter', False) and rsi_col in df_higher.columns:
        rsi_min = config.get('htf_rsi_min', 40)
        rsi_max = config.get('htf_rsi_max', 60)

        rsi_ok = (df_higher[rsi_col] > rsi_min) & (df_higher[rsi_col] < rsi_max)
        df_higher['can_long'] = df_higher['can_long'] & rsi_ok
        df_higher['can_short'] = df_higher['can_short'] & rsi_ok

    # 2. Sincronizar timeframes: Cada vela trade TF debe saber qué permite el higher TF
    df_trade = df_trade.copy()

    # Crear columna timestamp redondeada al TF superior
    # Asumimos que df_trade y df_higher tienen columna 'timestamp' como datetime
    # Si no existe, usar el índice
    if 'timestamp' not in df_trade.columns:
        df_trade['timestamp'] = df_trade.index
    if 'timestamp' not in df_higher.columns:
        df_higher['timestamp'] = df_higher.index

    # Determinar el intervalo del TF superior para hacer merge correcto
    # Esto es una simplificación - en producción deberías detectar automáticamente
    df_trade['timestamp_htf'] = pd.to_datetime(df_trade['timestamp']).dt.floor('1H')

    # Merge para traer can_long y can_short al timeframe de operación
    df_merged = df_trade.merge(
        df_higher[['timestamp', 'can_long', 'can_short']],
        left_on='timestamp_htf',
        right_on='timestamp',
        how='left',
        suffixes=('', '_htf')
    )

    # Forward fill por si hay gaps
    df_merged['can_long'] = df_merged['can_long'].fillna(method='ffill').fillna(False)
    df_merged['can_short'] = df_merged['can_short'].fillna(method='ffill').fillna(False)

    # 3. Generar señales en timeframe de operación usando generate_signals_multi_indicator
    df_signals = generate_signals_multi_indicator(df_merged, config)

    # 4. Filtrar señales según permiso del timeframe superior
    final_signal = pd.Series(0, index=df_signals.index)

    # Solo permitir LONG si can_long = True
    long_signals = (df_signals['señal'] == 1) & df_signals['can_long']
    final_signal[long_signals] = 1

    # Solo permitir SHORT si can_short = True
    short_signals = (df_signals['señal'] == -1) & df_signals['can_short']
    final_signal[short_signals] = -1

    df_signals['señal'] = final_signal

    # Limpiar columnas temporales
    df_signals = df_signals.drop(columns=['timestamp_htf', 'timestamp_htf', 'can_long', 'can_short'], errors='ignore')

    return df_signals
