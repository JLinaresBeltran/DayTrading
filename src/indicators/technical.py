"""
Módulo para calcular indicadores técnicos usando pandas-ta.
Todos los indicadores se añaden directamente al DataFrame.
"""

import pandas as pd
import numpy as np

# Intentar importar pandas_ta, si no está disponible, usar implementaciones manuales
try:
    import pandas_ta as ta
    HAS_PANDAS_TA = True
except ImportError:
    HAS_PANDAS_TA = False
    print("ADVERTENCIA: pandas_ta no está instalado. Usando implementaciones manuales de indicadores.")


def calcular_ema_manual(series, period):
    """Calcula EMA manualmente sin pandas-ta."""
    return series.ewm(span=period, adjust=False).mean()


def calcular_atr_manual(df, period=14):
    """Calcula ATR manualmente sin pandas-ta."""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR es el promedio móvil del True Range
    atr = tr.rolling(window=period).mean()

    return atr


def calcular_rsi_manual(series, period=14):
    """Calcula RSI manualmente sin pandas-ta."""
    delta = series.diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calcular_macd_manual(series, fast=12, slow=26, signal=9):
    """Calcula MACD manualmente sin pandas-ta."""
    ema_fast = calcular_ema_manual(series, fast)
    ema_slow = calcular_ema_manual(series, slow)

    macd = ema_fast - ema_slow
    signal_line = calcular_ema_manual(macd, signal)
    histogram = macd - signal_line

    return macd, signal_line, histogram


def calcular_bollinger_bands_manual(series, period=20, std_dev=2):
    """Calcula Bandas de Bollinger manualmente sin pandas-ta."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()

    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)

    # Bandwidth y %B
    bandwidth = (upper - lower) / sma
    percent_b = (series - lower) / (upper - lower)

    return lower, sma, upper, bandwidth, percent_b


def calcular_stochastic_manual(df, k_period=14, d_period=3, smooth_k=3):
    """Calcula Estocástico manualmente sin pandas-ta."""
    high = df['high']
    low = df['low']
    close = df['close']

    # Estocástico %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()

    k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    # Suavizar %K si smooth_k > 1
    if smooth_k > 1:
        k = k.rolling(window=smooth_k).mean()

    # Estocástico %D (promedio móvil de %K)
    d = k.rolling(window=d_period).mean()

    return k, d


def calcular_adx_manual(df, period=14):
    """
    Calcula ADX (Average Directional Index) manualmente sin pandas-ta.

    El ADX mide la fuerza de la tendencia (no la dirección).
    - ADX > 25: Tendencia fuerte
    - ADX < 20: Tendencia débil o mercado lateral
    - ADX > 50: Tendencia muy fuerte

    Args:
        df: DataFrame con columnas high, low, close
        period: Período de suavizado (típicamente 14)

    Returns:
        tuple: (adx, plus_di, minus_di)
    """
    high = df['high']
    low = df['low']
    close = df['close']

    # Calcular True Range (TR)
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Calcular movimientos direccionales
    plus_dm = high.diff()  # High(t) - High(t-1)
    minus_dm = -low.diff()  # Low(t-1) - Low(t)

    # Solo contar movimientos positivos
    plus_dm = plus_dm.where((plus_dm > 0) & (plus_dm > minus_dm), 0)
    minus_dm = minus_dm.where((minus_dm > 0) & (minus_dm > plus_dm), 0)

    # Suavizar TR y DM con EMA (Wilder's smoothing)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di_smooth = plus_dm.ewm(alpha=1/period, adjust=False).mean()
    minus_di_smooth = minus_dm.ewm(alpha=1/period, adjust=False).mean()

    # Calcular +DI y -DI (Directional Indicators)
    plus_di = 100 * (plus_di_smooth / atr)
    minus_di = 100 * (minus_di_smooth / atr)

    # Calcular DX (Directional Movement Index)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    # Calcular ADX (promedio suavizado del DX)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    return adx, plus_di, minus_di


def agregar_indicadores(df, config=None):
    """
    Añade indicadores técnicos al DataFrame usando pandas-ta.

    Indicadores incluidos:
    - EMA (21, 50, 200)
    - RSI (14)
    - Bandas de Bollinger (20, std 2)
    - MACD (12, 26, 9)
    - ATR (14)
    - ADX (14) - Average Directional Index para medir fuerza de tendencia
    - Estocástico (14, 3, 3)
    - Canales de Donchian (20, 40, 60) - para estrategia de Breakout

    Args:
        df: DataFrame con columnas OHLCV (open, high, low, close, volume)
        config: Diccionario con parámetros personalizados (opcional)

    Returns:
        DataFrame con indicadores añadidos
    """
    # Parámetros por defecto
    if config is None:
        config = {
            'ema_short': 21,
            'ema_long': 51,    # EMA 51 para estrategia v21 (antes 50)
            'ema_trend': 200,  # EMA de tendencia para filtro de régimen
            'rsi_period': 14,
            'bb_length': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_length': 14,
            'adx_period': 14,  # ADX para medir fuerza de tendencia
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3,
            'donchian_period': 20  # Canal de Donchian para breakout
        }

    # Hacer una copia para evitar modificar el original
    df = df.copy()

    if HAS_PANDAS_TA:
        # Usar pandas_ta si está disponible
        # EMA (Exponential Moving Average) - Solo si están en config
        if 'ema_short' in config:
            df.ta.ema(length=config['ema_short'], append=True)
        if 'ema_long' in config:
            df.ta.ema(length=config['ema_long'], append=True)
        if 'ema_trend' in config:
            df.ta.ema(length=config['ema_trend'], append=True)  # EMA 200 para filtro de régimen
        if 'ema_filter' in config:
            df.ta.ema(length=config['ema_filter'], append=True)  # EMA filtro para estrategia Long-Only

        # RSI (Relative Strength Index)
        df.ta.rsi(length=config['rsi_period'], append=True)

        # Bandas de Bollinger
        df.ta.bbands(length=config['bb_length'], std=config['bb_std'], append=True)

        # MACD (Moving Average Convergence Divergence)
        df.ta.macd(
            fast=config['macd_fast'],
            slow=config['macd_slow'],
            signal=config['macd_signal'],
            append=True
        )

        # ATR (Average True Range) - para gestión de riesgo
        df.ta.atr(length=config['atr_length'], append=True)

        # ADX (Average Directional Index) - para medir fuerza de tendencia
        df.ta.adx(length=config['adx_period'], append=True)

        # Estocástico
        df.ta.stoch(
            k=config['stoch_k'],
            d=config['stoch_d'],
            smooth_k=config['stoch_smooth'],
            append=True
        )
    else:
        # Usar implementaciones manuales
        # EMA
        if 'ema_short' in config:
            df[f"EMA_{config['ema_short']}"] = calcular_ema_manual(df['close'], config['ema_short'])
        if 'ema_long' in config:
            df[f"EMA_{config['ema_long']}"] = calcular_ema_manual(df['close'], config['ema_long'])
        if 'ema_trend' in config:
            df[f"EMA_{config['ema_trend']}"] = calcular_ema_manual(df['close'], config['ema_trend'])
        if 'ema_filter' in config:
            df[f"EMA_{config['ema_filter']}"] = calcular_ema_manual(df['close'], config['ema_filter'])

        # RSI
        df[f"RSI_{config['rsi_period']}"] = calcular_rsi_manual(df['close'], config['rsi_period'])

        # Bandas de Bollinger
        bb_lower, bb_middle, bb_upper, bb_bandwidth, bb_percent_b = calcular_bollinger_bands_manual(
            df['close'], config['bb_length'], config['bb_std']
        )
        df[f"BBL_{config['bb_length']}_{config['bb_std']}.0"] = bb_lower
        df[f"BBM_{config['bb_length']}_{config['bb_std']}.0"] = bb_middle
        df[f"BBU_{config['bb_length']}_{config['bb_std']}.0"] = bb_upper
        df[f"BBB_{config['bb_length']}_{config['bb_std']}.0"] = bb_bandwidth
        df[f"BBP_{config['bb_length']}_{config['bb_std']}.0"] = bb_percent_b

        # MACD
        macd, macd_signal, macd_histogram = calcular_macd_manual(
            df['close'], config['macd_fast'], config['macd_slow'], config['macd_signal']
        )
        df[f"MACD_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"] = macd
        df[f"MACDs_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"] = macd_signal
        df[f"MACDh_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}"] = macd_histogram

        # ATR
        df[f"ATRr_{config['atr_length']}"] = calcular_atr_manual(df, config['atr_length'])

        # ADX
        adx, plus_di, minus_di = calcular_adx_manual(df, config['adx_period'])
        df[f"ADX_{config['adx_period']}"] = adx
        df[f"DMP_{config['adx_period']}"] = plus_di  # +DI
        df[f"DMN_{config['adx_period']}"] = minus_di  # -DI

        # Estocástico
        stoch_k, stoch_d = calcular_stochastic_manual(
            df, config['stoch_k'], config['stoch_d'], config['stoch_smooth']
        )
        df[f"STOCHk_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}"] = stoch_k
        df[f"STOCHd_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}"] = stoch_d

    # Canales de Donchian - Para estrategia de Breakout
    # El Canal de Donchian traza el precio más alto y más bajo de los últimos N períodos
    # NOTA: pandas-ta sobrescribe columnas al llamar múltiples veces donchian()
    # Por lo tanto, calculamos manualmente cada período
    if 'donchian_period' in config:
        periods = config['donchian_period'] if isinstance(config['donchian_period'], list) else [config['donchian_period']]

        for period in periods:
            # Calcular manualmente Canal de Donchian para cada período
            # DONCHI_h = Donchian Channel Upper (máximo de N períodos)
            # DONCHI_l = Donchian Channel Lower (mínimo de N períodos)
            # DONCHIm = Donchian Channel Middle (promedio de upper y lower)
            df[f'DONCHI_h_{period}'] = df['high'].rolling(window=period).max()
            df[f'DONCHI_l_{period}'] = df['low'].rolling(window=period).min()
            df[f'DONCHIm_{period}'] = (df[f'DONCHI_h_{period}'] + df[f'DONCHI_l_{period}']) / 2

    # Limpiar valores NaN (primeras filas donde no se pueden calcular indicadores)
    # Nota: No eliminamos filas, solo rellenamos con método forward-fill para las primeras
    df.fillna(method='bfill', inplace=True)

    return df


def agregar_indicadores_personalizados(df, indicadores_config):
    """
    Añade indicadores personalizados según configuración específica.

    Args:
        df: DataFrame con datos OHLCV
        indicadores_config: Lista de diccionarios con configuración de indicadores
            Ejemplo:
            [
                {'tipo': 'ema', 'length': 9},
                {'tipo': 'sma', 'length': 200},
                {'tipo': 'rsi', 'length': 7}
            ]

    Returns:
        DataFrame con indicadores añadidos
    """
    df = df.copy()

    for config in indicadores_config:
        tipo = config['tipo'].lower()

        if tipo == 'ema':
            df.ta.ema(length=config['length'], append=True)
        elif tipo == 'sma':
            df.ta.sma(length=config['length'], append=True)
        elif tipo == 'rsi':
            df.ta.rsi(length=config['length'], append=True)
        elif tipo == 'macd':
            df.ta.macd(
                fast=config.get('fast', 12),
                slow=config.get('slow', 26),
                signal=config.get('signal', 9),
                append=True
            )
        elif tipo == 'atr':
            df.ta.atr(length=config['length'], append=True)
        elif tipo == 'bbands':
            df.ta.bbands(
                length=config.get('length', 20),
                std=config.get('std', 2),
                append=True
            )
        elif tipo == 'stoch':
            df.ta.stoch(
                k=config.get('k', 14),
                d=config.get('d', 3),
                smooth_k=config.get('smooth_k', 3),
                append=True
            )
        else:
            print(f"Advertencia: Indicador '{tipo}' no reconocido")

    df.fillna(method='bfill', inplace=True)

    return df


def obtener_nombres_indicadores(config=None):
    """
    Obtiene los nombres de las columnas de indicadores que se crearán.

    Args:
        config: Diccionario con parámetros de indicadores

    Returns:
        Lista de nombres de columnas
    """
    if config is None:
        config = {
            'ema_short': 21,
            'ema_long': 50,
            'rsi_period': 14,
            'bb_length': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_length': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'stoch_smooth': 3
        }

    nombres = [
        f"EMA_{config['ema_short']}",
        f"EMA_{config['ema_long']}",
        f"RSI_{config['rsi_period']}",
        f"BBL_{config['bb_length']}_{config['bb_std']}.0",  # Banda inferior
        f"BBM_{config['bb_length']}_{config['bb_std']}.0",  # Banda media
        f"BBU_{config['bb_length']}_{config['bb_std']}.0",  # Banda superior
        f"BBB_{config['bb_length']}_{config['bb_std']}.0",  # Bandwidth
        f"BBP_{config['bb_length']}_{config['bb_std']}.0",  # Percent B
        f"MACD_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}",
        f"MACDh_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}",  # Histograma
        f"MACDs_{config['macd_fast']}_{config['macd_slow']}_{config['macd_signal']}",  # Señal
        f"ATR_{config['atr_length']}",
        f"STOCHk_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}",
        f"STOCHd_{config['stoch_k']}_{config['stoch_d']}_{config['stoch_smooth']}"
    ]

    return nombres


def verificar_indicadores(df):
    """
    Verifica que todos los indicadores se hayan calculado correctamente.

    Args:
        df: DataFrame con indicadores

    Returns:
        Diccionario con estadísticas de los indicadores
    """
    # Columnas de indicadores (excluir OHLCV originales)
    columnas_base = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    columnas_indicadores = [col for col in df.columns if col not in columnas_base]

    estadisticas = {
        'total_indicadores': len(columnas_indicadores),
        'indicadores': columnas_indicadores,
        'valores_nulos': {},
        'rango_valores': {}
    }

    for col in columnas_indicadores:
        estadisticas['valores_nulos'][col] = df[col].isna().sum()
        estadisticas['rango_valores'][col] = {
            'min': df[col].min(),
            'max': df[col].max(),
            'promedio': df[col].mean()
        }

    return estadisticas


if __name__ == "__main__":
    # Test básico del módulo
    from src.data.binance_client import BinanceClientManager
    from src.data.data_fetcher import obtener_datos_binance

    print("=== Test de Indicadores Técnicos ===\n")

    # Obtener datos
    manager = BinanceClientManager()
    client = manager.get_public_client()

    print("1. Descargando datos...")
    df = obtener_datos_binance(
        client=client,
        simbolo='BTCUSDT',
        intervalo='5m',
        inicio='7 days ago UTC'
    )

    print(f"   Datos originales: {len(df)} filas, {len(df.columns)} columnas")

    # Añadir indicadores
    print("\n2. Calculando indicadores técnicos...")
    df = agregar_indicadores(df)

    print(f"   Datos con indicadores: {len(df)} filas, {len(df.columns)} columnas")

    # Mostrar últimas 5 filas
    print("\n3. Últimas 5 filas con indicadores:")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(df.tail())

    # Verificar indicadores
    print("\n4. Verificación de indicadores:")
    stats = verificar_indicadores(df)
    print(f"   Total de indicadores: {stats['total_indicadores']}")
    print(f"   Indicadores calculados: {stats['indicadores']}")

    # Verificar valores nulos
    valores_nulos = sum(stats['valores_nulos'].values())
    if valores_nulos > 0:
        print(f"   ⚠️  Valores nulos encontrados: {valores_nulos}")
    else:
        print("   ✓ Sin valores nulos")

    # Mostrar algunos valores de indicadores clave
    print("\n5. Valores de indicadores clave (última fila):")
    ultima_fila = df.iloc[-1]
    print(f"   Precio: ${ultima_fila['close']:.2f}")
    print(f"   EMA_21: ${ultima_fila['EMA_21']:.2f}")
    print(f"   EMA_50: ${ultima_fila['EMA_50']:.2f}")
    print(f"   RSI_14: {ultima_fila['RSI_14']:.2f}")
    print(f"   ATR_14: ${ultima_fila['ATR_14']:.2f}")

    print("\n✓ Test completado exitosamente")
