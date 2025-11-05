"""
Módulo para obtener datos históricos de Binance.
Descarga velas (OHLCV) y las procesa en DataFrames de pandas.
"""

import pandas as pd
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException


def obtener_datos_binance(client, simbolo='BTCUSDT', intervalo='5m', inicio='1 year ago UTC'):
    """
    Descarga datos históricos de Binance usando la API REST.

    Args:
        client: Cliente de Binance (de binance_client.py)
        simbolo: Par de trading (ej. 'BTCUSDT', 'ETHUSDT')
        intervalo: Intervalo de velas (ej. '1m', '5m', '15m', '1h', '1d')
        inicio: Fecha de inicio (ej. '1 year ago UTC', '3 months ago UTC', '2023-01-01')

    Returns:
        DataFrame de pandas con columnas: timestamp, open, high, low, close, volume
    """
    print(f"Descargando datos de {simbolo} ({intervalo}) desde {inicio}...")

    # Mapeo de intervalos a constantes de Binance
    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH
    }

    if intervalo not in interval_map:
        raise ValueError(f"Intervalo '{intervalo}' no válido. Opciones: {list(interval_map.keys())}")

    binance_interval = interval_map[intervalo]

    try:
        # Descargar datos históricos
        klines = client.get_historical_klines(
            symbol=simbolo,
            interval=binance_interval,
            start_str=inicio
        )

        # Procesar datos en DataFrame
        df = _procesar_klines(klines)

        print(f"✓ Descargados {len(df)} registros desde {df['timestamp'].iloc[0]} hasta {df['timestamp'].iloc[-1]}")

        return df

    except BinanceAPIException as e:
        raise Exception(f"Error al descargar datos de Binance: {e}")


def _procesar_klines(klines):
    """
    Convierte los datos raw de Binance en un DataFrame limpio.

    Estructura de cada kline de Binance:
    [
        0: Open time (timestamp en ms),
        1: Open,
        2: High,
        3: Low,
        4: Close,
        5: Volume,
        6: Close time,
        7: Quote asset volume,
        8: Number of trades,
        9: Taker buy base asset volume,
        10: Taker buy quote asset volume,
        11: Ignore
    ]

    Args:
        klines: Lista de klines raw de Binance

    Returns:
        DataFrame con columnas limpias
    """
    # Crear DataFrame
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Seleccionar solo las columnas necesarias
    df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']]

    # Convertir timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')

    # Convertir precios y volumen a float
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # Reorganizar columnas
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Resetear índice
    df.reset_index(drop=True, inplace=True)

    return df


def obtener_ultimas_velas(client, simbolo='BTCUSDT', intervalo='5m', limite=500):
    """
    Descarga las últimas N velas (útil para "priming" en paper/live trading).

    Args:
        client: Cliente de Binance
        simbolo: Par de trading
        intervalo: Intervalo de velas
        limite: Número de velas a descargar (máx. 1000)

    Returns:
        DataFrame con las últimas velas
    """
    if limite > 1000:
        raise ValueError("El límite máximo es 1000 velas por solicitud")

    interval_map = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY
    }

    binance_interval = interval_map.get(intervalo, Client.KLINE_INTERVAL_5MINUTE)

    try:
        # Descargar últimas velas
        klines = client.get_klines(
            symbol=simbolo,
            interval=binance_interval,
            limit=limite
        )

        # Procesar en DataFrame
        df = _procesar_klines(klines)

        print(f"✓ Descargadas últimas {len(df)} velas de {simbolo}")

        return df

    except BinanceAPIException as e:
        raise Exception(f"Error al descargar últimas velas: {e}")


def guardar_datos(df, archivo='data/historical_data.csv'):
    """
    Guarda el DataFrame en un archivo CSV.

    Args:
        df: DataFrame a guardar
        archivo: Ruta del archivo CSV
    """
    df.to_csv(archivo, index=False)
    print(f"✓ Datos guardados en {archivo}")


def cargar_datos(archivo='data/historical_data.csv'):
    """
    Carga datos desde un archivo CSV.

    Args:
        archivo: Ruta del archivo CSV

    Returns:
        DataFrame con los datos
    """
    df = pd.read_csv(archivo)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"✓ Datos cargados desde {archivo} ({len(df)} registros)")
    return df


if __name__ == "__main__":
    # Test básico del módulo
    from src.data.binance_client import BinanceClientManager

    print("=== Test de Data Fetcher ===\n")

    # Crear cliente
    manager = BinanceClientManager()
    client = manager.get_public_client()

    # Descargar datos históricos (últimos 7 días)
    print("1. Descargando datos históricos:")
    df = obtener_datos_binance(
        client=client,
        simbolo='BTCUSDT',
        intervalo='5m',
        inicio='7 days ago UTC'
    )

    print("\n2. Primeros 5 registros:")
    print(df.head())

    print("\n3. Últimos 5 registros:")
    print(df.tail())

    print("\n4. Información del DataFrame:")
    print(f"   Filas: {len(df)}")
    print(f"   Columnas: {list(df.columns)}")
    print(f"   Tipos de datos:")
    print(df.dtypes)

    # Descargar últimas velas
    print("\n5. Descargando últimas 100 velas:")
    df_recent = obtener_ultimas_velas(
        client=client,
        simbolo='BTCUSDT',
        intervalo='5m',
        limite=100
    )
    print(f"   Rango: {df_recent['timestamp'].iloc[0]} a {df_recent['timestamp'].iloc[-1]}")

    print("\n✓ Test completado exitosamente")
