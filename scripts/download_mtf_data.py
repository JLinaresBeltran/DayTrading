#!/usr/bin/env python3
"""
Script para descargar datos históricos Multi-Timeframe.
Descarga ETHUSDT en timeframes 15m y 1h para la estrategia MTF v001.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger('download_mtf_data', 'logs/download_mtf_data.log')


def descargar_timeframe(symbol, timeframe, days_back):
    """
    Descarga datos históricos para un timeframe específico.

    Args:
        symbol: Par de trading (ej: 'ETHUSDT')
        timeframe: Timeframe ('15m' o '1h')
        days_back: Días hacia atrás

    Returns:
        DataFrame con datos OHLCV
    """
    print(f"📥 Descargando {symbol} {timeframe}...")
    print(f"   Período: últimos {days_back} días")

    try:
        # Inicializar cliente
        client_manager = BinanceClientManager()
        client = client_manager.get_public_client()

        # Descargar datos usando la función directa
        inicio = f"{days_back} days ago UTC"
        df = obtener_datos_binance(
            client=client,
            simbolo=symbol,
            intervalo=timeframe,
            inicio=inicio
        )

        if df is None or len(df) == 0:
            print(f"   ❌ Error: No se pudieron descargar datos")
            return None

        print(f"   ✓ Descargados: {len(df)} velas")
        print(f"   ✓ Rango: {df['timestamp'].min()} a {df['timestamp'].max()}")

        # Guardar CSV
        os.makedirs('data', exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d')
        filename = f'data/{symbol}_{timeframe}_OHLCV_{timestamp}.csv'

        df.to_csv(filename, index=False)
        print(f"   ✓ Guardado: {filename}")
        print()

        return df

    except Exception as e:
        logger.error(f"Error descargando {symbol} {timeframe}: {str(e)}")
        print(f"   ❌ Error: {str(e)}")
        return None


def main():
    """
    Descarga datos Multi-Timeframe para estrategia MTF v001.
    """
    print("=" * 80)
    print("DESCARGA DE DATOS MULTI-TIMEFRAME")
    print("Estrategia MTF v001: 15m (ejecución) + 1h (régimen)")
    print("=" * 80)
    print()

    symbol = 'ETHUSDT'

    # Configuración de timeframes
    timeframes = [
        {'tf': '15m', 'days': 180},  # 6 meses de datos en 15m
        {'tf': '1h', 'days': 365}    # 1 año de datos en 1h (para EMA_200)
    ]

    print("📊 Configuración:")
    print(f"  - Par: {symbol}")
    print(f"  - Timeframe 1 (ejecución): 15m - 180 días")
    print(f"  - Timeframe 2 (régimen): 1h - 365 días")
    print()

    print("🔌 Conectando a Binance (API pública)...")
    print()

    resultados = []

    # Descargar cada timeframe
    for config in timeframes:
        df = descargar_timeframe(
            symbol=symbol,
            timeframe=config['tf'],
            days_back=config['days']
        )

        if df is not None:
            resultados.append({
                'timeframe': config['tf'],
                'velas': len(df),
                'desde': df['timestamp'].min(),
                'hasta': df['timestamp'].max(),
                'precio_min': df['close'].min(),
                'precio_max': df['close'].max(),
                'precio_ultimo': df['close'].iloc[-1]
            })

    # Resumen final
    print("=" * 80)
    if len(resultados) == len(timeframes):
        print("✅ DESCARGA COMPLETADA EXITOSAMENTE")
    else:
        print("⚠️  DESCARGA PARCIAL - Algunos timeframes fallaron")
    print("=" * 80)
    print()

    if len(resultados) > 0:
        print("📊 Resumen de datos descargados:")
        print()

        for res in resultados:
            print(f"  {res['timeframe'].upper()}:")
            print(f"    - Velas: {res['velas']:,}")
            print(f"    - Rango: {res['desde']} a {res['hasta']}")
            print(f"    - Precio: ${res['precio_min']:.2f} - ${res['precio_max']:.2f}")
            print(f"    - Último: ${res['precio_ultimo']:.2f}")
            print()

    # Verificar archivos
    print("📁 Archivos disponibles en data/:")
    data_files = [f for f in os.listdir('data') if f.startswith('ETHUSDT') and f.endswith('.csv')]

    for f in sorted(data_files):
        print(f"  ✓ {f}")

    print()

    if len(resultados) == len(timeframes):
        print("=" * 80)
        print("🎉 Todo listo para ejecutar la optimización")
        print("=" * 80)
        print()
        print("Siguiente paso:")
        print("  python scripts/phase2_optimize_mtf_v001.py")
        print()
    else:
        print("⚠️  Algunos datos no se descargaron correctamente.")
        print("   Revisa los logs en logs/download_mtf_data.log")


if __name__ == "__main__":
    main()
