#!/usr/bin/env python3
"""
Script para descargar datos históricos de 1 hora para ETHUSDT.
Complementa los datos de 15m ya existentes.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from datetime import datetime

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import DataFetcher
from src.utils.logger import setup_logger

# Configurar logger
logger = setup_logger('download_1h_data', 'logs/download_1h_data.log')


def descargar_datos_1h():
    """
    Descarga datos históricos de 1 hora para ETHUSDT.
    """
    print("=" * 80)
    print("DESCARGA DE DATOS HISTÓRICOS - TIMEFRAME 1H")
    print("=" * 80)
    print()

    # Configuración
    symbol = 'ETHUSDT'
    timeframe = '1h'
    days_back = 365  # 1 año de datos (suficiente para EMA_200)

    print(f"📊 Configuración:")
    print(f"  - Par: {symbol}")
    print(f"  - Timeframe: {timeframe}")
    print(f"  - Período: últimos {days_back} días")
    print()

    try:
        # ============================================
        # 1. INICIALIZAR CLIENTE DE BINANCE
        # ============================================
        print("🔌 Paso 1: Conectando a Binance (API pública)...")

        client_manager = BinanceClientManager()
        client = client_manager.get_public_client()

        print("  ✓ Conexión establecida")
        print()

        # ============================================
        # 2. DESCARGAR DATOS
        # ============================================
        print(f"📥 Paso 2: Descargando datos históricos...")
        print(f"  Esto puede tomar unos minutos...")
        print()

        fetcher = DataFetcher(client)

        df = fetcher.get_historical_klines(
            symbol=symbol,
            interval=timeframe,
            days_back=days_back
        )

        if df is None or len(df) == 0:
            print("❌ ERROR: No se pudieron descargar datos")
            return

        print(f"  ✓ Descargados: {len(df)} velas")
        print(f"  ✓ Desde: {df['timestamp'].min()}")
        print(f"  ✓ Hasta: {df['timestamp'].max()}")
        print()

        # ============================================
        # 3. GUARDAR CSV
        # ============================================
        print("💾 Paso 3: Guardando datos...")

        # Crear directorio si no existe
        os.makedirs('data', exist_ok=True)

        # Nombre del archivo con timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d')
        filename = f'data/{symbol}_{timeframe}_OHLCV_{timestamp}.csv'

        df.to_csv(filename, index=False)

        print(f"  ✓ Archivo guardado: {filename}")
        print()

        # ============================================
        # 4. VERIFICAR DATOS
        # ============================================
        print("✅ Paso 4: Verificación de datos")
        print()

        # Verificar columnas
        required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            print(f"  ⚠️  Columnas faltantes: {missing_cols}")
        else:
            print(f"  ✓ Todas las columnas requeridas están presentes")

        # Verificar valores nulos
        null_counts = df[required_cols].isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls > 0:
            print(f"  ⚠️  Valores nulos encontrados:")
            for col, count in null_counts.items():
                if count > 0:
                    print(f"      - {col}: {count}")
        else:
            print(f"  ✓ No hay valores nulos")

        # Estadísticas básicas
        print()
        print("📈 Estadísticas de precio (close):")
        print(f"  - Mínimo: ${df['close'].min():.2f}")
        print(f"  - Máximo: ${df['close'].max():.2f}")
        print(f"  - Promedio: ${df['close'].mean():.2f}")
        print(f"  - Último: ${df['close'].iloc[-1]:.2f}")

        print()
        print("=" * 80)
        print("🎉 Descarga completada exitosamente")
        print("=" * 80)
        print()
        print("Archivos de datos disponibles:")
        print(f"  ✓ ETHUSDT_15m_OHLCV_*.csv (ya existente)")
        print(f"  ✓ {filename.replace('data/', '')} (recién descargado)")
        print()
        print("Ahora puedes ejecutar la optimización:")
        print("  python scripts/phase2_optimize_mtf_v001.py")

    except Exception as e:
        logger.error(f"Error al descargar datos: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    descargar_datos_1h()
