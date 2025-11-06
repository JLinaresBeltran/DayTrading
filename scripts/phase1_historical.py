#!/usr/bin/env python3
"""
FASE 1: Lógica y Datos Históricos
Descarga datos de Binance, calcula indicadores técnicos y genera señales.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.strategy.signal_generator import generar_señales, contar_señales, obtener_señales_recientes


def main():
    print("=" * 70)
    print("FASE 1: LÓGICA Y DATOS HISTÓRICOS")
    print("=" * 70)

    # Configuración
    SIMBOLO = 'ETHUSDT'
    INTERVALO = '5m'
    INICIO = '1 year ago UTC'

    # 1. Crear cliente de Binance
    print(f"\n1. Conectando a Binance...")
    manager = BinanceClientManager()
    client = manager.get_public_client()
    print("   ✓ Cliente público creado")

    # 2. Descargar datos históricos
    print(f"\n2. Descargando datos históricos de {SIMBOLO} ({INTERVALO})...")
    df = obtener_datos_binance(
        client=client,
        simbolo=SIMBOLO,
        intervalo=INTERVALO,
        inicio=INICIO
    )
    print(f"   ✓ {len(df)} velas descargadas")
    print(f"   ✓ Período: {df['timestamp'].iloc[0]} hasta {df['timestamp'].iloc[-1]}")

    # 3. Calcular indicadores técnicos
    print(f"\n3. Calculando indicadores técnicos...")
    df = agregar_indicadores(df)
    print(f"   ✓ Indicadores calculados: EMA, RSI, BB, MACD, ATR, Stochastic")
    print(f"   ✓ Total de columnas: {len(df.columns)}")

    # 4. Generar señales de trading
    print(f"\n4. Generando señales de trading...")
    df = generar_señales(df)

    # Estadísticas de señales
    stats = contar_señales(df)
    print(f"   ✓ Total de registros: {stats['total']}")
    print(f"   ✓ Señales de COMPRA: {stats['compras']} ({stats['pct_compras']:.2f}%)")
    print(f"   ✓ Señales de VENTA: {stats['ventas']} ({stats['pct_ventas']:.2f}%)")
    print(f"   ✓ Señales NEUTRALES: {stats['neutrales']} ({stats['pct_neutrales']:.2f}%)")

    # 5. Mostrar últimas señales
    print(f"\n5. Últimas 20 señales generadas:")
    print("-" * 70)
    import pandas as pd
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(obtener_señales_recientes(df, n=20))

    # 6. Guardar datos (opcional)
    print(f"\n6. Guardando datos...")
    from src.data.data_fetcher import guardar_datos
    guardar_datos(df, 'data/historical_data.csv')

    print("\n" + "=" * 70)
    print("✓ FASE 1 COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print("\nPróximos pasos:")
    print("  - Ejecutar Fase 2: python scripts/phase2_backtest.py")
    print("  - Revisar los datos en: data/historical_data.csv")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Operación cancelada por el usuario")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
