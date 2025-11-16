"""
Módulo para generar señales de trading basadas en indicadores técnicos.
Señales: 1 (COMPRA), -1 (VENTA), 0 (NADA/NEUTRAL)

NOTA: Las estrategias individuales se han movido a src/strategy/strategies/
Este archivo mantiene las importaciones para compatibilidad con código existente
y proporciona funciones de utilidad.
"""

import pandas as pd
import numpy as np

# Importar todas las estrategias desde el módulo strategies
from .strategies import (
    generar_señales,
    generar_senales_triple_capa,
    generar_senales_momentum_v1,
    generar_senales_hibrido_v1,
    generar_senales_bajista_v1,
    generar_señales_avanzadas,
    generar_señales_con_filtro_tendencia,
)

# Re-exportar todas las estrategias para mantener compatibilidad
__all__ = [
    'generar_señales',
    'generar_senales_triple_capa',
    'generar_senales_momentum_v1',
    'generar_senales_hibrido_v1',
    'generar_senales_bajista_v1',
    'generar_señales_avanzadas',
    'generar_señales_con_filtro_tendencia',
    'obtener_señales_recientes',
    'contar_señales',
]


# ==========================================
# FUNCIONES DE UTILIDAD
# ==========================================


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


# ==========================================
# TEST MODULE
# ==========================================


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
