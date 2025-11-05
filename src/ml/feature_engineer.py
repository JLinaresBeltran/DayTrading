"""
Módulo de Feature Engineering para ML en Trading

Este módulo prepara datos para modelos de Machine Learning:
1. Extrae features técnicos relevantes
2. Crea target binario basado en retornos futuros
3. Maneja la creación del dataset para entrenamiento
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def crear_features(df: pd.DataFrame, mejorado: bool = False) -> pd.DataFrame:
    """
    Extrae features técnicos para el modelo ML.

    Features Básicos:
    1. Tendencia: EMA_200
    2. Momentum: RSI_14
    3. Volatilidad: ATRr_14 (ATR de pandas-ta)
    4. Presión: MACDh_12_26_9 (Histograma MACD)
    5. Reversión: STOCHk_14_3_3 (Estocástico %K)

    Features Mejorados (si mejorado=True):
    6. Bollinger %B: Posición relativa dentro de Bollinger Bands
    7. EMA Crossover: Señal de cruces de EMAs
    8. RSI Z-Score: Desviación normalizada del RSI
    9. Volume Momentum: Cambio % de volumen
    10. Price to EMA200: Precio relativo a media de tendencia

    Args:
        df: DataFrame con indicadores técnicos calculados
        mejorado: Si True, incluye 5 features adicionales (default: False)

    Returns:
        DataFrame con columnas de features seleccionadas

    Raises:
        ValueError: Si faltan indicadores requeridos
    """
    df = df.copy()

    # Features requeridas
    required_features = ['EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3']

    # Validar que existan todos los indicadores
    missing = [f for f in required_features if f not in df.columns]
    if missing:
        raise ValueError(f"Indicadores faltantes: {missing}")

    # Normalización básica de features
    feature_df = df[['close', 'volume', 'EMA_21', 'EMA_50'] + required_features].copy()

    # Normalizar volumen (escala logarítmica)
    feature_df['volume_norm'] = np.log1p(feature_df['volume'])

    # RSI ya está normalizado (0-100)
    # MACD Histogram puede ser negativo, dejarlo como está
    # ATR normalizar por el precio (%)
    feature_df['atr_pct'] = (feature_df['ATRr_14'] / feature_df['close']) * 100

    # Estocástico ya está normalizado (0-100)

    # Crear features de cambio (momentum adicional)
    feature_df['close_pct_change'] = feature_df['close'].pct_change() * 100

    # ===== FEATURES MEJORADOS (Iteración 15+16) =====
    if mejorado:
        # 1. Bollinger %B (posición relativa en bandas)
        if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
            bb_high = df['BBU_20_2.0']
            bb_low = df['BBL_20_2.0']
            bb_range = bb_high - bb_low
            bb_range = bb_range.replace(0, 1)  # Evitar división por cero
            feature_df['bollinger_pct_b'] = ((feature_df['close'] - bb_low) / bb_range) * 100
            feature_df['bollinger_pct_b'] = feature_df['bollinger_pct_b'].clip(0, 100)
        else:
            feature_df['bollinger_pct_b'] = 50  # Default si no existen bandas

        # 2. EMA Crossover (EMA_21 > EMA_50 = 1, else 0)
        feature_df['ema_21_50_cross'] = (feature_df['EMA_21'] > feature_df['EMA_50']).astype(int)

        # 3. RSI Z-Score (desviación normalizada)
        rsi_mean = feature_df['RSI_14'].rolling(window=14).mean()
        rsi_std = feature_df['RSI_14'].rolling(window=14).std()
        rsi_std = rsi_std.replace(0, 1)  # Evitar división por cero
        feature_df['rsi_zscore'] = (feature_df['RSI_14'] - rsi_mean) / rsi_std

        # 4. Volume Momentum (cambio % de volumen)
        feature_df['volume_momentum'] = feature_df['volume'].pct_change() * 100
        feature_df['volume_momentum'] = feature_df['volume_momentum'].fillna(0)

        # 5. Price to EMA200 ratio (como %)
        feature_df['price_to_ema200_pct'] = ((feature_df['close'] / feature_df['EMA_200']) - 1) * 100

        # ===== FEATURES DE TIMING Y MOMENTUM (Iteración 16) =====
        # 6. RSI Momentum (cambio de RSI en 5 velas)
        feature_df['rsi_momentum_5'] = feature_df['RSI_14'].diff(5).fillna(0)

        # 7. RSI Momentum (cambio de RSI en 10 velas)
        feature_df['rsi_momentum_10'] = feature_df['RSI_14'].diff(10).fillna(0)

        # 8. Stochastic Momentum (cambio de %K en 5 velas)
        feature_df['stoch_momentum_k'] = feature_df['STOCHk_14_3_3'].diff(5).fillna(0)

        # 9. Price Acceleration (cambio de velocidad)
        price_change = feature_df['close_pct_change'].copy()
        feature_df['price_acceleration'] = price_change.diff().fillna(0)

        # 10. EMA Distance Ratio (relación entre precio y EMAs)
        ema_50_200_dist = feature_df['EMA_50'] - feature_df['EMA_200']
        ema_50_200_dist = ema_50_200_dist.replace(0, 1)  # Evitar división
        price_to_ema50 = feature_df['close'] - feature_df['EMA_50']
        feature_df['ema_distance_ratio'] = (price_to_ema50 / ema_50_200_dist) * 100

        # Rellenar NaN en nuevos features
        feature_df['rsi_zscore'] = feature_df['rsi_zscore'].fillna(0)
        feature_df['ema_distance_ratio'] = feature_df['ema_distance_ratio'].fillna(0)

    return feature_df


def crear_target(
    df: pd.DataFrame,
    horizonte: int = 10,
    ganancia_min: float = 0.01
) -> pd.Series:
    """
    Crea target binario basado en máximo futuro.

    Target = 1 si en las próximas H velas:
        max_price >= current_price * (1 + G)
    Target = 0 en caso contrario

    Args:
        df: DataFrame con columna 'close' y 'high'
        horizonte: Número de velas futuras a considerar (default: 10)
        ganancia_min: Ganancia mínima requerida en % (default: 0.01 = 1%)

    Returns:
        Series con target binario (1, 0)
    """
    df = df.copy()

    # Calcular el máximo en las próximas H velas
    max_futuro = df['high'].rolling(window=horizonte).max().shift(-horizonte)

    # Precio de referencia (cierre actual)
    precio_ref = df['close']

    # Target: 1 si alcanza TP, 0 si no
    target = (max_futuro >= precio_ref * (1 + ganancia_min)).astype(int)

    return target


def crear_target_triple_barrier(
    df: pd.DataFrame,
    sl_pct: float = -0.02,
    tp_pct: float = 0.04,
    horizonte: int = 30
) -> pd.Series:
    """
    Crea target realista basado en Triple Barrier (TP, SL, Horizonte).

    ITERACIÓN 16: Target realista que considera:
    - Stop Loss (SL): sl_pct (default -2.0%)
    - Take Profit (TP): tp_pct (default +4.0%)
    - Horizonte: máx velas (default 30 = 7.5h en 15m)
    - Risk/Reward Ratio: 1:2 (1% riesgo por 2% ganancia)

    Target = 1 si precio alcanza TP ANTES de SL o Horizonte
    Target = 0 si precio alcanza SL ANTES de TP, o se agota Horizonte

    Ventaja: Modelo aprende movimientos NET positivos realistas

    Args:
        df: DataFrame con 'close', 'high', 'low'
        sl_pct: Stop Loss en % (default -0.02 = -2.0%)
        tp_pct: Take Profit en % (default 0.04 = +4.0%)
        horizonte: Número de velas futuras (default 30 = 7.5h)

    Returns:
        Series con target binario (1 si TP hit primero, 0 en otro caso)
    """
    df = df.copy()

    # Inicializar target como 0 (default)
    target = np.zeros(len(df), dtype=int)

    # Precios de referencia para cada vela
    entry_price = df['close'].values

    # Calcular SL y TP en términos de precio
    sl_price = entry_price * (1 + sl_pct)  # Precio más bajo (SL)
    tp_price = entry_price * (1 + tp_pct)  # Precio más alto (TP)

    # Para cada vela, buscar la PRIMERA ocurrencia de TP o SL en próximas H velas
    for i in range(len(df) - horizonte):
        # Próximas H velas
        future_high = df['high'].iloc[i:i+horizonte].values
        future_low = df['low'].iloc[i:i+horizonte].values

        # ¿Se alcanza TP? (precio sube)
        tp_hit = (future_high >= tp_price[i]).any()
        # ¿Se alcanza SL? (precio cae)
        sl_hit = (future_low <= sl_price[i]).any()

        # Lógica de Triple Barrier:
        # - Si TP se alcanza primero → target = 1
        # - Si SL se alcanza primero → target = 0
        # - Si ambos o ninguno en horizonte → target = 0 (conservative)

        if tp_hit and not sl_hit:
            # TP alcanzado SIN SL
            target[i] = 1
        elif tp_hit and sl_hit:
            # AMBOS alcanzados: buscar cuál primero
            # Encontrar índice del primer TP
            idx_tp = np.where(future_high >= tp_price[i])[0]
            # Encontrar índice del primer SL
            idx_sl = np.where(future_low <= sl_price[i])[0]

            if len(idx_tp) > 0 and len(idx_sl) > 0:
                if idx_tp[0] < idx_sl[0]:
                    target[i] = 1  # TP primero
                # else: target[i] = 0 (SL primero, ya inicializado)
        # else: SL hit sin TP → target = 0 (ya inicializado)

    return pd.Series(target, index=df.index)


def preparar_dataset_ml(
    df: pd.DataFrame,
    horizonte: int = 10,
    ganancia_min: float = 0.01,
    mejorado: bool = False
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Pipeline completo de preparación de dataset para ML.

    Flujo:
    1. Crear features técnicos (básicos o mejorados)
    2. Crear target binario
    3. Eliminar filas con NaN
    4. Retornar features y target listos para entrenamiento

    Args:
        df: DataFrame con indicadores técnicos
        horizonte: Número de velas para predecir (default: 10 = 2.5h en 15m)
        ganancia_min: Ganancia mínima requerida (default: 0.01 = 1%)
        mejorado: Si True, usa features mejorados (default: False para backwards compatibility)

    Returns:
        Tupla (X, y, feature_names)
        - X: DataFrame con features
        - y: Series con target binario
        - feature_names: Lista con nombres de features
    """
    # Crear features
    feature_df = crear_features(df, mejorado=mejorado)

    # Crear target
    target = crear_target(df, horizonte=horizonte, ganancia_min=ganancia_min)

    # Combinar en un DataFrame
    dataset = feature_df.copy()
    dataset['target'] = target

    # Eliminar NaN (primeras filas con indicadores incompletos + últimas H filas)
    dataset = dataset.dropna()

    # Separar X (features) e y (target)
    if mejorado:
        # Features mejorados (Iteración 15)
        feature_names = [
            'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3',
            'volume_norm', 'atr_pct', 'close_pct_change',
            'bollinger_pct_b', 'ema_21_50_cross', 'rsi_zscore',
            'volume_momentum', 'price_to_ema200_pct'
        ]
    else:
        # Features básicos (Iteración 14)
        feature_names = [
            'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3',
            'volume_norm', 'atr_pct', 'close_pct_change'
        ]

    X = dataset[feature_names].copy()
    y = dataset['target'].copy()

    # Validar
    assert len(X) == len(y), "Longitud de X e y no coinciden"
    assert X.isnull().sum().sum() == 0, f"Existen NaN en features: {X.isnull().sum()}"
    assert y.isnull().sum() == 0, "Existen NaN en target"

    return X, y, feature_names


def preparar_dataset_ml_triple_barrier(
    df: pd.DataFrame,
    sl_pct: float = -0.02,
    tp_pct: float = 0.04,
    horizonte: int = 30,
    mejorado: bool = True
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Pipeline ITERACIÓN 16: Preparación con Triple Barrier Target.

    Combina:
    - Features mejorados (IT15+16) con timing/momentum
    - Target realista: Triple Barrier (TP/SL/Horizonte)

    Args:
        df: DataFrame con indicadores técnicos
        sl_pct: Stop Loss en % (default -0.02 = -2%)
        tp_pct: Take Profit en % (default 0.04 = +4%)
        horizonte: Número de velas (default 30 = 7.5h)
        mejorado: Usar features mejorados (siempre True para IT16)

    Returns:
        Tupla (X, y, feature_names) con features IT16 y target triple barrier
    """
    # Crear features mejorados (siempre para IT16)
    feature_df = crear_features(df, mejorado=True)

    # Crear target con Triple Barrier (NUEVO en IT16)
    target = crear_target_triple_barrier(df, sl_pct=sl_pct, tp_pct=tp_pct, horizonte=horizonte)

    # Combinar
    dataset = feature_df.copy()
    dataset['target'] = target

    # Eliminar NaN
    dataset = dataset.dropna()

    # Features IT16 incluyen momentum/timing (18 features)
    feature_names = [
        # Básicos (8)
        'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3',
        'volume_norm', 'atr_pct', 'close_pct_change',
        # IT15 mejorados (5)
        'bollinger_pct_b', 'ema_21_50_cross', 'rsi_zscore',
        'volume_momentum', 'price_to_ema200_pct',
        # IT16 timing/momentum (5)
        'rsi_momentum_5', 'rsi_momentum_10', 'stoch_momentum_k',
        'price_acceleration', 'ema_distance_ratio'
    ]

    X = dataset[feature_names].copy()
    y = dataset['target'].copy()

    # Validar
    assert len(X) == len(y), "Longitud de X e y no coinciden"
    assert X.isnull().sum().sum() == 0, f"Existen NaN en features: {X.isnull().sum()}"
    assert y.isnull().sum() == 0, "Existen NaN en target"

    return X, y, feature_names


def calcular_stats_target(y: pd.Series) -> Dict[str, float]:
    """
    Calcula estadísticas del target.

    Args:
        y: Series con target binario

    Returns:
        Dict con estadísticas
    """
    total = len(y)
    positivos = (y == 1).sum()
    negativos = (y == 0).sum()

    return {
        'total': total,
        'positivos': positivos,
        'negativos': negativos,
        'pct_positivos': (positivos / total * 100) if total > 0 else 0,
        'pct_negativos': (negativos / total * 100) if total > 0 else 0,
        'balance_ratio': positivos / negativos if negativos > 0 else np.inf
    }


if __name__ == "__main__":
    """Test del módulo"""
    print("✓ Módulo feature_engineer.py cargado correctamente")
    print("\nFunciones disponibles:")
    print("  - crear_features(df)")
    print("  - crear_target(df, horizonte=10, ganancia_min=0.01)")
    print("  - preparar_dataset_ml(df, horizonte=10, ganancia_min=0.01)")
    print("  - calcular_stats_target(y)")
