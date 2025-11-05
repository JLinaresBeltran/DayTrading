"""
Módulo para cargar y usar modelos LSTM para predicción de precios.
Nota: El modelo debe estar pre-entrenado y guardado.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def load_lstm_model(model_path='models/lstm_model.h5'):
    """
    Carga un modelo LSTM pre-entrenado.

    Args:
        model_path: Ruta al modelo guardado

    Returns:
        Modelo LSTM cargado
    """
    try:
        from tensorflow.keras.models import load_model
        model = load_model(model_path)
        print(f"✓ Modelo LSTM cargado desde {model_path}")
        return model
    except Exception as e:
        print(f"⚠ No se pudo cargar modelo LSTM: {e}")
        return None


def get_lstm_prediction(df, model, lookback=100):
    """
    Obtiene una predicción del modelo LSTM.

    Args:
        df: DataFrame con datos OHLCV e indicadores
        model: Modelo LSTM cargado
        lookback: Número de velas para usar como entrada

    Returns:
        Señal (-1, 0, 1) basada en predicción
    """
    if model is None:
        return 0  # Neutral si no hay modelo

    try:
        # Seleccionar features
        features = ['close', 'volume', 'RSI_14', 'MACD_12_26_9', 'ATR_14']
        data = df[features].tail(lookback).copy()

        # Normalizar
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)

        # Reshape para LSTM: (1, lookback, n_features)
        X = scaled_data.reshape(1, lookback, len(features))

        # Predecir
        prediction = model.predict(X, verbose=0)[0][0]

        # Convertir predicción a señal
        current_price = df['close'].iloc[-1]
        price_change_threshold = 0.005  # 0.5%

        if prediction > current_price * (1 + price_change_threshold):
            return 1  # Predicción alcista -> COMPRA
        elif prediction < current_price * (1 - price_change_threshold):
            return -1  # Predicción bajista -> VENTA
        else:
            return 0  # Neutral

    except Exception as e:
        print(f"Error en predicción LSTM: {e}")
        return 0


if __name__ == "__main__":
    print("=== Test de LSTM Model ===\n")
    print("Nota: Este módulo requiere un modelo pre-entrenado en models/lstm_model.h5")
    print("Para entrenar un modelo, consulta la documentación del proyecto.\n")
    print("✓ Módulo cargado correctamente")
