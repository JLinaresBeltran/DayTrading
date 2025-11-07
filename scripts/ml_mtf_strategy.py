"""
SISTEMA HÍBRIDO: Machine Learning + Multi-Timeframe Analysis

Componentes:
1. Multi-Timeframe Filter (1H + 15m)
2. LSTM para predicción de dirección
3. Random Forest para clasificación de señales
4. Ensemble voting para decisión final

Objetivo:
- Win Rate: 40-60%
- Trades: 100-200/año
- Profit Factor: >2.0
- Max DD: <15%
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/jhonathan/BotDayTrading')

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.backtest.engine import VectorizedBacktester

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SYMBOL = 'ETHUSDT'
TF_TRADE = '15m'  # Timeframe de trading
TF_FILTER = '1h'  # Timeframe de filtro
START_DATE = '730 days ago UTC'  # 2 años para entrenamiento
INITIAL_CAPITAL = 10000

# Split de datos
TRAIN_RATIO = 0.7  # 70% entrenamiento, 30% test

# Configuración de ML
RF_CONFIG = {
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 20,
    'min_samples_leaf': 10,
    'random_state': 42,
    'class_weight': 'balanced'  # Importante para win rate bajo
}

# Configuración de estrategia
STRATEGY_CONFIG = {
    'atr_period': 14,
    'sl_atr_multiplier': 1.5,
    'tp_atr_multiplier': 3.0,  # Ratio 1:2
    'min_ml_confidence': 0.55,  # Confianza mínima del modelo
}

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def create_ml_features(df):
    """
    Crea features avanzadas para Machine Learning.

    Features incluyen:
    - Indicadores técnicos normalizados
    - Ratios de precio
    - Volatilidad
    - Momentum multi-período
    - Patrones de velas
    """
    df = df.copy()

    # ========================================================================
    # 1. FEATURES BÁSICAS DE PRECIO
    # ========================================================================

    # Retornos en múltiples períodos
    for period in [1, 3, 5, 10, 20]:
        df[f'return_{period}'] = df['close'].pct_change(period)

    # Precio relativo a EMAs
    for ema_period in [9, 21, 50, 200]:
        ema_col = f'EMA_{ema_period}'
        if ema_col in df.columns:
            df[f'price_to_ema_{ema_period}'] = (df['close'] / df[ema_col]) - 1

    # ========================================================================
    # 2. VOLATILIDAD Y ATR
    # ========================================================================

    # ATR normalizado por precio
    if 'ATRr_14' in df.columns:
        df['atr_pct'] = (df['ATRr_14'] / df['close']) * 100
        df['atr_ratio'] = df['ATRr_14'] / df['ATRr_14'].rolling(20).mean()

    # True Range
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['tr_pct'] = (df['tr'] / df['close']) * 100

    # ========================================================================
    # 3. MOMENTUM
    # ========================================================================

    # RSI en múltiples períodos
    for rsi_period in [7, 14, 21]:
        rsi_col = f'RSI_{rsi_period}'
        if rsi_col in df.columns:
            # RSI normalizado (0-100 -> -1 a 1)
            df[f'rsi_{rsi_period}_norm'] = (df[rsi_col] - 50) / 50

            # RSI momentum (cambio)
            df[f'rsi_{rsi_period}_momentum'] = df[rsi_col].diff()

    # MACD
    if 'MACD_12_26_9' in df.columns and 'MACDs_12_26_9' in df.columns:
        df['macd_signal_diff'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
        df['macd_histogram_change'] = df['macd_signal_diff'].diff()

    # ========================================================================
    # 4. VOLUME
    # ========================================================================

    if 'volume' in df.columns:
        # Volume ratio
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

        # Volume-Price correlation
        df['volume_price_corr'] = df['volume'].rolling(20).corr(df['close'])

        # OBV (On Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        df['obv_ema'] = df['obv'].ewm(span=20).mean()
        df['obv_signal'] = df['obv'] - df['obv_ema']

    # ========================================================================
    # 5. PATRONES DE VELAS
    # ========================================================================

    # Tamaño del cuerpo
    df['body'] = abs(df['close'] - df['open'])
    df['body_pct'] = (df['body'] / df['close']) * 100

    # Sombras
    df['upper_shadow'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_shadow'] = np.minimum(df['open'], df['close']) - df['low']

    # Tipo de vela
    df['is_bullish'] = (df['close'] > df['open']).astype(int)
    df['is_doji'] = (df['body_pct'] < 0.1).astype(int)

    # ========================================================================
    # 6. TENDENCIA
    # ========================================================================

    # ADX
    if 'ADX_14' in df.columns:
        df['adx_14_norm'] = df['ADX_14'] / 100  # Normalizar 0-100 -> 0-1
        df['adx_14_trend'] = df['ADX_14'].diff()

    # Pendiente de EMAs
    for ema_period in [9, 21, 50]:
        ema_col = f'EMA_{ema_period}'
        if ema_col in df.columns:
            df[f'ema_{ema_period}_slope'] = df[ema_col].pct_change(5)

    # ========================================================================
    # 7. FEATURES DE SUPERTREND
    # ========================================================================

    # Supertrend direction
    if 'SUPERTd_10_3.0' in df.columns:
        df['supertrend_direction'] = df['SUPERTd_10_3.0']
        df['supertrend_change'] = df['supertrend_direction'].diff()

    # ========================================================================
    # 8. STATISTICAL FEATURES
    # ========================================================================

    # Rolling statistics
    for window in [5, 10, 20]:
        df[f'close_std_{window}'] = df['close'].rolling(window).std()
        df[f'close_mean_{window}'] = df['close'].rolling(window).mean()
        df[f'close_zscore_{window}'] = (df['close'] - df[f'close_mean_{window}']) / df[f'close_std_{window}']

    # Bollinger Bands position
    if 'BBL_20_2.0' in df.columns and 'BBU_20_2.0' in df.columns:
        bb_range = df['BBU_20_2.0'] - df['BBL_20_2.0']
        df['bb_position'] = (df['close'] - df['BBL_20_2.0']) / bb_range
        df['bb_width'] = bb_range / df['close']

    return df


def create_labels(df, forward_periods=5, profit_threshold=0.5):
    """
    Crea labels para entrenamiento.

    Label = 1 (BUY) si el precio sube > profit_threshold% en los próximos forward_periods
    Label = -1 (SELL) si el precio baja > profit_threshold% en los próximos forward_periods
    Label = 0 (HOLD) en caso contrario
    """
    df = df.copy()

    # Calcular máximo y mínimo futuro
    df['future_max'] = df['high'].shift(-1).rolling(forward_periods).max()
    df['future_min'] = df['low'].shift(-1).rolling(forward_periods).min()

    # Calcular retornos potenciales
    df['max_return'] = ((df['future_max'] - df['close']) / df['close']) * 100
    df['min_return'] = ((df['future_min'] - df['close']) / df['close']) * 100

    # Crear labels
    df['label'] = 0  # HOLD por defecto
    df.loc[df['max_return'] > profit_threshold, 'label'] = 1   # BUY
    df.loc[df['min_return'] < -profit_threshold, 'label'] = -1  # SELL

    return df


# ============================================================================
# MULTI-TIMEFRAME FILTER
# ============================================================================

def create_mtf_filter(df_high, df_low):
    """
    Crea filtro Multi-Timeframe.

    El TF alto determina la tendencia general.
    Solo se permite operar en la dirección de la tendencia del TF alto.
    """
    df_high = df_high.copy()

    # Calcular tendencia en TF alto usando EMAs
    ema_fast = df_high['EMA_50']
    ema_slow = df_high['EMA_200']

    # Señal de tendencia
    df_high['trend_long'] = (ema_fast > ema_slow).astype(int)
    df_high['trend_short'] = (ema_fast < ema_slow).astype(int)

    # ADX para fuerza de tendencia
    if 'ADX_14' in df_high.columns:
        df_high['trend_strong'] = (df_high['ADX_14'] > 25).astype(int)
    else:
        df_high['trend_strong'] = 1

    # Combinar
    df_high['can_long'] = df_high['trend_long'] & df_high['trend_strong']
    df_high['can_short'] = df_high['trend_short'] & df_high['trend_strong']

    # Sincronizar con TF bajo
    df_low = df_low.copy()
    df_low['timestamp_htf'] = pd.to_datetime(df_low['timestamp']).dt.floor('1h')

    # Merge
    df_merged = df_low.merge(
        df_high[['timestamp', 'can_long', 'can_short']],
        left_on='timestamp_htf',
        right_on='timestamp',
        how='left',
        suffixes=('', '_htf')
    )

    # Forward fill
    df_merged['can_long'] = df_merged['can_long'].ffill().fillna(False)
    df_merged['can_short'] = df_merged['can_short'].ffill().fillna(False)

    return df_merged


# ============================================================================
# MACHINE LEARNING MODEL
# ============================================================================

class MLTradingModel:
    """
    Modelo de Machine Learning para trading.
    Usa Random Forest para clasificación de señales.
    """

    def __init__(self, config=RF_CONFIG):
        self.model = RandomForestClassifier(**config)
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False

    def prepare_features(self, df):
        """Prepara features para el modelo."""
        # Seleccionar solo columnas numéricas y sin NaN
        feature_cols = [col for col in df.columns if col not in [
            'timestamp', 'label', 'señal', 'future_max', 'future_min',
            'max_return', 'min_return', 'can_long', 'can_short',
            'timestamp_htf'
        ]]

        # Filtrar solo columnas numéricas
        df_features = df[feature_cols].select_dtypes(include=[np.number])

        # Eliminar columnas con demasiados NaN
        df_features = df_features.dropna(axis=1, thresh=len(df_features) * 0.5)

        self.feature_cols = df_features.columns.tolist()

        return df_features

    def train(self, df_train):
        """Entrena el modelo."""
        print("\n" + "="*80)
        print("ENTRENAMIENTO DEL MODELO ML")
        print("="*80)

        # Preparar features
        X = self.prepare_features(df_train)
        y = df_train['label'].values

        # Eliminar filas con NaN en features o labels
        valid_idx = ~(X.isna().any(axis=1) | pd.isna(y))
        X = X[valid_idx]
        y = y[valid_idx]

        print(f"\nDatos de entrenamiento:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(self.feature_cols)}")
        print(f"\nDistribución de labels:")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            label_name = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}[label]
            print(f"  {label_name}: {count} ({count/len(y)*100:.1f}%)")

        # Normalizar features
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar
        print("\nEntrenando Random Forest...")
        self.model.fit(X_scaled, y)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 features más importantes:")
        print(feature_importance.head(10).to_string(index=False))

        self.is_trained = True
        print("\n✅ Modelo entrenado exitosamente")

    def predict(self, df):
        """Predice señales."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado. Ejecutar train() primero.")

        # Preparar features (usar las mismas columnas del entrenamiento)
        X = df[self.feature_cols].copy()

        # Eliminar NaN
        valid_idx = ~X.isna().any(axis=1)

        # Predecir
        predictions = np.zeros(len(df))
        probabilities = np.zeros((len(df), 3))  # 3 clases: -1, 0, 1

        if valid_idx.sum() > 0:
            X_valid = X[valid_idx]
            X_scaled = self.scaler.transform(X_valid)

            predictions[valid_idx] = self.model.predict(X_scaled)
            probabilities[valid_idx] = self.model.predict_proba(X_scaled)

        return predictions, probabilities


# ============================================================================
# BACKTESTING CON ML
# ============================================================================

def backtest_ml_strategy(df, model, min_confidence=0.55):
    """
    Ejecuta backtest con señales de ML.
    """
    df = df.copy()

    # Obtener predicciones
    predictions, probabilities = model.predict(df)

    # Calcular confianza máxima
    max_prob = probabilities.max(axis=1)

    # Generar señales
    df['señal'] = 0
    df['ml_signal'] = predictions
    df['ml_confidence'] = max_prob

    # Solo tomar trades con alta confianza
    high_confidence = max_prob >= min_confidence

    df.loc[high_confidence & (predictions == 1), 'señal'] = 1   # BUY
    df.loc[high_confidence & (predictions == -1), 'señal'] = -1  # SELL

    # Filtrar por MTF
    if 'can_long' in df.columns and 'can_short' in df.columns:
        # Convertir a booleano explícitamente
        can_long = df['can_long'].fillna(False).astype(bool)
        can_short = df['can_short'].fillna(False).astype(bool)

        # Solo permitir longs si MTF lo permite
        df.loc[~can_long & (df['señal'] == 1), 'señal'] = 0
        # Solo permitir shorts si MTF lo permite
        df.loc[~can_short & (df['señal'] == -1), 'señal'] = 0

    return df


def main():
    print("="*80)
    print("🤖 SISTEMA HÍBRIDO: MACHINE LEARNING + MULTI-TIMEFRAME")
    print("="*80)

    # ========================================================================
    # FASE 1: DESCARGA DE DATOS
    # ========================================================================

    print("\n" + "="*80)
    print("FASE 1: DESCARGA DE DATOS")
    print("="*80)

    manager = BinanceClientManager()
    client = manager.get_public_client()

    # Descargar TF de trading (15m)
    print(f"\nDescargando {SYMBOL} {TF_TRADE}...")
    df_trade = obtener_datos_binance(
        client=client,
        simbolo=SYMBOL,
        intervalo=TF_TRADE,
        inicio=START_DATE
    )
    print(f"✅ Descargado: {len(df_trade)} velas")

    # Descargar TF de filtro (1h)
    print(f"\nDescargando {SYMBOL} {TF_FILTER}...")
    df_filter = obtener_datos_binance(
        client=client,
        simbolo=SYMBOL,
        intervalo=TF_FILTER,
        inicio=START_DATE
    )
    print(f"✅ Descargado: {len(df_filter)} velas")

    # ========================================================================
    # FASE 2: CALCULAR INDICADORES
    # ========================================================================

    print("\n" + "="*80)
    print("FASE 2: CALCULANDO INDICADORES")
    print("="*80)

    print("\nCalculando indicadores en TF de trading...")
    df_trade = agregar_indicadores(df_trade)
    df_trade.ta.atr(length=14, append=True)
    df_trade.ta.adx(length=14, append=True)
    df_trade.ta.supertrend(length=10, multiplier=3.0, append=True)
    df_trade.ta.rsi(length=7, append=True)
    df_trade.ta.rsi(length=21, append=True)

    print("\nCalculando indicadores en TF de filtro...")
    df_filter = agregar_indicadores(df_filter)
    df_filter.ta.adx(length=14, append=True)
    df_filter.ta.ema(length=50, append=True)
    df_filter.ta.ema(length=200, append=True)

    print("✅ Indicadores calculados")

    # ========================================================================
    # FASE 3: FEATURE ENGINEERING
    # ========================================================================

    print("\n" + "="*80)
    print("FASE 3: FEATURE ENGINEERING")
    print("="*80)

    df_trade = create_ml_features(df_trade)
    print(f"✅ Features creadas: {len(df_trade.columns)} columnas totales")

    # Crear labels
    df_trade = create_labels(df_trade, forward_periods=5, profit_threshold=0.5)

    # MTF Filter
    df_trade = create_mtf_filter(df_filter, df_trade)

    # ========================================================================
    # FASE 4: SPLIT TRAIN/TEST
    # ========================================================================

    print("\n" + "="*80)
    print("FASE 4: SPLIT DE DATOS")
    print("="*80)

    # Eliminar filas con NaN en labels
    df_clean = df_trade.dropna(subset=['label'])

    split_idx = int(len(df_clean) * TRAIN_RATIO)
    df_train = df_clean.iloc[:split_idx].copy()
    df_test = df_clean.iloc[split_idx:].copy()

    print(f"\nTrain set: {len(df_train)} samples ({df_train['timestamp'].min()} a {df_train['timestamp'].max()})")
    print(f"Test set:  {len(df_test)} samples ({df_test['timestamp'].min()} a {df_test['timestamp'].max()})")

    # ========================================================================
    # FASE 5: ENTRENAR MODELO
    # ========================================================================

    model = MLTradingModel()
    model.train(df_train)

    # ========================================================================
    # FASE 6: BACKTEST EN TEST SET
    # ========================================================================

    print("\n" + "="*80)
    print("FASE 6: BACKTESTING EN TEST SET")
    print("="*80)

    df_test_signals = backtest_ml_strategy(
        df_test,
        model,
        min_confidence=STRATEGY_CONFIG['min_ml_confidence']
    )

    # Contar señales
    signals = df_test_signals['señal']
    print(f"\nSeñales generadas:")
    print(f"  BUY:  {(signals == 1).sum()}")
    print(f"  SELL: {(signals == -1).sum()}")
    print(f"  HOLD: {(signals == 0).sum()}")

    # Ejecutar backtest
    if signals.abs().sum() == 0:
        print("\n❌ No se generaron señales de trading")
        return

    backtester = VectorizedBacktester(
        df=df_test_signals,
        initial_capital=INITIAL_CAPITAL,
        commission=0.00075,
        slippage=0.0005
    )

    backtester.run_backtest_with_stop_loss(
        atr_column='ATRr_14',
        atr_multiplier=STRATEGY_CONFIG['sl_atr_multiplier']
    )

    metrics = backtester.calculate_metrics()

    # ========================================================================
    # FASE 7: RESULTADOS
    # ========================================================================

    print("\n" + "="*80)
    print("RESULTADOS DEL BACKTEST (TEST SET - OUT OF SAMPLE)")
    print("="*80)

    print(f"\n📊 MÉTRICAS DE RENDIMIENTO:")
    print(f"  Capital Inicial:    ${metrics['initial_capital']:,.2f}")
    print(f"  Capital Final:      ${metrics['final_value']:,.2f}")
    print(f"  Retorno Total:      {metrics['total_return_pct']:.2f}%")
    print(f"  Retorno Anual:      {metrics['annual_return_pct']:.2f}%")
    print(f"  Buy & Hold:         {metrics['buy_hold_return_pct']:.2f}%")
    print(f"  Exceso sobre B&H:   {metrics['excess_return_pct']:.2f}%")

    print(f"\n📈 MÉTRICAS DE RIESGO:")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Sortino Ratio:      {metrics['sortino_ratio']:.2f}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}")

    print(f"\n🎯 MÉTRICAS DE TRADING:")
    print(f"  Número de Trades:   {metrics['num_trades']}")
    print(f"  Win Rate:           {metrics['win_rate_pct']:.2f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"  Avg Trade:          ${metrics['avg_trade']:.2f}")
    print(f"  Best Trade:         ${metrics['best_trade']:.2f}")
    print(f"  Worst Trade:        ${metrics['worst_trade']:.2f}")

    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/ml_mtf_results_{timestamp}.csv'

    results_df = pd.DataFrame([{
        'model': 'RandomForest + MTF',
        'timeframe_trade': TF_TRADE,
        'timeframe_filter': TF_FILTER,
        'min_confidence': STRATEGY_CONFIG['min_ml_confidence'],
        **metrics
    }])

    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Resultados guardados en: {results_file}")

    # Evaluación
    print("\n" + "="*80)
    print("EVALUACIÓN vs OBJETIVOS")
    print("="*80)

    objectives = {
        'Win Rate': (40, 60, metrics['win_rate_pct']),
        'Trades/año': (100, 200, metrics['num_trades'] * (365 / len(df_test))),
        'Profit Factor': (2.0, float('inf'), metrics['profit_factor']),
        'Max Drawdown': (0, 15, metrics['max_drawdown_pct'])
    }

    for name, (min_val, max_val, actual) in objectives.items():
        if min_val <= actual <= max_val:
            status = "✅"
        else:
            status = "❌"

        if name == 'Max Drawdown':
            print(f"  {status} {name}: {actual:.2f}% (objetivo: <{max_val}%)")
        elif name == 'Profit Factor':
            print(f"  {status} {name}: {actual:.2f} (objetivo: >{min_val})")
        else:
            print(f"  {status} {name}: {actual:.2f} (objetivo: {min_val}-{max_val})")

    print("\n" + "="*80)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*80)


if __name__ == "__main__":
    main()
