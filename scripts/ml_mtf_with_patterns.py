"""
SISTEMA AVANZADO: ML + MTF + PATRONES DE VELAS + PATRONES CHARTISTAS

Componentes:
1. Multi-Timeframe Filter (1H + 15m)
2. Random Forest con features avanzadas
3. Patrones de velas japonesas (Doji, Hammer, Engulfing, etc.)
4. Patrones chartistas (Soportes, Resistencias, Estructura de mercado)
5. Ensemble voting para decisión final

Objetivo mejorado:
- Win Rate: 45-65% (con patrones)
- Trades: 80-150/año
- Profit Factor: >2.5
- Max DD: <12%
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

sys.path.append('/Users/jhonathan/BotDayTrading')

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.indicators.candlestick_patterns import add_candlestick_patterns
from src.backtest.engine import VectorizedBacktester

# Importar funciones de feature engineering del script anterior
import scripts.ml_mtf_strategy as ml_base

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

SYMBOL = 'ETHUSDT'
TF_TRADE = '15m'
TF_FILTER = '1h'
START_DATE = '730 days ago UTC'
INITIAL_CAPITAL = 10000

TRAIN_RATIO = 0.7

# Configuración de ML mejorada
RF_CONFIG = {
    'n_estimators': 200,  # Más árboles para capturar patrones
    'max_depth': 15,      # Mayor profundidad
    'min_samples_split': 15,
    'min_samples_leaf': 8,
    'random_state': 42,
    'class_weight': 'balanced',
    'max_features': 'sqrt',  # Usar sqrt de features en cada split
}

STRATEGY_CONFIG = {
    'atr_period': 14,
    'sl_atr_multiplier': 1.5,
    'tp_atr_multiplier': 3.5,  # Ratio 1:2.3
    'min_ml_confidence': 0.60,  # Mayor exigencia
    'require_pattern_confirmation': True,  # Requerir confirmación de patrón
    'min_pattern_score': 1,  # Score mínimo de patrones
}

# ============================================================================
# FEATURE ENGINEERING MEJORADO
# ============================================================================

def create_enhanced_features(df):
    """
    Crea features mejoradas incluyendo patrones.
    """
    df = df.copy()

    # 1. Features técnicas básicas (del script anterior)
    df = ml_base.create_ml_features(df)

    # 2. Patrones de velas y chartistas
    df = add_candlestick_patterns(df)

    # 3. Features adicionales de patrones

    # Confirmación de reversión en soporte/resistencia
    df['bounce_from_support'] = (
        df['near_support'] &
        (df['bullish_pattern_count'] > 0) &
        (df['close'] > df['open'])  # Vela alcista
    ).astype(int)

    df['rejection_from_resistance'] = (
        df['near_resistance'] &
        (df['bearish_pattern_count'] > 0) &
        (df['close'] < df['open'])  # Vela bajista
    ).astype(int)

    # Momentum de estructura de mercado
    df['market_structure_momentum'] = df['market_structure'].rolling(5).mean()

    # Strength index (combinación de ADX + patrones + estructura)
    if 'ADX_14' in df.columns:
        df['trend_strength_index'] = (
            (df['ADX_14'] / 100) * 0.5 +  # 50% ADX
            (df['pattern_score'].clip(-2, 2) / 4 + 0.5) * 0.3 +  # 30% patrones
            (df['market_structure'] + 1) / 2 * 0.2  # 20% estructura
        )

    # Señal combinada de múltiples patrones
    df['strong_bullish_signal'] = (
        (df['bullish_pattern_count'] >= 2) |
        (df['pattern_morning_star'] == 1) |
        (df['pattern_three_white_soldiers'] == 1) |
        (df['bounce_from_support'] == 1)
    ).astype(int)

    df['strong_bearish_signal'] = (
        (df['bearish_pattern_count'] >= 2) |
        (df['pattern_evening_star'] == 1) |
        (df['pattern_three_black_crows'] == 1) |
        (df['rejection_from_resistance'] == 1)
    ).astype(int)

    return df


# ============================================================================
# MODELO ML MEJORADO CON ENSEMBLE
# ============================================================================

class EnhancedMLModel:
    """
    Modelo ensemble con Random Forest + Gradient Boosting.
    """

    def __init__(self):
        self.rf_model = RandomForestClassifier(**RF_CONFIG)
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.is_trained = False

    def prepare_features(self, df):
        """Prepara features."""
        # Excluir columnas no deseadas
        exclude_cols = [
            'timestamp', 'label', 'señal', 'future_max', 'future_min',
            'max_return', 'min_return', 'can_long', 'can_short',
            'timestamp_htf', 'timestamp_htf'
        ]

        # Añadir columnas de soporte/resistencia a excluir
        exclude_cols.extend([col for col in df.columns if col.startswith('support_')])
        exclude_cols.extend([col for col in df.columns if col.startswith('resistance_')])

        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Filtrar solo numéricas
        df_features = df[feature_cols].select_dtypes(include=[np.number])

        # Eliminar columnas con muchos NaN
        df_features = df_features.dropna(axis=1, thresh=len(df_features) * 0.5)

        self.feature_cols = df_features.columns.tolist()

        return df_features

    def train(self, df_train):
        """Entrena ambos modelos."""
        print("\n" + "="*80)
        print("ENTRENAMIENTO DEL MODELO ML MEJORADO (ENSEMBLE)")
        print("="*80)

        # Preparar datos
        X = self.prepare_features(df_train)
        y = df_train['label'].values

        # Eliminar NaN
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

        # Normalizar
        X_scaled = self.scaler.fit_transform(X)

        # Entrenar Random Forest
        print("\n[1/2] Entrenando Random Forest...")
        self.rf_model.fit(X_scaled, y)

        # Entrenar Gradient Boosting
        print("[2/2] Entrenando Gradient Boosting...")
        self.gb_model.fit(X_scaled, y)

        # Feature importance (RF)
        feature_importance = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': self.rf_model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\n🏆 Top 15 features más importantes:")
        print(feature_importance.head(15).to_string(index=False))

        # Mostrar importancia de patrones
        pattern_features = feature_importance[
            feature_importance['feature'].str.contains('pattern|structure|support|resistance|bounce|rejection', na=False)
        ]
        if len(pattern_features) > 0:
            print("\n📊 Importancia de features de patrones:")
            print(pattern_features.head(10).to_string(index=False))

        self.is_trained = True
        print("\n✅ Modelo ensemble entrenado exitosamente")

    def predict(self, df):
        """Predice usando ensemble (promedio de ambos modelos)."""
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        X = df[self.feature_cols].copy()
        valid_idx = ~X.isna().any(axis=1)

        predictions = np.zeros(len(df))
        probabilities = np.zeros((len(df), 3))

        if valid_idx.sum() > 0:
            X_valid = X[valid_idx]
            X_scaled = self.scaler.transform(X_valid)

            # Predicciones de RF
            rf_pred = self.rf_model.predict(X_scaled)
            rf_proba = self.rf_model.predict_proba(X_scaled)

            # Predicciones de GB
            gb_pred = self.gb_model.predict(X_scaled)
            gb_proba = self.gb_model.predict_proba(X_scaled)

            # Ensemble: Promedio de probabilidades
            avg_proba = (rf_proba + gb_proba) / 2

            # Predicción final basada en promedio
            final_pred = np.argmax(avg_proba, axis=1)
            # Convertir índices a labels (-1, 0, 1)
            label_map = {0: -1, 1: 0, 2: 1}  # Ajustar según orden de clases
            classes = self.rf_model.classes_
            label_map = {i: label for i, label in enumerate(classes)}

            predictions[valid_idx] = [label_map[p] for p in final_pred]
            probabilities[valid_idx] = avg_proba

        return predictions, probabilities


# ============================================================================
# BACKTESTING CON CONFIRMACIÓN DE PATRONES
# ============================================================================

def backtest_with_pattern_confirmation(df, model, config):
    """
    Backtest con confirmación de patrones.
    """
    df = df.copy()

    # Obtener predicciones ML
    predictions, probabilities = model.predict(df)
    max_prob = probabilities.max(axis=1)

    # Generar señales base
    df['señal'] = 0
    df['ml_signal'] = predictions
    df['ml_confidence'] = max_prob

    # Filtro de confianza
    high_confidence = max_prob >= config['min_ml_confidence']

    # LONGS
    long_signals = (
        high_confidence &
        (predictions == 1)
    )

    # SHORTS
    short_signals = (
        high_confidence &
        (predictions == -1)
    )

    # Confirmación de patrones (si está activada)
    if config.get('require_pattern_confirmation', False):
        min_score = config.get('min_pattern_score', 1)

        # Reforzar longs con patrones alcistas
        long_signals = long_signals & (
            (df['pattern_score'] >= min_score) |
            (df['strong_bullish_signal'] == 1) |
            (df['market_structure'] == 1)
        )

        # Reforzar shorts con patrones bajistas
        short_signals = short_signals & (
            (df['pattern_score'] <= -min_score) |
            (df['strong_bearish_signal'] == 1) |
            (df['market_structure'] == -1)
        )

    # Asignar señales
    df.loc[long_signals, 'señal'] = 1
    df.loc[short_signals, 'señal'] = -1

    # Filtrar por MTF
    if 'can_long' in df.columns and 'can_short' in df.columns:
        can_long = df['can_long'].fillna(False).astype(bool)
        can_short = df['can_short'].fillna(False).astype(bool)

        df.loc[~can_long & (df['señal'] == 1), 'señal'] = 0
        df.loc[~can_short & (df['señal'] == -1), 'señal'] = 0

    return df


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("🚀 SISTEMA AVANZADO: ML + MTF + PATRONES DE VELAS")
    print("="*80)

    # Descargar datos
    print("\n📥 DESCARGANDO DATOS...")
    manager = BinanceClientManager()
    client = manager.get_public_client()

    df_trade = obtener_datos_binance(client, SYMBOL, TF_TRADE, START_DATE)
    df_filter = obtener_datos_binance(client, SYMBOL, TF_FILTER, START_DATE)

    print(f"✅ Descargado: {len(df_trade)} velas ({TF_TRADE}), {len(df_filter)} velas ({TF_FILTER})")

    # Calcular indicadores
    print("\n⚙️ CALCULANDO INDICADORES...")
    df_trade = agregar_indicadores(df_trade)
    df_trade.ta.atr(length=14, append=True)
    df_trade.ta.adx(length=14, append=True)
    df_trade.ta.supertrend(length=10, multiplier=3.0, append=True)
    df_trade.ta.rsi(length=7, append=True)
    df_trade.ta.rsi(length=21, append=True)

    df_filter = agregar_indicadores(df_filter)
    df_filter.ta.adx(length=14, append=True)
    df_filter.ta.ema(length=50, append=True)
    df_filter.ta.ema(length=200, append=True)

    # Feature engineering
    print("\n🔧 FEATURE ENGINEERING CON PATRONES...")
    df_trade = create_enhanced_features(df_trade)
    print(f"✅ Features totales: {len(df_trade.columns)}")

    # Labels
    df_trade = ml_base.create_labels(df_trade, forward_periods=5, profit_threshold=0.5)

    # MTF
    df_trade = ml_base.create_mtf_filter(df_filter, df_trade)

    # Split
    print("\n✂️  SPLIT DE DATOS...")
    df_clean = df_trade.dropna(subset=['label'])
    split_idx = int(len(df_clean) * TRAIN_RATIO)
    df_train = df_clean.iloc[:split_idx].copy()
    df_test = df_clean.iloc[split_idx:].copy()

    print(f"Train: {len(df_train)} samples")
    print(f"Test:  {len(df_test)} samples")

    # Entrenar
    model = EnhancedMLModel()
    model.train(df_train)

    # Backtest
    print("\n" + "="*80)
    print("📊 BACKTESTING CON CONFIRMACIÓN DE PATRONES")
    print("="*80)

    df_test_signals = backtest_with_pattern_confirmation(df_test, model, STRATEGY_CONFIG)

    signals = df_test_signals['señal']
    print(f"\nSeñales generadas:")
    print(f"  BUY:  {(signals == 1).sum()}")
    print(f"  SELL: {(signals == -1).sum()}")
    print(f"  HOLD: {(signals == 0).sum()}")

    if signals.abs().sum() == 0:
        print("\n❌ No se generaron señales")
        return

    # Ejecutar backtest
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

    # Resultados
    print("\n" + "="*80)
    print("🏆 RESULTADOS FINALES (OUT OF SAMPLE)")
    print("="*80)

    print(f"\n📊 RENDIMIENTO:")
    print(f"  Capital Final:      ${metrics['final_value']:,.2f}")
    print(f"  Retorno Total:      {metrics['total_return_pct']:.2f}%")
    print(f"  Retorno Anual:      {metrics['annual_return_pct']:.2f}%")
    print(f"  Buy & Hold:         {metrics['buy_hold_return_pct']:.2f}%")

    print(f"\n📈 RIESGO:")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:.2f}%")
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:.2f}")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:.2f}")

    print(f"\n🎯 TRADING:")
    print(f"  Trades:             {metrics['num_trades']}")
    print(f"  Win Rate:           {metrics['win_rate_pct']:.2f}%")
    print(f"  Profit Factor:      {metrics['profit_factor']:.2f}")
    print(f"  Avg Trade:          ${metrics['avg_trade']:.2f}")

    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/ml_patterns_results_{timestamp}.csv'

    results_df = pd.DataFrame([{
        'model': 'RF+GB Ensemble + Patterns',
        'timeframe': TF_TRADE,
        **metrics,
        **STRATEGY_CONFIG
    }])

    results_df.to_csv(results_file, index=False)
    print(f"\n💾 Resultados guardados: {results_file}")

    # Evaluación
    print("\n" + "="*80)
    print("✅ EVALUACIÓN vs OBJETIVOS")
    print("="*80)

    trades_per_year = metrics['num_trades'] * (365 / (len(df_test) * 15 / (60 * 24)))

    objectives = [
        ('Win Rate', 45, 65, metrics['win_rate_pct'], '%'),
        ('Trades/año', 80, 150, trades_per_year, ''),
        ('Profit Factor', 2.5, 999, metrics['profit_factor'], ''),
        ('Max Drawdown', 0, 12, metrics['max_drawdown_pct'], '%'),
    ]

    for name, min_val, max_val, actual, unit in objectives:
        if min_val <= actual <= max_val:
            status = "✅"
        else:
            status = "❌"

        if 'Drawdown' in name:
            print(f"  {status} {name}: {actual:.2f}{unit} (objetivo: <{max_val}{unit})")
        else:
            print(f"  {status} {name}: {actual:.2f}{unit} (objetivo: {min_val}-{max_val}{unit})")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
