"""
FASE 2: BACKTESTING CON MACHINE LEARNING v1
============================================

Pipeline ML para generar señales de trading:
1. Descargar datos históricos (ETH/USDT, 15m, 1 año)
2. Calcular indicadores técnicos
3. Preparar dataset ML (features + target)
4. Split Train/Test (80%/20% temporal)
5. Entrenar RandomForestClassifier
6. Generar señales en test set
7. Backtesting con Stop Loss dinámico ATR
8. Generar reporte completo
"""

import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, roc_curve
)

# Agregar path al proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.backtest.engine import VectorizedBacktester
from src.ml.feature_engineer import (
    preparar_dataset_ml,
    calcular_stats_target,
    crear_features
)
from src.utils.logger import setup_logger

# Logger
logger = setup_logger("phase2_ml", "logs/phase2_ml.log")


def generar_senales_ml(
    df: pd.DataFrame,
    modelo: RandomForestClassifier,
    threshold: float = 0.70
) -> pd.DataFrame:
    """
    Genera señales usando predicciones del modelo ML.

    Args:
        df: DataFrame con features técnicos
        modelo: Modelo entrenado
        threshold: Umbral de confianza para señal (default: 0.70)

    Returns:
        DataFrame con columnas 'señal' y 'position'
    """
    df = df.copy()

    # Features usados en entrenamiento
    feature_names = [
        'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3',
        'volume_norm', 'atr_pct', 'close_pct_change'
    ]

    # Preparar features (mismo preprocesamiento que entrenamiento)
    feature_df = crear_features(df)
    X = feature_df[feature_names].copy()

    # Rellenar NaN en features (solo forward fill)
    X = X.fillna(method='ffill').fillna(method='bfill')

    # Predecir probabilidades
    try:
        proba_buy = modelo.predict_proba(X)[:, 1]  # Probabilidad de clase 1
    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        proba_buy = np.zeros(len(X))

    # Generar señales basadas en threshold
    df['ml_probability'] = proba_buy
    df['señal'] = np.where(proba_buy > threshold, 1, 0)

    # Calcular posición (tracking)
    df['position'] = df['señal'].copy()
    df.loc[df['señal'] == 0, 'position'] = df.loc[df['señal'] == 0, 'position'].shift(1)
    df['position'] = df['position'].fillna(0).astype(int)

    return df


def main():
    """Ejecuta el pipeline completo de ML"""

    logger.info("=" * 80)
    logger.info("FASE 2: BACKTESTING CON MACHINE LEARNING v1")
    logger.info("ESTRATEGIA: RandomForest con Features Técnicos")
    logger.info("=" * 80)

    # ========================================================================
    # 1. CONECTAR A BINANCE Y DESCARGAR DATOS
    # ========================================================================
    logger.info("\n1. Conectando a Binance...")
    try:
        client = BinanceClientManager().get_public_client()
        logger.info("   ✓ Cliente creado")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # Configuración
    symbol = "ETHUSDT"
    interval = "15m"
    start_date = "1 year ago UTC"

    logger.info(f"\n2. Descargando datos...")
    logger.info(f"   ACTIVO: {symbol}")
    logger.info(f"   TIMEFRAME: {interval}")
    logger.info(f"   PERÍODO: {start_date}")

    try:
        df = obtener_datos_binance(client, symbol, interval, start_date)
        logger.info(f"   ✓ Descargados {len(df)} registros")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # ========================================================================
    # 2. CALCULAR INDICADORES TÉCNICOS
    # ========================================================================
    logger.info("\n3. Calculando indicadores técnicos...")
    try:
        df = agregar_indicadores(df)
        logger.info(f"   ✓ Indicadores calculados")
        logger.info(f"   Total de columnas: {len(df.columns)}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # ========================================================================
    # 3. PREPARAR DATASET ML
    # ========================================================================
    logger.info("\n4. Preparando dataset ML...")

    # Parámetros del target
    horizonte = 10  # velas (2.5h en 15m)
    ganancia_min = 0.01  # 1%

    try:
        X, y, feature_names = preparar_dataset_ml(df, horizonte=horizonte, ganancia_min=ganancia_min)
        logger.info(f"   ✓ Dataset preparado")
        logger.info(f"   Features: {len(feature_names)}")
        logger.info(f"   Muestras: {len(X)}")

        # Estadísticas del target
        stats_target = calcular_stats_target(y)
        logger.info(f"\n   TARGET DISTRIBUTION:")
        logger.info(f"   - Total: {stats_target['total']}")
        logger.info(f"   - Positivos (Buy): {stats_target['positivos']} ({stats_target['pct_positivos']:.1f}%)")
        logger.info(f"   - Negativos (No Buy): {stats_target['negativos']} ({stats_target['pct_negativos']:.1f}%)")
        logger.info(f"   - Balance Ratio: {stats_target['balance_ratio']:.2f}")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # ========================================================================
    # 4. SPLIT TRAIN/TEST (80/20 TEMPORAL)
    # ========================================================================
    logger.info("\n5. Split Train/Test...")

    # Split temporal (no aleatorio para preservar orden cronológico)
    split_idx = int(len(X) * 0.80)

    X_train = X.iloc[:split_idx].copy()
    y_train = y.iloc[:split_idx].copy()

    X_test = X.iloc[split_idx:].copy()
    y_test = y.iloc[split_idx:].copy()

    logger.info(f"   Train: {len(X_train)} muestras ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"   Test: {len(X_test)} muestras ({len(X_test)/len(X)*100:.1f}%)")

    # Estadísticas por set
    stats_train = calcular_stats_target(y_train)
    stats_test = calcular_stats_target(y_test)

    logger.info(f"\n   TRAIN SET:")
    logger.info(f"   - Positivos: {stats_train['positivos']} ({stats_train['pct_positivos']:.1f}%)")
    logger.info(f"   - Negativos: {stats_train['negativos']} ({stats_train['pct_negativos']:.1f}%)")

    logger.info(f"\n   TEST SET:")
    logger.info(f"   - Positivos: {stats_test['positivos']} ({stats_test['pct_positivos']:.1f}%)")
    logger.info(f"   - Negativos: {stats_test['negativos']} ({stats_test['pct_negativos']:.1f}%)")

    # ========================================================================
    # 5. ENTRENAR MODELO
    # ========================================================================
    logger.info("\n6. Entrenando RandomForestClassifier...")

    # Configuración del modelo
    model_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }

    try:
        modelo = RandomForestClassifier(**model_params)
        modelo.fit(X_train, y_train)
        logger.info(f"   ✓ Modelo entrenado")
        logger.info(f"   Parámetros: {model_params}")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # ========================================================================
    # 6. EVALUAR MODELO EN TRAIN/TEST
    # ========================================================================
    logger.info("\n7. Evaluando modelo...")

    # Predicciones Train
    y_train_pred = modelo.predict(X_train)
    y_train_proba = modelo.predict_proba(X_train)[:, 1]

    auc_train = roc_auc_score(y_train, y_train_proba)
    acc_train = accuracy_score(y_train, y_train_pred)

    logger.info(f"\n   TRAIN SET:")
    logger.info(f"   - AUC Score: {auc_train:.4f}")
    logger.info(f"   - Accuracy: {acc_train:.4f}")

    # Predicciones Test
    y_test_pred = modelo.predict(X_test)
    y_test_proba = modelo.predict_proba(X_test)[:, 1]

    auc_test = roc_auc_score(y_test, y_test_proba)
    acc_test = accuracy_score(y_test, y_test_pred)
    prec_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)

    logger.info(f"\n   TEST SET:")
    logger.info(f"   - AUC Score: {auc_test:.4f}")
    logger.info(f"   - Accuracy: {acc_test:.4f}")
    logger.info(f"   - Precision: {prec_test:.4f}")
    logger.info(f"   - Recall: {recall_test:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_test_pred)
    logger.info(f"\n   CONFUSION MATRIX (Test):")
    logger.info(f"   - True Negatives: {cm[0, 0]}")
    logger.info(f"   - False Positives: {cm[0, 1]}")
    logger.info(f"   - False Negatives: {cm[1, 0]}")
    logger.info(f"   - True Positives: {cm[1, 1]}")

    # ========================================================================
    # 7. IMPORTANCIA DE FEATURES
    # ========================================================================
    logger.info(f"\n8. Importancia de Features:")

    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in feature_importance.iterrows():
        logger.info(f"   {row['feature']:.<25} {row['importance']:>6.4f} ({row['importance']*100:>5.1f}%)")

    # ========================================================================
    # 8. GENERAR SEÑALES EN TEST SET
    # ========================================================================
    logger.info("\n9. Generando señales en test set...")

    # Reconstruir DataFrame con índices del test set para backtesting
    df_test = df.iloc[split_idx:].copy()

    # Generar señales
    threshold_conf = 0.70
    df_test = generar_senales_ml(df_test, modelo, threshold=threshold_conf)

    num_signals = (df_test['señal'] == 1).sum()
    logger.info(f"   ✓ Señales generadas")
    logger.info(f"   Total de señales de compra: {num_signals} ({num_signals/len(df_test)*100:.2f}%)")
    logger.info(f"   Threshold de confianza: {threshold_conf}")

    # ========================================================================
    # 9. BACKTESTING CON STOP LOSS ATR
    # ========================================================================
    logger.info("\n10. Ejecutando backtesting...")

    try:
        backtester = VectorizedBacktester(
            df=df_test,
            initial_capital=10000,
            commission=0.00075,
            slippage=0.0005
        )

        results = backtester.run_backtest_with_stop_loss(
            atr_column='ATRr_14',
            atr_multiplier=2.0
        )

        metrics = backtester.calculate_metrics()

        logger.info(f"   ✓ Backtesting completado")

    except Exception as e:
        logger.error(f"   ✗ Error en backtesting: {e}")
        return

    # ========================================================================
    # 10. REPORTE FINAL
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("REPORTE FINAL - ITERACIÓN 14: MÓDULO ML v1")
    logger.info("=" * 80)

    logger.info(f"\nDATASET:")
    logger.info(f"  Activo: {symbol}")
    logger.info(f"  Timeframe: {interval}")
    logger.info(f"  Período: {start_date}")
    logger.info(f"  Total velas: {len(df):,}")
    logger.info(f"  Train set: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Test set: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    logger.info(f"\nFEATURES (5):")
    logger.info(f"  1. EMA_200 (Tendencia)")
    logger.info(f"  2. RSI_14 (Momentum)")
    logger.info(f"  3. ATRr_14 (Volatilidad)")
    logger.info(f"  4. MACDh_12_26_9 (Presión)")
    logger.info(f"  5. STOCHk_14_3_3 (Reversión)")
    logger.info(f"  + Features derivados: volume_norm, atr_pct, close_pct_change")

    logger.info(f"\nTARGET:")
    logger.info(f"  Horizonte: {horizonte} velas (2.5h en 15m)")
    logger.info(f"  Ganancia mínima: {ganancia_min*100:.1f}%")

    logger.info(f"\nMODELO: RandomForestClassifier")
    logger.info(f"  Estimadores: {model_params['n_estimators']}")
    logger.info(f"  Max depth: {model_params['max_depth']}")
    logger.info(f"  Threshold: {threshold_conf}")

    logger.info(f"\nMÉTRICAS DEL MODELO (Test Set):")
    logger.info(f"  AUC Score: {auc_test:.4f}")
    logger.info(f"  Accuracy: {acc_test:.4f} ({acc_test*100:.1f}%)")
    logger.info(f"  Precision: {prec_test:.4f} ({prec_test*100:.1f}%)")
    logger.info(f"  Recall: {recall_test:.4f} ({recall_test*100:.1f}%)")

    logger.info(f"\nTOP 3 FEATURES POR IMPORTANCIA:")
    for idx, (_, row) in enumerate(feature_importance.head(3).iterrows(), 1):
        logger.info(f"  {idx}. {row['feature']:<25} {row['importance']*100:>5.1f}%")

    logger.info(f"\nBACKTESTING (Test Set, últimas {len(df_test):,} velas):")
    logger.info(f"  Capital inicial: ${metrics['initial_capital']:,.0f}")
    logger.info(f"  Capital final: ${metrics['final_value']:,.0f}")
    logger.info(f"  Retorno neto: ${metrics['net_profit']:,.0f}")
    logger.info(f"  Retorno %: {metrics['total_return_pct']:+.2f}%")
    logger.info(f"  Annual return: {metrics['annual_return_pct']:+.2f}%")

    logger.info(f"\nRISK METRICS:")
    logger.info(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    logger.info(f"  Calmar Ratio: {metrics['calmar_ratio']:.2f}")
    logger.info(f"  Max Drawdown: {metrics['max_drawdown_pct']:-.2f}%")

    logger.info(f"\nTRADES:")
    logger.info(f"  Total trades: {metrics['num_trades']}")
    logger.info(f"  Win rate: {metrics['win_rate_pct']:.1f}%")
    logger.info(f"  Profit factor: {metrics['profit_factor']:.2f}")
    logger.info(f"  Avg trade: ${metrics['avg_trade']:+,.0f}")
    logger.info(f"  Best trade: ${metrics['best_trade']:+,.0f}")
    logger.info(f"  Worst trade: ${metrics['worst_trade']:+,.0f}")

    logger.info(f"\nBENCHMARK (Buy & Hold):")
    logger.info(f"  Return: {metrics['buy_hold_return_pct']:+.2f}%")
    logger.info(f"  Excess return (vs B&H): {metrics['excess_return_pct']:+.2f}%")

    # ========================================================================
    # CRITERIOS DE ÉXITO
    # ========================================================================
    logger.info(f"\n" + "=" * 80)
    logger.info("CRITERIOS DE ÉXITO DE ML")
    logger.info("=" * 80)

    success_auc = auc_test >= 0.60
    success_profit = metrics['profit_factor'] >= 1.0

    logger.info(f"\nAUC Score ≥ 0.60: {auc_test:.4f} {'✓ APROBADO' if success_auc else '✗ REPROBADO'}")
    logger.info(f"Profit Factor ≥ 1.0: {metrics['profit_factor']:.2f} {'✓ APROBADO' if success_profit else '✗ REPROBADO'}")

    if success_auc and success_profit:
        logger.info(f"\n✓✓✓ MÓDULO ML VIABLE - APROBADO PARA AUDITORIA ✓✓✓")
    else:
        logger.info(f"\n✗✗✗ MÓDULO ML REQUIERE AJUSTES - RECHAZADO ✗✗✗")

    # ========================================================================
    # GUARDAR LOG DE TRADES
    # ========================================================================
    logger.info(f"\n" + "=" * 80)
    logger.info("Guardando artefactos...")
    logger.info("=" * 80)

    try:
        # Crear directorio de resultados si no existe
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)

        # Guardar log de trades
        if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
            trades_path = results_dir / f"trades_log_ml_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            backtester.trades_log.to_csv(trades_path, index=False)
            logger.info(f"✓ Trades log guardado en: {trades_path}")

        # Guardar importancia de features
        features_path = results_dir / f"feature_importance_ml_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        feature_importance.to_csv(features_path, index=False)
        logger.info(f"✓ Feature importance guardado en: {features_path}")

        # Guardar métricas en JSON
        metrics_save = {
            'model': {
                'auc_score': float(auc_test),
                'accuracy': float(acc_test),
                'precision': float(prec_test),
                'recall': float(recall_test),
                'n_estimators': model_params['n_estimators'],
                'max_depth': model_params['max_depth'],
            },
            'backtest': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                        for k, v in metrics.items()},
            'dataset': {
                'total': len(X),
                'train': len(X_train),
                'test': len(X_test),
                'symbol': symbol,
                'interval': interval,
            },
            'success': {
                'auc_>=_0.60': success_auc,
                'profit_factor_>= 1.0': success_profit,
                'approved': success_auc and success_profit
            }
        }

        metrics_path = results_dir / f"metrics_ml_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_save, f, indent=2)
        logger.info(f"✓ Métricas guardadas en: {metrics_path}")

    except Exception as e:
        logger.error(f"Error guardando artefactos: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("EJECUCIÓN COMPLETADA")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
