"""
FASE 2: BACKTESTING CON MACHINE LEARNING v2 (ITERACIÓN 15 - OPTIMIZACIÓN)
==========================================================================

Optimización crítica del Pipeline ML para corregir Profit Factor = 0:

CAMBIOS CLAVE vs Iteración 14:
1. Features mejorados (5 adicionales): Bollinger %B, EMA Crossover, RSI Z-Score, etc.
2. Target redefinido: 2.0% ganancia mínima (vs 1.0%)
3. Class balancing: class_weight='balanced' en RandomForest
4. Parámetros de riesgo optimizados:
   - ATR multiplier: 3.0 (vs 2.0) → Más "aire" para los trades
   - Threshold de señal: 0.60 (vs 0.70) → Más Recall

CRITERIOS DE ÉXITO:
- AUC Score ≥ 0.65
- Profit Factor ≥ 1.0 (CRÍTICO)
- Win Rate > 0% (CRÍTICO)
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
    confusion_matrix, classification_report
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
logger = setup_logger("phase2_ml_v15", "logs/phase2_ml_v15.log")


def generar_senales_ml_mejoradas(
    df: pd.DataFrame,
    modelo: RandomForestClassifier,
    threshold: float = 0.60
) -> pd.DataFrame:
    """
    Genera señales usando predicciones del modelo ML mejorado.

    Args:
        df: DataFrame con features técnicos mejorados
        modelo: Modelo entrenado
        threshold: Umbral de confianza para señal (default: 0.60)

    Returns:
        DataFrame con columnas 'señal' y 'position'
    """
    df = df.copy()

    # Features mejorados (13 features)
    feature_names = [
        'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3',
        'volume_norm', 'atr_pct', 'close_pct_change',
        'bollinger_pct_b', 'ema_21_50_cross', 'rsi_zscore',
        'volume_momentum', 'price_to_ema200_pct'
    ]

    # Preparar features (mismo preprocesamiento que entrenamiento)
    feature_df = crear_features(df, mejorado=True)
    X = feature_df[feature_names].copy()

    # Rellenar NaN en features
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
    """Ejecuta el pipeline ML optimizado - Iteración 15"""

    logger.info("=" * 80)
    logger.info("FASE 2: BACKTESTING CON MACHINE LEARNING v2")
    logger.info("ITERACIÓN 15: OPTIMIZACIÓN (Features Mejorados + Class Balancing + Risk Adj)")
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
    # 3. PREPARAR DATASET ML CON FEATURES MEJORADOS
    # ========================================================================
    logger.info("\n4. Preparando dataset ML con FEATURES MEJORADOS...")

    # ITERACIÓN 15: Parámetros optimizados
    horizonte = 10           # velas (2.5h en 15m)
    ganancia_min = 0.02      # 2.0% (vs 1.0% en IT14)

    try:
        X, y, feature_names = preparar_dataset_ml(
            df,
            horizonte=horizonte,
            ganancia_min=ganancia_min,
            mejorado=True  # NUEVO: Features mejorados
        )
        logger.info(f"   ✓ Dataset preparado")
        logger.info(f"   Features: {len(feature_names)}")
        logger.info(f"   Muestras: {len(X)}")

        logger.info(f"\n   FEATURES UTILIZADOS ({len(feature_names)}):")
        for i, name in enumerate(feature_names, 1):
            logger.info(f"     {i:2d}. {name}")

        # Estadísticas del target
        stats_target = calcular_stats_target(y)
        logger.info(f"\n   TARGET DISTRIBUTION (2.0% ganancia mínima):")
        logger.info(f"   - Total: {stats_target['total']}")
        logger.info(f"   - Positivos (Buy): {stats_target['positivos']} ({stats_target['pct_positivos']:.1f}%)")
        logger.info(f"   - Negativos (No Buy): {stats_target['negativos']} ({stats_target['pct_negativos']:.1f}%)")
        logger.info(f"   - Balance Ratio: {stats_target['balance_ratio']:.2f}")

    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================================================
    # 4. SPLIT TRAIN/TEST (80/20 TEMPORAL)
    # ========================================================================
    logger.info("\n5. Split Train/Test...")

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
    # 5. ENTRENAR MODELO CON CLASS BALANCING
    # ========================================================================
    logger.info("\n6. Entrenando RandomForestClassifier CON CLASS BALANCING...")

    # ITERACIÓN 15: Parámetros optimizados
    model_params = {
        'n_estimators': 100,
        'max_depth': 8,  # Reducido de 10 para menos overfitting
        'min_samples_split': 15,  # Aumentado de 10
        'min_samples_leaf': 10,  # Aumentado de 5
        'class_weight': 'balanced',  # NUEVO: Balanceo de clases
        'random_state': 42,
        'n_jobs': -1,
        'verbose': 0
    }

    try:
        modelo = RandomForestClassifier(**model_params)
        modelo.fit(X_train, y_train)
        logger.info(f"   ✓ Modelo entrenado")
        logger.info(f"   Parámetros: class_weight='balanced' + max_depth=8")
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

    logger.info(f"\n   TOP 10 FEATURES:")
    for idx, (_, row) in enumerate(feature_importance.head(10).iterrows(), 1):
        logger.info(f"   {idx:2d}. {row['feature']:.<30} {row['importance']*100:>5.1f}%")

    # ========================================================================
    # 8. GENERAR SEÑALES EN TEST SET
    # ========================================================================
    logger.info("\n9. Generando señales en test set...")

    # ITERACIÓN 15: Threshold reducido para más Recall
    threshold_conf = 0.60  # Reducido de 0.70

    df_test = df.iloc[split_idx:].copy()
    df_test = generar_senales_ml_mejoradas(df_test, modelo, threshold=threshold_conf)

    num_signals = (df_test['señal'] == 1).sum()
    logger.info(f"   ✓ Señales generadas")
    logger.info(f"   Total de señales de compra: {num_signals} ({num_signals/len(df_test)*100:.2f}%)")
    logger.info(f"   Threshold de confianza: {threshold_conf}")

    # ========================================================================
    # 9. BACKTESTING CON STOP LOSS OPTIMIZADO
    # ========================================================================
    logger.info("\n10. Ejecutando backtesting...")

    # ITERACIÓN 15: ATR multiplier aumentado para más "aire"
    atr_mult = 3.0  # Aumentado de 2.0

    try:
        backtester = VectorizedBacktester(
            df=df_test,
            initial_capital=10000,
            commission=0.00075,
            slippage=0.0005
        )

        results = backtester.run_backtest_with_stop_loss(
            atr_column='ATRr_14',
            atr_multiplier=atr_mult  # NUEVO: 3.0
        )

        metrics = backtester.calculate_metrics()

        logger.info(f"   ✓ Backtesting completado")

    except Exception as e:
        logger.error(f"   ✗ Error en backtesting: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # ========================================================================
    # 10. REPORTE FINAL
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("REPORTE FINAL - ITERACIÓN 15: ML v2 OPTIMIZACIÓN")
    logger.info("=" * 80)

    logger.info(f"\nDATASET:")
    logger.info(f"  Activo: {symbol}")
    logger.info(f"  Timeframe: {interval}")
    logger.info(f"  Período: {start_date}")
    logger.info(f"  Total velas: {len(df):,}")
    logger.info(f"  Train set: {len(X_train):,} ({len(X_train)/len(X)*100:.1f}%)")
    logger.info(f"  Test set: {len(X_test):,} ({len(X_test)/len(X)*100:.1f}%)")

    logger.info(f"\nFEATURES ({len(feature_names)}):")
    logger.info(f"  Básicos (8):  EMA_200, RSI_14, ATRr_14, MACDh, STOCHk, volume_norm, atr_pct, close_pct_change")
    logger.info(f"  Mejorados (5): bollinger_pct_b, ema_21_50_cross, rsi_zscore, volume_momentum, price_to_ema200_pct")

    logger.info(f"\nTARGET:")
    logger.info(f"  Ganancia mínima: {ganancia_min*100:.1f}% (↑ de 1% en IT14)")
    logger.info(f"  Horizonte: {horizonte} velas (2.5h en 15m)")

    logger.info(f"\nMODELO: RandomForestClassifier + CLASS BALANCING")
    logger.info(f"  Estimadores: {model_params['n_estimators']}")
    logger.info(f"  Max depth: {model_params['max_depth']} (↓ de 10)")
    logger.info(f"  Class weight: 'balanced' (NUEVO)")
    logger.info(f"  Min samples leaf: {model_params['min_samples_leaf']} (↑ de 5)")

    logger.info(f"\nMÉTRICAS DEL MODELO (Test Set):")
    logger.info(f"  AUC Score: {auc_test:.4f} (Target: ≥0.65)")
    logger.info(f"  Accuracy: {acc_test:.4f} ({acc_test*100:.1f}%)")
    logger.info(f"  Precision: {prec_test:.4f} ({prec_test*100:.1f}%)")
    logger.info(f"  Recall: {recall_test:.4f} ({recall_test*100:.1f}%)")

    logger.info(f"\nTOP 3 FEATURES POR IMPORTANCIA:")
    for idx, (_, row) in enumerate(feature_importance.head(3).iterrows(), 1):
        logger.info(f"  {idx}. {row['feature']:<30} {row['importance']*100:>5.1f}%")

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
    logger.info(f"  Profit factor: {metrics['profit_factor']:.2f} (Target: ≥1.0)")
    logger.info(f"  Avg trade: ${metrics['avg_trade']:+,.0f}")
    logger.info(f"  Best trade: ${metrics['best_trade']:+,.0f}")
    logger.info(f"  Worst trade: ${metrics['worst_trade']:+,.0f}")

    logger.info(f"\nBENCHMARK (Buy & Hold):")
    logger.info(f"  Return B&H: {metrics['buy_hold_return_pct']:+.2f}%")
    logger.info(f"  Return Estrategia: {metrics['total_return_pct']:+.2f}%")
    logger.info(f"  Excess return: {metrics['excess_return_pct']:+.2f}%")

    # ========================================================================
    # CRITERIOS DE ÉXITO - ITERACIÓN 15
    # ========================================================================
    logger.info(f"\n" + "=" * 80)
    logger.info("CRITERIOS DE ÉXITO - ITERACIÓN 15")
    logger.info("=" * 80)

    success_auc = auc_test >= 0.65
    success_profit = metrics['profit_factor'] >= 1.0

    logger.info(f"\nAUC Score ≥ 0.65: {auc_test:.4f} {'✓ APROBADO' if success_auc else '✗ REPROBADO'}")
    logger.info(f"Profit Factor ≥ 1.0: {metrics['profit_factor']:.2f} {'✓ APROBADO' if success_profit else '✗ REPROBADO'}")

    if success_auc and success_profit:
        logger.info(f"\n✓✓✓ ITERACIÓN 15 APROBADA - ML VIABLE PARA FASE 5 ✓✓✓")
    else:
        logger.info(f"\n✗✗✗ ITERACIÓN 15 PARCIALMENTE COMPLETADA ✗✗✗")
        if not success_profit:
            logger.info(f"NOTA: Profit Factor sigue siendo desafío. Recomendación: revisar entrada de trades")

    # ========================================================================
    # GUARDAR ARTEFACTOS Y REPORTES
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
            trades_path = results_dir / f"trades_log_ml_v15_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            backtester.trades_log.to_csv(trades_path, index=False)
            logger.info(f"✓ Trades log guardado: {trades_path.name}")

        # Guardar importancia de features
        features_path = results_dir / f"feature_importance_ml_v15_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        feature_importance.to_csv(features_path, index=False)
        logger.info(f"✓ Feature importance guardado: {features_path.name}")

        # Guardar métricas en JSON
        metrics_save = {
            'iteration': 15,
            'date': datetime.now().isoformat(),
            'config': {
                'features_mejorados': True,
                'ganancia_min': ganancia_min,
                'horizonte': horizonte,
                'atr_multiplier': atr_mult,
                'threshold': threshold_conf,
                'class_weight': 'balanced',
                'max_depth': model_params['max_depth'],
            },
            'model': {
                'auc_score': float(auc_test),
                'accuracy': float(acc_test),
                'precision': float(prec_test),
                'recall': float(recall_test),
                'n_features': len(feature_names),
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
                'auc_>=_0.65': success_auc,
                'profit_factor_>= 1.0': success_profit,
                'approved': success_auc and success_profit
            }
        }

        metrics_path = results_dir / f"metrics_ml_v15_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_save, f, indent=2)
        logger.info(f"✓ Métricas guardadas: {metrics_path.name}")

        # Guardar resultado en formato solicitado: backtest_results_eth_v15_opt_ml.csv
        if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
            backtest_csv_path = results_dir / "backtest_results_eth_v15_opt_ml.csv"
            backtester.trades_log.to_csv(backtest_csv_path, index=False)
            logger.info(f"✓ Reporte solicitado guardado: {backtest_csv_path.name}")

    except Exception as e:
        logger.error(f"Error guardando artefactos: {e}")

    logger.info("\n" + "=" * 80)
    logger.info("EJECUCIÓN COMPLETADA")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
