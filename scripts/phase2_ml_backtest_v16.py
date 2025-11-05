#!/usr/bin/env python3
"""
ITERACIÓN 16: MACHINE LEARNING v3 (TRIPLE BARRIER + GRADIENT BOOSTING)
======================================================================

CAMBIOS CRÍTICOS:
1. Target: Triple Barrier (TP +4%, SL -2%, H=30 velas) - REALISTA
2. Features: 18 features con timing/momentum (IT15+16)
3. Modelo: GradientBoostingClassifier (vs RandomForest IT15)
4. Objetivo: Resolver Profit Factor = 0

CRITERIOS DE ÉXITO:
- Profit Factor ≥ 1.0 (CRÍTICO)
- Win Rate > 0% (CRÍTICO)
- AUC ≥ 0.60 (deseable)
"""

import os, sys, json, logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.binance_client import BinanceClientManager
from src.data.data_fetcher import obtener_datos_binance
from src.indicators.technical import agregar_indicadores
from src.backtest.engine import VectorizedBacktester
from src.ml.feature_engineer import preparar_dataset_ml_triple_barrier, calcular_stats_target, crear_features
from src.utils.logger import setup_logger

logger = setup_logger("phase2_ml_v16", "logs/phase2_ml_v16.log")

def generar_senales_ml_v16(df: pd.DataFrame, modelo: GradientBoostingClassifier, threshold: float = 0.60) -> pd.DataFrame:
    """Genera señales IT16"""
    df = df.copy()
    feature_names = [
        'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3',
        'volume_norm', 'atr_pct', 'close_pct_change',
        'bollinger_pct_b', 'ema_21_50_cross', 'rsi_zscore',
        'volume_momentum', 'price_to_ema200_pct',
        'rsi_momentum_5', 'rsi_momentum_10', 'stoch_momentum_k',
        'price_acceleration', 'ema_distance_ratio'
    ]
    feature_df = crear_features(df, mejorado=True)
    X = feature_df[feature_names].copy()
    X = X.fillna(method='ffill').fillna(method='bfill')
    proba_buy = modelo.predict_proba(X)[:, 1]
    df['ml_probability'] = proba_buy
    df['señal'] = np.where(proba_buy > threshold, 1, 0)
    df['position'] = df['señal'].copy()
    df.loc[df['señal'] == 0, 'position'] = df.loc[df['señal'] == 0, 'position'].shift(1)
    df['position'] = df['position'].fillna(0).astype(int)
    return df

def main():
    logger.info("="*80)
    logger.info("ITERACIÓN 16: ML v3 - TRIPLE BARRIER + GRADIENT BOOSTING")
    logger.info("="*80)

    # DATOS
    logger.info("\n1. Descargando datos...")
    try:
        client = BinanceClientManager().get_public_client()
        df = obtener_datos_binance(client, "ETHUSDT", "15m", "1 year ago UTC")
        df = agregar_indicadores(df)
        logger.info(f"   ✓ {len(df)} velas descargadas")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # DATASET ML CON TRIPLE BARRIER
    logger.info("\n2. Preparando dataset con Triple Barrier...")
    try:
        X, y, feature_names = preparar_dataset_ml_triple_barrier(
            df, sl_pct=-0.02, tp_pct=0.04, horizonte=30
        )
        logger.info(f"   ✓ {len(X)} muestras, {len(feature_names)} features")
        stats = calcular_stats_target(y)
        logger.info(f"   Positivos: {stats['positivos']} ({stats['pct_positivos']:.1f}%)")
        logger.info(f"   Negativos: {stats['negativos']} ({stats['pct_negativos']:.1f}%)")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return

    # SPLIT
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    logger.info(f"\n3. Train/Test split")
    logger.info(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    # MODELO: GRADIENT BOOSTING
    logger.info(f"\n4. Entrenando GradientBoostingClassifier...")
    try:
        modelo = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            min_samples_leaf=20, subsample=0.8, random_state=42
        )
        modelo.fit(X_train, y_train)
        logger.info(f"   ✓ Modelo entrenado")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # EVALUACIÓN
    logger.info(f"\n5. Evaluando modelo...")
    y_test_proba = modelo.predict_proba(X_test)[:, 1]
    y_test_pred = modelo.predict(X_test)
    auc = roc_auc_score(y_test, y_test_proba)
    acc = accuracy_score(y_test, y_test_pred)
    prec = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    logger.info(f"   AUC: {auc:.4f}, Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {recall:.4f}")

    # IMPORTANCIA FEATURES
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': modelo.feature_importances_
    }).sort_values('importance', ascending=False)
    logger.info(f"\n6. TOP 5 Features:")
    for i, (_, row) in enumerate(feature_importance.head(5).iterrows(), 1):
        logger.info(f"   {i}. {row['feature']:.<30} {row['importance']*100:>5.1f}%")

    # SEÑALES
    logger.info(f"\n7. Generando señales...")
    df_test = df.iloc[split_idx:].copy()
    df_test = generar_senales_ml_v16(df_test, modelo, threshold=0.60)
    num_signals = (df_test['señal'] == 1).sum()
    logger.info(f"   ✓ {num_signals} señales ({num_signals/len(df_test)*100:.2f}%)")

    # BACKTESTING
    logger.info(f"\n8. Ejecutando backtesting...")
    try:
        backtester = VectorizedBacktester(df=df_test, initial_capital=10000, commission=0.00075, slippage=0.0005)
        results = backtester.run_backtest_with_stop_loss(atr_column='ATRr_14', atr_multiplier=3.0)
        metrics = backtester.calculate_metrics()
        logger.info(f"   ✓ Backtesting completado")
    except Exception as e:
        logger.error(f"   ✗ Error: {e}")
        return

    # REPORTE FINAL
    logger.info(f"\n{'='*80}")
    logger.info(f"REPORTE IT16: TRIPLE BARRIER + GRADIENT BOOSTING")
    logger.info(f"{'='*80}")
    logger.info(f"\nDATASET: {len(df):,} velas (Train: {len(X_train):,}, Test: {len(X_test):,})")
    logger.info(f"FEATURES: {len(feature_names)} (básicos + IT15 + timing/momentum)")
    logger.info(f"TARGET: Triple Barrier (TP +4%, SL -2%, H=30 velas)")
    logger.info(f"\nMODELO: GradientBoostingClassifier")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    logger.info(f"  Precision: {prec:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"\nBACKTESTING:")
    logger.info(f"  Capital Inicial: ${metrics['initial_capital']:,.0f}")
    logger.info(f"  Capital Final: ${metrics['final_value']:,.0f}")
    logger.info(f"  Retorno: {metrics['total_return_pct']:+.2f}%")
    logger.info(f"  Trades: {metrics['num_trades']}")
    logger.info(f"  Win Rate: {metrics['win_rate_pct']:.1f}%")
    logger.info(f"  Profit Factor: {metrics['profit_factor']:.2f}")
    logger.info(f"  Sharpe: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max DD: {metrics['max_drawdown_pct']:.2f}%")

    # CRITERIOS
    logger.info(f"\n{'='*80}")
    logger.info(f"CRITERIOS DE ÉXITO IT16")
    logger.info(f"{'='*80}")
    success_auc = auc >= 0.60
    success_profit = metrics['profit_factor'] >= 1.0
    success_wr = metrics['win_rate_pct'] > 0
    logger.info(f"\nAUC ≥ 0.60: {auc:.4f} {'✓' if success_auc else '✗'}")
    logger.info(f"Profit Factor ≥ 1.0: {metrics['profit_factor']:.2f} {'✓ CRÍTICO' if success_profit else '✗ CRÍTICO'}")
    logger.info(f"Win Rate > 0%: {metrics['win_rate_pct']:.1f}% {'✓ CRÍTICO' if success_wr else '✗ CRÍTICO'}")

    if success_profit and success_wr:
        logger.info(f"\n✓✓✓ ITERACIÓN 16 APROBADA ✓✓✓")
    else:
        logger.info(f"\n✗✗✗ ITERACIÓN 16 SIN APROBACIÓN ✗✗✗")

    # GUARDAR ARTEFACTOS
    logger.info(f"\n{'='*80}")
    logger.info(f"Guardando artefactos...")
    try:
        results_dir = project_root / "results"
        results_dir.mkdir(exist_ok=True)
        if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
            trades_path = results_dir / f"trades_log_ml_v16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            backtester.trades_log.to_csv(trades_path, index=False)
            logger.info(f"✓ Trades log: {trades_path.name}")
        feature_importance.to_csv(results_dir / f"feature_importance_ml_v16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
        metrics_save = {
            'iteration': 16,
            'date': datetime.now().isoformat(),
            'target': 'Triple Barrier (TP +4%, SL -2%, H=30)',
            'model': {'auc': float(auc), 'accuracy': float(acc), 'precision': float(prec), 'recall': float(recall)},
            'backtest': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in metrics.items()},
            'success': {'profit_factor_>=_1.0': success_profit, 'win_rate_>_0': success_wr, 'approved': success_profit and success_wr}
        }
        metrics_path = results_dir / f"metrics_ml_v16_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_save, f, indent=2)

        # Archivo solicitado
        backtest_csv = results_dir / "backtest_results_eth_v16_triple_barrier.csv"
        if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
            backtester.trades_log.to_csv(backtest_csv, index=False)
        logger.info(f"✓ Reporte: {backtest_csv.name}")
    except Exception as e:
        logger.error(f"Error guardando: {e}")

    logger.info(f"\n{'='*80}")
    logger.info("EJECUCIÓN COMPLETADA")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    main()
