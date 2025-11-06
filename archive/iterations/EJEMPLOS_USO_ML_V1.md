# EJEMPLOS DE USO - MÓDULO ML v1

## Tabla de Contenidos

1. [Importación del módulo](#importación)
2. [Feature Engineering](#feature-engineering)
3. [Preparación de Dataset](#preparación)
4. [Entrenamiento del Modelo](#entrenamiento)
5. [Generación de Señales](#generación)
6. [Integración con Backtester](#integración)

---

## Importación

### Opción 1: Usar el script completo (recomendado para pruebas)

```bash
python3 scripts/phase2_ml_backtest.py
```

Esto ejecuta el pipeline **completo automáticamente**:
- Descarga datos
- Calcula indicadores
- Prepara features
- Entrena modelo
- Genera señales
- Ejecuta backtesting
- Genera reportes

### Opción 2: Usar el módulo directamente en código Python

```python
import sys
from pathlib import Path

# Agregar path al proyecto
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ml.feature_engineer import (
    crear_features,
    crear_target,
    preparar_dataset_ml,
    calcular_stats_target
)
```

---

## Feature Engineering

### Ejemplo 1: Extraer Features de un DataFrame

```python
import pandas as pd
from src.indicators.technical import agregar_indicadores
from src.ml.feature_engineer import crear_features

# Suponemos que 'df' tiene columnas OHLCV básicas
# Primero agregar indicadores técnicos
df = agregar_indicadores(df)
# df ahora tiene: OHLCV + 19 columnas de indicadores

# Extraer features para ML
feature_df = crear_features(df)

print(f"Shape: {feature_df.shape}")
# Output: (35040, 8) = 35040 muestras, 8 features

print(feature_df.columns.tolist())
# Output: ['close', 'volume', 'EMA_200', 'RSI_14', 'ATRr_14',
#          'MACDh_12_26_9', 'STOCHk_14_3_3', 'volume_norm',
#          'atr_pct', 'close_pct_change']
```

### Ejemplo 2: Calcular estadísticas de features

```python
# Ver distribución de features
print(feature_df.describe())

# Ver correlaciones
print(feature_df.corr())

# Detectar valores faltantes
print(feature_df.isnull().sum())
```

---

## Preparación

### Ejemplo 3: Crear target binario

```python
from src.ml.feature_engineer import crear_target

# Crear target con parámetros por defecto
target = crear_target(df)

print(f"Distribución del target:")
print(f"  Positivos (Buy): {(target == 1).sum()} ({(target == 1).sum()/len(target)*100:.1f}%)")
print(f"  Negativos (No Buy): {(target == 0).sum()} ({(target == 0).sum()/len(target)*100:.1f}%)")

# Output:
# Distribución del target:
#   Positivos (Buy): 10139 (28.9%)
#   Negativos (No Buy): 24900 (71.1%)
```

### Ejemplo 4: Crear target con parámetros personalizados

```python
# Target con ganancia mínima 2% en horizonte 15 velas
target_2pct = crear_target(df, horizonte=15, ganancia_min=0.02)

print(f"Con 2% ganancia y horizonte 15:")
print(f"  Positivos: {(target_2pct == 1).sum()} ({(target_2pct == 1).sum()/len(target_2pct)*100:.1f}%)")

# Probar diferentes horizontes
for h in [5, 10, 15, 20]:
    t = crear_target(df, horizonte=h, ganancia_min=0.01)
    print(f"Horizonte {h:2d}: {(t == 1).sum()} positivos ({(t == 1).sum()/len(t)*100:.1f}%)")

# Output:
# Horizonte  5: 5823 positivos (16.6%)
# Horizonte 10: 10139 positivos (28.9%)
# Horizonte 15: 13258 positivos (37.8%)
# Horizonte 20: 15982 positivos (45.6%)
```

### Ejemplo 5: Pipeline completo de preparación

```python
from src.ml.feature_engineer import preparar_dataset_ml, calcular_stats_target

# Ejecutar pipeline completo
X, y, feature_names = preparar_dataset_ml(
    df,
    horizonte=10,
    ganancia_min=0.01
)

print(f"\n=== DATASET PREPARADO ===")
print(f"Features shape: {X.shape}")  # (35039, 8)
print(f"Target shape: {y.shape}")    # (35039,)
print(f"\nNombres de features:")
for i, name in enumerate(feature_names, 1):
    print(f"  {i}. {name}")

# Ver estadísticas
stats = calcular_stats_target(y)
print(f"\n=== ESTADÍSTICAS DEL TARGET ===")
print(f"Total: {stats['total']}")
print(f"Positivos: {stats['positivos']} ({stats['pct_positivos']:.1f}%)")
print(f"Negativos: {stats['negativos']} ({stats['pct_negativos']:.1f}%)")
print(f"Balance Ratio: {stats['balance_ratio']:.2f}")

# Output:
# === DATASET PREPARADO ===
# Features shape: (35039, 8)
# Target shape: (35039,)
#
# Nombres de features:
#   1. EMA_200
#   2. RSI_14
#   3. ATRr_14
#   4. MACDh_12_26_9
#   5. STOCHk_14_3_3
#   6. volume_norm
#   7. atr_pct
#   8. close_pct_change
#
# === ESTADÍSTICAS DEL TARGET ===
# Total: 35039
# Positivos: 10139 (28.9%)
# Negativos: 24900 (71.1%)
# Balance Ratio: 0.41
```

---

## Entrenamiento

### Ejemplo 6: Entrenar modelo RandomForest

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

# Split Train/Test
split_idx = int(len(X) * 0.80)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Crear y entrenar modelo
modelo = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print("Entrenando modelo...")
modelo.fit(X_train, y_train)
print("✓ Modelo entrenado")

# Evaluar
y_pred = modelo.predict(X_test)
y_proba = modelo.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_proba)
acc = accuracy_score(y_test, y_pred)

print(f"\nMétricas:")
print(f"  AUC Score: {auc:.4f}")
print(f"  Accuracy: {acc:.4f}")
```

### Ejemplo 7: Ajustar hiperparámetros con GridSearchCV

```python
from sklearn.model_selection import GridSearchCV

# Grid de parámetros
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [6, 8, 10, 12],
    'min_samples_leaf': [5, 10, 20],
}

# Grid search
gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross validation
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("Ejecutando grid search...")
gs.fit(X_train, y_train)

print(f"\nMejores parámetros:")
print(gs.best_params_)
print(f"AUC Score: {gs.best_score_:.4f}")

# Usar mejor modelo
mejor_modelo = gs.best_estimator_
```

### Ejemplo 8: Ver importancia de features

```python
import pandas as pd

# Extraer importancia
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': modelo.feature_importances_
}).sort_values('importance', ascending=False)

print("Importancia de Features:")
print(feature_importance.to_string(index=False))

# Gráfico
import matplotlib.pyplot as plt

feature_importance.plot(x='feature', y='importance', kind='barh')
plt.title('Feature Importance - RandomForest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Output:
# Importancia de Features:
#           feature  importance
#           atr_pct       0.2507
#          ATRr_14       0.1598
#       volume_norm       0.1543
#          EMA_200       0.1526
#           RSI_14       0.0782
#    MACDh_12_26_9       0.0777
#  close_pct_change       0.0653
#    STOCHk_14_3_3       0.0614
```

---

## Generación

### Ejemplo 9: Generar señales en test set

```python
from src.backtest.engine import VectorizedBacktester
import numpy as np

# Función para generar señales (del script)
def generar_senales_ml(df, modelo, threshold=0.70):
    df = df.copy()

    feature_names = [
        'EMA_200', 'RSI_14', 'ATRr_14', 'MACDh_12_26_9', 'STOCHk_14_3_3',
        'volume_norm', 'atr_pct', 'close_pct_change'
    ]

    # Preparar features
    feature_df = crear_features(df)
    X = feature_df[feature_names].copy()
    X = X.fillna(method='ffill').fillna(method='bfill')

    # Predecir
    proba_buy = modelo.predict_proba(X)[:, 1]

    # Generar señales
    df['ml_probability'] = proba_buy
    df['señal'] = np.where(proba_buy > threshold, 1, 0)

    # Calcular posición
    df['position'] = df['señal'].copy()
    df.loc[df['señal'] == 0, 'position'] = df.loc[df['señal'] == 0, 'position'].shift(1)
    df['position'] = df['position'].fillna(0).astype(int)

    return df

# Generar señales en test set
df_test = df.iloc[split_idx:].copy()
df_test = generar_senales_ml(df_test, modelo, threshold=0.70)

# Estadísticas de señales
num_signals = (df_test['señal'] == 1).sum()
print(f"\nSeñales Generadas:")
print(f"  Total de señales: {num_signals}")
print(f"  % del dataset: {num_signals/len(df_test)*100:.2f}%")
print(f"  Probabilidad promedio: {df_test['ml_probability'].mean():.2f}")
print(f"  Probabilidad máxima: {df_test['ml_probability'].max():.2f}")

# Ver ejemplo de señales
signals_df = df_test[df_test['señal'] == 1][['timestamp', 'close', 'ml_probability', 'señal']].head()
print(f"\nEjemplos de señales:")
print(signals_df)
```

### Ejemplo 10: Probar diferentes thresholds

```python
# Comparar impacto de threshold
thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

for threshold in thresholds:
    df_temp = generar_senales_ml(df_test.copy(), modelo, threshold=threshold)
    num_signals = (df_temp['señal'] == 1).sum()
    avg_proba = df_temp.loc[df_temp['señal'] == 1, 'ml_probability'].mean()

    print(f"Threshold {threshold:.2f}: {num_signals:3d} señales (Proba: {avg_proba:.3f})")

# Output:
# Threshold 0.50: 856 señales (Proba: 0.624)
# Threshold 0.60: 432 señales (Proba: 0.668)
# Threshold 0.70:  48 señales (Proba: 0.750)
# Threshold 0.80:   8 señales (Proba: 0.825)
# Threshold 0.90:   1 señal   (Proba: 0.912)
```

---

## Integración

### Ejemplo 11: Backtesting completo con modelo

```python
from src.backtest.engine import VectorizedBacktester

# Ejecutar backtesting
backtester = VectorizedBacktester(
    df=df_test,
    initial_capital=10000,
    commission=0.00075,  # 0.075% Binance
    slippage=0.0005      # 0.05%
)

results = backtester.run_backtest_with_stop_loss(
    atr_column='ATRr_14',
    atr_multiplier=2.0
)

# Calcular métricas
metrics = backtester.calculate_metrics()

print(f"\n=== RESULTADOS DE BACKTESTING ===")
print(f"Capital inicial: ${metrics['initial_capital']:,.0f}")
print(f"Capital final: ${metrics['final_value']:,.0f}")
print(f"Retorno: {metrics['total_return_pct']:+.2f}%")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
print(f"Win Rate: {metrics['win_rate_pct']:.1f}%")
print(f"Max Drawdown: {metrics['max_drawdown_pct']:-.2f}%")
```

### Ejemplo 12: Guardar resultados

```python
import json
from datetime import datetime

# Guardar métricas en JSON
metrics_json = {
    'timestamp': datetime.now().isoformat(),
    'model': {
        'auc_score': float(auc),
        'accuracy': float(acc),
        'threshold': 0.70,
    },
    'backtest': {k: float(v) if isinstance(v, (int, float)) else str(v)
                 for k, v in metrics.items()}
}

with open('results/metrics_ml.json', 'w') as f:
    json.dump(metrics_json, f, indent=2)

# Guardar log de trades
if hasattr(backtester, 'trades_log') and not backtester.trades_log.empty:
    backtester.trades_log.to_csv('results/trades_log.csv', index=False)
    print(f"✓ Guardados {len(backtester.trades_log)} trades")

# Guardar feature importance
feature_importance.to_csv('results/feature_importance.csv', index=False)

print("✓ Artefactos guardados")
```

---

## Casos de Uso Avanzados

### Ejemplo 13: Ensemble con múltiples modelos

```python
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier

# Entrenar múltiples modelos
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)

# Crear ensemble
ensemble = VotingClassifier(
    estimators=[('rf', rf), ('gb', gb)],
    voting='soft'  # Usar probabilidades
)

ensemble.fit(X_train, y_train)

# Predecir
y_proba_ensemble = ensemble.predict_proba(X_test)[:, 1]
auc_ensemble = roc_auc_score(y_test, y_proba_ensemble)

print(f"AUC RandomForest: {roc_auc_score(y_test, modelo.predict_proba(X_test)[:, 1]):.4f}")
print(f"AUC Ensemble: {auc_ensemble:.4f}")
```

### Ejemplo 14: Walk-Forward Validation

```python
from sklearn.metrics import roc_auc_score

# Validación progresiva
window_size = 2000  # 2000 muestras por ventana
step = 1000        # Avanzar 1000 en cada iteración

scores = []
for i in range(0, len(X) - window_size - step, step):
    X_train_wf = X.iloc[i:i+window_size]
    y_train_wf = y.iloc[i:i+window_size]

    X_test_wf = X.iloc[i+window_size:i+window_size+step]
    y_test_wf = y.iloc[i+window_size:i+window_size+step]

    # Entrenar
    modelo_wf = RandomForestClassifier(n_estimators=50, random_state=42)
    modelo_wf.fit(X_train_wf, y_train_wf)

    # Evaluar
    score = roc_auc_score(y_test_wf, modelo_wf.predict_proba(X_test_wf)[:, 1])
    scores.append(score)

    print(f"Ventana {i:5d}: AUC = {score:.4f}")

print(f"\nAUC Promedio: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

## Troubleshooting

### Problema: "KeyError: 'EMA_200'"
**Causa:** Los indicadores no fueron calculados
**Solución:** Asegurar que llamaste a `agregar_indicadores(df)` antes

### Problema: "Shape mismatch" en predicción
**Causa:** Los features no tienen el mismo orden
**Solución:** Usar `X[feature_names]` para seleccionar en orden correcto

### Problema: "Todos los targets son 0"
**Causa:** Ganancia mínima demasiado alta o horizonte muy corto
**Solución:** Reducir `ganancia_min` o aumentar `horizonte`

### Problema: "AUC es 0.50 (random)"
**Causa:** Features insuficientes o target muy ruidoso
**Solución:** Revisar features con `crear_features()`, mejorar target

---

## Referencia Rápida

```python
# Importar todo lo necesario
from src.ml.feature_engineer import preparar_dataset_ml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Preparar datos
X, y, names = preparar_dataset_ml(df)

# 2. Split
split_idx = int(len(X) * 0.80)
X_tr, X_ts = X.iloc[:split_idx], X.iloc[split_idx:]
y_tr, y_ts = y.iloc[:split_idx], y.iloc[split_idx:]

# 3. Entrenar
model = RandomForestClassifier()
model.fit(X_tr, y_tr)

# 4. Evaluar
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_ts, model.predict_proba(X_ts)[:, 1])

# 5. Generar señales
proba = model.predict_proba(X_ts)[:, 1]
signals = (proba > 0.70).astype(int)

# 6. Backtestear
backtester = VectorizedBacktester(df_test)
results = backtester.run_backtest_with_stop_loss()
metrics = backtester.calculate_metrics()
```

---

**Documento de ejemplos para Iteración 14 - Módulo ML v1**
