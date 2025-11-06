# ITERACIÓN 14: MÓDULO DE MACHINE LEARNING v1

**Fecha:** 2025-11-04
**Proyecto:** Bot de Trading Algorítmico (Fase 5)
**Auditoría:** Quant-Auditor

---

## ÍNDICE

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura del Pipeline](#arquitectura-del-pipeline)
3. [Módulos Implementados](#módulos-implementados)
4. [Dataset y Preparación](#dataset-y-preparación)
5. [Modelo de Clasificación](#modelo-de-clasificación)
6. [Resultados del Backtesting](#resultados-del-backtesting)
7. [Análisis de Resultados](#análisis-de-resultados)
8. [Criterios de Éxito](#criterios-de-éxito)
9. [Recomendaciones](#recomendaciones)
10. [Artefactos Generados](#artefactos-generados)

---

## RESUMEN EJECUTIVO

Se implementó un **pipeline completo de Machine Learning** para generar señales de trading de criptomonedas (ETH/USDT, 15m). El sistema entrena un **RandomForestClassifier** usando features técnicos para predecir oportunidades de compra con horizonte de 2.5 horas y ganancia mínima de 1%.

### Estado de Criterios de Éxito

| Criterio | Valor | Requerido | Estado |
|----------|-------|-----------|--------|
| **AUC Score** | 0.6389 | ≥ 0.60 | ✅ **APROBADO** |
| **Profit Factor** | 0.00 | ≥ 1.0 | ❌ **REPROBADO** |
| **Estado General** | - | Ambos | ❌ **RECHAZADO** |

**Conclusión:** El módulo ML cumple con la primera métrica (modelo discriminante) pero **falla en rentabilidad**. Requiere ajustes en:
- Generación de features
- Definición del target
- Parámetros del modelo
- Umbral de confianza

---

## ARQUITECTURA DEL PIPELINE

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. DESCARGA DE DATOS HISTÓRICOS                                 │
│ - Binance API (ETH/USDT, 15m, 1 año)                           │
│ - 35,040 velas                                                  │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. CÁLCULO DE INDICADORES TÉCNICOS                             │
│ - EMA_200, RSI_14, ATRr_14, MACD, Estocástico, etc.           │
│ - DataFrame con 25 columnas (OHLCV + indicadores)             │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. FEATURE ENGINEERING                                          │
│ - Extracción de 5 features core + 3 derivados                 │
│ - Normalización y preprocesamiento                             │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. CREACIÓN DE TARGET BINARIO                                  │
│ - Target = 1 si precio alcanza +1% en próximas 10 velas       │
│ - Target = 0 si no alcanza TP                                 │
│ - Distribución: 28.9% positivos, 71.1% negativos              │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. SPLIT TRAIN/TEST                                             │
│ - Train: 80% (28,031 muestras) - Histórico                    │
│ - Test: 20% (7,008 muestras) - Reciente                       │
│ - Split temporal (no aleatorio) para evitar data leakage      │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. ENTRENAMIENTO DEL MODELO                                     │
│ - RandomForestClassifier (100 árboles, max_depth=10)          │
│ - Train AUC: 0.8137 / Train Accuracy: 0.7638                  │
│ - Entrenamiento completado en <1 segundo                       │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 7. GENERACIÓN DE SEÑALES                                        │
│ - Predicción en test set con threshold = 0.70                 │
│ - Signal = 1 si P(buy) > 0.70                                 │
│ - Total de 48 señales de compra (0.68% del dataset)           │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 8. BACKTESTING CON STOP LOSS DINÁMICO                          │
│ - Stop Loss: entry_price - (ATR × 2.0)                        │
│ - Capital inicial: $10,000                                     │
│ - Comisión: 0.075%, Slippage: 0.05%                           │
│ - 6 trades ejecutados                                          │
└─────────────┬───────────────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 9. REPORTE DE RESULTADOS                                        │
│ - Retorno: -15.56% (Capital final: $8,444)                    │
│ - Profit Factor: 0.00 (Win Rate: 0%)                          │
│ - vs Buy & Hold: +18.36% superior (pero negativos ambos)      │
└─────────────────────────────────────────────────────────────────┘
```

---

## MÓDULOS IMPLEMENTADOS

### 1. `src/ml/feature_engineer.py`

**Propósito:** Feature engineering y preparación de dataset para ML

**Funciones principales:**

#### `crear_features(df: DataFrame) → DataFrame`
Extrae 8 features técnicos del DataFrame:

| Feature | Tipo | Rango | Importancia |
|---------|------|-------|-------------|
| **EMA_200** | Tendencia | Variable | 15.3% |
| **RSI_14** | Momentum | 0-100 | 7.8% |
| **ATRr_14** | Volatilidad | Variable | 16.0% |
| **MACDh_12_26_9** | Presión | Variable | 7.8% |
| **STOCHk_14_3_3** | Reversión | 0-100 | 6.1% |
| **volume_norm** | Volumen | Log(volume) | 15.4% |
| **atr_pct** | ATR % | Variable | 25.1% ⭐ |
| **close_pct_change** | Momentum | % | 6.5% |

**Nota:** `atr_pct` (ATR como % del precio) es el feature más importante (25.1%).

#### `crear_target(df, horizonte=10, ganancia_min=0.01) → Series`
Genera target binario basado en máximo futuro:

```python
target = 1  si max(high[t:t+10]) >= close[t] * (1 + 0.01)
target = 0  en caso contrario
```

**Parámetros:**
- `horizonte`: 10 velas (2.5 horas en 15m)
- `ganancia_min`: 1.0% ganancia mínima requerida

#### `preparar_dataset_ml(df, horizonte=10, ganancia_min=0.01) → (X, y, feature_names)`
Pipeline completo:
1. Crear features
2. Crear target
3. Eliminar NaN
4. Retornar X (features), y (target), nombres de features

**Output:**
- X: DataFrame con 8 features, 35,039 muestras
- y: Series binaria
- feature_names: Lista con nombres de 8 features

#### `calcular_stats_target(y: Series) → Dict`
Estadísticas del target para análisis de balance:

```
Total: 35,039
Positivos: 10,139 (28.9%)
Negativos: 24,900 (71.1%)
Balance Ratio: 0.41 (desbalanceado)
```

---

### 2. `scripts/phase2_ml_backtest.py`

**Propósito:** Entrenamiento de modelo y backtesting con reportes

**Flujo principal:**

1. **Descarga de datos** → 35,040 velas ETH/USDT 15m (1 año)
2. **Indicadores técnicos** → 25 columnas
3. **Preparación ML** → 35,039 muestras con 8 features
4. **Split Train/Test** → 80% / 20% temporal
5. **Entrenamiento** → RandomForestClassifier (100 árboles)
6. **Evaluación** → Métricas de clasificación
7. **Generación de señales** → threshold = 0.70
8. **Backtesting** → Stop Loss dinámico ATR
9. **Reportes** → JSON, CSV, logs

**Función principal: `generar_senales_ml(df, modelo, threshold=0.70)`**

Convierte predicciones del modelo en señales de trading:
- Input: DataFrame con features técnicos, modelo entrenado
- Output: DataFrame con columnas 'señal' (1/0), 'ml_probability', 'position'
- Lógica: `señal = 1` si `P(buy) > threshold`, else `0`

**Parámetros del Modelo:**
```python
{
    'n_estimators': 100,        # 100 árboles de decisión
    'max_depth': 10,            # Profundidad máxima
    'min_samples_split': 10,    # Min muestras para split
    'min_samples_leaf': 5,      # Min muestras por hoja
    'random_state': 42          # Reproducibilidad
}
```

**Backtesting:**
```python
VectorizedBacktester(
    df=df_test,
    initial_capital=10000,
    commission=0.00075,  # 0.075% Binance
    slippage=0.0005      # 0.05% estimado
)

results = run_backtest_with_stop_loss(
    atr_column='ATRr_14',
    atr_multiplier=2.0
)
```

---

## DATASET Y PREPARACIÓN

### Dimensiones del Dataset

```
Total velas descargadas:     35,040
Período:                     2024-11-04 → 2025-11-04 (365 días)
Timeframe:                   15 minutos
Activo:                      ETH/USDT
```

### Distribución Train/Test (Split Temporal)

```
TRAIN SET (80%):
├─ Muestras: 28,031
├─ Período: 2024-11-04 → ~2025-08-10
├─ Positivos: 8,593 (30.7%)
└─ Negativos: 19,438 (69.3%)

TEST SET (20%):
├─ Muestras: 7,008
├─ Período: ~2025-08-10 → 2025-11-04 (período de backtesting)
├─ Positivos: 1,546 (22.1%)
└─ Negativos: 5,462 (77.9%)
```

**Distribución del Target:**

```
TRAIN SET:
  Positivos (Buy): 8,593 / 28,031 = 30.7%
  Negativos (No Buy): 19,438 / 28,031 = 69.3%
  Balance Ratio: 0.44

TEST SET:
  Positivos (Buy): 1,546 / 7,008 = 22.1%
  Negativos (No Buy): 5,462 / 7,008 = 77.9%
  Balance Ratio: 0.28
```

**Análisis del Desbalance:**
- Dataset fuertemente desbalanceado (71% negativos)
- Test set más desbalanceado que train set
- Bajo recall (7.5%) indica dificultad para detectar oportunidades de compra

---

## MODELO DE CLASIFICACIÓN

### Arquitectura: RandomForestClassifier

**Justificación:**
- Robusto con datos desbalanceados
- Proporciona importancia de features
- Rápido en inferencia (importante para trading)
- No requiere normalización de features
- Interpretable (tree-based)

**Hiperparámetros:**

| Parámetro | Valor | Justificación |
|-----------|-------|---------------|
| n_estimators | 100 | Balance entre accuraccy y velocidad |
| max_depth | 10 | Previene overfitting |
| min_samples_split | 10 | Evita splits en muestras muy pequeñas |
| min_samples_leaf | 5 | Regularización adicional |
| random_state | 42 | Reproducibilidad |

### Evaluación del Modelo

#### Train Set (28,031 muestras)

```
AUC Score:  0.8137 ✓ Excelente discriminación
Accuracy:   0.7638 (76.4%)
```

**Interpretación:** El modelo aprende bien el patrón en datos históricos.

#### Test Set (7,008 muestras)

```
AUC Score:   0.6389 ✓ APROBADO (≥ 0.60)
Accuracy:    0.7788 (77.9%)
Precision:   0.4915 (49.2%) ← Muchos falsos positivos
Recall:      0.0750 (7.5%)  ← Pocas oportunidades de compra detectadas
```

**Matriz de Confusión (Test Set):**

```
                Predicho NO    Predicho SÍ
Actual NO:        5,342            120      (TNR: 97.8%, FPR: 2.2%)
Actual SÍ:        1,430            116      (FNR: 92.5%, TPR: 7.5%)
```

**Interpretación Detallada:**

1. **AUC = 0.6389**: Modelo es ligeramente mejor que azar (AUC > 0.5)
   - Mejora esperada: 27.8% sobre predicción aleatoria
   - Pero es apenas superior al mínimo requerido (0.60)

2. **Precision = 49.2%**: De los 236 trades predichos, ~49% son correctos
   - 120 falsos positivos (4.2% del negativo real)
   - Los falsos positivos generan pérdidas

3. **Recall = 7.5%**: Detecta solo 116 de 1,546 oportunidades reales (7.5%)
   - Muy conservador, se pierde 92.5% de oportunidades
   - Por eso pocos trades en backtest

### Importancia de Features

```
Rank Feature              Importancia  Tipo
─────────────────────────────────────────────
 1.  atr_pct             25.1%        ✓ Volatilidad
 2.  ATRr_14             16.0%        ✓ Volatilidad
 3.  volume_norm         15.4%        ✓ Volumen
 4.  EMA_200             15.3%        ✓ Tendencia
 5.  RSI_14               7.8%        - Momentum
 6.  MACDh_12_26_9        7.8%        - Presión
 7.  close_pct_change     6.5%        - Cambio %
 8.  STOCHk_14_3_3        6.1%        - Reversión
```

**Análisis:**
- **Volatilidad domina** (41.1% = ATR + volumen)
- **Features core contribuyen menos** (7.8% cada uno)
- **Hipótesis:** Modelo se enfoca en "cuándo está volátil" no "cuándo subirá"

---

## RESULTADOS DEL BACKTESTING

### Configuración

```
Capital inicial:      $10,000
Período de backtest:  ~2025-08-10 → 2025-11-04 (3 meses)
Comisión Binance:     0.075% (0.001875 por lado)
Slippage estimado:    0.05%
Stop Loss:            ATR × 2.0 (dinámico)
Threshold de señal:   0.70 (probabilidad de compra)
```

### Resultados Cuantitativos

#### Capital y Retornos

```
Capital Inicial:         $10,000
Capital Final:           $8,444
Pérdida Neta:            -$1,556
Retorno Total:           -15.56%
Retorno Anualizado:      +0.00% (período muy corto)
```

#### Análisis de Riesgo

```
Sharpe Ratio:           -0.38      (negativo = no rentable)
Sortino Ratio:          -0.03      (penaliza volatilidad negativa)
Calmar Ratio:            0.00      (cero = no hay ganancia)
Max Drawdown:           -15.56%    (caída máxima = pérdida total)
```

**Interpretación:** El portfolio está en territorio negativo sin recuperación.

#### Análisis de Operaciones

```
Total de Trades:        6
Winning Trades:         0 (0.0%)    ✗ Ningún trade ganador
Losing Trades:          6 (100.0%)
Profit Factor:          0.00        ✗ REPROBADO (≥ 1.0 requerido)
Avg Trade:              -$259
Best Trade:             -$196
Worst Trade:            -$448
```

**Detalles de Trades:**

| # | Entry | Entry $ | Exit $ | Exit Reason | PnL % | PnL $ |
|---|-------|---------|--------|-------------|-------|-------|
| 1 | 4741 | $4,740.62 | $4,644.39 | SL | -2.15% | -$215 |
| 2 | 3871 | $3,871.30 | $3,782.31 | SL | -2.42% | -$237 |
| 3 | 3590 | $3,590.00 | $3,426.16 | SL | -4.69% | -$448 |
| 4 | 3320 | $3,320.39 | $3,253.19 | SL | -2.15% | -$196 |
| 5 | 3229 | $3,228.92 | $3,155.95 | SL | -2.38% | -$212 |
| 6 | 3216 | $3,216.23 | $3,128.53 | SL | -2.85% | -$248 |

**Todos los trades terminan en Stop Loss** (100% SL)

### Benchmark: Buy & Hold

```
Retorno B&H:          -33.92%
Retorno Estrategia:   -15.56%
Excess Return:        +18.36%
```

**Análisis:** La estrategia es **menos mala** que buy-and-hold, pero sigue siendo pérdida.

---

## ANÁLISIS DE RESULTADOS

### ¿Por qué falló el módulo?

#### 1. **Baja Generación de Señales (0.68%)**
   - **Problema:** Solo 48 señales en 7,008 velas
   - **Causa:** Threshold muy alto (0.70) con modelo débil
   - **Impacto:** Muy pocos trades (solo 6 en 3 meses)

#### 2. **Recall muy bajo (7.5%)**
   - **Problema:** Detecta solo 7.5% de oportunidades reales
   - **Causa:** Modelo extremadamente conservador
   - **Impacto:** Pierde 92.5% de trades potencialmente ganadores

#### 3. **Todos los trades son pérdidas**
   - **Problema:** 0% win rate
   - **Causa:**
     - Target definition sub-óptima (1% ganancia es bajo)
     - Ruido en el dataset
     - Features insuficientes para capturar momentum

#### 4. **Desbalance del Dataset**
   - **Problema:** 71% negativos, 29% positivos
   - **Impacto:** El modelo tiende a predecir "no comprar"
   - **Solución necesaria:** Class weighting o SMOTE

#### 5. **Volatilidad domina sobre tendencia**
   - **Problema:** ATR es 41% de importancia
   - **Causa:** Features de tendencia (EMA, RSI, MACD) son débiles
   - **Impacto:** Se compra en momentos volátiles sin dirección

### Degradación Train → Test

```
AUC Train:    0.8137
AUC Test:     0.6389
Degradación:  22.0% ← Sobreentrenamiento (overfitting)

Recall Train: NO DISPONIBLE
Recall Test:  0.0750 ← Muy bajo en generalización
```

**Conclusión:** El modelo aprende en train pero no generaliza bien a datos nuevos.

---

## CRITERIOS DE ÉXITO

### Evaluación Final

| Criterio | Valor | Requerido | Status | Análisis |
|----------|-------|-----------|--------|----------|
| **AUC Score ≥ 0.60** | **0.6389** | 0.60 | ✅ **APROBADO** | Apenas supera el mínimo |
| **Profit Factor ≥ 1.0** | **0.00** | 1.00 | ❌ **REPROBADO** | Sin ganancias netas |
| **MÓDULO VIABLE** | - | Ambos ✓ | ❌ **RECHAZADO** | Falla criterio de rentabilidad |

### Veredicto Final

```
╔════════════════════════════════════════════════════════════════╗
║         ❌ MÓDULO ML REPROBADO - REQUIERE AJUSTES            ║
║                                                                ║
║ El modelo es discriminante (AUC > 0.60) pero NO es rentable.  ║
║ Profit Factor = 0 indica pérdida total en test set.           ║
║ Necesita optimización fundamental antes de aprobación.        ║
╚════════════════════════════════════════════════════════════════╝
```

---

## RECOMENDACIONES

### CRÍTICAS (Resolver primero)

1. **Optimizar Feature Engineering**
   - Agregar features de momentum más robustos
   - Crear features de cambio de tendencia (crossovers)
   - Incluir features de volatilidad relativa (Bollinger %B)
   - Probar normalización Z-score por feature

2. **Redefina el Target**
   - Aumentar ganancia mínima a 2% o 3% (1% muy bajo)
   - Incluir Stop Loss en definición: target=1 solo si TP sin SL
   - Probar horizontes diferentes (5, 15, 20 velas)
   - Crear múltiples targets (1%, 2%, 3% ganancias)

3. **Balanceo de Dataset**
   - Usar `class_weight='balanced'` en RandomForest
   - Probar SMOTE para oversampling de clase minoritaria
   - Ajustar threshold de predicción en función del recall

4. **Reducir Overfitting**
   - Aumentar `min_samples_leaf` (de 5 a 20)
   - Reducir `max_depth` (de 10 a 6-8)
   - Usar validación cruzada 5-fold
   - Implementar early stopping

### IMPORTANTES (Implementar después)

5. **Tuning de Hiperparámetros**
   ```python
   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [6, 8, 10, 12],
       'min_samples_leaf': [5, 10, 20],
       'threshold': [0.50, 0.60, 0.70, 0.80]
   }
   ```

6. **Mejorar Generación de Señales**
   - Aumentar umbral dinámicamente según confianza
   - Implementar "no trade" si probabilidad < 0.55
   - Usar ensambles: RandomForest + Gradient Boosting

7. **Análisis de Transacciones**
   - Estudiar por qué todos los trades van al SL
   - Verificar si entrada es temprana o el mercado es adverso
   - Análisis técnico manual de los 6 trades perdedores

### FUTURO (Iteraciones posteriores)

8. **Modelos Alternativos**
   - Gradient Boosting (XGBoost, LightGBM)
   - LSTM para secuencias temporales
   - Ensambles múltiples modelos

9. **Optimización de Riesgo**
   - Ajustar ATR multiplier (actualmente 2.0)
   - Implementar position sizing dinámico
   - Usar trailing stops en lugar de stops fijos

10. **Validación Robusta**
    - Walk-forward validation
    - Out-of-sample testing en datos 2023-2024
    - Prueba en otros pares (BTC/USDT, ADA/USDT)

---

## ARTEFACTOS GENERADOS

### Archivos de Código

```
✓ src/ml/feature_engineer.py (203 líneas)
  └─ Módulo completo de feature engineering y preparación

✓ scripts/phase2_ml_backtest.py (462 líneas)
  └─ Script completo de entrenamiento y backtesting

✓ ITERACION_14_REPORTE_ML_v1.md (Este archivo)
  └─ Reporte completo de auditoría
```

### Artefactos de Salida

```
results/
├─ metrics_ml_v1_20251104_164541.json       (986 bytes)
│  └─ Métricas del modelo y backtesting (JSON)
│
├─ feature_importance_ml_v1_20251104_164541.csv (265 bytes)
│  └─ Importancia de features para cada uno de los 8 features
│
└─ trades_log_ml_v1_20251104_164541.csv     (706 bytes)
   └─ Log detallado de 6 trades ejecutados

logs/
└─ phase2_ml.log (4.8 KB)
   └─ Log completo de ejecución con timestamps
```

### Estructura de Directorios Creada

```
BotDayTrading/
├─ src/
│  └─ ml/                              ← NUEVO
│     └─ feature_engineer.py           ← NUEVO
│
├─ scripts/
│  ├─ phase1_historical.py
│  ├─ phase2_backtest.py
│  ├─ phase2_ml_backtest.py            ← NUEVO
│  ├─ phase3_paper.py
│  ├─ phase4_live.py
│  └─ phase5_deployment.py
│
├─ results/                             ← NUEVO
│  ├─ metrics_ml_v1_20251104_164541.json
│  ├─ feature_importance_ml_v1_20251104_164541.csv
│  └─ trades_log_ml_v1_20251104_164541.csv
│
└─ ITERACION_14_REPORTE_ML_v1.md        ← ESTE ARCHIVO
```

### Cómo Reproducir

```bash
# Instalar dependencias (si no están)
pip install -r requirements.txt

# Ejecutar pipeline completo
python3 scripts/phase2_ml_backtest.py

# Ver resultados
cat logs/phase2_ml.log
cat results/metrics_ml_v1_*.json
cat results/trades_log_ml_v1_*.csv
```

---

## CONCLUSIONES Y PRÓXIMAS ACCIONES

### Logros Alcanzados

✅ Pipeline ML completo implementado y funcional
✅ Modelo RandomForest entrenado y evaluado
✅ AUC Score ≥ 0.60 (criterio de discriminación cumplido)
✅ Feature engineering robusto con 8 features relevantes
✅ Backtesting vectorizado con Stop Loss dinámico
✅ Reporting completo con métricas y logs

### Puntos Críticos Identificados

❌ Profit Factor = 0 (pérdida total)
❌ Win Rate = 0% (ningún trade ganador)
❌ Recall = 7.5% (muy pocas señales generadas)
❌ Degradación Train→Test (overfitting)
❌ Desbalance de dataset sin mitigación

### Próximos Pasos

**ITERACIÓN 15 (Optimización ML v2):**
1. Redefinir target (2-3% ganancia mínima)
2. Agregar features de momentum/reversión
3. Implementar class_weight='balanced'
4. Hacer grid search de hiperparámetros
5. Validación cruzada 5-fold
6. Backtesting en walk-forward validation

**Criterios de Éxito para Iteración 15:**
- AUC Score ≥ 0.65
- Profit Factor ≥ 1.5
- Win Rate ≥ 40%

---

## AUDITORÍA COMPLETADA

**Fecha:** 2025-11-04
**Auditor:** Claude Code (Quant-Auditor)
**Status:** ❌ **RECHAZADO - REQUIERE AJUSTES**

**Firma de Auditoría:**
```
Iteración: 14 (ML v1)
Commits: [Pendiente auditor]
Aprobación: [Pendiente dirección ejecutiva]
```

---

**Documento preparado para Quant-Auditor**
*Por favor, revisar y aprobar o solicitar correcciones*
