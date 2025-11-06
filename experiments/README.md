# 🧪 Directorio de Experimentación

Este directorio está dedicado al desarrollo y prueba de estrategias experimentales que aún no forman parte del pipeline oficial de 5 fases.

## 📁 Estructura

```
experiments/
├── strategies/          # Estrategias en desarrollo/prueba
├── ml_backtests/        # Scripts de Machine Learning experimentales
├── results/             # Resultados de experimentos
├── notebooks/           # Jupyter notebooks para análisis
└── README.md           # Este archivo
```

## 🎯 Propósito

**Este directorio es un sandbox para:**
- Probar nuevas estrategias de trading antes de integrarlas al pipeline oficial
- Experimentar con diferentes parámetros de indicadores
- Desarrollar modelos de Machine Learning
- Realizar análisis exploratorios de datos
- Comparar múltiples variantes de una estrategia

## ⚠️ Importante

### NO incluir en producción
- Los scripts aquí NO son parte del pipeline oficial (phase1-phase5)
- Los resultados son experimentales y no garantizan rentabilidad
- Código aquí puede ser inestable o incompleto

### Buenas prácticas
1. **Documentar experimentos**: Añade comentarios sobre qué intentas lograr
2. **Nombrar con versionado**: Usa sufijos como `_v1`, `_v2`, `_v15` para trackear iteraciones
3. **Guardar resultados**: Exporta métricas y logs a `experiments/results/`
4. **No modificar datos originales**: Usa copias de datos para experimentar

## 📊 Scripts ML Experimentales

### `ml_backtests/phase2_ml_backtest.py` (v1 - Original)
- **Propósito**: Primera implementación de ML con RandomForest
- **Líneas**: 472
- **Estado**: Base experimental, superada por v15

### `ml_backtests/phase2_ml_backtest_v15.py` (v15 - Optimizado)
- **Propósito**: Versión optimizada con mejores features
- **Líneas**: 511
- **Estado**: Mejor performance que v1

### `ml_backtests/phase2_ml_backtest_v16.py` (v16 - Lightweight)
- **Propósito**: Versión ligera y simplificada
- **Líneas**: 224
- **Estado**: Más rápido pero menos features

## 🔬 Flujo de Trabajo Recomendado

### 1. Desarrollo de Nueva Estrategia
```bash
# Crear archivo en experiments/strategies/
touch experiments/strategies/mi_estrategia_v1.py

# Desarrollar y probar
python experiments/strategies/mi_estrategia_v1.py
```

### 2. Backtest Experimental
```bash
# Usar scripts ML si aplica
python experiments/ml_backtests/phase2_ml_backtest_v16.py

# O crear tu propio backtest custom
python experiments/strategies/backtest_mi_estrategia.py
```

### 3. Análisis de Resultados
```bash
# Guardar resultados con timestamp
python mi_estrategia.py > experiments/results/mi_estrategia_$(date +%Y%m%d_%H%M%S).log

# Revisar métricas
cat experiments/results/metrics_*.json
```

### 4. Integración al Pipeline Oficial
Si tu estrategia demuestra ser rentable:
1. Mover código a `src/strategy/strategies/`
2. Integrar con el backtester oficial (`src/backtest/engine.py`)
3. Actualizar `scripts/phase2_backtest.py` para incluirla
4. Documentar en `PHASE_GUIDE.md`

## 📈 Métricas a Evaluar

Antes de promover una estrategia experimental a producción, asegúrate de que cumple:

- ✅ **Win Rate** > 40%
- ✅ **Sharpe Ratio** > 1.0
- ✅ **Profit Factor** > 1.5
- ✅ **Max Drawdown** < 20%
- ✅ **Número de trades** > 100 (validación estadística)

## 🚀 Tips de Desarrollo

### Usar datos históricos existentes
```python
import pandas as pd
from src.indicators.technical import agregar_indicadores

# Cargar datos ya descargados
df = pd.read_csv('ETHUSDT_15m_OHLCV_2025-11-05.csv')
df = agregar_indicadores(df)

# Experimentar con tu estrategia
# ...
```

### Comparar múltiples estrategias
```python
from src.backtest.engine import VectorizedBacktester

estrategias = [
    ('EMA Cross', generar_senales_ema),
    ('Triple Layer', generar_senales_triple_capa),
    ('Mi Estrategia', mi_nueva_estrategia),
]

for nombre, func in estrategias:
    df_signals = func(df)
    backtester = VectorizedBacktester(df_signals)
    results = backtester.run_backtest()
    print(f"{nombre}: ROI={results['roi']:.2f}%")
```

### Guardar configuración ganadora
```python
import json

# Si encuentras parámetros óptimos
optimal_params = {
    'ema_short': 9,
    'ema_long': 21,
    'rsi_threshold': 55
}

with open('experiments/results/optimal_params_mi_estrategia.json', 'w') as f:
    json.dump(optimal_params, f, indent=2)
```

## 📚 Recursos

- **Documentación oficial**: Ver `/PHASE_GUIDE.md`
- **Estrategias oficiales**: Ver `src/strategy/strategies/`
- **Ejemplos de backtest**: Ver `scripts/phase2_backtest.py`
- **Indicadores disponibles**: Ver `src/indicators/technical.py`

## 🤝 Contribuciones

Si desarrollas una estrategia exitosa:
1. Documenta claramente la lógica
2. Incluye resultados de backtest
3. Propón su integración al pipeline oficial
4. Comparte aprendizajes con el equipo

---

**Última actualización**: 2025-11-06
**Autor**: Equipo de Trading Algorítmico
