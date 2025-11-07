# ESTRATEGIA HÍBRIDA DE DAY TRADING - 4 CAPAS

## Descripción General

La **Estrategia Híbrida v1** es un sistema de trading algorítmico **Long-Only** (solo posiciones largas) diseñado para day trading de criptomonedas. Combina 4 capas de análisis técnico para generar señales de alta probabilidad:

1. **Filtro de Régimen** (Tendencia Macro) - EMA(200)
2. **Filtro de Momentum** (Fuerza del Mercado) - RSI(14)
3. **Señal de Entrada/Salida** (Timing Preciso) - MACD(12,26,9)
4. **Gestión de Riesgo** (Stop Loss Dinámico) - ATR(14)

---

## Filosofía de la Estrategia

La estrategia solo opera cuando **todas las 4 capas están alineadas**, reduciendo significativamente las falsas señales y operando únicamente en condiciones de mercado favorables:

✅ **Régimen correcto** (precio > EMA_200)
✅ **Momentum alcista confirmado** (RSI > 50)
✅ **Timing de entrada apropiado** (cruce alcista MACD)
✅ **Gestión de riesgo adaptativa** (ATR Stop Loss dinámico)

---

## Arquitectura de 4 Capas

### CAPA 1: Filtro de Régimen (EMA_200)

**Objetivo**: Identificar la tendencia macro y operar solo en mercados alcistas.

**Indicador**: EMA de 200 períodos (Exponential Moving Average)

**Lógica**:
- **Régimen ALCISTA**: `precio > EMA_200` → Permite abrir posiciones LONG
- **Régimen BAJISTA**: `precio < EMA_200` → Fuera del mercado (no operar)

**Rationale**: La EMA(200) actúa como un filtro de tendencia de largo plazo. Operar por encima de esta media asegura que estamos siguiendo la tendencia principal del mercado, evitando operar contra la corriente.

---

### CAPA 2: Filtro de Momentum (RSI)

**Objetivo**: Confirmar que existe momentum alcista antes de entrar.

**Indicador**: RSI de 14 períodos (Relative Strength Index)

**Lógica**:
- **Entrada permitida**: `RSI > rsi_momentum_level` (default: 50)
- **Entrada bloqueada**: `RSI < rsi_momentum_level`

**Rationale**: No queremos comprar en debilidad. El RSI > 50 confirma que el mercado tiene fuerza alcista. Esto evita entrar en momentos de debilidad donde el precio podría seguir cayendo.

**Nota crítica**: En iteraciones anteriores (Iteración 10.1) se intentó "comprar debilidad" (RSI < 30) con **Win Rate 0%**. Esta estrategia híbrida invierte esa lógica: compramos **fuerza**, no debilidad.

---

### CAPA 3: Señal de Entrada/Salida (MACD)

**Objetivo**: Timing preciso de entrada y salida basado en cruces de momentum.

**Indicador**: MACD(12, 26, 9) - Moving Average Convergence Divergence

**Lógica de COMPRA (abrir posición LONG)**:
```
Cruce alcista del MACD:
- MACD[-1] < Signal[-1]  (MACD anterior estaba por debajo de su señal)
- MACD >= Signal         (MACD actual cruza hacia arriba)
```

**Lógica de VENTA (cerrar posición LONG / Take Profit)**:
```
Cruce bajista del MACD:
- MACD[-1] > Signal[-1]  (MACD anterior estaba por encima de su señal)
- MACD <= Signal         (MACD actual cruza hacia abajo)
```

**Rationale**: El MACD es un indicador de momentum que captura cambios en la tendencia. El cruce alcista indica inicio de momentum alcista, mientras que el cruce bajista indica pérdida de momentum (momento de tomar ganancias).

---

### CAPA 4: Gestión de Riesgo (ATR Stop Loss)

**Objetivo**: Proteger el capital con Stop Loss dinámico adaptado a la volatilidad.

**Indicador**: ATR de 14 períodos (Average True Range)

**Lógica**:
```
Stop Loss (SL) = Precio_Entrada - (ATR × atr_multiplier)

Donde:
- atr_multiplier es un parámetro configurable (default: 2.0)
- ATR mide la volatilidad actual del mercado
```

**Ventajas del Stop Loss Dinámico**:
- En mercados **volátiles** (ATR alto): SL más amplio → Evita stop outs prematuros
- En mercados **tranquilos** (ATR bajo): SL más ajustado → Protege mejor el capital
- **Adaptativo**: Se ajusta automáticamente a las condiciones del mercado

**Implementación**: El Stop Loss se calcula y verifica en el motor de backtesting (`src/backtest/engine.py`), no en el generador de señales.

---

## Generación de Señales - Lógica de Confluencia

La estrategia genera 3 tipos de señales:

| Señal | Valor | Significado | Condiciones |
|-------|-------|-------------|-------------|
| **COMPRA** | `1` | Abrir posición LONG | Todas las capas alineadas: (Régimen alcista) AND (RSI > nivel) AND (Cruce alcista MACD) |
| **VENTA** | `-1` | Cerrar posición LONG | Cruce bajista MACD (Take Profit / Stop de tendencia) |
| **NEUTRAL** | `0` | Sin acción | No se cumplen las condiciones de entrada/salida |

### Pseudocódigo de Confluencia:

```python
# SEÑAL DE COMPRA (1) - Confluencia de 4 capas
if (precio > EMA_200) AND            # CAPA 1: Régimen alcista
   (RSI > rsi_momentum_level) AND    # CAPA 2: Momentum confirmado
   (MACD cruza hacia arriba Signal): # CAPA 3: Timing de entrada
    señal = 1  # COMPRA

# SEÑAL DE VENTA (-1) - Proteger ganancias
if (MACD cruza hacia abajo Signal):  # CAPA 3: Pérdida de momentum
    señal = -1  # VENTA

# SEÑAL NEUTRAL (0)
else:
    señal = 0  # NEUTRAL (esperar)
```

---

## Parámetros de Configuración

### Parámetros por Defecto (config.json)

```json
{
  "strategy": {
    "ema_trend": 200,          // EMA de tendencia (Capa 1)
    "rsi_period": 14,          // Período del RSI (Capa 2)
    "rsi_momentum_level": 50,  // Nivel mínimo de RSI para entrar (Capa 2)
    "macd_fast": 12,           // MACD rápido (Capa 3)
    "macd_slow": 26,           // MACD lento (Capa 3)
    "macd_signal": 9,          // MACD señal (Capa 3)
    "atr_length": 14           // ATR para Stop Loss (Capa 4)
  },
  "risk": {
    "atr_sl_multiplier": 2.0,  // Multiplicador para Stop Loss
    "capital_per_trade": 15,   // Capital por operación ($)
    "max_open_positions": 3,   // Máximo de posiciones abiertas simultáneas
    "max_daily_loss_pct": 0.05 // Límite de pérdida diaria (5%)
  }
}
```

### Parámetros Optimizables

Los siguientes parámetros pueden optimizarse mediante grid search (Fase 2):

- `rsi_momentum_level`: Nivel mínimo de RSI (valores típicos: 45, 50, 55)
- `atr_sl_multiplier`: Multiplicador del Stop Loss (valores típicos: 1.5, 2.0, 2.5)
- `macd_fast`, `macd_slow`, `macd_signal`: Períodos del MACD (si se desea ajustar fino)

---

## Comparación con Estrategias Anteriores

| Iteración | Estrategia | Tipo | Win Rate | Sharpe Ratio | Return | Diagnóstico |
|-----------|------------|------|----------|--------------|--------|-------------|
| **10.1** | Estocástico (comprar debilidad) | Long-Only | **0%** | N/A | -100% | ❌ Comprar debilidad no funciona |
| **11.1** | Donchian Breakout (momentum) | Long-Only | **5%** | N/A | N/A | ❌ Baja frecuencia, pocos trades |
| **12** | **Híbrida 4 Capas** | Long-Only | **27.51%** | **-0.12** | **-33.30%** | ⚠️ Mejora en Win Rate pero aún no rentable |
| **13** | Bajista Invertida (Short-Only) | Short-Only | TBD | TBD | TBD | 🔄 En evaluación |

**Conclusión**: La estrategia híbrida (Iteración 12) mostró mejoras significativas en Win Rate (27.51%) comparado con iteraciones anteriores, pero aún no es rentable. Se requiere optimización de parámetros o ajustes en la gestión de riesgo.

---

## Implementación Técnica

### Módulo de Código

La estrategia está implementada en:
```
src/strategy/signal_generator.py
```

Función principal:
```python
def generar_senales_hibrido_v1(df, config=None):
    """
    Genera señales de trading usando ESTRATEGIA HÍBRIDA DE 4 CAPAS.

    Args:
        df: DataFrame con indicadores calculados (EMA_200, RSI_14, MACD, ATR)
        config: Diccionario con parámetros de estrategia

    Returns:
        DataFrame con columnas 'señal' y 'position' añadidas
    """
```

### Columnas Requeridas en el DataFrame

Antes de llamar a `generar_senales_hibrido_v1()`, el DataFrame debe contener:

- `close`: Precio de cierre
- `EMA_200`: EMA de 200 períodos (Capa 1)
- `RSI_14`: RSI de 14 períodos (Capa 2)
- `MACD_12_26_9`: Línea MACD (Capa 3)
- `MACDs_12_26_9`: Línea de señal MACD (Capa 3)
- `ATRr_14`: ATR de 14 períodos (Capa 4) - Nota: pandas-ta usa 'ATRr' en lugar de 'ATR'

Estas columnas se generan automáticamente con:
```python
from src.indicators.technical import agregar_indicadores
df = agregar_indicadores(df, config=config)
```

---

## Uso y Ejecución

### Fase 1: Análisis Histórico (Sin Riesgo)

```bash
# Descargar datos históricos y generar señales
python scripts/phase1_historical.py
```

### Fase 2: Backtesting con Estrategia Híbrida (Sin Riesgo)

```bash
# Ejecutar backtest y optimización de parámetros
python scripts/phase2_hibrido_v1.py
```

Este script:
1. Descarga datos históricos de ETH/USDT (15m, 1 año)
2. Calcula indicadores técnicos
3. Genera señales con `generar_senales_hibrido_v1()`
4. Ejecuta backtest vectorizado con Stop Loss ATR
5. Optimiza parámetros (`rsi_momentum_level`, `atr_multiplier`) mediante grid search
6. Muestra métricas de rendimiento (Sharpe, Win Rate, Drawdown, etc.)
7. Guarda parámetros óptimos en `config/optimal_params.json`

### Fase 3: Paper Trading (Sin Riesgo - Solo Alertas)

```bash
# Trading en papel con datos en vivo
python scripts/phase3_paper.py
```

### Fase 4: Live Trading (RIESGO REAL)

```bash
# Trading con dinero real (requiere API keys)
python scripts/phase4_live.py
```

---

## Gestión de Riesgo y Límites

La estrategia implementa múltiples capas de protección de capital:

### 1. Stop Loss Dinámico (ATR)
```
SL = Precio_Entrada - (ATR × atr_multiplier)
```
- Se verifica en cada candle: `if df['low'] <= stop_loss`
- Si se toca el SL, la posición se cierra inmediatamente

### 2. Capital por Operación
- **Default**: $15 por trade
- **Rationale**: MIN_NOTIONAL de Binance es ~$10-15
- **Configurable**: `capital_per_trade` en `config.json`

### 3. Límite de Posiciones Abiertas
- **Default**: Máximo 3 posiciones simultáneas
- **Rationale**: Evitar sobreexposición al mercado
- **Configurable**: `max_open_positions` en `config.json`

### 4. Límite de Pérdida Diaria
- **Default**: 5% de pérdida diaria máxima
- **Rationale**: Protección contra días catastróficos
- **Configurable**: `max_daily_loss_pct` en `config.json`

---

## Ventajas y Limitaciones

### ✅ Ventajas

1. **Multi-Capa**: Reduce falsas señales mediante confluencia de múltiples indicadores
2. **Adaptativo**: Stop Loss dinámico se ajusta a la volatilidad del mercado
3. **Seguimiento de Tendencia**: Solo opera en régimen alcista (precio > EMA_200)
4. **Momentum Confirmado**: Filtra entradas cuando no hay fuerza (RSI < 50)
5. **Timing Preciso**: MACD proporciona puntos de entrada/salida claros
6. **Backtesteable**: Estrategia completamente vectorizada para backtesting rápido

### ⚠️ Limitaciones

1. **Mercados Laterales**: Puede generar señales falsas en rangos sin tendencia
2. **Lag de Indicadores**: EMA(200) y MACD tienen retraso inherente
3. **No Rentable (aún)**: Win Rate 27.51% necesita mejoras (objetivo: >40%)
4. **Solo Long**: No aprovecha tendencias bajistas (versión Short-Only en desarrollo)
5. **Requiere Optimización**: Parámetros por defecto no son óptimos para todos los activos

---

## Próximos Pasos y Mejoras

### Optimización de Parámetros (Fase 2)
- Ejecutar grid search con más valores de `rsi_momentum_level`
- Probar diferentes `atr_multipliers` para Stop Loss
- Evaluar períodos alternativos de MACD (ej: 5,35,5 para timeframes cortos)

### Filtros Adicionales
- **Filtro de Volatilidad**: Evitar operar cuando ATR es demasiado bajo (mercado lateral)
- **Filtro de Volumen**: Solo entrar cuando hay volumen suficiente
- **Multi-Timeframe**: Confirmar tendencia en timeframe superior (1h, 4h)

### Gestión de Riesgo Avanzada
- **Trailing Stop**: Mover Stop Loss a break-even después de X% de ganancia
- **Take Profit parcial**: Cerrar 50% en TP1, dejar 50% correr con trailing stop
- **Position Sizing Dinámico**: Ajustar tamaño según volatilidad (Kelly Criterion)

### Integración con IA (Fase 5)
- **LSTM Price Prediction**: Añadir predicción de precio como filtro adicional
- **Sentiment Analysis**: Confirmar señales con análisis de noticias/Twitter
- **Reinforcement Learning**: Entrenar agente RL para optimizar timing de salida

---

## Conclusión

La **Estrategia Híbrida de 4 Capas** representa un avance significativo sobre estrategias de indicador único. Al combinar filtros de régimen, momentum y timing, reduce drásticamente las falsas señales y opera solo en condiciones favorables.

Si bien aún no es rentable en su forma actual (Win Rate 27.51%, Sharpe -0.12), proporciona una base sólida para optimización y mejoras incrementales. La arquitectura modular permite añadir capas adicionales (volumen, volatilidad, IA) sin romper la lógica existente.

**Filosofía clave**: En trading algorítmico, menos es más. No generar señales es mejor que generar señales perdedoras. Esta estrategia prioriza calidad sobre cantidad.

---

## Referencias y Recursos

- **Código fuente**: `src/strategy/signal_generator.py` (línea 373-539)
- **Documentación del proyecto**: `CLAUDE.md`
- **Guía de fases**: `PHASE_GUIDE.md`
- **Configuración**: `config/config.json`
- **Resultados de backtest**: `backtest_output_v12_hibrido.log`

---

**Versión**: 1.0
**Fecha**: 2025
**Autor**: Bot Day Trading Project
**Iteración**: 12 (Módulo Híbrido v1)
