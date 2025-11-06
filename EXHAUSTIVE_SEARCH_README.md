# 🚀 BÚSQUEDA EXHAUSTIVA DE ESTRATEGIAS ÓPTIMAS

Sistema de optimización masiva que prueba **miles de combinaciones** de estrategias para encontrar aquellas que cumplan tus criterios específicos.

---

## 🎯 OBJETIVOS DE BÚSQUEDA

El script busca estrategias que cumplan **TODOS** estos criterios:

| Métrica | Objetivo |
|---------|----------|
| **Profit Factor** | >= 2.0 (ratio 1:2) |
| **Retorno Total** | >= 100% |
| **Max Drawdown** | <= 12% |
| **Número de Trades** | > 220 |
| **Win Rate** | >= 35% |
| **Sharpe Ratio** | >= 0.5 |

---

## 📊 COMBINACIONES A PROBAR

### **Total estimado: ~10,000+ estrategias**

### **1. Filtros de Régimen (Tendencia)**
- **Tipos**: EMA, SMA, None
- **Períodos**: 50, 100, 150, 200
- **Dirección**: Long-Only, Short-Only, Hybrid
- **Combinaciones**: 3 × 4 × 3 = **36**

### **2. Indicadores de Entrada**
- **EMA Cross**: 3 fast × 3 slow = 9 combinaciones
- **RSI**: 2 oversold × 2 overbought = 4 combinaciones
- **MACD**: 1 configuración estándar
- **Bollinger Bands**: 1 configuración estándar
- **Donchian**: 2 períodos
- **Combinaciones de indicadores**: 11 grupos diferentes
  - `['ema_cross']`
  - `['ema_cross', 'rsi']`
  - `['ema_cross', 'macd']`
  - `['ema_cross', 'rsi', 'macd']`
  - `['macd']`
  - `['macd', 'rsi']`
  - `['donchian']`
  - `['donchian', 'rsi']`
  - `['bb']`
  - `['bb', 'rsi']`
  - `['ema_cross', 'bb']`

### **3. Filtros Adicionales**
- **Volumen**: Con/Sin filtro
- **ATR (Volatilidad)**: Con/Sin filtro, 2 umbrales
- **Combinaciones**: 2 × 4 = **8**

### **4. Gestión de Riesgo (SL:TP)**
- **Ratios probados**: 8 combinaciones
  - SL 1.5, TP 3.0 (ratio 1:2)
  - SL 1.5, TP 4.5 (ratio 1:3)
  - SL 2.0, TP 4.0 (ratio 1:2)
  - SL 2.0, TP 6.0 (ratio 1:3)
  - SL 2.5, TP 5.0 (ratio 1:2)
  - SL 2.5, TP 7.5 (ratio 1:3)
  - SL 3.0, TP 6.0 (ratio 1:2)
  - SL 3.0, TP 9.0 (ratio 1:3)
- **ATR Periods**: 14, 20

**Total aproximado**: 36 × 11 × 9 × 8 × 8 × 2 = **~50,000 combinaciones teóricas**

*(El script optimiza y reduce a ~10,000 combinaciones prácticas)*

---

## 🚀 CÓMO EJECUTAR

### **1. Asegúrate de tener las dependencias instaladas**

```bash
pip install -r requirements.txt
```

### **2. Navega al directorio del proyecto**

```bash
cd /path/to/DayTrading
```

### **3. Ejecuta el script**

```bash
python3 scripts/exhaustive_search.py
```

### **4. Confirma la ejecución**

El script te preguntará si deseas continuar:

```
📊 Combinaciones totales a probar: 10,368
⏱️  Tiempo estimado: 17.3 minutos (asumiendo 10 tests/sec)

¿Deseas continuar? (y/n):
```

Escribe `y` y presiona Enter.

---

## ⏱️ TIEMPO DE EJECUCIÓN

- **Velocidad estimada**: 10-20 tests/segundo
- **10,000 estrategias**: ~10-20 minutos
- **50,000 estrategias**: ~40-80 minutos

*Nota: Depende del hardware. En CPU potente puede ser más rápido.*

---

## 📂 ARCHIVOS GENERADOS

Al finalizar, encontrarás estos archivos en la carpeta `results/`:

### **1. `exhaustive_search_all.csv`**
- **Contiene**: Todas las estrategias probadas
- **Columnas**: ~40 columnas con parámetros y métricas
- **Uso**: Análisis completo, identificar patrones

### **2. `exhaustive_search_best.csv`** ⭐
- **Contiene**: Estrategias que cumplen **TODOS** los criterios
- **Filtros aplicados**:
  - Profit Factor >= 2.0
  - Return >= 100%
  - Max DD <= 12%
  - Trades > 220
  - Win Rate >= 35%
  - Sharpe >= 0.5
- **Uso**: Estas son las **estrategias ganadoras**

### **3. `exhaustive_search_relaxed.csv`**
- **Contiene**: Estrategias con criterios relajados (si no hay resultados perfectos)
- **Filtros aplicados**:
  - Profit Factor >= 1.5
  - Return >= 50%
  - Max DD <= 15%
  - Trades > 150
  - Win Rate >= 30%
  - Sharpe >= 0.3
- **Uso**: Alternativas si no se encuentran estrategias perfectas

---

## 📊 INTERPRETACIÓN DE RESULTADOS

### **Ejemplo de salida en consola:**

```
🏆 TOP 10 MEJORES ESTRATEGIAS
================================================================================

#4523 - PF: 2.15 | Return: 127.3% | DD: 10.2%
   Config: EMA(150) + ema_cross, rsi
   SL:TP = 2.0:6.0 | Trades: 245 | WR: 38.4%

#7891 - PF: 2.08 | Return: 115.8% | DD: 11.5%
   Config: SMA(100) + macd, rsi
   SL:TP = 2.5:7.5 | Trades: 267 | WR: 36.7%
```

### **Qué significan las métricas:**

- **PF (Profit Factor)**: Ratio de ganancias brutas / pérdidas brutas
  - PF 2.0 = Ganas $2 por cada $1 que pierdes
- **Return**: Retorno total porcentual sobre $10,000 iniciales
- **DD (Max Drawdown)**: Pérdida máxima desde un pico
- **Config**: Filtro de régimen + indicadores usados
- **SL:TP**: Multiplicadores de ATR para Stop Loss y Take Profit
- **Trades**: Número total de operaciones
- **WR (Win Rate)**: Porcentaje de trades ganadores

---

## 🔧 PERSONALIZAR LA BÚSQUEDA

### **Cambiar criterios de filtrado:**

Edita el archivo `scripts/exhaustive_search.py`, líneas 37-44:

```python
CRITERIA = {
    'min_profit_factor': 2.0,      # Cambia a 1.8 si quieres ser menos estricto
    'min_return_pct': 100.0,       # Cambia a 80.0 para menos rentabilidad
    'max_drawdown_pct': 12.0,      # Cambia a 15.0 para tolerar más riesgo
    'min_num_trades': 220,         # Cambia a 150 para menos trades
    'min_win_rate_pct': 35.0,      # Cambia a 30.0 para menos Win Rate
    'min_sharpe_ratio': 0.5,       # Cambia a 0.3 para menos Sharpe
}
```

### **Cambiar activo o timeframe:**

Líneas 27-29:

```python
SYMBOL = 'ETHUSDT'      # Cambia a 'BTCUSDT', 'BNBUSDT', etc.
INTERVAL = '15m'        # Cambia a '5m', '1h', etc.
START_DATE = '365 days ago UTC'  # Cambia a '180 days ago UTC', etc.
```

### **Agregar más combinaciones:**

Edita el diccionario `STRATEGY_GRID` (líneas 47-115) para agregar más valores a probar.

---

## 🐛 SOLUCIÓN DE PROBLEMAS

### **Error: `ModuleNotFoundError: No module named 'pandas'`**
```bash
pip install pandas numpy scikit-learn pandas-ta
```

### **Error: `AttributeError: 'DataFrame' object has no attribute 'ta'`**
```bash
pip install pandas-ta
```

### **Error: `Columna 'ATRr_14' no encontrada`**
Asegúrate de que `agregar_indicadores()` calcula todos los indicadores necesarios.

### **El script es demasiado lento**
- Reduce el número de combinaciones en `STRATEGY_GRID`
- Usa menos períodos de EMA
- Usa menos ratios SL:TP

---

## 📈 QUÉ HACER DESPUÉS

### **Si encuentras estrategias ganadoras:**

1. **Validar en datos out-of-sample**
   - Prueba en periodo más reciente no usado en búsqueda
   - Prueba en otro activo (BTC, BNB)

2. **Walk-Forward Optimization**
   - Divide datos en 3 periodos
   - Optimiza en periodo 1, valida en periodo 2, confirma en periodo 3

3. **Paper Trading (Fase 3)**
   - Implementa en `phase3_paper.py`
   - Prueba con datos en vivo sin riesgo

4. **Deployment (Fase 4-5)**
   - Si todo funciona, avanza a trading real

### **Si NO encuentras estrategias que cumplan criterios:**

El script generará automáticamente resultados con **criterios relajados**. Analiza:

- ¿Qué métricas están más cerca del objetivo?
- ¿Qué patrones tienen las mejores estrategias?
- ¿Debes ajustar los criterios o probar otros indicadores?

---

## 📞 SOPORTE

Si tienes problemas, revisa:

1. `logs/` - Archivos de log con errores detallados
2. `CLAUDE.md` - Documentación del proyecto
3. `README.md` - Información general

---

## ⚠️ ADVERTENCIAS IMPORTANTES

1. **Overfitting**: Probar muchas estrategias aumenta el riesgo de encontrar patrones por azar
   - **Solución**: Siempre valida en datos fuera de muestra
   - **Solución**: Usa Walk-Forward Optimization

2. **No hay garantías**: Resultados pasados no garantizan rendimiento futuro
   - Mercados cambian constantemente
   - Estrategia puede degradarse con el tiempo

3. **Uso responsable**: Este es un proyecto educativo
   - No arriesgues dinero que no puedes perder
   - Usa siempre testnet primero
   - Empieza con capital pequeño

---

## 🎉 BUENA SUERTE

¡Que encuentres la estrategia perfecta! 🚀

Si encuentras resultados interesantes, considera documentarlos para análisis futuro.

---

**Autor**: Claude Code
**Fecha**: 2025-11-06
**Versión**: 1.0
