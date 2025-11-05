# Bot de Trading Algorítmico para Criptomonedas

Bot de day trading algorítmico para criptomonedas usando la API de Binance. Incluye backtesting, optimización de parámetros, paper trading, ejecución automática y modelos de IA.

## Características

- Descarga de datos históricos desde Binance
- Cálculo de indicadores técnicos (EMA, RSI, MACD, BB, ATR, Stochastic)
- Generación de señales de trading
- Backtesting vectorizado con comisiones y slippage
- Optimización de parámetros con Grid Search
- Paper trading en tiempo real (WebSockets)
- Trading automático con gestión de riesgo (OCO orders)
- Integración de modelos de IA (LSTM, sentiment analysis)
- Sistema robusto de logging

## Stack Técnico

- **Datos**: pandas, numpy
- **Visualización**: matplotlib
- **API Binance**: python-binance
- **Indicadores Técnicos**: pandas-ta
- **Machine Learning**: scikit-learn
- **Deep Learning**: tensorflow
- **NLP**: transformers
- **Configuración**: python-dotenv

## Estructura del Proyecto

```
BotDayTrading/
├── config/                     # Configuración
│   ├── config.json            # Tu configuración (credenciales)
│   └── config.example.json    # Ejemplo de configuración
├── src/                       # Código fuente modular
│   ├── data/                  # Obtención de datos
│   ├── indicators/            # Indicadores técnicos
│   ├── strategy/              # Estrategia y gestión de riesgo
│   ├── backtest/              # Motor de backtesting
│   ├── trading/               # Paper trading y trading real
│   ├── ai/                    # Modelos de IA
│   └── utils/                 # Utilidades (logger, metrics)
├── scripts/                   # Scripts ejecutables por fase
│   ├── phase1_historical.py   # Datos históricos
│   ├── phase2_backtest.py     # Backtesting y optimización
│   ├── phase3_paper.py        # Paper trading
│   ├── phase4_live.py         # Trading real
│   └── phase5_deployment.py   # Despliegue con IA
├── models/                    # Modelos ML guardados
├── data/                      # Datos en cache
├── logs/                      # Archivos de log
├── PHASE_GUIDE.md             # Guía detallada de cada fase
├── requirements.txt           # Dependencias
└── README.md                  # Este archivo
```

## Instalación

### 1. Clonar el repositorio

```bash
git clone <tu-repo>
cd BotDayTrading
```

### 2. Crear entorno virtual (recomendado)

```bash
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar credenciales

Copia el archivo de ejemplo y edita con tus credenciales:

```bash
cp config/config.example.json config/config.json
nano config/config.json  # O usa tu editor favorito
```

Añade tus credenciales de Binance:
```json
{
  "binance": {
    "api_key": "TU_API_KEY_AQUI",
    "api_secret": "TU_API_SECRET_AQUI",
    "testnet": true
  }
}
```

**IMPORTANTE**: Para las Fases 1-3 no necesitas credenciales de API (solo datos públicos).

## Uso por Fases

El bot sigue un desarrollo estricto de 5 fases. **No saltar fases**.

### FASE 1: Datos Históricos y Señales

Descarga datos de Binance y genera señales de trading.

```bash
python scripts/phase1_historical.py
```

**Salida esperada**:
- Datos OHLCV de BTCUSDT (5m, 1 año)
- Indicadores técnicos calculados
- Señales de COMPRA/VENTA/NEUTRAL
- Archivo guardado en `data/historical_data.csv`

**Sin riesgo**: Solo descarga y análisis de datos.

### FASE 2: Backtesting y Optimización

Encuentra los mejores parámetros para tu estrategia.

```bash
python scripts/phase2_backtest.py
```

**Salida esperada**:
- Grid search sobre diferentes combinaciones de parámetros
- Top 5 mejores configuraciones (ordenadas por Sharpe Ratio)
- Métricas detalladas (Sharpe, Drawdown, Win Rate, etc.)
- Parámetros óptimos guardados en `config/optimal_params.json`

**Sin riesgo**: Solo simulación con datos históricos.

### FASE 3: Paper Trading (Alertas en Vivo)

Conecta a datos en vivo pero sin ejecutar órdenes.

```bash
python scripts/phase3_paper.py
```

**Salida esperada**:
- Conexión a WebSocket de Binance
- Alertas en consola cuando aparecen señales de COMPRA/VENTA
- Validación de estrategia en tiempo real

**Sin riesgo**: Solo alertas, no ejecuta operaciones.

**Nota**: Esta fase requiere que hayas completado la Fase 2 (necesita `optimal_params.json`).

### FASE 4: Trading Real (Capital Mínimo)

Ejecuta órdenes reales con capital mínimo ($15 por operación).

```bash
python scripts/phase4_live.py
```

**ADVERTENCIA**: Esta fase opera con dinero real.

**Requisitos previos**:
1. Completar Fases 1-3
2. Configurar credenciales de API en `config.json`
3. Verificar que `testnet: true` para pruebas
4. Tener saldo mínimo en cuenta de Binance

**Características**:
- Gestión de riesgo con ATR
- Órdenes OCO automáticas (Take Profit + Stop Loss)
- Capital fijo de $15 por operación
- Límite de pérdida diaria (5% del capital)

**PRIMERO PRUEBA EN TESTNET**: Configura `"testnet": true` en `config.json`.

### FASE 5: Despliegue Completo con IA

Bot completo con modelos de IA en producción.

```bash
python scripts/phase5_deployment.py
```

**Características adicionales**:
- Predicciones con modelo LSTM
- Análisis de sentimiento de noticias
- Lógica de decisión combinada (Técnica + LSTM + Sentimiento)
- Auto-reconexión de WebSockets
- Logging robusto
- Listo para VPS

**Nota**: Requiere modelos pre-entrenados en `models/`.

## Gestión de Riesgo (Obligatorio)

### Reglas de Seguridad

1. **Capital por operación**: Máximo $15 (configurable en `config.json`)
2. **Stop Loss**: Siempre activo (2x ATR por defecto)
3. **Take Profit**: Siempre definido (3x ATR por defecto)
4. **Pérdida diaria máxima**: 5% del capital total
5. **Máximo posiciones abiertas**: 3 simultáneas

### Límites en Código

El `RiskManager` aplica estos límites automáticamente:
- Verifica pérdida diaria antes de cada operación
- Calcula TP/SL dinámicamente basado en ATR
- Formatea cantidades según requisitos de Binance
- Mantiene registro de PnL

## Configuración Avanzada

### Archivo `config/config.json`

```json
{
  "binance": {
    "api_key": "",
    "api_secret": "",
    "testnet": true
  },
  "trading": {
    "symbol": "BTCUSDT",
    "interval": "5m",
    "capital_per_trade": 15,
    "atr_sl_multiplier": 2.0,
    "atr_tp_multiplier": 3.0,
    "max_open_positions": 3,
    "max_daily_loss_pct": 0.05
  },
  "strategy": {
    "ema_short": 21,
    "ema_long": 50,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30
  },
  "ai": {
    "use_lstm": false,
    "use_sentiment": false
  }
}
```

### Parámetros Clave

- `testnet`: `true` para pruebas, `false` para producción
- `capital_per_trade`: Capital por operación en USD
- `atr_sl_multiplier`: Multiplicador de ATR para Stop Loss
- `atr_tp_multiplier`: Multiplicador de ATR para Take Profit
- `max_daily_loss_pct`: Pérdida diaria máxima (decimal)

## Obtener Credenciales de Binance

### Para Testnet (Recomendado para pruebas)

1. Ve a https://testnet.binance.vision/
2. Click en "Generate API Key"
3. Guarda tu API Key y Secret
4. Configura `"testnet": true` en `config.json`

### Para Producción (Dinero Real)

1. Inicia sesión en https://www.binance.com
2. Ve a Account > API Management
3. Crea una nueva API Key
4. Configura permisos: "Enable Spot & Margin Trading"
5. Añade restricción de IP (recomendado para VPS)
6. Guarda API Key y Secret
7. Configura `"testnet": false` en `config.json`

## Logs y Monitoreo

### Archivos de Log

- `logs/bot.log`: Log general del bot
- `logs/trades.log`: Log específico de operaciones

### Ver logs en tiempo real

```bash
tail -f logs/bot.log
```

### Formato de logs

```
2025-11-03 18:30:00 - TradingBot - INFO - Bot iniciado
2025-11-03 18:35:00 - TradingBot - INFO - SEÑAL: COMPRA @ $60500.50 | RSI: 45.6, MACD: 120.3
2025-11-03 18:35:01 - TradeLogger - INFO - BUY BTCUSDT - Precio: $60500.50, SL: $60200.00, TP: $60900.00
```

## Despliegue en VPS (Fase 5)

### Conectar a VPS

```bash
ssh root@tu-vps-ip
```

### Instalar dependencias

```bash
apt update && apt install python3 python3-pip git -y
```

### Clonar y configurar

```bash
git clone <tu-repo>
cd BotDayTrading
pip3 install -r requirements.txt
cp config/config.example.json config/config.json
nano config/config.json  # Configurar credenciales
```

### Ejecutar con Screen

```bash
screen -S trading_bot
python3 scripts/phase5_deployment.py

# Detach: Ctrl+A, luego D
# Reconectar: screen -r trading_bot
```

### Ejecutar con Nohup

```bash
nohup python3 scripts/phase5_deployment.py > output.log 2>&1 &

# Ver logs
tail -f logs/bot.log
```

## Troubleshooting

### Error: "Invalid API-key"

- Verifica que API_KEY y API_SECRET estén correctos
- Verifica que la IP esté en whitelist (si configuraste restricción)
- Para testnet, usa credenciales de testnet

### Error: "Insufficient balance"

- Verifica saldo en Binance
- Reduce `capital_per_trade` en configuración

### Error: "Filter failure: MIN_NOTIONAL"

- El valor de la operación es menor al mínimo de Binance ($10-15)
- Aumenta `capital_per_trade` a mínimo $15

### WebSocket desconectado

- La auto-reconexión debería manejar esto (Fase 5)
- Verifica conexión a internet
- Revisa logs para errores específicos

### Indicadores con valores NaN

- Normal en las primeras filas (warm-up period)
- El código usa `fillna(method='bfill')` para manejar esto
- Asegúrate de tener suficientes datos (mínimo 200 velas)

## Desarrollo y Testing

### Test de módulos individuales

Cada módulo tiene su propio `if __name__ == "__main__":` para testing:

```bash
python src/data/binance_client.py
python src/indicators/technical.py
python src/strategy/signal_generator.py
python src/backtest/engine.py
```

### Ejecutar tests

```bash
# Test de cliente de Binance
python -m src.data.binance_client

# Test de indicadores técnicos
python -m src.indicators.technical

# Test de generación de señales
python -m src.strategy.signal_generator
```

## Contribuir

1. Fork del repositorio
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Añadir nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crea un Pull Request

## Disclaimer

**ADVERTENCIA**: Este bot es una herramienta educativa y experimental. El trading algorítmico conlleva riesgos significativos de pérdida de capital.

- Nunca operes con capital que no puedas permitirte perder
- Prueba SIEMPRE en testnet antes de usar dinero real
- La gestión de riesgo es CRÍTICA para la supervivencia
- Los resultados pasados no garantizan resultados futuros
- El autor no se hace responsable de pérdidas financieras

## Licencia

MIT License - Ver archivo LICENSE para detalles

## Soporte

- Documentación detallada: `PHASE_GUIDE.md`
- Issues: [GitHub Issues]
- Preguntas: [Discussions]

## Roadmap

- [ ] Entrenar modelo LSTM con datos históricos
- [ ] Integrar API de noticias para sentiment analysis
- [ ] Añadir más estrategias (breakout, mean reversion)
- [ ] Dashboard web con métricas en tiempo real
- [ ] Backtesting walk-forward
- [ ] Multi-par trading
- [ ] Telegram bot para notificaciones

---

**Última actualización**: 2025-11-03

**Versión**: 1.0.0

**Autor**: Jhonathan

**Estado del proyecto**: ✅ Funcional (Fases 1-2 completas, Fases 3-5 en desarrollo)
