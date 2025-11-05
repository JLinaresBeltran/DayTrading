# Guía de Fases del Bot de Trading Algorítmico

Esta guía detalla las 5 fases del desarrollo del bot de trading algorítmico para day trading de criptomonedas usando la API de Binance.

## Stack Técnico Confirmado

- **Análisis de Datos**: pandas, numpy, matplotlib
- **API Binance**: python-binance (REST API y WebSockets)
- **Indicadores Técnicos**: pandas-ta
- **Machine Learning**: scikit-learn, tensorflow
- **Análisis de Sentimiento**: transformers
- **Configuración**: python-dotenv

---

## FASE 1: Lógica y Datos Históricos

### Objetivo
Definir la estrategia de trading y obtener datos históricos a través de la API REST de Binance para realizar backtesting.

### Componentes a Implementar

#### 1. `src/data/binance_client.py`
- Configurar el cliente de Binance (`Client` de `python-binance`)
- No requiere API key para datos públicos
- Métodos para conexión y validación

#### 2. `src/data/data_fetcher.py`
- **Función**: `obtener_datos_binance(simbolo, intervalo, inicio)`
  - Parámetros:
    - `simbolo`: String (ej. 'BTCUSDT')
    - `intervalo`: Constante de Binance (ej. `Client.KLINE_INTERVAL_5MINUTE`)
    - `inicio`: String (ej. "1 year ago UTC")
  - Proceso:
    1. Llamar a `client.get_historical_klines()`
    2. Procesar respuesta JSON
    3. Crear DataFrame con columnas: `timestamp`, `open`, `high`, `low`, `close`, `volume`
    4. Convertir `timestamp` a datetime
    5. Convertir precios/volumen a float
  - Retorna: DataFrame de pandas limpio

#### 3. `src/indicators/technical.py`
- **Función**: `agregar_indicadores(df)`
  - Usa la extensión `df.ta` de pandas-ta
  - Indicadores a calcular (con `append=True`):
    - **EMA**: Longitudes 21 y 50 (`df.ta.ema(length=21, append=True)`)
    - **RSI**: Longitud 14 (`df.ta.rsi(length=14, append=True)`)
    - **Bollinger Bands**: Longitud 20, std 2 (`df.ta.bbands(length=20, std=2, append=True)`)
    - **MACD**: 12, 26, 9 (`df.ta.macd(fast=12, slow=26, signal=9, append=True)`)
    - **ATR**: Longitud 14 (`df.ta.atr(length=14, append=True)`)
    - **Stochastic**: 14, 3, 3 (`df.ta.stoch(k=14, d=3, smooth_k=3, append=True)`)
  - Retorna: DataFrame con indicadores añadidos

#### 4. `src/strategy/signal_generator.py`
- **Función**: `generar_señales(df)`
  - Crea columna `señal` con valores:
    - `1`: COMPRA
    - `-1`: VENTA
    - `0`: NADA (neutral)
  - Lógica de ejemplo:
    - **COMPRA (1)**: `EMA_21 > EMA_50` Y `RSI_14 < 70`
    - **VENTA (-1)**: `EMA_21 < EMA_50` Y `RSI_14 > 30`
    - **NADA (0)**: Resto de casos
  - Retorna: DataFrame con columna `señal`

#### 5. `scripts/phase1_historical.py`
Script principal ejecutable:
```python
# Pseudocódigo
1. Configurar Client() de Binance
2. Llamar a obtener_datos_binance('BTCUSDT', '5m', '1 year ago UTC')
3. Llamar a agregar_indicadores(df)
4. Llamar a generar_señales(df)
5. Imprimir df.tail(20) para verificar
```

### Salida Esperada
DataFrame con columnas:
- Datos OHLCV originales
- Indicadores técnicos (EMA_21, EMA_50, RSI_14, BBL_20_2.0, BBM_20_2.0, BBU_20_2.0, MACD_12_26_9, ATR_14, STOCHk_14_3_3, STOCHd_14_3_3)
- Columna `señal` (1, -1, 0)

### Verificación
- Ejecutar: `python scripts/phase1_historical.py`
- Debe mostrar últimos 20 registros con todas las columnas
- Sin errores de tipo de datos

---

## FASE 2: Backtesting y Optimización

### Objetivo
Construir un motor de backtesting robusto que simule comisiones y slippage, y optimizar los parámetros de la estrategia.

### Componentes a Implementar

#### 1. `src/backtest/engine.py`
- **Clase**: `VectorizedBacktester`
  - **Constructor** `__init__(df, capital_inicial, comision)`:
    - `df`: DataFrame con señales (1, -1, 0)
    - `capital_inicial`: Float (ej. 10000)
    - `comision`: Float (ej. 0.00075 para Binance = 0.075%)

  - **Método**: `run_backtest()`:
    ```python
    1. Calcular retornos: df['returns'] = df['close'].pct_change()
    2. Calcular retornos de estrategia: df['strategy_returns'] = df['señal'].shift(1) * df['returns']
    3. Detectar cambios de señal (operaciones): df['trade'] = df['señal'].diff().abs() > 0
    4. Restar comisiones en cada operación: df.loc[df['trade'], 'strategy_returns'] -= comision
    5. Calcular PnL acumulado: df['cumulative_returns'] = (1 + df['strategy_returns']).cumprod()
    6. Calcular valor del portafolio: df['portfolio_value'] = capital_inicial * df['cumulative_returns']
    ```

  - **Método**: `calculate_metrics()`:
    - Retorna diccionario con:
      - **Ganancia Neta**: `portfolio_value.iloc[-1] - capital_inicial`
      - **Max Drawdown**: Máxima caída desde el pico más alto
      - **Ratio de Sharpe**: `(returns.mean() / returns.std()) * sqrt(252)` (anualizado)

#### 2. `src/backtest/optimizer.py`
- **Función**: `run_strategy_backtest(params)`
  - Parámetros: Diccionario `{'ema_corta': 21, 'ema_larga': 50, 'rsi_periodo': 14}`
  - Proceso:
    1. Obtener datos históricos
    2. Calcular indicadores con parámetros personalizados
    3. Generar señales con parámetros personalizados
    4. Ejecutar backtest
    5. Retornar métricas

- **Función**: `optimize_parameters(param_grid)`
  - Usa `sklearn.model_selection.ParameterGrid`
  - Ejemplo de grid:
    ```python
    {
        'ema_corta': [10, 21, 30],
        'ema_larga': [50, 100, 200],
        'rsi_periodo': [10, 14, 20]
    }
    ```
  - Itera sobre todas las combinaciones
  - Guarda resultados (parámetros + Sharpe)
  - Retorna top 5 mejores combinaciones ordenadas por Sharpe

#### 3. `scripts/phase2_backtest.py`
Script principal ejecutable:
```python
# Pseudocódigo
1. Definir param_grid
2. Llamar a optimize_parameters(param_grid)
3. Imprimir top 5 mejores resultados
4. Guardar mejores parámetros en config/optimal_params.json
```

### Salida Esperada
```
Top 5 Mejores Estrategias:
1. EMA(21,50) RSI(14) - Sharpe: 1.85, Ganancia: $2,340, Drawdown: -12.5%
2. EMA(10,100) RSI(14) - Sharpe: 1.62, Ganancia: $1,890, Drawdown: -15.2%
...
```

### Verificación
- Ejecutar: `python scripts/phase2_backtest.py`
- Debe mostrar resultados ordenados por Sharpe
- Guardar parámetros óptimos para Fase 3

---

## FASE 3: Paper Trading y Alertas en Vivo

### Objetivo
Conectar el bot a datos en vivo usando WebSockets de Binance, pero sin operar. El bot solo genera alertas para validación manual.

### Componentes a Implementar

#### 1. `src/trading/paper_trader.py`
- **Clase**: `PaperTrader`
  - **Atributos**:
    - `symbol`: String ('BTCUSDT')
    - `interval`: String ('5m')
    - `optimal_params`: Diccionario con parámetros óptimos de Fase 2
    - `df`: DataFrame con contexto histórico (últimas 500 velas)

  - **Método**: `initialize_context()`:
    - Descarga últimas 500 velas vía REST API
    - Calcula todos los indicadores
    - **CRUCIAL**: Este "priming" proporciona contexto histórico para cálculos correctos

  - **Método**: `handle_kline_message(msg)`:
    ```python
    1. Verificar si vela está cerrada: if msg['k']['x']:
    2. Extraer datos: timestamp, open, high, low, close, volume
    3. Añadir nueva fila al DataFrame
    4. Eliminar primera fila: df = df.iloc[1:]  # Mantener tamaño constante
    5. Recalcular TODOS los indicadores (pandas-ta es rápido)
    6. Aplicar generar_señales() solo a última fila
    7. Obtener señales: señal_actual = df['señal'].iloc[-1]
                        señal_anterior = df['señal'].iloc[-2]
    8. Si señal_actual != señal_anterior:
         - Imprimir alerta detallada con precio, RSI, MACD, etc.
    ```

  - **Método**: `start()`:
    - Inicializar `BinanceSocketManager`
    - Crear `kline_socket` con callback `handle_kline_message`
    - Iniciar socket: `socket.start()`

#### 2. `scripts/phase3_paper.py`
Script principal ejecutable:
```python
# Pseudocódigo
1. Cargar parámetros óptimos de config/optimal_params.json
2. Crear instancia de PaperTrader('BTCUSDT', '5m', optimal_params)
3. Llamar a trader.initialize_context()
4. Llamar a trader.start()
5. Mantener corriendo con while True: time.sleep(1)
```

### Salida Esperada
```
[2025-11-03 14:35:21] Conectado a WebSocket. Esperando señales...
[2025-11-03 14:40:00] NUEVA SEÑAL DE COMPRA @ $60,500.50
                      RSI: 45.6 | MACD: 120.3 | EMA21/50: CRUCE ALCISTA
[2025-11-03 15:15:00] NUEVA SEÑAL DE VENTA @ $61,200.30
                      RSI: 72.4 | MACD: -85.2 | EMA21/50: CRUCE BAJISTA
```

### Verificación
- Ejecutar: `python scripts/phase3_paper.py`
- Debe conectarse al WebSocket sin errores
- Debe generar alertas en tiempo real cuando cambian señales
- NO ejecuta órdenes reales

---

## FASE 4: Ejecución Semiautomática

### Objetivo
Implementar ejecución de órdenes reales con capital mínimo ($15) y gestión de riesgo usando órdenes OCO (One-Cancels-the-Other) para Take Profit y Stop Loss.

### ADVERTENCIA
Esta fase opera con dinero real. Usa capital mínimo y cuenta de prueba (testnet) primero.

### Componentes a Implementar

#### 1. `src/strategy/risk_manager.py`
- **Clase**: `RiskManager`
  - **Configuración**:
    - `ATR_SL_MULTIPLIER = 2.0`: Stop Loss a 2x ATR del precio de entrada
    - `ATR_TP_MULTIPLIER = 3.0`: Take Profit a 3x ATR del precio de entrada
    - `CAPITAL_POR_OPERACION = 15.0`: Capital fijo por operación (mínimo Binance)

  - **Método**: `calculate_position_size(current_price, capital)`:
    ```python
    quantity = capital / current_price
    # Formatear a decimales correctos según el símbolo (usar binance_client.get_symbol_info())
    return formatted_quantity
    ```

  - **Método**: `calculate_sl_tp(current_price, atr_value, side)`:
    ```python
    if side == 'BUY':
        sl_price = current_price - (atr_value * ATR_SL_MULTIPLIER)
        tp_price = current_price + (atr_value * ATR_TP_MULTIPLIER)
    elif side == 'SELL':
        sl_price = current_price + (atr_value * ATR_SL_MULTIPLIER)
        tp_price = current_price - (atr_value * ATR_TP_MULTIPLIER)

    return round(sl_price, 2), round(tp_price, 2)
    ```

#### 2. `src/trading/live_trader.py`
- **Clase**: `LiveTrader` (extiende `PaperTrader`)
  - **Atributos adicionales**:
    - `client`: Cliente autenticado de Binance (requiere API_KEY y API_SECRET)
    - `en_posicion`: Boolean (False por defecto)
    - `risk_manager`: Instancia de RiskManager

  - **Método**: `handle_kline_message(msg)` (sobreescrito):
    ```python
    # Heredar lógica de PaperTrader
    super().handle_kline_message(msg)

    # Lógica adicional de ejecución
    if señal_actual == 1 and señal_anterior != 1 and not en_posicion:
        execute_buy_order()

    if señal_actual == -1 and en_posicion:
        execute_sell_order()
    ```

  - **Método**: `execute_buy_order()`:
    ```python
    try:
        # 1. Obtener precio actual y ATR
        current_price = df['close'].iloc[-1]
        atr_value = df['ATR_14'].iloc[-1]

        # 2. Calcular tamaño de posición
        quantity = risk_manager.calculate_position_size(current_price, CAPITAL_POR_OPERACION)

        # 3. Calcular SL y TP
        sl_price, tp_price = risk_manager.calculate_sl_tp(current_price, atr_value, 'BUY')

        # 4. Ejecutar orden de mercado (entrada)
        order = client.create_order(
            symbol='BTCUSDT',
            side='BUY',
            type='MARKET',
            quantity=quantity
        )
        print(f"ORDEN DE COMPRA EJECUTADA @ {current_price}")

        # 5. Colocar orden OCO (TP/SL)
        oco_order = client.create_oco_order(
            symbol='BTCUSDT',
            side='SELL',
            quantity=quantity,
            price=str(tp_price),              # Take Profit (LIMIT_MAKER)
            stopPrice=str(sl_price),          # Stop Loss trigger
            stopLimitPrice=str(sl_price * 0.995),  # Stop Loss limit (5% slippage)
            stopLimitTimeInForce='GTC'
        )
        print(f"OCO COLOCADO - TP: ${tp_price}, SL: ${sl_price}")

        # 6. Actualizar estado
        en_posicion = True

    except BinanceAPIException as e:
        print(f"ERROR AL EJECUTAR ORDEN: {e}")
    ```

  - **Método**: `execute_sell_order()`:
    ```python
    try:
        # Cerrar posición manualmente (si la señal cambia antes del OCO)
        order = client.create_order(
            symbol='BTCUSDT',
            side='SELL',
            type='MARKET',
            quantity=quantity
        )
        print(f"POSICIÓN CERRADA MANUALMENTE @ {df['close'].iloc[-1]}")

        # Cancelar OCO pendiente (si existe)
        # TODO: Implementar cancelación de órdenes abiertas

        en_posicion = False

    except BinanceAPIException as e:
        print(f"ERROR AL CERRAR POSICIÓN: {e}")
    ```

#### 3. `scripts/phase4_live.py`
Script principal ejecutable:
```python
# Pseudocódigo
1. Cargar API_KEY y API_SECRET desde config/config.json (o .env)
2. Validar credenciales con client.get_account()
3. Cargar parámetros óptimos
4. Crear instancia de LiveTrader('BTCUSDT', '5m', optimal_params)
5. Inicializar contexto
6. Iniciar bot
```

### Configuración Requerida

#### `config/config.json`:
```json
{
  "binance": {
    "api_key": "TU_API_KEY_AQUI",
    "api_secret": "TU_API_SECRET_AQUI",
    "testnet": false
  },
  "trading": {
    "symbol": "BTCUSDT",
    "interval": "5m",
    "capital_per_trade": 15,
    "atr_sl_multiplier": 2.0,
    "atr_tp_multiplier": 3.0
  },
  "optimal_params": {
    "ema_corta": 21,
    "ema_larga": 50,
    "rsi_periodo": 14
  }
}
```

### Salida Esperada
```
[2025-11-03 16:20:15] Bot iniciado con capital por operación: $15
[2025-11-03 16:20:15] Conectado a cuenta de Binance. Balance: $500.00
[2025-11-03 16:25:00] ORDEN DE COMPRA EJECUTADA @ $60,500.50
[2025-11-03 16:25:01] OCO COLOCADO - TP: $60,950.75, SL: $60,200.25
[2025-11-03 16:45:00] OCO EJECUTADO - TAKE PROFIT ALCANZADO @ $60,950.75
                      Ganancia: +$11.25 (+0.74%)
```

### Verificación
- **PRIMERO**: Probar en Binance Testnet (`"testnet": true` en config)
- Ejecutar: `python scripts/phase4_live.py`
- Validar que órdenes se ejecuten correctamente
- Verificar que OCO funcione (TP/SL)
- Monitorear balance de cuenta

### Mejoras Necesarias para Producción
- Implementar WebSocket de `userDataStream` para detectar ejecuciones de OCO
- Auto-resetear `en_posicion` cuando OCO se ejecuta
- Manejo de reconexión de WebSocket
- Sistema de logging robusto

---

## FASE 5: Despliegue Completo e IA

### Objetivo
Integrar modelos avanzados de IA (LSTM para predicción de precios, transformers para análisis de sentimiento), mejorar la robustez del sistema y preparar para despliegue en VPS.

### Componentes a Implementar

#### 1. `src/ai/lstm_model.py`
- **Función**: `load_lstm_model(model_path)`
  ```python
  from tensorflow.keras.models import load_model
  return load_model(model_path)
  ```

- **Función**: `get_lstm_prediction(df, model)`
  ```python
  # 1. Extraer últimas 100 velas
  recent_data = df[['close', 'volume', 'RSI_14', 'MACD_12_26_9', 'ATR_14']].tail(100)

  # 2. Normalizar datos (usar mismo scaler del entrenamiento)
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler()
  scaled_data = scaler.fit_transform(recent_data)

  # 3. Reshape para LSTM: (1, 100, 5)
  X = scaled_data.reshape(1, 100, 5)

  # 4. Predecir próxima vela
  prediction = model.predict(X, verbose=0)

  # 5. Convertir a señal simple
  # Si predicción > precio_actual: señal = 1 (COMPRA)
  # Si predicción < precio_actual: señal = -1 (VENTA)
  # Si diferencia < 0.5%: señal = 0 (NADA)

  return lstm_signal  # 1, -1, o 0
  ```

#### 2. `src/ai/sentiment.py`
- **Función**: `load_sentiment_analyzer()`
  ```python
  from transformers import pipeline
  return pipeline(
      "sentiment-analysis",
      model="nlptown/bert-base-multilingual-uncased-sentiment"
  )
  ```

- **Función**: `get_sentiment_score(news_headlines, analyzer)`
  ```python
  # 1. Obtener titulares (simulado o de API de noticias)
  # Para demo, simular 5 titulares aleatorios
  news = [
      "Bitcoin reaches new all-time high",
      "Crypto market shows strong momentum",
      "Investors cautious about volatility",
      "Institutional adoption increasing",
      "Regulatory concerns emerge"
  ]

  # 2. Analizar sentimiento (1-5 estrellas)
  results = analyzer(news)

  # 3. Extraer puntajes numéricos y promediar
  scores = [int(r['label'].split()[0]) for r in results]
  avg_score = sum(scores) / len(scores)

  return avg_score  # Float entre 1.0 y 5.0
  ```

#### 3. `src/utils/logger.py`
- **Configuración**:
  ```python
  import logging
  from logging.handlers import RotatingFileHandler

  def setup_logger(name, log_file, level=logging.INFO):
      formatter = logging.Formatter(
          '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
      )

      # File handler con rotación (max 10MB, 5 backups)
      file_handler = RotatingFileHandler(
          log_file,
          maxBytes=10*1024*1024,
          backupCount=5
      )
      file_handler.setFormatter(formatter)

      # Console handler
      console_handler = logging.StreamHandler()
      console_handler.setFormatter(formatter)

      logger = logging.getLogger(name)
      logger.setLevel(level)
      logger.addHandler(file_handler)
      logger.addHandler(console_handler)

      return logger
  ```

#### 4. `src/trading/live_trader.py` (actualizado con IA)
- **Modificar**: `handle_kline_message(msg)`
  ```python
  # Después de calcular señal técnica
  señal_tecnica = df['señal'].iloc[-1]

  # Obtener señal de LSTM
  lstm_signal = get_lstm_prediction(df, lstm_model)

  # Obtener sentimiento
  sentiment_score = get_sentiment_score([], sentiment_analyzer)

  # LÓGICA DE DECISIÓN COMBINADA
  # COMPRA: Todas las señales deben estar alineadas
  if (señal_tecnica == 1 and
      lstm_signal == 1 and
      sentiment_score > 3.0 and
      not en_posicion):

      logger.info(f"SEÑAL DE COMPRA CONFIRMADA - Técnica: 1, LSTM: 1, Sentimiento: {sentiment_score}")
      execute_buy_order()

  # VENTA: Si cualquier señal es negativa o muy bajista
  if (en_posicion and
      (señal_tecnica == -1 or
       lstm_signal == -1 or
       sentiment_score < 2.0)):

      logger.warning(f"SEÑAL DE VENTA ACTIVADA - Técnica: {señal_tecnica}, LSTM: {lstm_signal}, Sentimiento: {sentiment_score}")
      execute_sell_order()
  ```

#### 5. Mejoras de Robustez

##### A. Auto-Reconexión de WebSocket
```python
def start_with_reconnection():
    while True:
        try:
            logger.info("Iniciando conexión WebSocket...")
            socket = bm.start_kline_socket(symbol, interval, handle_kline_message)

            # Mantener conexión
            while True:
                time.sleep(1)

        except Exception as e:
            logger.error(f"Error en WebSocket: {e}")
            logger.info("Reconectando en 30 segundos...")
            time.sleep(30)
```

##### B. WebSocket de Usuario (para detectar ejecuciones de OCO)
```python
def handle_user_message(msg):
    # Detectar cuando OCO se ejecuta
    if msg['e'] == 'executionReport':
        if msg['X'] == 'FILLED':  # Orden completada
            logger.info(f"Orden ejecutada: {msg['s']} {msg['S']} @ {msg['L']}")

            # Si es una orden de SELL (cierre), resetear posición
            if msg['S'] == 'SELL':
                en_posicion = False
                logger.info("Posición cerrada por OCO")

# Iniciar user data stream
user_socket = bm.start_user_socket(handle_user_message)
```

##### C. Configuración Externa con `configparser`
```python
import configparser

config = configparser.ConfigParser()
config.read('config/config.ini')

API_KEY = config['binance']['api_key']
API_SECRET = config['binance']['api_secret']
SYMBOL = config['trading']['symbol']
CAPITAL = config.getfloat('trading', 'capital_per_trade')
```

#### 6. `scripts/phase5_deployment.py`
Script completo para producción:
```python
"""
Bot de Trading Algorítmico - Fase 5 (Producción)
Integra IA, logging robusto y auto-reconexión
"""

import sys
import signal
from src.trading.live_trader import LiveTrader
from src.utils.logger import setup_logger
from src.ai.lstm_model import load_lstm_model
from src.ai.sentiment import load_sentiment_analyzer

# Configurar logger
logger = setup_logger('TradingBot', 'logs/bot.log')

# Manejar señales de sistema (CTRL+C)
def signal_handler(sig, frame):
    logger.info("Cerrando bot de forma segura...")
    # Cerrar posiciones abiertas si las hay
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == "__main__":
    try:
        logger.info("="*50)
        logger.info("BOT DE TRADING ALGORÍTMICO - FASE 5")
        logger.info("="*50)

        # Cargar modelos de IA
        logger.info("Cargando modelos de IA...")
        lstm_model = load_lstm_model('models/lstm_model.h5')
        sentiment_analyzer = load_sentiment_analyzer()

        # Inicializar trader
        logger.info("Inicializando trader...")
        trader = LiveTrader(
            symbol='BTCUSDT',
            interval='5m',
            lstm_model=lstm_model,
            sentiment_analyzer=sentiment_analyzer
        )

        # Iniciar bot
        logger.info("Iniciando bot con auto-reconexión...")
        trader.start_with_reconnection()

    except Exception as e:
        logger.critical(f"Error crítico: {e}", exc_info=True)
        sys.exit(1)
```

### Despliegue en VPS

#### Paso 1: Configurar VPS
```bash
# Conectar por SSH
ssh root@tu-vps-ip

# Instalar Python 3.10+
sudo apt update
sudo apt install python3.10 python3-pip

# Clonar proyecto
git clone https://github.com/tu-repo/BotDayTrading.git
cd BotDayTrading

# Instalar dependencias
pip3 install -r requirements.txt
```

#### Paso 2: Configurar credenciales
```bash
# Crear archivo de configuración
cp config/config.example.json config/config.json
nano config/config.json  # Editar con tus credenciales
```

#### Paso 3: Ejecutar con Screen o Nohup
```bash
# Opción 1: Screen (recomendado)
screen -S trading_bot
python3 scripts/phase5_deployment.py
# Presionar Ctrl+A, luego D para detach
# Reconectar con: screen -r trading_bot

# Opción 2: Nohup
nohup python3 scripts/phase5_deployment.py > output.log 2>&1 &

# Ver logs en tiempo real
tail -f logs/bot.log
```

#### Paso 4: Monitoreo
- **Logs**: `tail -f logs/bot.log`
- **Procesos**: `ps aux | grep python`
- **Órdenes en Binance**: Panel web o API

### Salida Esperada (Producción)
```
2025-11-03 18:30:00 - TradingBot - INFO - ==================================================
2025-11-03 18:30:00 - TradingBot - INFO - BOT DE TRADING ALGORÍTMICO - FASE 5
2025-11-03 18:30:00 - TradingBot - INFO - ==================================================
2025-11-03 18:30:01 - TradingBot - INFO - Cargando modelos de IA...
2025-11-03 18:30:05 - TradingBot - INFO - Inicializando trader...
2025-11-03 18:30:06 - TradingBot - INFO - Iniciando bot con auto-reconexión...
2025-11-03 18:35:00 - TradingBot - INFO - SEÑAL DE COMPRA CONFIRMADA - Técnica: 1, LSTM: 1, Sentimiento: 4.2
2025-11-03 18:35:01 - TradingBot - INFO - ORDEN DE COMPRA EJECUTADA @ $60,500.50
2025-11-03 18:35:02 - TradingBot - INFO - OCO COLOCADO - TP: $60,950.75, SL: $60,200.25
2025-11-03 19:10:00 - TradingBot - INFO - Orden ejecutada: BTCUSDT SELL @ 60950.75
2025-11-03 19:10:00 - TradingBot - INFO - Posición cerrada por OCO
```

### Verificación Final
- Bot se ejecuta 24/7 sin intervención
- Auto-reconexión funciona tras errores de red
- Órdenes OCO se ejecutan correctamente
- Logs detallados de todas las operaciones
- Modelos de IA influyen en decisiones

---

## Resumen de Progresión

| Fase | Componentes | Ejecutable | Propósito |
|------|-------------|------------|-----------|
| 1 | data, indicators, strategy | `phase1_historical.py` | Datos y señales básicas |
| 2 | backtest, optimizer, metrics | `phase2_backtest.py` | Encontrar mejores parámetros |
| 3 | paper_trader | `phase3_paper.py` | Validar en vivo sin riesgo |
| 4 | live_trader, risk_manager | `phase4_live.py` | Trading real con capital mínimo |
| 5 | ai (LSTM, sentiment), logger | `phase5_deployment.py` | Producción con IA en VPS |

## Gestión de Riesgo (Máxima Prioridad)

### Reglas Obligatorias
1. **Capital por operación**: Máximo $15 (0.5-1% del capital total)
2. **Stop Loss**: Siempre activo (2x ATR)
3. **Take Profit**: Siempre definido (3x ATR)
4. **Máximo 3 operaciones simultáneas**: Diversificación limitada
5. **Pérdida diaria máxima**: -5% del capital total (detener bot)
6. **Testnet primero**: Probar TODO en entorno de pruebas antes de dinero real

### Límites de Seguridad en Código
```python
# En risk_manager.py
MAX_DAILY_LOSS = 0.05  # 5%
MAX_OPEN_POSITIONS = 3
MIN_SHARPE_RATIO = 0.8  # No operar si backtesting < 0.8

# Verificar antes de cada operación
if daily_loss > MAX_DAILY_LOSS:
    logger.critical("LÍMITE DE PÉRDIDA DIARIA ALCANZADO. DETENIENDO BOT.")
    sys.exit(1)
```

## Próximos Pasos

Una vez completadas las 5 fases:
1. **Monitorear durante 1 semana** con capital mínimo
2. **Analizar resultados** (ganancia, drawdown, Sharpe ratio)
3. **Optimizar estrategia** según datos reales (no solo backtesting)
4. **Escalar gradualmente** el capital si resultados son positivos
5. **Diversificar** a otros pares (ETHUSDT, BNBUSDT, etc.)

## Soporte y Troubleshooting

### Errores Comunes

**Error: "Invalid API-key"**
- Verificar que API_KEY y API_SECRET estén correctos en config.json
- Verificar que la IP del VPS esté en whitelist de Binance

**Error: "Insufficient balance"**
- Verificar saldo en cuenta de Binance
- Reducir `capital_per_trade` en configuración

**Error: "Filter failure: MIN_NOTIONAL"**
- El valor de la operación es menor al mínimo de Binance ($10-15)
- Aumentar `capital_per_trade` a mínimo $15

**WebSocket desconectado**
- La auto-reconexión debería manejar esto
- Si persiste, verificar conexión a internet del VPS

### Logs de Debugging
```bash
# Aumentar nivel de logging a DEBUG
# En src/utils/logger.py
logger.setLevel(logging.DEBUG)

# Ver logs detallados
tail -f logs/bot.log | grep DEBUG
```

---

**IMPORTANTE**: Este bot es una herramienta educativa y experimental. El trading algorítmico conlleva riesgos significativos. Nunca operes con capital que no puedas permitirte perder. La gestión de riesgo es CRÍTICA para la supervivencia a largo plazo.
