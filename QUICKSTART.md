# Inicio Rápido - Bot de Trading

## Instalación en 3 Pasos

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Configurar (opcional para Fases 1-2)
```bash
cp config/config.example.json config/config.json
# Edita config.json solo si vas a las Fases 3-5
```

### 3. Ejecutar Fase 1
```bash
python scripts/phase1_historical.py
```

## Progresión de Fases

### ✅ Fase 1: Datos Históricos (SIN RIESGO)
```bash
python scripts/phase1_historical.py
```
- Descarga datos de BTCUSDT (1 año, 5m)
- Calcula indicadores (EMA, RSI, MACD, etc.)
- Genera señales de COMPRA/VENTA
- **Sin riesgo**: Solo análisis de datos

### ✅ Fase 2: Backtesting (SIN RIESGO)
```bash
python scripts/phase2_backtest.py
```
- Prueba diferentes combinaciones de parámetros
- Encuentra la mejor configuración
- Guarda parámetros óptimos
- **Sin riesgo**: Solo simulación

### ⚠️ Fase 3: Paper Trading
```bash
python scripts/phase3_paper.py
```
- Alertas en tiempo real
- Datos en vivo (WebSockets)
- **Sin riesgo**: No ejecuta órdenes
- **Requiere**: Completar Fase 2

### 🔴 Fase 4: Trading Real
```bash
python scripts/phase4_live.py
```
- **DINERO REAL**: Ejecuta órdenes
- Capital mínimo: $15 por operación
- Stop Loss + Take Profit automáticos
- **Requiere**: API Key de Binance

### 🚀 Fase 5: Producción con IA
```bash
python scripts/phase5_deployment.py
```
- Modelos LSTM + Sentiment Analysis
- Auto-reconexión
- Listo para VPS 24/7
- **Requiere**: Modelos entrenados

## ⚠️ IMPORTANTE

1. **Fases 1-2**: Totalmente seguras, sin riesgo
2. **Fase 3**: Requiere parámetros de Fase 2
3. **Fases 4-5**: DINERO REAL - usa testnet primero
4. **Gestión de riesgo**: SIEMPRE activa
5. **Testnet**: Prueba SIEMPRE antes de producción

## Documentación Completa

- **README.md**: Documentación completa
- **PHASE_GUIDE.md**: Guía detallada de cada fase
- **config/config.example.json**: Ejemplo de configuración

## Obtener Credenciales de Binance

### Testnet (Recomendado):
1. https://testnet.binance.vision/
2. "Generate API Key"
3. Copia API Key y Secret
4. Pega en `config/config.json`
5. `"testnet": true`

### Producción (Dinero Real):
1. https://www.binance.com
2. Account > API Management
3. Crea API Key con permisos de trading
4. `"testnet": false`

## Soporte

- Errores comunes: Ver README.md > Troubleshooting
- Documentación: PHASE_GUIDE.md
- Logs: `tail -f logs/bot.log`

---

**¡Comienza con la Fase 1!** Es 100% segura y te dará una buena idea del potencial del bot.
