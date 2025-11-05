# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an algorithmic day trading bot for cryptocurrencies using the Binance API. The project follows a strict 5-phase development progression from historical data analysis to production deployment with AI models. Each phase must be completed before moving to the next.

## Commands

### Development and Testing

```bash
# Install dependencies
pip install -r requirements.txt

# Phase 1: Historical data and signals (NO RISK)
python scripts/phase1_historical.py

# Phase 2: Backtesting and optimization (NO RISK)
python scripts/phase2_backtest.py

# Phase 3: Paper trading with live data (NO RISK - alerts only)
python scripts/phase3_paper.py

# Phase 4: Live trading with real money (REQUIRES API KEYS)
python scripts/phase4_live.py

# Phase 5: Production deployment with AI (FULL SYSTEM)
python scripts/phase5_deployment.py

# Test individual modules
python src/data/binance_client.py
python src/indicators/technical.py
python src/strategy/signal_generator.py
python src/backtest/engine.py

# Monitor logs
tail -f logs/bot.log
```

### Configuration

Configuration is managed through `config/config.json`. Copy `config/config.example.json` to get started:
```bash
cp config/config.example.json config/config.json
```

Phases 1-2 do not require API credentials (public data only). Phases 3-5 require Binance API keys.

## Architecture

### Phase-Based Development

The codebase is strictly organized into 5 phases that must not be skipped:

1. **Phase 1** (Historical Data): Download OHLCV data, calculate technical indicators, generate signals
2. **Phase 2** (Backtesting): Vectorized backtest with commission/slippage, parameter optimization via grid search
3. **Phase 3** (Paper Trading): WebSocket connection to live data, signal alerts without execution
4. **Phase 4** (Live Trading): Real order execution with OCO orders (Take Profit + Stop Loss)
5. **Phase 5** (Production): AI integration (LSTM + sentiment), auto-reconnection, VPS deployment

### Module Structure

**Data Layer** (`src/data/`):
- `binance_client.py`: Manages Binance client instances (public vs authenticated, testnet vs production)
- `data_fetcher.py`: Downloads historical OHLCV data via REST API

**Indicators** (`src/indicators/`):
- `technical.py`: Calculates technical indicators using pandas-ta (EMA, RSI, MACD, Bollinger Bands, ATR, Stochastic)

**Strategy** (`src/strategy/`):
- `signal_generator.py`: Generates trading signals (1=BUY, -1=SELL, 0=NEUTRAL) based on indicator logic
- `risk_manager.py`: Calculates position sizing, Stop Loss/Take Profit using ATR multipliers, enforces risk limits

**Backtesting** (`src/backtest/`):
- `engine.py`: Vectorized backtester with commission/slippage simulation
- `optimizer.py`: Grid search parameter optimization using sklearn.model_selection.ParameterGrid

**Trading** (`src/trading/`):
- `paper_trader.py`: WebSocket-based paper trading (alerts only)
- `live_trader.py`: Real order execution with OCO orders (extends paper_trader.py)

**AI** (`src/ai/`):
- `lstm_model.py`: LSTM price prediction model loading and inference
- `sentiment.py`: News sentiment analysis using transformers

**Utilities** (`src/utils/`):
- `logger.py`: Rotating file logger with console output
- `metrics.py`: Performance metrics calculation (Sharpe, drawdown, win rate, etc.)

### Key Design Patterns

**Signal Generation**: All signal generators return DataFrames with a `señal` column containing 1, -1, or 0.

**Vectorized Backtesting**: Uses pandas operations for speed. Applies `shift(1)` to signals to prevent look-ahead bias.

**Risk Management**: ATR-based dynamic Stop Loss (2x ATR) and Take Profit (3x ATR). Fixed capital per trade ($15 default).

**Configuration-Driven**: All parameters (strategy, risk, AI) are in `config/config.json`. Phase 2 saves optimized parameters to `config/optimal_params.json`.

**WebSocket Context**: Paper trading and live trading maintain a rolling window of 500 candles to calculate indicators correctly with historical context.

**OCO Orders**: Phase 4+ uses Binance OCO (One-Cancels-Other) orders to place Take Profit and Stop Loss simultaneously.

## Critical Development Guidelines

### Safety and Risk Management

- **NEVER skip phases**: Each phase validates the previous one
- **ALWAYS use testnet first**: Set `"testnet": true` in config before real trading
- **Capital limits are mandatory**: $15 per trade, max 3 open positions, 5% daily loss limit
- **Stop Loss is non-negotiable**: Every trade must have SL/TP calculated from ATR
- Phases 1-3 are completely safe (no real money). Only Phases 4-5 execute real trades.

### Working with Signals

When modifying signal generation logic:
1. Ensure the `señal` column contains only 1, -1, or 0
2. Test with historical data first (Phase 1)
3. Validate with backtesting (Phase 2) before live deployment
4. Check that all required indicator columns exist in DataFrame

### Configuration Management

- `config/config.json` contains active configuration (never commit with real API keys)
- `config/config.example.json` is the template
- Phase 2 outputs `config/optimal_params.json` with optimized strategy parameters
- The `BinanceClientManager` automatically selects testnet vs production credentials

### Testing Individual Modules

Each module has a `if __name__ == "__main__":` block for standalone testing. Run modules directly to test:
```bash
python -m src.data.binance_client  # or python src/data/binance_client.py
```

### Binance API Specifics

- Public endpoints (Phases 1-2) don't require API keys
- WebSocket kline messages arrive continuously; only process when `msg['k']['x'] == True` (candle closed)
- Use `get_symbol_info()` to get correct price/quantity precision for order formatting
- MIN_NOTIONAL on Binance is ~$10-15; ensure `capital_per_trade >= 15`

### AI Integration (Phase 5)

- LSTM model expects last 100 candles with features: close, volume, RSI_14, MACD, ATR_14
- Sentiment analysis uses `nlptown/bert-base-multilingual-uncased-sentiment`
- Decision logic: All three signals (technical, LSTM, sentiment) must align for BUY
- Models are loaded once at startup to avoid repeated loading overhead

## Common Patterns and Code Examples

### Adding a New Technical Indicator

Indicators are added in `src/indicators/technical.py` using pandas-ta:
```python
df.ta.sma(length=200, append=True)  # Adds SMA_200 column
```

### Modifying Signal Logic

Signal generation is in `src/strategy/signal_generator.py`. Example buy condition:
```python
condicion_compra = (
    (df['EMA_21'] > df['EMA_50']) &  # Trend
    (df['RSI_14'] < 70)               # Not overbought
)
df.loc[condicion_compra, 'señal'] = 1
```

### Running a Custom Backtest

```python
from src.backtest.engine import VectorizedBacktester

backtester = VectorizedBacktester(
    df=df,  # Must have 'señal' column
    initial_capital=10000,
    commission=0.00075
)
results = backtester.run_backtest()
metrics = backtester.calculate_metrics()
```

## File Locations

- **Data cache**: `data/historical_data.csv` (Phase 1 output)
- **Logs**: `logs/bot.log`, `logs/trades.log`
- **Models**: `models/lstm_model.h5` (for Phase 5)
- **Config**: `config/config.json` (active), `config/optimal_params.json` (Phase 2 output)

## Important Constraints

- The project is in Spanish (variable names, comments, print statements)
- Uses `señal` (not `signal`) for the signal column name
- All phases print progress and results to console for visibility
- Error handling should be comprehensive - this is financial software
- Logging is critical for production (Phase 5) - use the logger module, not print statements
