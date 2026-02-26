# Project Kairos – Trading Backtest API

Kairos is a Python-based backtesting platform with a FastAPI REST interface for evaluating algorithmic trading strategies on historical market data. It supports Moving Average crossover strategies, configurable with custom parameters. The engine computes key performance metrics such as returns, Sharpe ratio, win rate, drawdown, and trade statistics.

Historical data is fetched through yfinance with automatic caching, and all results are stored using SQLite or PostgreSQL via SQLAlchemy.

**Intraday data limits:** For timeframes 1h, 15m, 5m, and 1m, Yahoo Finance only provides data for the last 60 days (7 days for 1m). If you request more candles than exist in that window (e.g. 1000 candles for 1h), you will receive and store only what Yahoo returns (e.g. around 390–630 bars for 1h depending on market hours). Use `1d` for longer history. Backtest runs are fully logged for comparison across tickers, timeframes, and strategies. The platform uses Python 3.9+, FastAPI, SQLAlchemy, yfinance, pandas, numpy, scikit-learn, and pytest.

## Features

The platform includes data fetching with automatic caching, Moving Average trading strategies, comprehensive backtesting with performance metrics, a REST API for triggering backtests and retrieving results, algorithm comparison capabilities, and database support for storing historical data and results.

## Technology Stack

The project is built using Python 3.9+ with FastAPI for the web framework, SQLAlchemy for database operations, yfinance for stock data fetching, pandas and numpy for data manipulation, scikit-learn for metrics calculation, and pytest for testing.

## Setup

`pip install -r requirements.txt`
`python run.py`

The API will be available at http://localhost:8000 with interactive docs at http://localhost:8000/docs.

## API Endpoints

### Fetch Historical Data

POST `/api/fetch-data` - Fetches and caches historical stock price data for a given ticker, timeframe, and number of candles. Returns success status and data fetch information.

### Run Backtest

POST `/api/backtest` - Executes a backtest with a specified strategy, ticker, timeframe, and parameters. Returns comprehensive performance metrics including total return, win rate, Sharpe ratio, and trade details.

### Compare Algorithms

POST `/api/compare-algo` - Compares two algorithms across multiple combinations of tickers, timeframes, and candle counts. Returns detailed per-combination results and an overall comparison summary showing which algorithm performed better.

### Get Backtest Results

GET `/api/results/{backtest_id}` - Retrieves detailed backtest results by ID including all metrics, trades, and model parameters.

### Health Check

GET `/api/health` - Returns API health status and version information.

### WhatsApp (Twilio)

POST `/api/whatsapp/start-hi` - Starts a background job that sends `"hi"` to a WhatsApp number every minute (or custom interval).

POST `/api/whatsapp/stop` - Stops the background WhatsApp job.

POST `/api/whatsapp/send-once` - Sends a single WhatsApp message immediately.

GET `/api/whatsapp/status` - Returns the current repeater status/state.

## Examples

Fetch data for a stock, run a backtest, and retrieve results:

```bash
curl -X POST "http://localhost:8000/api/fetch-data" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "timeframe": "1d", "candles": 1000}'

curl -X POST "http://localhost:8000/api/backtest" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "strategy": "MOVING_AVERAGE", "timeframe": "1d", "candles": 1000, "initial_capital": 10000.0}'

curl "http://localhost:8000/api/results/1"
```

### WhatsApp example (send "hi" every minute)

1) Set environment variables (example values):

- `TWILIO_ACCOUNT_SID`
- `TWILIO_AUTH_TOKEN`
- `TWILIO_WHATSAPP_FROM` (example: `whatsapp:+14155238886` for Twilio sandbox)

Then run:

```bash
curl -X POST "http://localhost:8000/api/whatsapp/start-hi" \
  -H "Content-Type: application/json" \
  -d '{ "to": "whatsapp:+15551234567", "interval_seconds": 60 }'
```

Stop it:

```bash
curl -X POST "http://localhost:8000/api/whatsapp/stop"
```

## Testing

Run the test suite with `pytest`

## Project Structure

The project follows a modular structure with separate directories for models (database schemas and Pydantic models), strategies (trading algorithm implementations), services (data fetching, processing, backtesting, and metrics), and API routes (FastAPI endpoints).

## Configuration

Configuration is managed in `config.py` with support for environment variable overrides via `.env` file. Key settings include database URL, default candles, timeframe, initial capital, and transaction costs.

### Data limits (Yahoo Finance)

For **intraday** timeframes (`1h`, `15m`, `5m`, `1m`), Yahoo Finance limits history to the last **60 days** (or **7 days** for `1m`). The API will return and store only the bars Yahoo provides, so `candles_fetched` may be less than requested (e.g. ~390–630 for 1h). For long backtests, use `timeframe: "1d"`.

## Performance Metrics

The backtesting engine calculates total return percentage, win rate, maximum drawdown, number of trades, average gain per trade, Sharpe ratio for risk-adjusted returns, and prediction accuracy metrics (MAE, RMSE, MAPE) for model evaluation.

## Strategies

The platform supports Moving Average crossover strategies with configurable windows. Each strategy can be customized with specific parameters and tested across different market conditions. More to be added later.

## Database

The system uses SQLAlchemy ORM with support for SQLite (default) and PostgreSQL. The database schema includes tables for stocks, price_data, strategies, and backtests, automatically initialized on first run.
