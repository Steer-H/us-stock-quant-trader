# us-stock-quant-trader

A live trading system for US stocks. Fetches real-time prices, runs a Transformer model for directional predictions, manages a portfolio with dynamic leverage, and exposes a web dashboard.

## What it does

- Pulls real-time prices from Yahoo Finance for ~40 stocks
- Runs a Transformer model to predict short-term direction
- Uses Kelly criterion with volatility adjustment for position sizing
- Anti-Martingale: scales up on wins, scales down on losses
- Serves a dashboard at localhost:8080 with charts, positions, and trade log

## Running it

```bash
pip install -r requirements.txt
screen -dmS trading python3 -u live_trading/web_server.py
curl http://localhost:8080/api/status
```

Then open http://localhost:8080.

## Structure

```
live_trading/    trading engine, Flask server, dashboard
ml_model/        StockTransformer model + training
config/          model and trading parameters
backtesting/     backtest engine
data_pipeline/   data fetching, cleaning, feature engineering
risk/            risk manager
docs/            detailed docs
scripts/         training, monitoring, sentiment scripts
```

## Docs

Detailed docs (Traditional Chinese) under `docs/`. Start with `docs/README.md`.

## Dev rules

Read `GUARDRAILS.md` before touching anything. Past mistakes and no-touch zones are documented there.
