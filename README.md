# SelfLearn Trader (MVP)

An end-to-end **self-learning trading app** that updates its model after each trade outcome.
- **Online learning** with `SGDClassifier` (logistic regression) via `partial_fit`
- **Contextual trading**: features from price/volume -> predicts next-bar direction
- **Epsilon-greedy**: explore vs. exploit; decides to go **long / flat / short**
- **Backtester** with **trade journal** CSV and **equity curve**
- **Streamlit UI** for point-and-click backtests
- **Paper broker** stub for later live trading; optional OANDA placeholders

> This MVP is designed for EUR/USD hourlies but works with any yfinance ticker.
> Example tickers: `EURUSD=X`, `GBPUSD=X`, `GC=F` (Gold), `^GSPC` (S&P 500).

## Quick start

```bash
# 1) Create a venv
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) Run a backtest
python run_backtest.py --symbol EURUSD=X --interval 1h --start 2023-01-01 --end 2025-08-01

# 4) Launch the UI
streamlit run streamlit_app.py
```

## What "self-learning" means here

After the app takes a trade, it **updates the model** with the actual result of that trade using `partial_fit`.
This means the policy adapts over time based on its successes and mistakes.

## File structure

```
selflearn-trader/
  app/
    __init__.py
    config.py
    utils.py
    features.py
    journal.py
    backtest.py
    paperbroker.py
    models/online_classifier.py
    dataio/yf.py
    strategy/contextual_bandit.py
    ui/st_app.py
  run_backtest.py
  streamlit_app.py
  requirements.txt
  config.example.toml
  README.md
```

## Notes
- This is **not financial advice**. Educational code only.
- For live trading, integrate a real broker API and add risk checks (max daily loss, etc.).
- Default thresholds are conservative; tune on your data.


## Upgrades (v2)

- **ATR-based position sizing** with configurable risk % and ATR multiple
- **Transaction costs**: spread (pips), slippage (pips), commission
- **Streaming scaler** (online normalization) for regime changes
- **Model persistence** between runs (saves to `models_ckpt/`)
- **Session features** + optional **killzone gating** (London 07–10 UTC, NY 12–15 UTC)

### New CLI example
```bash
python run_backtest.py \
  --symbol EURUSD=X --interval 1h --start 2023-01-01 --end 2025-08-01 \
  --threshold 0.56 --epsilon 0.08 --warmup 300 --cash 10000 \
  --risk 0.005 --atr_mult 1.0 --spread_pips 0.2 --slippage_pips 0.1 --commission 0 \
  --gate_kz
```

