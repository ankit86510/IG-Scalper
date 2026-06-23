# IG Scalper

Automated CFD trading bot for IG Markets with AI-powered pattern recognition and multi-timeframe Fair Value Gap (FVG) analysis.

## Features

- **Two Strategy Modes** — switch via `strategy_type` in config:
  - `ai_pattern` — AI Pattern Recognizer with candlestick/chart pattern detection, momentum, and trend analysis
  - `fvg` — Multi-timeframe FVG strategy (60min → 15min → 5min cascade) with bias-based signal generation
- **Smart Data Aggregation** — TwelveData (primary), Yahoo Finance, AlphaVantage, IG REST fallback chain
- **Rate Limit Aware** — adaptive polling respects TwelveData free tier (800/day, 8/min)
- **Position Management** — real-time sync with broker, trailing stops, ATR-based risk
- **Support/Resistance Detection** — adjusts stops and targets to S/R zones
- **Kill Switch** — set `KILL_SWITCH=1` env var to gracefully stop

## Quick Start

```bash
# 1. Clone and configure
git clone https://github.com/ankit86510/IG-Scalper.git
cd IG-Scalper
cp .env.example .env  # Fill in your credentials

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run (choose one)
python runners/run_ai_autonomous.py   # AI autonomous trading
python runners/run_live.py            # Classic EMA strategy
```

## Docker (Recommended)

```bash
docker build -t ig-bot .
docker run -d \
  --name ig-scalper \
  --env-file .env \
  --restart unless-stopped \
  -v ./logs:/app/logs \
  -v ./data:/app/data \
  ig-bot python runners/run_ai_autonomous.py
```

Monitor: `docker logs ig-scalper -f`
Stop: `docker stop ig-scalper`

## Configuration

All settings in `config/settings_ai.yaml`:

```yaml
strategy_type: "fvg"          # "ai_pattern" or "fvg"

symbols:
  - "CS.D.CFEGOLD.CEB.IP"    # Gold (XAU/USD)

risk:
  invest_per_trade: 1000
  max_loss_pct_invest: 5.0
  max_daily_loss_pct: 3.0

execution:
  use_trailing_stop: true
  trailing_activation_pct: 0.3
  trailing_distance_pct: 0.5
```

## Environment Variables (.env)

```
IG_API_KEY=your_api_key
IG_USERNAME=your_username
IG_PASSWORD=your_password
IG_ACCOUNT_TYPE=DEMO          # DEMO or LIVE
TWELVE_DATA_KEY=your_key
ALPHA_VANTAGE_KEY=your_key    # optional
FINNHUB_KEY=your_key          # optional
```

## Project Structure

```
├── broker/          # IG API client, order execution, market rules
├── config/          # YAML configuration files
├── core/            # Logging, risk calculations, config loader
├── data/            # Data providers (TwelveData, Yahoo, IG, etc.)
├── runners/         # Entry points (autonomous, live, backtest)
├── strategy/        # Trading strategies
│   ├── ai_pattern_recognizer.py   # AI pattern + momentum strategy
│   ├── fvg_strategy.py            # Multi-timeframe FVG strategy
│   ├── fvg_detector.py            # FVG detection engine
│   ├── fvg_bias.py                # Bias calculation from FVGs
│   ├── fvg_signal.py              # Signal generation
│   └── fvg_scheduler.py           # Cycle scheduling + rate budget
├── tests/           # Property-based tests (Hypothesis)
└── Dockerfile
```

## Strategies

### FVG Multi-Timeframe (Recommended)

Detects Fair Value Gaps across 60min/15min/5min timeframes. Generates signals when price enters an unfilled FVG zone with aligned multi-timeframe bias. Cycle runs every 5 minutes.

### AI Pattern Recognizer

Uses candlestick pattern detection, momentum analysis, and trend strength. Generates signals above configurable confidence threshold with S/R-adjusted stops.

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Property-based tests only
python -m pytest tests/test_fvg_properties.py tests/test_position_sync_bug_condition.py -v
```

## Important Notes

- **Use IG Demo account first** — test thoroughly before switching to Live
- **TwelveData free tier** — 800 requests/day, bot auto-manages budget
- **Risk** — position sizing limits loss to configured % of invested capital per trade
- **Timezone** — all timestamps in Europe/Rome (CEST/CET)
