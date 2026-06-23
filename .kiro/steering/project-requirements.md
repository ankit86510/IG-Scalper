# IG Scalper — Project Requirements & Constraints

Always consult these requirements before making any change to the codebase.

## System Overview

This is an automated CFD trading bot that connects to IG Markets (Demo/Live) and executes scalping/swing trades on Gold spot (XAU/USD) and other instruments using AI-powered pattern recognition.

## Architecture Requirements

- **Language**: Python 3.11+
- **Deployment**: Docker container with `--env-file .env`
- **Timezone**: All logs and bar timestamps MUST be in `Europe/Rome` (CEST/CET)
- **Config**: YAML at `config/settings_ai.yaml` with env var expansion `${VAR}`
- **Credentials**: NEVER hardcoded — always via `.env` file and `${VAR}` placeholders

## Data Provider Requirements

### Primary: TwelveData REST API
- **Plan**: Free Basic — 800 requests/day, 8 requests/minute
- **Symbol**: `XAU/USD` (Gold spot), `EUR/USD`, `GBP/USD`
- **Timezone param**: `timezone=Europe/Rome` in every request
- **Rate limiting**: Adaptive based on number of active symbols
  - Formula: `poll_interval = (num_symbols × 86400) / 720`
  - Internal cache prevents redundant fetches within same bar period
  - Sliding window enforces per-minute cap (7 effective)
  - Daily budget: 720 effective (90% safety margin of 800)
- **CRITICAL**: Never exceed 800/day or 8/min under any circumstance

### Fallback chain
1. TwelveData (spot) — primary
2. Yahoo Finance — forex/indices only (NO futures like GC=F for Gold)
3. AlphaVantage — rate-limited backup
4. IG REST `/prices` — final fallback (has its own daily limit on Demo)

### Excluded sources for Gold
- Yahoo Finance `GC=F` is FUTURES, not spot — produces ~$20-30 price mismatch
- Finnhub free tier does NOT support forex/commodities (only crypto + US stocks)

## Trading Requirements

- **Timeframe**: 5 minutes (primary), configurable in YAML
- **Lookback**: 200 bars minimum for AI analysis (TwelveData free max is 250/request)
- **Strategy**: AI Pattern Recognizer with S/R detection
- **Risk per trade**: Configurable in YAML (`risk.invest_per_trade`)
- **Stop loss**: ATR-based × multiplier, adjusted to S/R zones
- **Take profit**: R:R ratio × stop, adjusted to S/R zones
- **Trailing stop**: Optional, managed via IG API position updates
- **Daily loss limit**: Stop trading if daily P&L exceeds configured %
- **Kill switch**: Check `KILL_SWITCH` env var every loop

## IG API Requirements

- **Authentication**: REST login → CST + X-SECURITY-TOKEN headers
- **Order placement**: `POST /positions/otc` with stop_distance + limit_distance
- **Market rules**: Always enforce `minNormalStopOrLimitDistance` and `minDealSize`
- **Position sizing**: Risk-based via `size_by_invested_capital()`
- **SSL**: `verify=False` for both Demo and Live (corporate proxy issues)

## Code Standards

- All strategies extend `strategy.base.Strategy` ABC
- `on_bar(df)` returns `{"side", "stop_pts", "tp_pts", "meta"}` or `None`
- Analysis uses **penultimate bar** (`iloc[-2]`) — last bar is forming/incomplete
- Logging via `core.logging_utils` — Windows-safe, Rome timezone
- No duplicate method definitions in any file
- Imports must be clean — no references to non-existent modules

## Docker Requirements

- Base image: `python:3.11-slim`
- Timezone: `TZ=Europe/Rome` with `/etc/localtime` symlink
- Runtime dirs: `logs/`, `data/` mounted as volumes
- `.env` passed via `--env-file`, NOT baked into image
- `.dockerignore` excludes `.env`, `logs/`, `data/*.json`, `__pycache__/`
