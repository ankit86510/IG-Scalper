# IG Scalper (Demo)

IG-only scalping bot using IG REST bars for analysis and execution. Strategies: EMA 9/21 + breakout filter with ATR-based stops.

## Quick start
- Python 3.11+
- Fill config/settings.yaml (IG creds via env placeholders)
- pip install -r requirements.txt
- python runners/run_live.py

## Docker
- docker build -t ig-bot .
- docker run -d --name ig-bot --env-file .env --restart unless-stopped -v "$(pwd)/logs:/app/logs" -v "$(pwd)/data:/app/data" ig-bot

## Notes
- Use IG Demo first
- Consider Lightstreamer for tick-driven 1m bars in production
- Risk: sizes are computed to limit loss to 6% of invested capital per trade
