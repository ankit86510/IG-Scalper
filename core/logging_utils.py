
import logging
import os

def setup_logging(level="INFO", sink="logs/bot.log"):
    os.makedirs(os.path.dirname(sink), exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(sink), logging.StreamHandler()]
    )
    return logging.getLogger("ig-scalper")
