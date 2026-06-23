import os
import yaml

# Load .env file if present (optional dependency - falls back gracefully)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed; use OS environment variables directly


def load_settings(path="config/settings_ai.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # Expand env placeholders like ${VAR}
    def expand(v):
        if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
            return os.environ.get(v[2:-1], "")
        if isinstance(v, dict):
            return {k: expand(val) for k, val in v.items()}
        if isinstance(v, list):
            return [expand(x) for x in v]
        return v
    return expand(cfg)
