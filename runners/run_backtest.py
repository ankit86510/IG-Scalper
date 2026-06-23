"""
Backtester for IG scalping strategies.

Usage:
    python runners/run_backtest.py --csv data/sample_1m.csv --strategy ema
    python runners/run_backtest.py --csv data/sample_1m.csv --strategy ma
    python runners/run_backtest.py --csv data/sample_1m.csv --strategy rsi
    python runners/run_backtest.py --csv data/sample_1m.csv --strategy stochastic
    python runners/run_backtest.py --csv data/sample_1m.csv --strategy psar

CSV must have columns: ts, open, high, low, close, volume
"""

import sys
import os
import argparse
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy.ema_cross_breakout import EMACrossBreakout
from strategy.moving_average_scalper import MovingAverageScalper
from strategy.rsi_scalper import RSIScalper
from strategy.stochastic_scalper import StochasticScalper
from strategy.parabolic_sar_scalper import ParabolicSARScalper


STRATEGIES = {
    "ema": lambda: EMACrossBreakout(fast=9, slow=21, atr_period=14, rr_take=1.5),
    "ma": lambda: MovingAverageScalper(fast_ma=5, slow_ma=20, trend_ma=200,
                                       stop_multiplier=1.0, rr_take=1.5),
    "rsi": lambda: RSIScalper(rsi_period=14, oversold=30, overbought=70,
                               ma_fast=5, ma_med=20, ma_slow=50,
                               stop_multiplier=1.0, rr_take=1.5),
    "stochastic": lambda: StochasticScalper(k_period=14, d_period=3,
                                             oversold=20, overbought=80,
                                             stop_multiplier=1.0, rr_take=1.5),
    "psar": lambda: ParabolicSARScalper(acceleration=0.02, maximum=0.2,
                                        stop_multiplier=1.0, rr_take=1.5),
}


def backtest(df: pd.DataFrame, strategy_name: str = "ema",
             initial_equity: float = 10000.0) -> dict:
    """
    Run a simple bar-by-bar backtest.

    Returns dict with equity, trades list, win_rate, profit_factor.
    """
    strat = STRATEGIES[strategy_name]()
    position = 0       # 0=flat, 1=long, -1=short
    entry = 0.0
    equity = initial_equity
    trades = []

    for i in range(len(df)):
        sub = df.iloc[:i + 1]
        sig = strat.on_bar(sub)
        px = df["close"].iloc[i]
        ts = df.index[i]

        if sig and sig["side"] == "BUY":
            if position <= 0:
                if position < 0:
                    pnl = (entry - px)
                    equity += pnl
                    trades.append({"ts": ts, "exit": px, "dir": "COVER", "pnl": pnl})
                position = 1
                entry = px
                trades.append({"ts": ts, "entry": px, "dir": "LONG", "pnl": 0})

        elif sig and sig["side"] == "SELL":
            if position >= 0:
                if position > 0:
                    pnl = (px - entry)
                    equity += pnl
                    trades.append({"ts": ts, "exit": px, "dir": "SELL", "pnl": pnl})
                position = -1
                entry = px
                trades.append({"ts": ts, "entry": px, "dir": "SHORT", "pnl": 0})

    # Close any open position at last bar
    if position == 1:
        px = df["close"].iloc[-1]
        pnl = px - entry
        equity += pnl
        trades.append({"ts": df.index[-1], "exit": px, "dir": "SELL", "pnl": pnl})
    elif position == -1:
        px = df["close"].iloc[-1]
        pnl = entry - px
        equity += pnl
        trades.append({"ts": df.index[-1], "exit": px, "dir": "COVER", "pnl": pnl})

    closed = [t for t in trades if "pnl" in t and t["pnl"] != 0]
    wins = [t for t in closed if t["pnl"] > 0]
    losses = [t for t in closed if t["pnl"] <= 0]
    win_rate = len(wins) / len(closed) * 100 if closed else 0
    gross_profit = sum(t["pnl"] for t in wins)
    gross_loss = abs(sum(t["pnl"] for t in losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    return {
        "strategy": strategy_name,
        "initial_equity": initial_equity,
        "final_equity": round(equity, 2),
        "net_pnl": round(equity - initial_equity, 2),
        "total_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "trades": trades,
    }


def main():
    parser = argparse.ArgumentParser(description="IG Scalper Backtester")
    parser.add_argument("--csv", default="data/sample_1m.csv",
                        help="Path to OHLCV CSV file")
    parser.add_argument("--strategy", default="ema",
                        choices=list(STRATEGIES.keys()),
                        help="Strategy to backtest")
    parser.add_argument("--equity", type=float, default=10000.0,
                        help="Starting equity")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"CSV not found: {args.csv}")
        print("Create a CSV with columns: ts,open,high,low,close,volume")
        sys.exit(1)

    df = pd.read_csv(args.csv, parse_dates=["ts"]).set_index("ts").sort_index()
    print(f"Loaded {len(df)} bars from {args.csv}")
    print(f"Running strategy: {args.strategy.upper()}")
    print("-" * 50)

    result = backtest(df, strategy_name=args.strategy,
                      initial_equity=args.equity)

    print(f"Strategy:      {result['strategy'].upper()}")
    print(f"Initial equity: £{result['initial_equity']:,.2f}")
    print(f"Final equity:   £{result['final_equity']:,.2f}")
    print(f"Net P&L:        £{result['net_pnl']:,.2f}")
    print(f"Total trades:   {result['total_trades']}")
    print(f"Wins/Losses:    {result['wins']}/{result['losses']}")
    print(f"Win rate:       {result['win_rate']}%")
    print(f"Profit factor:  {result['profit_factor']}")


if __name__ == "__main__":
    main()
