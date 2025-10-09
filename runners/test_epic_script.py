import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.config import load_settings
from broker.ig_client import IGClient
from data.ig_price_bars import bars_from_ig
from strategy.ema_cross_breakout import EMACrossBreakout
import pandas as pd
import json


# Test script to verify EPICs work and see actual data
def main():
    cfg = load_settings()

    ig = IGClient(
        api_key=cfg["ig"]["api_key"],
        username=cfg["ig"]["username"],
        password=cfg["ig"]["password"],
        demo=True,
        verify_ssl=False
    )

    print("Logging in...")
    ig.login()
    print("✅ Logged in successfully\n")

    # Test EPICs
    test_epics = [
        "CS.D.EURUSD.CFD.IP",  # EUR/USD
        "CS.D.GBPUSD.CFD.IP",  # GBP/USD
        "IX.D.SPTRD.DAILY.IP",  # S&P 500
        "CS.D.CFEGOLD.CEB.IP",  # Gold
    ]

    # Initialize strategy
    strat = EMACrossBreakout(fast=9, slow=21, atr_period=14, rr_take=1.5)

    for epic in test_epics:
        print(f"\n{'=' * 70}")
        print(f"Testing: {epic}")
        print('=' * 70)

        try:
            # Get market details
            mkt = ig.market_details(epic)
            print(f"✅ Market: {mkt['instrument']['name']}")

            # Access dealing rules safely
            rules = mkt.get('dealingRules', {})

            # Min deal size
            min_size_info = rules.get('minDealSize', {})
            print(f"   Min deal size: {min_size_info.get('value', 'N/A')}")
            print(f"   Size step: {min_size_info.get('step', 'N/A')}")

            # Min stop distance - can be in different places
            min_stop = None
            if 'minStopOrLimitDistance' in rules:
                min_stop = rules['minStopOrLimitDistance'].get('value')
            elif 'minControlledRiskStopDistance' in rules:
                min_stop = rules['minControlledRiskStopDistance'].get('value')
            elif 'minNormalStopOrLimitDistance' in rules:
                min_stop = rules['minNormalStopOrLimitDistance'].get('value')

            print(f"   Min stop distance: {min_stop if min_stop else 'N/A'}")

            # Get 1-minute bars
            print(f"\n   Fetching price data...")
            prices = ig.get_prices(epic, resolution="MINUTE", max=100)
            df = bars_from_ig(prices)

            if df.empty:
                print("   ⚠️  No price data available")
                continue

            print(f"   ✅ Got {len(df)} bars")
            print(f"   Date range: {df.index[0]} to {df.index[-1]}")
            print(f"   Last close: {df['close'].iloc[-1]:.5f}")

            # Test strategy signal
            print(f"\n   Testing strategy...")
            signal = strat.on_bar(df)

            if signal:
                print(f"   🎯 SIGNAL DETECTED!")
                print(f"      Side: {signal['side']}")
                print(f"      Stop points: {signal['stop_pts']:.2f}")
                print(f"      TP points: {signal['tp_pts']:.2f}")
                print(f"      Meta: {signal.get('meta', {})}")
            else:
                print(f"   ⚪ No signal (waiting for setup)")

            # Show last few bars
            print(f"\n   Last 3 bars:")
            print(df[['open', 'high', 'low', 'close']].tail(3).to_string())

        except Exception as e:
            print(f"   ❌ Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("Test complete!")
    print("\nNext steps:")
    print("1. Choose an EPIC that has price data")
    print("2. Run the debug version of run_live.py")
    print("3. Check logs/bot.log for detailed signal information")


if __name__ == "__main__":
    main()