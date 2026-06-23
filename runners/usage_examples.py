"""
Complete Usage Examples for Multi-Source Data Provider
Shows how to use the system in different scenarios
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.multi_data_provider import create_data_aggregator
from broker.ig_client import IGClient


# =====================================================================
# EXAMPLE 1: Basic Usage (FREE - Yahoo Finance Only)
# =====================================================================

def example_1_basic_free():
    """
    Simplest setup - uses only Yahoo Finance (100% FREE)
    Perfect for: Testing, learning, casual trading
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Basic Setup (FREE - Yahoo Finance)")
    print("=" * 70 + "\n")

    # Create aggregator with no API keys
    aggregator = create_data_aggregator()

    # Get Gold data
    print("Fetching Gold (XAUUSD) data...")
    gold_df = aggregator.get_bars("CS.D.CFEGOLD.CEB.IP", timeframe="5min", limit=100)

    if not gold_df.empty:
        print(f"✓ Success! Got {len(gold_df)} bars")
        print(f"  Latest price: ${gold_df['close'].iloc[-1]:.2f}")
        print(f"  Timestamp: {gold_df.index[-1]}")
    else:
        print("✗ Failed to get data")

    # Get S&P 500 data
    print("\nFetching S&P 500 data...")
    sp500_df = aggregator.get_bars("IX.D.SPTRD.DAILY.IP", timeframe="5min", limit=100)

    if not sp500_df.empty:
        print(f"✓ Success! Got {len(sp500_df)} bars")
        print(f"  Latest price: ${sp500_df['close'].iloc[-1]:.2f}")

    # Show statistics
    stats = aggregator.get_stats()
    print(f"\n📊 Statistics:")
    print(f"  Cached symbols: {stats['cached_symbols']}")
    print(f"  Active providers: {stats['providers_active']}")
    print(f"  Source usage: {stats.get('source_usage', {})}")


# =====================================================================
# EXAMPLE 2: With TwelveData API (Free Tier)
# =====================================================================

def example_2_with_twelvedata():
    """
    Using TwelveData free tier (800 requests/day)
    Perfect for: More reliable real-time data
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 2: With TwelveData Free Tier (800 req/day)")
    print("=" * 70 + "\n")

    # Create aggregator with TwelveData key
    aggregator = create_data_aggregator(
        twelve_data_key=os.getenv("TWELVE_DATA_KEY")  # Add to .env
    )

    instruments = [
        ("CS.D.CFEGOLD.CEB.IP", "Gold"),
        ("CS.D.EURUSD.CFD.IP", "EUR/USD"),
        ("IX.D.FTSE.CFD.IP", "FTSE 100")
    ]

    for epic, name in instruments:
        print(f"Fetching {name}...")
        df = aggregator.get_bars(epic, timeframe="5min", limit=50)

        if not df.empty:
            print(f"  ✓ {len(df)} bars | Latest: {df['close'].iloc[-1]:.4f}")
        else:
            print(f"  ✗ No data")

        time.sleep(0.5)  # Be nice to the API

    # Show which sources were used
    stats = aggregator.get_stats()
    print(f"\n📊 Data sources used: {stats.get('source_usage', {})}")


# =====================================================================
# EXAMPLE 3: Production Setup with Fallbacks
# =====================================================================

def example_3_production_setup():
    """
    Full production setup with multiple fallbacks
    Perfect for: 24/7 automated trading
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Production Setup (Multiple Sources)")
    print("=" * 70 + "\n")

    # Create aggregator with multiple API keys
    aggregator = create_data_aggregator(
        twelve_data_key=os.getenv("TWELVE_DATA_KEY"),
        alpha_vantage_key=os.getenv("ALPHA_VANTAGE_KEY"),
        fmp_key=os.getenv("FMP_KEY")
    )

    print(f"Active providers: {aggregator.get_stats()['providers_active']}")
    print("\nTesting failover mechanism...\n")

    # Test Gold with all sources
    epic = "CS.D.CFEGOLD.CEB.IP"

    for i in range(3):
        print(f"Attempt {i + 1}:")
        df = aggregator.get_bars(epic, timeframe="5min", limit=20)

        if not df.empty:
            # Check which source was used
            cache_key = f"{epic}:5min"
            source = aggregator.cache.get(cache_key, {}).get("source", "Unknown")
            print(f"  ✓ Data from: {source}")
            print(f"  Latest: ${df['close'].iloc[-1]:.2f}")

        time.sleep(2)


# =====================================================================
# EXAMPLE 4: Real-Time Trading Bot Integration
# =====================================================================

def example_4_trading_bot():
    """
    How to integrate with actual trading bot
    Perfect for: Live trading implementation
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Trading Bot Integration")
    print("=" * 70 + "\n")

    # Initialize IG client (optional - for execution only)
    try:
        from dotenv import load_dotenv
        load_dotenv()

        ig = IGClient(
            api_key=os.getenv("IG_API_KEY"),
            username=os.getenv("IG_USERNAME"),
            password=os.getenv("IG_PASSWORD"),
            demo=True,
            verify_ssl=False
        )
        ig.login()
        print("✓ Connected to IG (for execution)")
    except Exception as e:
        print(f"⚠ IG not available: {e}")
        ig = None

    # Create aggregator with IG as fallback
    aggregator = create_data_aggregator(
        ig_client=ig,
        twelve_data_key=os.getenv("TWELVE_DATA_KEY")
    )

    # Symbols to monitor
    symbols = [
        "CS.D.CFEGOLD.CEB.IP",
        "CS.D.EURUSD.CFD.IP",
        "IX.D.SPTRD.DAILY.IP"
    ]

    print(f"\nMonitoring {len(symbols)} instruments...")
    print("(Press Ctrl+C to stop)\n")

    try:
        for cycle in range(5):  # Just 5 cycles for demo
            print(f"--- Cycle {cycle + 1} ---")

            for epic in symbols:
                # Get latest data
                df = aggregator.get_bars(epic, timeframe="5min", limit=100)

                if not df.empty:
                    price = df['close'].iloc[-1]
                    change = ((df['close'].iloc[-1] - df['close'].iloc[-2]) /
                              df['close'].iloc[-2] * 100)

                    # Simple analysis
                    sma_20 = df['close'].tail(20).mean()
                    signal = "BUY" if price > sma_20 else "SELL"

                    print(f"  {epic[:20]:20} | ${price:8.2f} | "
                          f"{change:+.2f}% | Signal: {signal}")

            print()
            time.sleep(10)  # Wait 10 seconds

    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")

    # Show final stats
    stats = aggregator.get_stats()
    print(f"\n📊 Session Statistics:")
    print(f"  Total cached: {stats['cached_symbols']}")
    print(f"  Sources used: {stats.get('source_usage', {})}")


# =====================================================================
# EXAMPLE 5: Multi-Timeframe Analysis
# =====================================================================

def example_5_multi_timeframe():
    """
    Analyze multiple timeframes simultaneously
    Perfect for: Trend confirmation across timeframes
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Multi-Timeframe Analysis")
    print("=" * 70 + "\n")

    aggregator = create_data_aggregator(
        twelve_data_key=os.getenv("TWELVE_DATA_KEY")
    )

    epic = "CS.D.CFEGOLD.CEB.IP"
    timeframes = ["5min", "15min", "60min"]

    print(f"Analyzing Gold across {len(timeframes)} timeframes...\n")

    results = {}

    for tf in timeframes:
        print(f"Fetching {tf} data...")
        df = aggregator.get_bars(epic, timeframe=tf, limit=100)

        if not df.empty:
            # Calculate key metrics
            current_price = df['close'].iloc[-1]
            sma_20 = df['close'].tail(20).mean()
            sma_50 = df['close'].tail(50).mean()

            trend = "BULLISH" if sma_20 > sma_50 else "BEARISH"
            position = "ABOVE" if current_price > sma_20 else "BELOW"

            results[tf] = {
                'price': current_price,
                'trend': trend,
                'position': position,
                'sma_20': sma_20,
                'sma_50': sma_50
            }

            print(f"  ✓ {trend} trend | Price {position} SMA(20)")

    # Summary
    print("\n📊 Multi-Timeframe Summary:")
    bullish_count = sum(1 for r in results.values() if r['trend'] == 'BULLISH')

    print(f"  Bullish timeframes: {bullish_count}/{len(results)}")
    print(f"  Current price: ${results['5min']['price']:.2f}")

    if bullish_count >= 2:
        print(f"  ✓ CONFLUENCE: Multiple timeframes confirm BULLISH")
    elif bullish_count <= 1:
        print(f"  ✓ CONFLUENCE: Multiple timeframes confirm BEARISH")
    else:
        print(f"  ⚠ MIXED SIGNALS: No clear trend")


# =====================================================================
# EXAMPLE 6: Data Quality Monitoring
# =====================================================================

def example_6_quality_monitoring():
    """
    Monitor data quality and source reliability
    Perfect for: Ensuring data integrity
    """
    print("\n" + "=" * 70)
    print("EXAMPLE 6: Data Quality Monitoring")
    print("=" * 70 + "\n")

    aggregator = create_data_aggregator(
        twelve_data_key=os.getenv("TWELVE_DATA_KEY"),
        fmp_key=os.getenv("FMP_KEY")
    )

    test_symbols = [
        ("CS.D.CFEGOLD.CEB.IP", "Gold"),
        ("CS.D.EURUSD.CFD.IP", "EUR/USD"),
        ("IX.D.SPTRD.DAILY.IP", "S&P 500")
    ]

    quality_report = []

    for epic, name in test_symbols:
        print(f"Testing {name}...")

        start_time = time.time()
        df = aggregator.get_bars(epic, timeframe="5min", limit=100)
        elapsed = time.time() - start_time

        if not df.empty:
            # Check data quality
            missing_data = df.isnull().sum().sum()
            duplicates = df.index.duplicated().sum()

            # Get source
            cache_key = f"{epic}:5min"
            source = aggregator.cache.get(cache_key, {}).get("source", "Unknown")

            quality = {
                'name': name,
                'bars': len(df),
                'source': source,
                'latency_ms': int(elapsed * 1000),
                'missing_data': missing_data,
                'duplicates': duplicates,
                'status': '✓ Good' if missing_data == 0 and duplicates == 0 else '⚠ Issues'
            }

            quality_report.append(quality)

            print(f"  {quality['status']} | {quality['bars']} bars | "
                  f"{quality['latency_ms']}ms | Source: {source}")
        else:
            print(f"  ✗ Failed")

    # Summary
    print("\n📊 Data Quality Report:")
    avg_latency = sum(q['latency_ms'] for q in quality_report) / len(quality_report)
    print(f"  Average latency: {avg_latency:.0f}ms")
    print(f"  Success rate: {len(quality_report)}/{len(test_symbols)}")

    sources_used = set(q['source'] for q in quality_report)
    print(f"  Sources used: {', '.join(sources_used)}")


# =====================================================================
# MAIN MENU
# =====================================================================

def main():
    """Run example selection menu"""
    print("\n" + "=" * 70)
    print("Multi-Source Data Provider - Usage Examples")
    print("=" * 70)

    examples = [
        ("Basic Usage (FREE - Yahoo Finance)", example_1_basic_free),
        ("With TwelveData Free Tier", example_2_with_twelvedata),
        ("Production Setup with Fallbacks", example_3_production_setup),
        ("Trading Bot Integration", example_4_trading_bot),
        ("Multi-Timeframe Analysis", example_5_multi_timeframe),
        ("Data Quality Monitoring", example_6_quality_monitoring),
    ]

    print("\nSelect an example to run:\n")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  0. Run all examples")
    print()

    try:
        choice = input("Enter choice (0-6): ").strip()

        if choice == "0":
            print("\nRunning all examples...\n")
            for name, func in examples:
                try:
                    func()
                    time.sleep(2)
                except Exception as e:
                    print(f"✗ Error in {name}: {e}")
        else:
            idx = int(choice) - 1
            if 0 <= idx < len(examples):
                examples[idx][1]()
            else:
                print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\nCancelled by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()