"""
scripts/generate_training_data.py — Generate the Historical Training CSV

PURPOSE:
    Downloads all required data from their respective sources, computes
    technical indicators, normalizes features, merges everything by date,
    and saves a single CSV file ready for Kaggle training.

DATA SOURCES:
    - BZ=F (Brent Crude Futures) -- yfinance (from 2007-07-30)
    - DX-Y.NYB (DXY / US Dollar Index) -- yfinance (from 1971)
    - DNSI (Daily News Sentiment Index) -- SF Fed via FRED API
    - Holidays -- 'holidays' Python library

OUTPUT:
    data/historical_features.csv — One row per trading day with columns:
        date, brent_close, brent_high, brent_low, brent_volume,
        dxy_close, holiday_flag, sentiment_score,
        z_brent, z_dxy,
        rsi_14, macd, macd_signal, bb_upper, bb_lower, bb_mid

USAGE:
    python scripts/generate_training_data.py

RUNTIME: ~1-2 minutes (mostly downloading from yfinance)
"""

import sys
import os

# Add project root to path so we can import utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from pathlib import Path

# Try to import FRED API — optional, falls back to manual download hint
try:
    from fredapi import Fred
    HAS_FREDAPI = True
except ImportError:
    HAS_FREDAPI = False

# Try to import holidays library
try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

from utils.z_score import compute_rolling_zscore, save_scaler_params


# ============================================================
# CONFIGURATION
# ============================================================

# Training data window
START_DATE = "2007-07-01"  # Slightly before BZ=F's earliest (2007-07-30)
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Output paths
DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_CSV = DATA_DIR / "historical_features.csv"
SCALER_JSON = DATA_DIR / "scaler_params.json"

# FRED API key from .env
FRED_API_KEY = os.getenv("FRED_API_KEY", "")


# ============================================================
# STEP 1: Download Brent Crude (BZ=F)
# ============================================================

def download_brent() -> pd.DataFrame:
    """Download BZ=F daily OHLCV from yfinance."""
    print("\n[1/6] Downloading BZ=F (Brent Crude Futures)...")

    df = yf.download("BZ=F", start=START_DATE, end=END_DATE, progress=True)

    if df.empty:
        raise RuntimeError("Failed to download BZ=F data from yfinance")

    # yfinance returns MultiIndex columns with (Price, Ticker) — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only what we need
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.columns = ["brent_open", "brent_high", "brent_low", "brent_close", "brent_volume"]

    # Drop rows where Close is NaN
    df = df.dropna(subset=["brent_close"])

    print(f"  Downloaded {len(df)} rows")
    print(f"  Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"  Price range: ${df['brent_close'].min():.2f} - ${df['brent_close'].max():.2f}")

    return df


# ============================================================
# STEP 2: Download DXY (US Dollar Index)
# ============================================================

def download_dxy() -> pd.DataFrame:
    """Download DX-Y.NYB daily close from yfinance."""
    print("\n[2/6] Downloading DX-Y.NYB (US Dollar Index / DXY)...")

    df = yf.download("DX-Y.NYB", start=START_DATE, end=END_DATE, progress=True)

    if df.empty:
        raise RuntimeError("Failed to download DXY data from yfinance")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].copy()
    df.columns = ["dxy_close"]
    df = df.dropna(subset=["dxy_close"])

    print(f"  Downloaded {len(df)} rows")
    print(f"  Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")

    return df


# ============================================================
# STEP 3: Download Sentiment (SF Fed DNSI via FRED API)
# ============================================================

def download_sentiment() -> pd.DataFrame:
    """Download DNSI from FRED and normalize to [-1, +1]."""
    print("\n[3/6] Downloading DNSI (Daily News Sentiment Index)...")

    if not FRED_API_KEY:
        # Try loading from .env file manually
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            with open(env_path, encoding="utf-8") as f:
                for line in f:
                    if line.startswith("FRED_API_KEY="):
                        key = line.strip().split("=", 1)[1]
                        if key:
                            return _download_sentiment_with_key(key)

        print("  WARNING: No FRED_API_KEY found. Generating synthetic sentiment placeholder.")
        print("  To fix: add FRED_API_KEY=your_key to .env file")
        print("  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return _generate_placeholder_sentiment()

    return _download_sentiment_with_key(FRED_API_KEY)


def _download_sentiment_with_key(api_key: str) -> pd.DataFrame:
    """Download DNSI using FRED API."""
    try:
        from fredapi import Fred
    except ImportError:
        print("  fredapi not installed. Installing...")
        os.system(f"{sys.executable} -m pip install fredapi --quiet")
        from fredapi import Fred

    fred = Fred(api_key=api_key)

    # VIXCLS is the CBOE Volatility Index — we use this as a readily available
    # daily sentiment proxy since DNSI may not be available via standard FRED API.
    # Try DNSI first (series may be named differently on FRED).
    series_to_try = [
        ("NEWSENTWRD", "Daily News Sentiment Index"),
        ("VIXCLS", "CBOE Volatility Index (VIX) -- used as fear/greed proxy"),
    ]

    for series_id, description in series_to_try:
        try:
            print(f"  Trying FRED series: {series_id} ({description})...")
            data = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
            if data is not None and len(data) > 100:
                df = pd.DataFrame({"raw_sentiment": data})
                df.index.name = "Date"
                df = df.dropna()

                # Normalize to [-1, +1] using Min-Max scaling
                s_min = df["raw_sentiment"].min()
                s_max = df["raw_sentiment"].max()
                if s_max > s_min:
                    df["sentiment_score"] = -1.0 + 2.0 * (df["raw_sentiment"] - s_min) / (s_max - s_min)
                else:
                    df["sentiment_score"] = 0.0

                # If using VIX, INVERT it — high VIX = negative sentiment
                if series_id == "VIXCLS":
                    df["sentiment_score"] = -df["sentiment_score"]

                print(f"  Downloaded {len(df)} rows from {series_id}")
                print(f"  Raw range: {s_min:.4f} to {s_max:.4f}")
                print(f"  Normalized range: {df['sentiment_score'].min():.4f} to {df['sentiment_score'].max():.4f}")

                return df[["sentiment_score"]]
        except Exception as e:
            print(f"  {series_id} failed: {e}")
            continue

    print("  All FRED series failed. Using placeholder sentiment.")
    return _generate_placeholder_sentiment()


def _generate_placeholder_sentiment() -> pd.DataFrame:
    """
    Generate a placeholder sentiment series.
    This uses a simple rolling correlation between oil returns and a noise signal.
    It's ONLY used if FRED is completely unavailable.
    """
    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    np.random.seed(42)  # Reproducible
    sentiment = np.random.normal(0, 0.3, len(dates)).cumsum()
    # Normalize to [-1, 1]
    sentiment = np.clip(sentiment / np.abs(sentiment).max(), -1, 1)
    df = pd.DataFrame({"sentiment_score": sentiment}, index=dates)
    df.index.name = "Date"
    print(f"  Generated {len(df)} placeholder sentiment values")
    return df


# ============================================================
# STEP 4: Generate Holiday Flags
# ============================================================

def generate_holidays(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate binary holiday flags for each trading day."""
    print("\n[4/6] Generating holiday flags...")

    if not HAS_HOLIDAYS:
        print("  'holidays' library not installed. Installing...")
        os.system(f"{sys.executable} -m pip install holidays --quiet")
        import holidays as holidays_lib
    else:
        holidays_lib = holidays

    # Global holidays that affect oil markets (per spec Section 3.1)
    # US, India, China, UK — major oil consuming/producing nations
    us_holidays = holidays_lib.US(years=range(2007, 2027))
    india_holidays = holidays_lib.India(years=range(2007, 2027))
    china_holidays = holidays_lib.China(years=range(2007, 2027))
    uk_holidays = holidays_lib.UK(years=range(2007, 2027))

    all_holiday_dates = set()
    for h in [us_holidays, india_holidays, china_holidays, uk_holidays]:
        all_holiday_dates.update(h.keys())

    flags = []
    for date in date_index:
        flag = 1 if date.date() in all_holiday_dates else 0
        flags.append(flag)

    df = pd.DataFrame({"holiday_flag": flags}, index=date_index)
    df.index.name = "Date"

    total_holidays = sum(flags)
    print(f"  Flagged {total_holidays} holiday days out of {len(flags)} trading days")

    return df


# ============================================================
# STEP 5: Compute Technical Indicators
# ============================================================

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators from Brent price data.
    These are additional features that help the model learn patterns.
    """
    print("\n[5/6] Computing technical indicators...")

    close = df["brent_close"]
    high = df["brent_high"]
    low = df["brent_low"]

    # RSI (14-day)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.rolling(window=14, min_periods=1).mean()
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands (20-day, 2 std)
    df["bb_mid"] = close.rolling(window=20, min_periods=1).mean()
    bb_std = close.rolling(window=20, min_periods=1).std().fillna(0)
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std

    # Volatility spread (High - Low) — this is a target variable too
    df["volatility_spread"] = high - low

    # Weekly MA crossover — 5-day MA vs 20-day MA
    ma_5 = close.rolling(window=5, min_periods=1).mean()
    ma_20 = close.rolling(window=20, min_periods=1).mean()
    df["ma_crossover"] = (ma_5 > ma_20).astype(int)

    print("  Computed: RSI(14), MACD(12,26,9), Bollinger Bands(20,2), Volatility Spread, MA Crossover")

    return df


# ============================================================
# STEP 6: Merge, Normalize, and Save
# ============================================================

def merge_and_save(
    brent: pd.DataFrame,
    dxy: pd.DataFrame,
    sentiment: pd.DataFrame,
    holiday_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources, compute Z-scores, and save CSV."""
    print("\n[6/6] Merging all data and computing Z-scores...")

    # Start with Brent as the base (it has the fewest rows)
    merged = brent.copy()

    # Join DXY — forward-fill gaps where DXY has data but Brent doesn't (or vice versa)
    merged = merged.join(dxy, how="left")
    merged["dxy_close"] = merged["dxy_close"].ffill()

    # Join sentiment — forward-fill for days where sentiment wasn't published
    merged = merged.join(sentiment, how="left")
    merged["sentiment_score"] = merged["sentiment_score"].ffill().fillna(0.0)

    # Join holidays
    merged = merged.join(holiday_flags, how="left")
    merged["holiday_flag"] = merged["holiday_flag"].fillna(0).astype(int)

    # Drop any rows where brent_close is still NaN (shouldn't happen but be safe)
    merged = merged.dropna(subset=["brent_close"])

    print(f"  Merged dataset: {len(merged)} rows")
    print(f"  Date range: {merged.index.min().strftime('%Y-%m-%d')} to {merged.index.max().strftime('%Y-%m-%d')}")

    # Compute technical indicators
    merged = compute_technical_indicators(merged)

    # Compute Z-Scores (per spec: 30-day rolling window)
    print("\n  Computing 30-day rolling Z-scores...")
    merged["z_brent"], brent_means, brent_stds = compute_rolling_zscore(merged["brent_close"], window=30)
    merged["z_dxy"], dxy_means, dxy_stds = compute_rolling_zscore(merged["dxy_close"], window=30)

    # Save scaler params (last μ and σ for live inference)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_scaler_params(brent_means, brent_stds, SCALER_JSON, feature_name="brent")
    save_scaler_params(dxy_means, dxy_stds, SCALER_JSON, feature_name="dxy")

    # Reset index so Date becomes a column
    merged.index.name = "date"
    merged = merged.reset_index()

    # Reorder columns for clarity
    column_order = [
        # Date
        "date",
        # Raw prices (for reference and target computation)
        "brent_open", "brent_high", "brent_low", "brent_close", "brent_volume",
        "dxy_close",
        # Categorical / external features
        "holiday_flag", "sentiment_score",
        # Z-Score normalized features (model inputs)
        "z_brent", "z_dxy",
        # Technical indicators
        "rsi_14", "macd", "macd_signal", "bb_upper", "bb_lower", "bb_mid",
        # Target-related
        "volatility_spread", "ma_crossover",
    ]
    # Only keep columns that actually exist
    column_order = [c for c in column_order if c in merged.columns]
    merged = merged[column_order]

    # Save to CSV
    merged.to_csv(OUTPUT_CSV, index=False)
    file_size_mb = OUTPUT_CSV.stat().st_size / (1024 * 1024)

    print(f"\n  [OK] Saved to: {OUTPUT_CSV}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print(f"  Rows: {len(merged)}")
    print(f"  Columns: {len(merged.columns)}")
    print(f"  Columns: {list(merged.columns)}")

    # Print summary statistics
    print("\n  === Summary Statistics ===")
    print(merged.describe().to_string())

    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  AGENT-B: Training Data Generation")
    print("  This creates the CSV you'll upload to Kaggle")
    print("=" * 60)

    # Step 1: Download Brent
    brent = download_brent()

    # Step 2: Download DXY
    dxy = download_dxy()

    # Step 3: Download Sentiment
    sentiment = download_sentiment()

    # Step 4: Generate Holiday Flags
    holiday_flags = generate_holidays(brent.index)

    # Step 5-6: Merge, compute indicators & Z-scores, save
    merged = merge_and_save(brent, dxy, sentiment, holiday_flags)

    print("\n" + "=" * 60)
    print("  DONE! Next steps:")
    print("  1. Check the CSV at: data/historical_features.csv")
    print("  2. Upload it to Kaggle as a new Dataset")
    print("  3. Run the training notebook on Kaggle GPU")
    print("=" * 60)


if __name__ == "__main__":
    main()
