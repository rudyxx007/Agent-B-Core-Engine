"""
scripts/generate_training_data.py — Generate the Historical Training CSV (v2)

PURPOSE:
    Downloads all required data from their respective sources, computes
    technical indicators, normalizes features, merges everything by date,
    and saves a single CSV file ready for Kaggle training.

DATA SOURCES:
    - BZ=F (Brent Crude Futures) -- yfinance (from 2007-07-30)
    - CL=F (WTI Crude Futures) -- yfinance (for Brent-WTI spread)
    - RB=F (RBOB Gasoline) -- yfinance (for crack spread)
    - HO=F (Heating Oil/Diesel proxy) -- yfinance (for crack spread)
    - DX-Y.NYB (DXY / US Dollar Index) -- yfinance
    - VIXCLS (VIX as sentiment proxy) -- FRED API
    - WCESTUS1 (Weekly Crude Inventories) -- EIA API v2
    - Holidays -- 'holidays' Python library

OUTPUT:
    data/historical_features.csv — One row per trading day with columns:
        date, brent_close, ..., brent_wti_spread, crack_spread_321,
        eia_inventory, eia_inventory_change, z_brent, z_dxy, ...

USAGE:
    python scripts/generate_training_data.py

RUNTIME: ~2-3 minutes (mostly downloading from yfinance + EIA)
"""

import sys
import os

# Add project root to path so we can import utils/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
import requests
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

# API keys from .env
FRED_API_KEY = os.getenv("FRED_API_KEY", "")
EIA_API_KEY = os.getenv("EIA_API_KEY", "")

def _load_env_key(key_name: str) -> str:
    """Try to load an API key from .env file."""
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key_name}="):
                    return line.split("=", 1)[1].strip()
    return ""


# ============================================================
# STEP 1: Download Brent Crude (BZ=F)
# ============================================================

def download_brent() -> pd.DataFrame:
    """Download BZ=F daily OHLCV from yfinance."""
    print("\n[1/9] Downloading BZ=F (Brent Crude Futures)...")

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
    print("\n[2/9] Downloading DX-Y.NYB (US Dollar Index / DXY)...")

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
# STEP 3: Download WTI + Products (for spreads & crack spread)
# ============================================================

def download_wti_and_products() -> pd.DataFrame:
    """Download CL=F (WTI), RB=F (Gasoline), HO=F (Heating Oil/Diesel)."""
    print("\n[3/9] Downloading WTI, Gasoline, Heating Oil futures...")

    tickers = {
        "CL=F": "wti_close",
        "RB=F": "gasoline_close",   # $/gallon
        "HO=F": "heating_oil_close",  # $/gallon (diesel proxy)
    }

    frames = []
    for ticker, col_name in tickers.items():
        df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=True)
        if df.empty:
            print(f"  WARNING: {ticker} returned no data!")
            continue

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[["Close"]].copy()
        df.columns = [col_name]
        df = df.dropna()
        frames.append(df)
        print(f"  {ticker} ({col_name}): {len(df)} rows")

    if not frames:
        raise RuntimeError("Failed to download WTI/product futures data")

    # Join all on date index
    result = frames[0]
    for f in frames[1:]:
        result = result.join(f, how="outer")

    return result


# ============================================================
# STEP 4A: Download VIX (Fear Gauge via FRED API)
# ============================================================

def download_vix() -> pd.DataFrame:
    """Download VIX from FRED (now used as an independent feature, not a proxy)."""
    print("\n[4/10] Downloading VIX (Fear Gauge via FRED)...")

    api_key = FRED_API_KEY or _load_env_key("FRED_API_KEY")

    if not api_key:
        print("  WARNING: No FRED_API_KEY found. Generating synthetic VIX placeholder.")
        return _generate_placeholder_vix()

    return _download_vix_with_key(api_key)


def _download_vix_with_key(api_key: str) -> pd.DataFrame:
    """Download VIX using FRED API."""
    try:
        from fredapi import Fred
    except ImportError:
        print("  fredapi not installed. Installing...")
        os.system(f"{sys.executable} -m pip install fredapi --quiet")
        from fredapi import Fred

    fred = Fred(api_key=api_key)
    series_id = "VIXCLS"
    try:
        data = fred.get_series(series_id, observation_start=START_DATE, observation_end=END_DATE)
        if data is not None and len(data) > 100:
            df = pd.DataFrame({"vix_close": data})
            df.index.name = "Date"
            df = df.dropna()
            print(f"  Downloaded {len(df)} rows of VIX")
            return df[["vix_close"]]
    except Exception as e:
        print(f"  VIX download failed: {e}")

    return _generate_placeholder_vix()


def _generate_placeholder_vix() -> pd.DataFrame:
    """Placeholder VIX if FRED is unavailable."""
    dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    df = pd.DataFrame({"vix_close": 20.0}, index=dates) # Default to 20
    df.index.name = "Date"
    return df


# ============================================================
# STEP 4B: Load FinGPT Sentiment
# ============================================================

def load_fingpt_sentiment() -> pd.DataFrame:
    """Load the historical FinGPT sentiment scores (from Colab Phase 2.5)."""
    print("\n[5/10] Loading FinGPT Historical Sentiment Scores...")
    csv_path = DATA_DIR / "fingpt_historical_scores.csv"
    
    if not csv_path.exists():
        print("  WARNING: fingpt_historical_scores.csv not found in data/!")
        print("  You must run the Colab notebook first and place the CSV in data/.")
        print("  Generating zero-filled placeholder for sentiment.")
        dates = pd.bdate_range(start=START_DATE, end=END_DATE)
        df = pd.DataFrame({"sentiment_score": 0.0}, index=dates)
        df.index.name = "Date"
        return df

    df = pd.read_csv(csv_path)
    df["Date"] = pd.to_datetime(df["date"])
    df = df.set_index("Date").sort_index()
    # Assume column is 'fingpt_sentiment'
    df = df.rename(columns={"fingpt_sentiment": "sentiment_score"})
    
    print(f"  Loaded {len(df)} sentiment scores from FinGPT")
    return df[["sentiment_score"]]


# ============================================================
# STEP 5: Download EIA Weekly Crude Oil Inventories
# ============================================================

def download_eia_inventory() -> pd.DataFrame:
    """
    Download US Weekly Crude Oil Ending Stocks from EIA API v2.
    This is the single most market-moving weekly oil data point.
    Released every Wednesday by the US Energy Information Administration.
    """
    print("\n[6/10] Downloading EIA Weekly Crude Oil Inventories...")

    api_key = EIA_API_KEY or _load_env_key("EIA_API_KEY")

    if not api_key:
        print("  WARNING: No EIA_API_KEY found.")
        print("  To fix: Register at https://www.eia.gov/opendata/ and add EIA_API_KEY=your_key to .env")
        print("  Generating zero-filled placeholder (this feature will be inactive).")
        dates = pd.bdate_range(start=START_DATE, end=END_DATE)
        df = pd.DataFrame({"eia_inventory": 0.0, "eia_inventory_change": 0.0}, index=dates)
        df.index.name = "Date"
        return df

    try:
        # EIA API v2 endpoint for weekly petroleum stocks
        url = "https://api.eia.gov/v2/petroleum/stoc/wstk/data/"
        params = {
            "api_key": api_key,
            "frequency": "weekly",
            "facets[series][]": "WCESTUS1",  # US Ending Stocks of Crude Oil
            "data[]": "value",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "start": START_DATE,
            "end": END_DATE,
            "length": 5000,
        }

        print(f"  Fetching from EIA API v2 (series: WCESTUS1)...")
        response = requests.get(url, params=params, timeout=30)

        if response.status_code != 200:
            print(f"  EIA API error: {response.status_code}")
            print(f"  Falling back to zero-filled placeholder.")
            dates = pd.bdate_range(start=START_DATE, end=END_DATE)
            return pd.DataFrame({"eia_inventory": 0.0, "eia_inventory_change": 0.0}, index=dates)

        data = response.json()
        records = data.get("response", {}).get("data", [])

        if not records:
            print("  No records returned from EIA API.")
            dates = pd.bdate_range(start=START_DATE, end=END_DATE)
            return pd.DataFrame({"eia_inventory": 0.0, "eia_inventory_change": 0.0}, index=dates)

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["period"])
        df["eia_inventory"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.set_index("Date").sort_index()
        df = df[["eia_inventory"]].dropna()

        # Compute week-over-week change (the actual market-moving number)
        df["eia_inventory_change"] = df["eia_inventory"].diff()

        # Normalize inventory to millions of barrels for readability
        # EIA reports in thousands of barrels
        df["eia_inventory"] = df["eia_inventory"] / 1000.0  # Now in millions of barrels
        df["eia_inventory_change"] = df["eia_inventory_change"] / 1000.0

        print(f"  Downloaded {len(df)} weekly readings")
        print(f"  Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
        print(f"  Inventory range: {df['eia_inventory'].min():.1f}M - {df['eia_inventory'].max():.1f}M barrels")

        return df[["eia_inventory", "eia_inventory_change"]]

    except Exception as e:
        print(f"  EIA download failed: {e}")
        dates = pd.bdate_range(start=START_DATE, end=END_DATE)
        return pd.DataFrame({"eia_inventory": 0.0, "eia_inventory_change": 0.0}, index=dates)


# ============================================================
# STEP 6: Generate Holiday Flags
# ============================================================

def generate_holidays(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate binary holiday flags for each trading day."""
    print("\n[7/10] Generating holiday flags...")

    if not HAS_HOLIDAYS:
        print("  'holidays' library not installed. Installing...")
        os.system(f"{sys.executable} -m pip install holidays --quiet")
        import holidays as holidays_lib
    else:
        holidays_lib = holidays

    # Global holidays that affect oil markets (per spec Section 3.1)
    us_holidays = holidays_lib.US(years=range(2007, 2028))
    india_holidays = holidays_lib.India(years=range(2007, 2028))
    china_holidays = holidays_lib.China(years=range(2007, 2028))
    uk_holidays = holidays_lib.UK(years=range(2007, 2028))

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
# STEP 7: Compute Technical Indicators
# ============================================================

def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators from Brent price data.
    These are additional features that help the model learn patterns.
    """
    print("\n[8/10] Computing technical indicators...")

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
# STEP 8: Compute Spreads & Crack Spread
# ============================================================

def compute_spreads(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute market structure features:
    1. Brent-WTI spread — regional supply dynamics
    2. 3-2-1 Crack spread — RIL refining margin (Brent-based)

    RIL Jamnagar produces: Diesel (#1), Gasoline, ATF, Naphtha, LPG
    The 3-2-1 crack spread (2 bbl gasoline + 1 bbl diesel from 3 bbl crude)
    directly measures refining profitability.

    When crack spreads widen → refiners buy more crude → upward Brent pressure.
    When crack spreads narrow → refiners cut runs → downward Brent pressure.
    """
    print("\n[9/10] Computing market structure spreads...")

    # Brent-WTI spread
    if "wti_close" in df.columns:
        df["brent_wti_spread"] = df["brent_close"] - df["wti_close"]
        avg_spread = df["brent_wti_spread"].mean()
        print(f"  Brent-WTI spread: avg=${avg_spread:.2f}/bbl")
    else:
        df["brent_wti_spread"] = 0.0
        print("  WARNING: WTI data missing, Brent-WTI spread set to 0")

    # 3-2-1 Crack Spread (Brent-based, for RIL)
    # Formula: ((2 * Gasoline_gal * 42) + (1 * HeatingOil_gal * 42) - (3 * Brent_bbl)) / 3
    # RB=F and HO=F are quoted in $/gallon, BZ=F in $/barrel
    # 42 gallons per barrel
    if "gasoline_close" in df.columns and "heating_oil_close" in df.columns:
        df["crack_spread_321"] = (
            (2 * df["gasoline_close"] * 42) +
            (1 * df["heating_oil_close"] * 42) -
            (3 * df["brent_close"])
        ) / 3
        avg_crack = df["crack_spread_321"].mean()
        print(f"  3-2-1 Crack spread (Brent-based): avg=${avg_crack:.2f}/bbl")
        print(f"  This is RIL Jamnagar's approximate refining margin")
    else:
        df["crack_spread_321"] = 0.0
        print("  WARNING: Product futures data missing, crack spread set to 0")

    return df


# ============================================================
# STEP 9: Merge, Normalize, and Save
# ============================================================

def merge_and_save(
    brent: pd.DataFrame,
    dxy: pd.DataFrame,
    wti_products: pd.DataFrame,
    vix: pd.DataFrame,
    fingpt: pd.DataFrame,
    holiday_flags: pd.DataFrame,
    eia_inventory: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources, compute Z-scores, and save CSV."""
    print("\n[10/10] Merging all data and computing Z-scores...")

    # Start with Brent as the base (it has the fewest rows)
    merged = brent.copy()

    # Join DXY — forward-fill gaps
    merged = merged.join(dxy, how="left")
    merged["dxy_close"] = merged["dxy_close"].ffill()

    # Join WTI & Products — forward-fill
    merged = merged.join(wti_products, how="left")
    for col in ["wti_close", "gasoline_close", "heating_oil_close"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()

    # Join VIX
    merged = merged.join(vix, how="left")
    merged["vix_close"] = merged["vix_close"].ffill()

    # Join FinGPT sentiment
    merged = merged.join(fingpt, how="left")
    merged["sentiment_score"] = merged["sentiment_score"].ffill().fillna(0.0)

    # Join holidays
    merged = merged.join(holiday_flags, how="left")
    merged["holiday_flag"] = merged["holiday_flag"].fillna(0).astype(int)

    # Join EIA inventory — forward-fill weekly data to daily
    merged = merged.join(eia_inventory, how="left")
    merged["eia_inventory"] = merged["eia_inventory"].ffill().fillna(0.0)
    merged["eia_inventory_change"] = merged["eia_inventory_change"].ffill().fillna(0.0)

    # Drop any rows where brent_close is still NaN
    merged = merged.dropna(subset=["brent_close"])

    print(f"  Merged dataset: {len(merged)} rows")
    print(f"  Date range: {merged.index.min().strftime('%Y-%m-%d')} to {merged.index.max().strftime('%Y-%m-%d')}")

    # Compute technical indicators
    merged = compute_technical_indicators(merged)

    # Compute spreads & crack spread
    merged = compute_spreads(merged)

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
        # WTI & Products (for reference, used to compute spreads)
        "wti_close", "gasoline_close", "heating_oil_close",
        # Categorical / external features
        "holiday_flag", "vix_close", "sentiment_score",
        # NEW: Market structure features
        "brent_wti_spread", "crack_spread_321",
        # NEW: EIA inventory
        "eia_inventory", "eia_inventory_change",
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

    # Print summary statistics for new features
    print("\n  === New Feature Summary ===")
    for col in ["brent_wti_spread", "crack_spread_321", "eia_inventory", "eia_inventory_change"]:
        if col in merged.columns:
            s = merged[col]
            print(f"  {col:<25}: mean={s.mean():>8.2f}, std={s.std():>8.2f}, min={s.min():>8.2f}, max={s.max():>8.2f}")

    return merged


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("  AGENT-B: Training Data Generation v2")
    print("  Now with: Brent-WTI spread, Crack spread, EIA inventory")
    print("  Now with: True FinGPT Sentiment & Independent VIX")
    print("=" * 60)

    # Step 1: Download Brent
    brent = download_brent()

    # Step 2: Download DXY
    dxy = download_dxy()

    # Step 3: Download WTI + Products
    wti_products = download_wti_and_products()

    # Step 4: Download VIX
    vix = download_vix()

    # Step 5: Load FinGPT
    fingpt = load_fingpt_sentiment()

    # Step 6: Download EIA Inventory
    eia_inventory = download_eia_inventory()

    # Step 7: Generate Holiday Flags
    holiday_flags = generate_holidays(brent.index)

    # Steps 8-10: Merge, compute indicators, spreads, Z-scores, save
    merged = merge_and_save(brent, dxy, wti_products, vix, fingpt, holiday_flags, eia_inventory)

    print("\n" + "=" * 60)
    print("  DONE! Next steps:")
    print("  1. Check the CSV at: data/historical_features.csv")
    print("  2. Upload it to Kaggle as a new Dataset")
    print("  3. Run the v7 training notebook on Kaggle GPU")
    print("=" * 60)


if __name__ == "__main__":
    main()
