"""
scripts/modal_app/data_fetcher.py — Live Data Fetcher for Inference

Downloads the last 150 days of market data from yfinance and FRED, computes
the required technical indicators, normalizes them using the exact statistics
saved in the deployed Hugging Face config.json, and returns a [1, 90, 12] tensor.
"""

import pandas as pd
import numpy as np
import yfinance as yf
import json
import torch
import warnings
from huggingface_hub import hf_hub_download

warnings.filterwarnings("ignore")

# Features expected by the v7 iTransformer (in exact order)
RAW_FEATURE_COLS = [
    "z_brent", "z_dxy", "holiday_flag", "vix_close", "sentiment_score",
    "rsi_14", "macd", "macd_signal", "brent_wti_spread", 
    "crack_spread_321", "eia_inventory", "eia_inventory_change"
]

def fetch_daily_features(today_sentiment: float):
    """
    1. Downloads ~150 trading days of data.
    2. Computes RSI, MACD, spreads, Z-scores.
    3. Normalizes using config.json from HF.
    4. Returns [1, 90, 12] Tensor and raw prices for logging.
    """
    print("Fetching last 150 days of market data...")
    
    # Download data sequentially to prevent SQLite database lock errors in Modal
    tickers = ["BZ=F", "DX-Y.NYB", "CL=F", "RB=F", "HO=F"]
    df_all = pd.DataFrame()
    for t in tickers:
        df_all[t] = yf.download(t, period="7mo", progress=False)['Close']
    df_all = df_all.ffill()
    
    # Map columns
    df = pd.DataFrame()
    df['brent_close'] = df_all['BZ=F']
    df['dxy_close'] = df_all['DX-Y.NYB']
    df['wti_close'] = df_all['CL=F']
    df['gasoline_close'] = df_all['RB=F']
    df['heating_oil_close'] = df_all['HO=F']
    
    # For live deployment without heavy API dependencies, we approximate VIX and EIA if needed,
    # or fetch from Yahoo. Yahoo has VIX.
    vix = yf.download("^VIX", period="7mo", progress=False)['Close']
    df['vix_close'] = vix['^VIX']
    
    # Fill any remaining NaNs
    df = df.ffill().bfill()
    
    # Add sentiment (we assume neutral 0.0 for history, and today's score for the final row)
    df['sentiment_score'] = 0.0
    df.iloc[-1, df.columns.get_loc('sentiment_score')] = today_sentiment
    
    # Simple holiday approximation (0 for regular days)
    df['holiday_flag'] = 0
    
    # Spreads
    df['brent_wti_spread'] = df['brent_close'] - df['wti_close']
    df['crack_spread_321'] = (
        (2 * df['gasoline_close'] * 42) + 
        (1 * df['heating_oil_close'] * 42) - 
        (3 * df['brent_close'])
    ) / 3
    
    # EIA approximation (hard to get live without API key, using neutral baseline for now)
    # Ideally, replace this with actual EIA API call
    df['eia_inventory'] = 400.0  
    df['eia_inventory_change'] = 0.0
    
    # Technicals
    import ta
    df['rsi_14'] = ta.momentum.RSIIndicator(df['brent_close'], window=14).rsi()
    macd_ind = ta.trend.MACD(df['brent_close'])
    df['macd'] = macd_ind.macd()
    df['macd_signal'] = macd_ind.macd_signal()
    
    # Rolling Z-scores (30-day)
    r_mean = df['brent_close'].rolling(30).mean()
    r_std = df['brent_close'].rolling(30).std().replace(0, 1)
    df['z_brent'] = (df['brent_close'] - r_mean) / r_std
    
    dxy_mean = df['dxy_close'].rolling(30).mean()
    dxy_std = df['dxy_close'].rolling(30).std().replace(0, 1)
    df['z_dxy'] = (df['dxy_close'] - dxy_mean) / dxy_std
    
    df = df.dropna()
    
    # ==========================================
    # Normalization via Hugging Face config.json
    # ==========================================
    REPO_ID = "rudyxx07/agent-b-itransformer"
    print("Downloading config.json from Hugging Face to apply exact normalizations...")
    config_path = hf_hub_download(repo_id=REPO_ID, filename="config.json")
    
    with open(config_path, "r") as f:
        config = json.load(f)
        feature_stats = config.get("feature_stats", {})
    
    # If config doesn't have feature_stats (e.g. from an older commit), we use current fold
    # But ideally it does because we added it in Phase 2.5
    for col in RAW_FEATURE_COLS:
        if col in ["z_brent", "z_dxy"]:
            continue # Already z-scored
            
        stats = feature_stats.get(col)
        if stats:
            mu, sigma = stats["mean"], stats["std"]
        else:
            # Fallback if missing
            mu, sigma = df[col].mean(), df[col].std()
            
        if sigma == 0: sigma = 1.0
        df[col] = (df[col] - mu) / sigma
        
    # Extract the last 90 days for the sequence length
    seq_len = 90
    if len(df) < seq_len:
        raise ValueError(f"Not enough data. Expected {seq_len}, got {len(df)}")
        
    final_sequence = df[RAW_FEATURE_COLS].iloc[-seq_len:].values.astype(np.float32)
    
    # Convert to Tensor [Batch, SeqLen, Features] -> [1, 90, 12]
    tensor = torch.FloatTensor(final_sequence).unsqueeze(0)
    
    # Extract raw prices for logging and un-normalizing
    raw_prices = {
        "brent_close": float(df['brent_close'].iloc[-1]),
        "dxy_close": float(df['dxy_close'].iloc[-1]),
        "holiday_flag": int(df['holiday_flag'].iloc[-1]),
        "brent_mean_30d": float(r_mean.iloc[-1]),
        "brent_std_30d": float(r_std.iloc[-1])
    }
    
    return tensor, raw_prices

if __name__ == "__main__":
    t, raw = fetch_daily_features(0.5)
    print(f"Tensor Shape: {t.shape}")
    print(f"Raw Prices: {raw}")
