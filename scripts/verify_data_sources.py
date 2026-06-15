"""
scripts/verify_data_sources.py — Empirically Verify Available Data Ranges (v2)

Fixed URLs for FRED and Stooq.
"""

import pandas as pd
import io
import urllib.request
import ssl
import sys

def check_fred_brent():
    """Download Brent Crude from FRED CSV export."""
    print("=" * 60)
    print("  TEST 1: Brent Crude Oil (FRED: DCOILBRENTEU)")
    print("=" * 60)
    
    # Direct FRED CSV download URL (simpler endpoint)
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU"
    
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            data = response.read().decode("utf-8")
        
        df = pd.read_csv(io.StringIO(data))
        df.columns = ["Date", "Price"]
        df = df[df["Price"] != "."]
        df["Price"] = df["Price"].astype(float)
        df["Date"] = pd.to_datetime(df["Date"])
        
        print(f"  Total data points: {len(df)}")
        print(f"  Earliest date:     {df['Date'].min().strftime('%Y-%m-%d')}")
        print(f"  Latest date:       {df['Date'].max().strftime('%Y-%m-%d')}")
        print(f"  Price range:       ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
        print()
        print("  First 5 rows:")
        for _, row in df.head(5).iterrows():
            print(f"    {row['Date'].strftime('%Y-%m-%d')}  ${row['Price']:.2f}")
        print()
        print("  Last 5 rows:")
        for _, row in df.tail(5).iterrows():
            print(f"    {row['Date'].strftime('%Y-%m-%d')}  ${row['Price']:.2f}")
        print()
        print("  [OK] FRED Brent Crude data verified.")
        return True
    except Exception as e:
        print(f"  [FAIL] Could not download: {e}")
        return False


def check_stooq_dxy():
    """Download actual DXY from Stooq.com."""
    print()
    print("=" * 60)
    print("  TEST 2: DXY - US Dollar Index (Stooq.com)")
    print("=" * 60)
    
    # Stooq ticker formats to try
    tickers = [
        ("dxy.f", "DXY Futures"),
        ("^dxy", "DXY Index (caret)"),
        ("dxy", "DXY plain"),
        ("usdx", "USDX"),
    ]
    
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    for ticker, label in tickers:
        url = f"https://stooq.com/q/d/l/?s={ticker}&d1=19700101&d2=20261231&i=d"
        print(f"  Trying ticker '{ticker}' ({label})...")
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
                data = response.read().decode("utf-8")
            
            if "No data" in data or len(data) < 100:
                print(f"    No data returned for '{ticker}'")
                continue
            
            df = pd.read_csv(io.StringIO(data))
            
            if "Date" not in df.columns or "Close" not in df.columns:
                print(f"    Unexpected columns: {list(df.columns)}")
                continue
            
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.dropna(subset=["Close"])
            
            print(f"    Total data points: {len(df)}")
            print(f"    Earliest date:     {df['Date'].min().strftime('%Y-%m-%d')}")
            print(f"    Latest date:       {df['Date'].max().strftime('%Y-%m-%d')}")
            print(f"    Close range:       {df['Close'].min():.2f} - {df['Close'].max():.2f}")
            print()
            print("    First 5 rows:")
            for _, row in df.head(5).iterrows():
                print(f"      {row['Date'].strftime('%Y-%m-%d')}  {row['Close']:.2f}")
            print()
            print("    Last 5 rows:")
            for _, row in df.tail(5).iterrows():
                print(f"      {row['Date'].strftime('%Y-%m-%d')}  {row['Close']:.2f}")
            print()
            print(f"  [OK] Stooq DXY data verified with ticker '{ticker}'.")
            return True
        except Exception as e:
            print(f"    Failed: {e}")
    
    print()
    print("  All Stooq tickers failed. Trying pandas_datareader with FRED...")
    print("  (This downloads DTWEXM which is NOT DXY but proves FRED works)")
    
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DTWEXM"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, context=ctx, timeout=30) as response:
            data = response.read().decode("utf-8")
        
        df = pd.read_csv(io.StringIO(data))
        df.columns = ["Date", "Value"]
        df = df[df["Value"] != "."]
        df["Value"] = df["Value"].astype(float)
        df["Date"] = pd.to_datetime(df["Date"])
        
        print(f"  DTWEXM (NOT DXY, just checking FRED connectivity):")
        print(f"    Data points: {len(df)}")
        print(f"    Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        print()
        print("  NOTE: DTWEXM is NOT DXY. We still need to find real DXY data.")
        return False
    except Exception as e:
        print(f"  FRED DTWEXM also failed: {e}")
        return False


def main():
    print()
    print("*" * 60)
    print("  AGENT-B: Data Source Verification (v2)")
    print("  Proving exact date ranges available")
    print("*" * 60)
    print()
    
    brent_ok = check_fred_brent()
    dxy_ok = check_stooq_dxy()
    
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Brent Crude (FRED):  {'VERIFIED' if brent_ok else 'FAILED'}")
    print(f"  DXY (Stooq):         {'VERIFIED' if dxy_ok else 'NEEDS MANUAL DOWNLOAD'}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
