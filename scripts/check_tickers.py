"""Quick script to verify earliest available data for BZ=F and DX-Y.NYB"""
import yfinance as yf

print("Downloading BZ=F (Brent Crude Futures)...")
bz = yf.download("BZ=F", start="1900-01-01", end="2026-06-15", progress=False)
print(f"  Earliest: {bz.index.min().strftime('%Y-%m-%d')}")
print(f"  Latest:   {bz.index.max().strftime('%Y-%m-%d')}")
print(f"  Total rows: {len(bz)}")
print(f"  First 5 rows:")
print(bz.head(5)[["Close"]].to_string())
print()

print("Downloading DX-Y.NYB (US Dollar Index / DXY)...")
dx = yf.download("DX-Y.NYB", start="1900-01-01", end="2026-06-15", progress=False)
print(f"  Earliest: {dx.index.min().strftime('%Y-%m-%d')}")
print(f"  Latest:   {dx.index.max().strftime('%Y-%m-%d')}")
print(f"  Total rows: {len(dx)}")
print(f"  First 5 rows:")
print(dx.head(5)[["Close"]].to_string())
print()

# Check overlap
overlap_start = max(bz.index.min(), dx.index.min())
overlap_end = min(bz.index.max(), dx.index.max())
print(f"=== OVERLAP (usable training range) ===")
print(f"  Start: {overlap_start.strftime('%Y-%m-%d')}")
print(f"  End:   {overlap_end.strftime('%Y-%m-%d')}")

bz_overlap = bz.loc[overlap_start:overlap_end]
print(f"  BZ=F rows in range: {len(bz_overlap)}")
