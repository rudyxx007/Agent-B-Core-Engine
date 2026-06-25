import yfinance as yf

tickers = {
    "BZ=F": "Brent Front Month",
    "CL=F": "WTI Front Month",
    "RB=F": "RBOB Gasoline",
    "HO=F": "Heating Oil (Diesel proxy)",
}

for t, name in tickers.items():
    df = yf.download(t, start="2007-01-01", end="2026-06-20", progress=False)
    if df.empty:
        print(f"{t} ({name}): NO DATA")
    else:
        print(f"{t} ({name}): {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')} | {len(df)} rows")

# Check BZ2=F for term structure
for t2 in ["BZ2=F", "BZH25.NYM", "QO=F"]:
    df2 = yf.download(t2, start="2007-01-01", end="2026-06-20", progress=False)
    if df2.empty:
        print(f"{t2} (2nd month Brent): NO DATA")
    else:
        print(f"{t2} (2nd month Brent): {df2.index.min().strftime('%Y-%m-%d')} to {df2.index.max().strftime('%Y-%m-%d')} | {len(df2)} rows")
