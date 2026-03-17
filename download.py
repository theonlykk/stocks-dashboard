"""
download.py
===========
Downloads historical daily price data for tickers in tickers.csv using yfinance,
automatically pulls GICS sector classifications, and outputs:

  data/prices.csv        — wide format: Date, TICKER1, TICKER2, ...  (adjusted close)
  data/sectors.csv       — Ticker, Sector
  data/valid_tickers.json — list of tickers that downloaded successfully

Usage:
    python download.py
    python download.py --tickers tickers.csv --start 2021-01-01 --output data

Requirements:
    pip install yfinance pandas
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TICKERS_CSV = "tickers.csv"
DEFAULT_START       = "2021-01-01"
DEFAULT_OUTPUT_DIR  = "data"
MIN_TRADING_DAYS    = 252          # tickers with fewer rows are dropped
MAX_MISSING_PCT     = 0.10         # drop tickers with >10% missing prices
SECTOR_RETRY_DELAY  = 0.25         # seconds between yfinance info calls


def load_tickers(path: str) -> list[str]:
    """Read tickers from a CSV. Accepts any of:
       - single column (no header)
       - column named 'Ticker', 'ticker', 'Symbol', 'symbol'
    """
    df = pd.read_csv(path, header=0)
    col = df.columns[0]
    # try to find a named ticker column
    for candidate in ["Ticker", "ticker", "Symbol", "symbol"]:
        if candidate in df.columns:
            col = candidate
            break
    tickers = df[col].dropna().str.strip().str.upper().tolist()
    tickers = [t for t in tickers if t]
    print(f"[*] Loaded {len(tickers)} tickers from {path}")
    return tickers


def download_prices(tickers: list[str], start: str) -> pd.DataFrame:
    """Download adjusted close prices for all tickers in one batch call."""
    print(f"[*] Downloading prices from {start} to today ...")
    raw = yf.download(
        tickers,
        start=start,
        auto_adjust=True,
        progress=True,
        threads=True,
    )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        # single ticker — wrap in DataFrame
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"
    print(f"[✓] Raw price matrix: {prices.shape[0]} rows × {prices.shape[1]} tickers")
    return prices


def filter_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop tickers with insufficient history or too many gaps."""
    n_rows = len(prices)
    keep = []
    dropped = []
    for col in prices.columns:
        series = prices[col].dropna()
        if len(series) < MIN_TRADING_DAYS:
            dropped.append((col, f"only {len(series)} trading days"))
            continue
        missing_pct = prices[col].isna().sum() / n_rows
        if missing_pct > MAX_MISSING_PCT:
            dropped.append((col, f"{missing_pct:.1%} missing"))
            continue
        keep.append(col)

    if dropped:
        print(f"[!] Dropping {len(dropped)} tickers:")
        for t, reason in dropped:
            print(f"      {t}: {reason}")

    prices = prices[keep].copy()
    # Forward-fill then back-fill small gaps
    prices = prices.ffill().bfill()
    print(f"[✓] Retained {len(keep)} tickers after quality filter")
    return prices


def fetch_sectors(tickers: list[str]) -> dict[str, str]:
    """
    Fetch GICS sector for each ticker from yfinance.
    Falls back to 'Unknown' if info unavailable.
    """
    print(f"[*] Fetching GICS sectors for {len(tickers)} tickers ...")
    sector_map = {}
    for i, ticker in enumerate(tickers, 1):
        try:
            info = yf.Ticker(ticker).info
            sector = info.get("sector") or info.get("sectorDisp") or "Unknown"
        except Exception:
            sector = "Unknown"
        sector_map[ticker] = sector
        if i % 20 == 0:
            print(f"    {i}/{len(tickers)} done ...")
        time.sleep(SECTOR_RETRY_DELAY)

    # summary
    from collections import Counter
    counts = Counter(sector_map.values())
    print(f"[✓] Sectors fetched:")
    for sector, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"      {sector}: {count}")

    return sector_map


def save_outputs(prices: pd.DataFrame, sector_map: dict, output_dir: str):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # prices.csv
    prices_path = out / "prices.csv"
    prices.to_csv(prices_path)
    print(f"[✓] Saved prices  → {prices_path}  ({prices.shape})")

    # sectors.csv
    sectors_df = pd.DataFrame(
        [(t, s) for t, s in sector_map.items()],
        columns=["Ticker", "Sector"]
    ).sort_values("Ticker")
    sectors_path = out / "sectors.csv"
    sectors_df.to_csv(sectors_path, index=False)
    print(f"[✓] Saved sectors → {sectors_path}")

    # valid_tickers.json
    valid_path = out / "valid_tickers.json"
    valid_tickers = sorted(prices.columns.tolist())
    valid_path.write_text(json.dumps(valid_tickers, indent=2))
    print(f"[✓] Saved valid_tickers → {valid_path}  ({len(valid_tickers)} tickers)")


def main():
    parser = argparse.ArgumentParser(description="Download equity prices for cointegration dashboard")
    parser.add_argument("--tickers", default=DEFAULT_TICKERS_CSV, help="Path to tickers CSV")
    parser.add_argument("--start",   default=DEFAULT_START,       help="Start date YYYY-MM-DD")
    parser.add_argument("--output",  default=DEFAULT_OUTPUT_DIR,  help="Output directory")
    args = parser.parse_args()

    # 1. Load tickers
    tickers = load_tickers(args.tickers)
    if not tickers:
        print("[!] No tickers found. Check your tickers.csv.")
        return

    # 2. Download prices
    prices = download_prices(tickers, args.start)

    # 3. Quality filter
    prices = filter_prices(prices)
    valid_tickers = prices.columns.tolist()

    # 4. Fetch GICS sectors (only for valid tickers)
    sector_map = fetch_sectors(valid_tickers)

    # 5. Save everything
    save_outputs(prices, sector_map, args.output)

    print("\n[✓] Done. Next step: python precompute.py")


if __name__ == "__main__":
    main()