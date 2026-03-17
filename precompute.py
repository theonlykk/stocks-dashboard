"""
precompute.py
=============
Builds rolling cointegration metrics for all equity pairs and saves
results as parquet files consumed by server.py.

Inputs  (from download.py):
    data/prices.csv      — wide format: Date index, ticker columns (adjusted close)
    data/sectors.csv     — Ticker, Sector
    data/valid_tickers.json

Outputs:
    precomputed/windows/window_63.parquet
    precomputed/windows/window_126.parquet
    precomputed/windows/window_252.parquet
    precomputed/sectors/<Sector>.parquet   (one per sector)
    precomputed/metadata.json

Key difference from CDS precompute:
    - Prices are log-transformed before OLS regression
    - Beta is log-price elasticity (dimensionless)
    - Spread = log(P1) - beta * log(P2)   (OLS residual on log prices)
    - No carry calculation (no CDS duration)

Usage:
    python precompute.py
    python precompute.py --fast          # 2 sectors only, for testing
    python precompute.py --no-ray        # single-threaded fallback
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR        = Path(__file__).resolve().parent
PRECOMPUTED_DIR = BASE_DIR / "precomputed"
SECTORS_DIR     = PRECOMPUTED_DIR / "sectors"
WINDOWS_DIR     = PRECOMPUTED_DIR / "windows"
META_PATH       = PRECOMPUTED_DIR / "metadata.json"
PRICES_CSV      = BASE_DIR / "data" / "prices.csv"
SECTORS_CSV     = BASE_DIR / "data" / "sectors.csv"
VALID_JSON      = BASE_DIR / "data" / "valid_tickers.json"

WINDOWS = [63, 126, 252]

REQUIRED_COLUMNS = [
    "sector", "window", "ticker1", "ticker2", "date",
    "price1", "price2", "beta", "adf_stat", "pvalue",
    "spread", "zscore", "std_spread", "half_life", "hurst",
]


# =============================================================================
# HELPERS
# =============================================================================

def sanitize_sector_name(name: str) -> str:
    s = name.replace(" ", "_")
    s = re.sub(r"[^\w]", "", s)
    return s


def rolling_beta(s1: pd.Series, s2: pd.Series, window: int) -> pd.Series:
    """OLS beta of s1 on s2 over rolling window."""
    betas = np.full(len(s1), np.nan)
    s1v, s2v = s1.values, s2.values
    for i in range(window, len(s1v) + 1):
        y = s1v[i - window:i]
        x = s2v[i - window:i]
        if np.any(np.isnan(y)) or np.any(np.isnan(x)):
            continue
        xm, ym = x - x.mean(), y - y.mean()
        denom = (xm * xm).sum()
        if denom < 1e-12:
            continue
        betas[i - 1] = (xm * ym).sum() / denom
    return pd.Series(betas, index=s1.index)


def rolling_residuals(s1: pd.Series, s2: pd.Series,
                      betas: pd.Series) -> pd.Series:
    """Spread = s1 - beta * s2."""
    return s1 - betas * s2


def rolling_adf(spread: pd.Series, window: int) -> pd.DataFrame:
    """
    Rolling ADF test on the spread series.
    Returns DataFrame with columns: date, adf_stat, pvalue
    """
    from statsmodels.tsa.stattools import adfuller

    dates, stats, pvals = [], [], []
    arr = spread.values
    idx = spread.index

    for i in range(window, len(arr) + 1):
        w = arr[i - window:i]
        if np.isnan(w).any():
            continue
        try:
            result = adfuller(w, maxlag=1, autolag=None, regression="c")
            dates.append(idx[i - 1])
            stats.append(float(result[0]))
            pvals.append(float(result[1]))
        except Exception:
            pass

    if not dates:
        return pd.DataFrame()

    return pd.DataFrame({"date": dates, "adf_stat": stats, "pvalue": pvals})


def compute_half_life(spread_arr: np.ndarray, window: int) -> np.ndarray:
    n = len(spread_arr)
    hl = np.full(n, np.nan)
    for i in range(window, n):
        w = spread_arr[i - window:i]
        if np.isnan(w).any():
            continue
        try:
            dy  = np.diff(w)
            lag = w[:-1]
            if np.std(lag) < 1e-12:
                continue
            b = np.polyfit(lag, dy, 1)[0]
            if b < 0:
                hl[i] = -1.0 / b
        except Exception:
            pass
    return hl


def compute_hurst(spread_arr: np.ndarray, window: int) -> np.ndarray:
    """H = 0.5 + 0.5 * lag-1 autocorrelation of spread differences."""
    n = len(spread_arr)
    hv = np.full(n, np.nan)
    for i in range(window, n):
        w = spread_arr[i - window:i]
        if np.isnan(w).any() or len(w) < 10:
            continue
        try:
            d = np.diff(w)
            if len(d) < 4:
                continue
            rho = np.corrcoef(d[:-1], d[1:])[0, 1]
            if not np.isnan(rho):
                hv[i] = float(np.clip(0.5 + 0.5 * rho, 0.0, 1.0))
        except Exception:
            pass
    return hv


def process_pair(t1: str, t2: str, log_prices: pd.DataFrame,
                 window: int, sector: str) -> pd.DataFrame:
    """Compute all metrics for one pair. Returns REQUIRED_COLUMNS DataFrame."""
    if t1 not in log_prices.columns or t2 not in log_prices.columns:
        return pd.DataFrame()

    s1 = log_prices[t1].dropna()
    s2 = log_prices[t2].dropna()

    # Align on shared dates
    common = s1.index.intersection(s2.index)
    if len(common) < window:
        return pd.DataFrame()
    s1, s2 = s1.loc[common], s2.loc[common]

    # Rolling beta and spread
    betas  = rolling_beta(s1, s2, window)
    spread = rolling_residuals(s1, s2, betas)

    # ADF
    adf_df = rolling_adf(spread, window)
    if adf_df.empty:
        return pd.DataFrame()

    adf_df = adf_df.set_index("date")
    idx    = adf_df.index

    # Align everything
    adf_df["beta"]   = betas.reindex(idx)
    adf_df["spread"] = spread.reindex(idx)
    adf_df["price1"] = s1.reindex(idx)   # log prices stored as price1/price2
    adf_df["price2"] = s2.reindex(idx)

    # Z-score
    roll_mean            = spread.rolling(window).mean()
    roll_std             = spread.rolling(window).std().replace(0, np.nan)
    adf_df["zscore"]     = ((spread - roll_mean) / roll_std).reindex(idx)
    adf_df["std_spread"] = roll_std.reindex(idx)

    # Half-life and Hurst
    spread_arr         = spread.values
    adf_df["half_life"] = pd.Series(
        compute_half_life(spread_arr, window), index=spread.index
    ).reindex(idx)
    adf_df["hurst"] = pd.Series(
        compute_hurst(spread_arr, window), index=spread.index
    ).reindex(idx)

    # Metadata
    adf_df["sector"]  = sector
    adf_df["window"]  = window
    adf_df["ticker1"] = t1
    adf_df["ticker2"] = t2
    adf_df = adf_df.reset_index()

    missing = [c for c in REQUIRED_COLUMNS if c not in adf_df.columns]
    if missing:
        return pd.DataFrame()

    return adf_df[REQUIRED_COLUMNS]


# =============================================================================
# RAY WORKER
# =============================================================================

def _ray_worker(t1, t2, lp_dict, window, sector):
    """Thin wrapper so Ray can call process_pair without importing the module."""
    import numpy as np
    import pandas as pd
    lp = pd.DataFrame(lp_dict)
    lp.index = pd.to_datetime(lp.index)
    return process_pair(t1, t2, lp, window, sector)


# =============================================================================
# PIPELINE
# =============================================================================

def write_metadata(sector_map: dict, windows: list) -> None:
    meta = {
        "windows": windows,
        "sectors": list(sector_map.keys()),
        "sector_files": {
            name: f"precomputed/sectors/{sanitize_sector_name(name)}.parquet"
            for name in sector_map
        },
        "window_files": {
            str(w): f"precomputed/windows/window_{w}.parquet" for w in windows
        },
    }
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[✓] Wrote metadata.json")


def run_sector_window_ray(sector: str, tickers: list,
                          log_prices: pd.DataFrame, window: int) -> pd.DataFrame:
    import ray

    @ray.remote
    def _remote(t1, t2, lp_dict, w, s):
        return _ray_worker(t1, t2, lp_dict, w, s)

    lp_dict = log_prices[tickers].to_dict(orient="dict")
    jobs = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            jobs.append(_remote.remote(tickers[i], tickers[j], lp_dict, window, sector))

    results = ray.get(jobs)
    valid = [r for r in results if not r.empty]
    return pd.concat(valid, ignore_index=True) if valid else pd.DataFrame()


def run_sector_window_single(sector: str, tickers: list,
                             log_prices: pd.DataFrame, window: int) -> pd.DataFrame:
    results = []
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    done = 0
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            r = process_pair(tickers[i], tickers[j], log_prices, window, sector)
            if not r.empty:
                results.append(r)
            done += 1
            if done % 50 == 0:
                print(f"      {done}/{n_pairs} pairs ...")
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def run_all(sector_map: dict, log_prices: pd.DataFrame,
            windows: list, use_ray: bool) -> None:

    if use_ray:
        import ray
        ray.init(ignore_reinit_error=True)

    try:
        for window in windows:
            window_results = []
            print(f"\n{'='*60}")
            print(f"Window: {window} days")
            print(f"{'='*60}")

            for sector, tickers in sector_map.items():
                # Only keep tickers present in log_prices
                tickers = [t for t in tickers if t in log_prices.columns]
                if len(tickers) < 2:
                    print(f"  [!] {sector}: fewer than 2 valid tickers, skipping")
                    continue

                n_pairs = len(tickers) * (len(tickers) - 1) // 2
                print(f"\n  [{sector}]  {len(tickers)} tickers, {n_pairs} pairs")

                if use_ray:
                    res = run_sector_window_ray(sector, tickers, log_prices, window)
                else:
                    res = run_sector_window_single(sector, tickers, log_prices, window)

                if res.empty:
                    print(f"    [!] No results — skipping")
                    continue

                # Per-sector parquet (overwrite each window, last window wins)
                SECTORS_DIR.mkdir(parents=True, exist_ok=True)
                path = SECTORS_DIR / f"{sanitize_sector_name(sector)}.parquet"
                res[REQUIRED_COLUMNS].to_parquet(path, index=False)
                print(f"    [✓] Saved sector parquet → {path.name}")
                window_results.append(res)

            if window_results:
                WINDOWS_DIR.mkdir(parents=True, exist_ok=True)
                wpath = WINDOWS_DIR / f"window_{window}.parquet"
                pd.concat(window_results, ignore_index=True)[REQUIRED_COLUMNS].to_parquet(
                    wpath, index=False
                )
                print(f"\n  [✓] Saved window parquet → {wpath.name}")

    finally:
        if use_ray:
            ray.shutdown()

    write_metadata(sector_map, windows)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",   action="store_true", help="Process 2 sectors only (testing)")
    parser.add_argument("--no-ray", action="store_true", help="Single-threaded (no Ray)")
    args = parser.parse_args()

    # Check inputs
    for p in [PRICES_CSV, SECTORS_CSV, VALID_JSON]:
        if not p.exists():
            print(f"[!] Missing: {p}  — run download.py first")
            sys.exit(1)

    # Load prices and log-transform
    print("[*] Loading prices ...")
    prices = pd.read_csv(PRICES_CSV, index_col=0, parse_dates=True)
    log_prices = np.log(prices.replace(0, np.nan))
    print(f"    {log_prices.shape[0]} rows × {log_prices.shape[1]} tickers")
    print(f"    Date range: {log_prices.index[0].date()} → {log_prices.index[-1].date()}")

    # Load valid tickers whitelist
    with open(VALID_JSON) as f:
        valid_set = set(json.load(f))
    print(f"[*] Valid tickers: {len(valid_set)}")

    # Load sector map
    sectors_df = pd.read_csv(SECTORS_CSV)
    sector_map = (
        sectors_df[sectors_df["Ticker"].isin(valid_set)]
        .groupby("Sector")["Ticker"]
        .apply(list)
        .to_dict()
    )
    # Drop sectors with <2 tickers
    sector_map = {s: t for s, t in sector_map.items() if len(t) >= 2}
    total = sum(len(t) for t in sector_map.values())
    print(f"[*] {len(sector_map)} sectors, {total} tickers total")
    for s, t in sorted(sector_map.items()):
        print(f"    {s}: {len(t)} tickers")

    if args.fast:
        keys = list(sector_map.keys())[:2]
        sector_map = {k: sector_map[k] for k in keys}
        print(f"[*] FAST MODE: processing {list(sector_map.keys())} only")

    use_ray = not args.no_ray
    try:
        import ray
    except ImportError:
        print("[!] Ray not installed — falling back to single-threaded")
        use_ray = False

    print(f"[*] Ray: {'enabled' if use_ray else 'disabled'}")
    print(f"[*] Windows: {WINDOWS}")
    print(f"[*] Starting precompute ...\n")

    run_all(sector_map, log_prices, WINDOWS, use_ray)

    print("\n" + "=" * 60)
    print("PRECOMPUTE COMPLETE")
    print("Next step: python server.py")
    print("=" * 60)


if __name__ == "__main__":
    main()