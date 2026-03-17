"""
Flask backend for Equity Cointegration Dashboard.

Endpoints
---------
GET  /                                          -> index.html
GET  /api/sectors                               -> ["Technology", ...]
GET  /api/windows                               -> [252, 126, 63]
GET  /api/pairs?sector=&window=                 -> [["T1","T2"], ...]
GET  /api/heatmap?sector=&window=&pvalue=&asof= -> {rows, cols, matrix}
GET  /api/pair?window=&t1=&t2=&entry_z=&pnl_target=&hurst_max=&pvalue=
                                                -> full time-series + trades + blotter

Run:  python server.py
"""

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR      = Path(__file__).resolve().parent
STATIC_DIR    = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
META_PATH     = BASE_DIR / "precomputed" / "metadata.json"
WINDOWS_DIR   = BASE_DIR / "precomputed" / "windows"

app = Flask(__name__, static_folder=None)


# ── helpers ───────────────────────────────────────────────────────────────────

def normalize_hyphens(s):
    if isinstance(s, str):
        return s.replace("\u2011", "-").replace("\u2010", "-").replace("\u2212", "-")
    return s


def load_metadata():
    with open(META_PATH) as f:
        return json.load(f)


@lru_cache(maxsize=8)
def load_window_df(window: int) -> pd.DataFrame:
    df = pd.read_parquet(WINDOWS_DIR / f"window_{window}.parquet")
    df["ticker1"] = df["ticker1"].apply(normalize_hyphens)
    df["ticker2"] = df["ticker2"].apply(normalize_hyphens)
    df["date"]    = pd.to_datetime(df["date"])
    return df


def safe(v):
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def safe_list(arr) -> list:
    return [safe(v) for v in arr]


# ── beta stability (coefficient of variation) ─────────────────────────────────

def compute_beta_cv(beta_series: pd.Series, window: int) -> pd.Series:
    """std(beta) / |mean(beta)| over rolling window. Lower = more stable."""
    min_p = max(10, window // 4)
    roll_std  = beta_series.rolling(window, min_periods=min_p).std()
    roll_mean = beta_series.rolling(window, min_periods=min_p).mean()
    return roll_std / roll_mean.abs().replace(0, np.nan)


def compute_shading_flags(pair_df: pd.DataFrame, window: int,
                           pvalue_thresh: float, hurst_max: float,
                           hurst_arr: np.ndarray = None) -> dict:
    pv = pair_df["pvalue"].to_numpy(dtype=float, na_value=float("nan"))
    if hurst_arr is None:
        hurst_arr = pair_df["hurst"].to_numpy(dtype=float, na_value=float("nan"))
    cointegrated = [bool(not math.isnan(p) and p < pvalue_thresh) for p in pv]
    stable_beta  = [
        bool(cointegrated[i] and not math.isnan(hurst_arr[i]) and hurst_arr[i] < hurst_max)
        for i in range(len(pv))
    ]
    return {"cointegrated": cointegrated, "stable_beta": stable_beta}


# ── trade label helper ────────────────────────────────────────────────────────

def label_from_counter(n: int) -> str:
    """0→A, 1→B, …, 25→Z, 26→AA, 27→AB, …"""
    ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    result = ""
    n += 1
    while n > 0:
        n, r = divmod(n - 1, 26)
        result = ALPHA[r] + result
    return result


# ── trade engine ──────────────────────────────────────────────────────────────

def build_trades_pyramid(pair_df: pd.DataFrame, t1: str, t2: str,
                         entry_z: float = 1.0, pnl_target: float = 0.5,
                         hurst_max: float = 0.5, window: int = 252,
                         hurst_arr: np.ndarray = None, pvalue_thresh: float = 0.05):
    """
    Trade engine — blotter edition (equity, no carry).

    Rules
    -----
    - Entry levels: entry_z, entry_z+1 (one slot per level per direction)
    - On a given bar, open ALL levels whose threshold z has been crossed,
      all at the current spread/z/prices (handles gaps cleanly)
    - Every leg is its own named trade (A, B, C, …) sequentially
    - Each trade has exactly one Entry row and one Exit row in the blotter
    - Exit: long leg at level L exits when z >= -L + pnl_target
            short leg at level L exits when z <=  L - pnl_target
    - Same-day exit + opposite-direction entry: allowed
    - Same-direction same-level re-entry: blocked until NEXT day
    - Gatekeepers at entry only: pvalue < pvalue_thresh AND hurst < hurst_max
    - PnL = log-spread change only; no carry (equity pairs, not CDS)
    """
    z_arr  = pair_df["zscore"].to_numpy(dtype=float, na_value=float("nan"))
    s_arr  = pair_df["spread"].to_numpy(dtype=float, na_value=float("nan"))
    b_arr  = pair_df["beta"].to_numpy(dtype=float, na_value=float("nan"))
    pv_arr = pair_df["pvalue"].to_numpy(dtype=float, na_value=float("nan"))
    p1_arr = pair_df["price1"].to_numpy(dtype=float, na_value=float("nan"))
    p2_arr = pair_df["price2"].to_numpy(dtype=float, na_value=float("nan"))
    dt_arr = pair_df["date"].to_numpy()
    n      = len(z_arr)

    if hurst_arr is None:
        hurst_arr = pair_df["hurst"].to_numpy(dtype=float, na_value=float("nan"))

    blotter  = []
    markers  = []

    trade_counter = -1
    levels = [entry_z, entry_z + 1.0]

    open_legs: dict = {}
    exited_today: set = set()

    for idx in range(n):
        z  = z_arr[idx]
        s  = s_arr[idx]
        b  = b_arr[idx]
        p1 = p1_arr[idx]
        p2 = p2_arr[idx]
        dt = pd.Timestamp(dt_arr[idx])
        dt_str = dt.strftime("%Y-%m-%d")

        if math.isnan(z) or math.isnan(s):
            exited_today = set()
            continue

        pv_val    = pv_arr[idx]
        hurst_val = hurst_arr[idx]
        beta   = 0.0 if math.isnan(b)  else float(b)
        p1f    = 0.0 if math.isnan(p1) else float(p1)
        p2f    = 0.0 if math.isnan(p2) else float(p2)

        can_enter = (
            not math.isnan(pv_val) and pv_val < pvalue_thresh and
            not math.isnan(hurst_val) and hurst_val < hurst_max
        )

        # ── 1. Exits (no gatekeeper — always honour open positions) ──────────
        newly_exited: set = set()

        for (direction, level), leg in list(open_legs.items()):
            if direction == "LONG"  and z < leg["exit_z"]:
                continue
            if direction == "SHORT" and z > leg["exit_z"]:
                continue

            # exit triggered — use locked beta from entry
            exit_economic_spread = p1f - leg["locked_beta"] * p2f
            if direction == "LONG":
                spread_pnl = round(exit_economic_spread - float(leg["entry_spread"]), 4)
            else:
                spread_pnl = round(float(leg["entry_spread"]) - exit_economic_spread, 4)

            total_profit = spread_pnl  # no carry for equities

            blotter.append({
                "TradeLabel":  leg["trade_label"],
                "RowType":     "Exit",
                "Date":        dt_str,
                "DaysHeld":    (dt - leg["entry_date"]).days,
                "Dir":         direction,
                "T1":          t1,
                "T2":          t2,
                "T1_Px":       round(p1f, 4),
                "T2_Px":       round(p2f, 4),
                "Spread":      round(exit_economic_spread, 4),
                "Z":           round(float(z), 3),
                "TradeBeta":   round(leg["locked_beta"], 4),
                "RollingBeta": round(beta, 4),
                "EntrySpread": round(float(leg["entry_spread"]), 4),
                "EntryZ":      round(float(leg["entry_z_val"]), 3),
                "SpreadPnL":   spread_pnl,
                "Profit":      total_profit,
            })
            markers.append({
                "date": dt_str, "spread": round(exit_economic_spread, 4),
                "z": round(float(z), 3), "type": "exit",
                "dir": direction, "leg_level": level,
                "trade_label": leg["trade_label"],
            })
            newly_exited.add((direction, level))
            del open_legs[(direction, level)]

        exited_today = newly_exited

        # ── 2. Entries ────────────────────────────────────────────────────────
        if not can_enter:
            continue

        for direction in ("LONG", "SHORT"):
            for level in levels:
                slot = (direction, level)

                if slot in open_legs:
                    continue
                if slot in exited_today:
                    continue

                if direction == "LONG"  and z > -level:
                    continue
                if direction == "SHORT" and z <  level:
                    continue

                trade_counter += 1
                trade_label = f"Trade {label_from_counter(trade_counter)}"
                exit_z = (-level + pnl_target) if direction == "LONG" \
                         else (level - pnl_target)

                locked_entry_spread = p1f - beta * p2f

                open_legs[slot] = {
                    "entry_date":   dt,
                    "entry_z_val":  z,
                    "entry_spread": locked_entry_spread,
                    "locked_beta":  beta,
                    "exit_z":       exit_z,
                    "trade_label":  trade_label,
                }

                blotter.append({
                    "TradeLabel":  trade_label,
                    "RowType":     "Entry",
                    "Date":        dt_str,
                    "DaysHeld":    None,
                    "Dir":         direction,
                    "T1":          t1,
                    "T2":          t2,
                    "T1_Px":       round(p1f, 4),
                    "T2_Px":       round(p2f, 4),
                    "Spread":      round(locked_entry_spread, 4),
                    "Z":           round(float(z), 3),
                    "TradeBeta":   round(beta, 4),
                    "RollingBeta": round(beta, 4),
                    "EntrySpread": None,
                    "EntryZ":      None,
                    "SpreadPnL":   None,
                    "Profit":      None,
                })
                markers.append({
                    "date": dt_str, "spread": round(locked_entry_spread, 4),
                    "z": round(float(z), 3), "type": "entry",
                    "dir": direction, "leg_level": level,
                    "trade_label": trade_label,
                })

    return blotter, markers


def compute_pnl_from_blotter(blotter: list, dates: list) -> list:
    """Cumulative PnL curve from exit rows only."""
    pnl_by_date = {}
    for row in blotter:
        if row["RowType"] == "Exit" and row["Profit"] is not None:
            d = row["Date"]
            pnl_by_date[d] = pnl_by_date.get(d, 0.0) + row["Profit"]
    cum, running = [], 0.0
    for d in dates:
        running += pnl_by_date.get(d, 0.0)
        cum.append(round(running, 4))
    return cum


def get_proposed(pair_df, t1, t2, entry_z=1.0, hurst_max=0.5,
                 window=252, hurst_arr=None):
    if pair_df.empty:
        return None
    last    = pair_df.iloc[-1]
    z, beta = last.get("zscore"), last.get("beta")
    if pd.isna(z) or pd.isna(beta):
        return None
    if hurst_arr is not None:
        last_hurst = float(hurst_arr[-1])
    else:
        last_hurst = float(pair_df["hurst"].iloc[-1]) if "hurst" in pair_df.columns else float("nan")
    if not math.isnan(last_hurst) and last_hurst >= hurst_max:
        return None
    if   z >  entry_z: return {"direction": "SHORT", "t1": t1, "t2": t2, "beta": float(beta)}
    elif z < -entry_z: return {"direction": "LONG",  "t1": t1, "t2": t2, "beta": float(beta)}
    return None


# ── static ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    resp = send_from_directory(TEMPLATES_DIR, "index.html")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    return resp

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

@app.route("/favicon.ico")
def favicon():
    return "", 204


# ── API ───────────────────────────────────────────────────────────────────────

@app.route("/api/sectors")
def api_sectors():
    return jsonify([normalize_hyphens(s) for s in load_metadata()["sectors"]])


@app.route("/api/windows")
def api_windows():
    return jsonify(load_metadata()["windows"])


@app.route("/api/pairs")
def api_pairs():
    sector = normalize_hyphens(request.args.get("sector", ""))
    window = int(request.args.get("window", 63))
    df = load_window_df(window)
    df = df[df["sector"] == sector]
    if df.empty:
        sector_norm = sector.strip().lower()
        df = load_window_df(window)
        df = df[df["sector"].str.strip().str.lower() == sector_norm]
    if df.empty:
        return jsonify([])
    pairs = sorted(set(zip(df["ticker1"], df["ticker2"])))
    return jsonify([[a, b] for a, b in pairs])


@app.route("/api/heatmap")
def api_heatmap():
    sector     = normalize_hyphens(request.args.get("sector", ""))
    window     = int(request.args.get("window", 63))
    pvalue_max = float(request.args.get("pvalue", 0.05))
    asof       = request.args.get("asof", "").strip()

    df = load_window_df(window)
    df = df[df["sector"] == sector]
    if df.empty:
        sector_norm = sector.strip().lower()
        df = load_window_df(window)
        df = df[df["sector"].str.strip().str.lower() == sector_norm]
    if df.empty:
        return jsonify({"rows": [], "cols": [], "matrix": []})

    if asof:
        try:
            asof_dt = pd.Timestamp(asof)
            df = df[df["date"] <= asof_dt]
        except Exception:
            pass

    if df.empty:
        return jsonify({"rows": [], "cols": [], "matrix": []})

    latest  = df.loc[df.groupby(["ticker1", "ticker2"])["date"].idxmax()]
    tickers = sorted(set(latest["ticker1"]) | set(latest["ticker2"]))
    idx     = {t: i for i, t in enumerate(tickers)}
    n       = len(tickers)
    mat     = [[None] * n for _ in range(n)]

    for _, row in latest.iterrows():
        i, j = idx[row["ticker1"]], idx[row["ticker2"]]
        v = safe(row["pvalue"])
        mat[i][j] = v
        mat[j][i] = v

    return jsonify({"rows": tickers, "cols": tickers, "matrix": mat,
                    "pvalue_max": pvalue_max})


@app.route("/api/pair")
def api_pair():
    t1            = normalize_hyphens(request.args.get("t1", ""))
    t2            = normalize_hyphens(request.args.get("t2", ""))
    t1, t2        = sorted([t1, t2])
    window        = int(request.args.get("window", 63))
    entry_z       = float(request.args.get("entry_z", 1.0))
    pnl_target    = float(request.args.get("pnl_target", 0.5))
    pvalue_thresh = float(request.args.get("pvalue", 0.05))
    hurst_max     = float(request.args.get("hurst_max", 0.75))

    df = load_window_df(window)
    mask = (
        ((df["ticker1"] == t1) & (df["ticker2"] == t2)) |
        ((df["ticker1"] == t2) & (df["ticker2"] == t1))
    )
    pair_df = df[mask].copy().sort_values("date").reset_index(drop=True)

    if pair_df.empty:
        return jsonify({"error": f"No data for {t1}/{t2} in window={window}"}), 404

    dates     = [d.strftime("%Y-%m-%d") for d in pair_df["date"]]
    hurst_arr = pair_df["hurst"].to_numpy(dtype=float, na_value=float("nan"))
    cv_ser    = compute_beta_cv(pair_df["beta"], window)

    shading = compute_shading_flags(pair_df, window, pvalue_thresh, hurst_max,
                                    hurst_arr=hurst_arr)

    blotter, markers = build_trades_pyramid(
        pair_df, t1, t2,
        entry_z=entry_z, pnl_target=pnl_target,
        hurst_max=hurst_max, window=window,
        hurst_arr=hurst_arr, pvalue_thresh=pvalue_thresh,
    )

    pnl      = compute_pnl_from_blotter(blotter, dates)
    proposed = get_proposed(pair_df, t1, t2, entry_z=entry_z,
                            hurst_max=hurst_max, hurst_arr=hurst_arr)

    return jsonify({
        "dates":          dates,
        "price1":         safe_list(pair_df["price1"]),
        "price2":         safe_list(pair_df["price2"]),
        "spread":         safe_list(pair_df["spread"]),
        "std_spread":     safe_list(pair_df["std_spread"]),
        "zscore":         safe_list(pair_df["zscore"]),
        "pnl":            pnl,
        "beta":           safe_list(pair_df["beta"]),
        "beta_cv":        safe_list(cv_ser),
        "half_life":      safe_list(pair_df["half_life"]),
        "hurst":          safe_list(pair_df["hurst"]),
        "pvalue":         safe_list(pair_df["pvalue"]),
        "blotter":        blotter,
        "markers":        markers,
        "shading":        shading,
        "proposed_trade": proposed,
        "params": {
            "entry_z":       entry_z,
            "pnl_target":    pnl_target,
            "hurst_max":     hurst_max,
            "pvalue_thresh": pvalue_thresh,
        },
    })


@app.route("/api/refresh")
def api_refresh():
    load_window_df.cache_clear()
    return jsonify({"status": "cache cleared"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=port)
