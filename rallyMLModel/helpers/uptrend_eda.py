import argparse
import os
from dataclasses import dataclass
from typing import Optional

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "../.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), "../.cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_FILE = "uptrendStocksQUBTUNAMBG.csv"
DEFAULT_OUTDIR = "eda_out"

PRICE_COLS = ["open", "high", "low", "close"]
VOLUME_COL = "volume"
TARGET_COL = "target"
TS_COL = "timestamp"


@dataclass(frozen=True)
class EdaConfig:
    file: str
    outdir: str
    show: bool
    max_points: int


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_uptrend_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    missing = [c for c in [TS_COL, *PRICE_COLS, VOLUME_COL, TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    df = df.copy()
    # Example format: "Mon May 12 04:00:00 BST 2025" (timezone abbreviations like BST are not stable in pandas)
    ts_clean = (
        df[TS_COL]
        .astype(str)
        .str.replace(r"\s[A-Z]{2,5}\s(\d{4})$", r" \1", regex=True)
    )
    df[TS_COL] = pd.to_datetime(ts_clean, errors="coerce", utc=True)
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)

    for col in [*PRICE_COLS, VOLUME_COL, TARGET_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=PRICE_COLS).reset_index(drop=True)
    df[TARGET_COL] = df[TARGET_COL].fillna(0).astype(int)
    return df


def _maybe_downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def _print_overview(df: pd.DataFrame, source: str) -> None:
    print(f"\n=== Uptrend EDA: {source} ===")
    print(f"Rows: {len(df):,}")
    print("Columns:", ", ".join(df.columns))
    print("Time range:", df[TS_COL].min(), "→", df[TS_COL].max())
    print("\nMissing values (top):")
    miss = df.isna().sum().sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        print("  (none)")
    else:
        print(miss.to_string())

    if TARGET_COL in df.columns:
        vc = df[TARGET_COL].value_counts(dropna=False).sort_index()
        pos = int(vc.get(1, 0))
        neg = int(vc.get(0, 0))
        tot = int(vc.sum())
        pos_rate = (pos / tot) if tot else 0.0
        print("\nTarget distribution:")
        print(vc.to_string())
        print(f"Positive rate: {pos_rate:.4f} ({pos:,}/{tot:,})")

    numeric_cols = [c for c in [*PRICE_COLS, VOLUME_COL] if c in df.columns]
    if numeric_cols:
        print("\nNumeric summary:")
        print(df[numeric_cols].describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).to_string())


def _savefig(outdir: str, name: str, show: bool) -> None:
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    if show:
        plt.show()
    plt.close()
    print(f"[saved] {path}")


def plot_price_and_target(df: pd.DataFrame, outdir: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(df[TS_COL], df["close"], linewidth=1.0, label="close")
    ax.set_title("Close over time (with target markers)")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.25)

    if TARGET_COL in df.columns:
        pos = df[df[TARGET_COL] == 1]
        if not pos.empty:
            ax.scatter(pos[TS_COL], pos["close"], s=10, c="red", alpha=0.7, label="target=1")
    ax.legend(loc="best")
    _savefig(outdir, "01_close_with_target.png", show)


def plot_volume(df: pd.DataFrame, outdir: str, show: bool) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[TS_COL], df[VOLUME_COL], linewidth=0.8, color="tab:green", label="volume")
    ax.set_title("Volume over time")
    ax.set_xlabel("time")
    ax.set_ylabel("volume")
    ax.grid(True, alpha=0.25)
    _savefig(outdir, "02_volume.png", show)


def plot_returns_distribution(df: pd.DataFrame, outdir: str, show: bool) -> None:
    close = df["close"].astype(float)
    ret1 = close.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(ret1.values, bins=80, color="tab:blue", alpha=0.85)
    ax.set_title("1-step returns distribution (close pct_change)")
    ax.set_xlabel("return")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    _savefig(outdir, "03_returns_hist.png", show)


def plot_correlation_heatmap(df: pd.DataFrame, outdir: str, show: bool) -> None:
    cols = [c for c in [*PRICE_COLS, VOLUME_COL] if c in df.columns]
    if len(cols) < 2:
        return
    corr = df[cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.set_title("Feature correlation (raw)")
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _savefig(outdir, "04_corr_heatmap.png", show)


def plot_intraday_profile(df: pd.DataFrame, outdir: str, show: bool) -> None:
    ts = df[TS_COL]
    if ts.isna().all():
        return

    tmp = df[[TS_COL, "close", VOLUME_COL]].copy()
    tmp["hour"] = ts.dt.hour
    grp = tmp.groupby("hour", dropna=True).agg(close_mean=("close", "mean"), vol_mean=(VOLUME_COL, "mean"))

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(grp.index, grp["close_mean"], marker="o", label="mean close")
    ax1.set_xlabel("hour (UTC)")
    ax1.set_ylabel("mean close")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(grp.index, grp["vol_mean"], marker="o", color="tab:green", label="mean volume")
    ax2.set_ylabel("mean volume")
    ax1.set_title("Average hour-of-day profile (UTC)")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="best")
    _savefig(outdir, "05_hourly_profile.png", show)


def plot_target_runs(df: pd.DataFrame, outdir: str, show: bool) -> None:
    if TARGET_COL not in df.columns:
        return
    y = df[TARGET_COL].astype(int).values
    if y.size == 0:
        return

    # run-length encoding for consecutive ones
    padded = np.concatenate(([0], y, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    run_lengths = (ends - starts).astype(int)

    fig, ax = plt.subplots(figsize=(10, 4))
    if run_lengths.size == 0:
        ax.text(0.5, 0.5, "No consecutive target=1 runs found", ha="center", va="center")
        ax.set_axis_off()
        _savefig(outdir, "06_target_runs.png", show)
        return

    ax.hist(run_lengths, bins=min(40, max(10, int(np.sqrt(run_lengths.size)))), color="tab:red", alpha=0.85)
    ax.set_title("Consecutive target=1 run length distribution")
    ax.set_xlabel("run length (bars)")
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.2)
    _savefig(outdir, "06_target_runs.png", show)


def run_eda(cfg: EdaConfig) -> None:
    _safe_makedirs(cfg.outdir)
    df = _read_uptrend_csv(cfg.file)
    _print_overview(df, cfg.file)

    df_plot = _maybe_downsample(df, cfg.max_points)
    plot_price_and_target(df_plot, cfg.outdir, cfg.show)
    plot_volume(df_plot, cfg.outdir, cfg.show)
    plot_returns_distribution(df, cfg.outdir, cfg.show)
    plot_correlation_heatmap(df, cfg.outdir, cfg.show)
    plot_intraday_profile(df, cfg.outdir, cfg.show)
    plot_target_runs(df, cfg.outdir, cfg.show)

    print("\nDone. Open the generated PNGs in:", cfg.outdir)


def parse_args(argv: Optional[list[str]] = None) -> EdaConfig:
    p = argparse.ArgumentParser(description="EDA + Visualisierung für uptrendStocks*.csv")
    p.add_argument("--file", default=DEFAULT_FILE, help="CSV file at repo root (default: uptrendStocksQUBTUNAMBG.csv)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for PNG plots")
    p.add_argument("--show", action="store_true", help="Show plots interactively (also saves PNGs)")
    p.add_argument(
        "--max-points",
        type=int,
        default=8000,
        help="Downsample time plots to at most N points (0=disabled)",
    )
    args = p.parse_args(argv)
    return EdaConfig(file=args.file, outdir=args.outdir, show=args.show, max_points=args.max_points)


if __name__ == "__main__":
    run_eda(parse_args())
