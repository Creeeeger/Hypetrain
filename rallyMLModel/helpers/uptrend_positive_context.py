import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), "../.mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(os.getcwd(), "../.cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

DEFAULT_FILE = "uptrendStocksQUBTUNAMBG.csv"
DEFAULT_OUTDIR = "pos_out"

TS_COL = "timestamp"
DEFAULT_TARGET_COL = "target"


@dataclass(frozen=True)
class Cfg:
    file: str
    outdir: str
    target_col: str
    pre_bars: int
    post_bars: int
    min_run: int
    merge_gap: int
    max_events: int
    overview_max_points: int
    run_source: str
    clean_min_green_ratio: float
    clean_min_run: int
    clean_max_red: int
    clean_min_return: float
    show: bool


def _safe_makedirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _read_uptrend_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, on_bad_lines="skip")
    required = [TS_COL, "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {df.columns.tolist()}")

    df = df.copy()
    ts_clean = (
        df[TS_COL]
        .astype(str)
        .str.replace(r"\s[A-Z]{2,5}\s(\d{4})$", r" \1", regex=True)
    )
    df[TS_COL] = pd.to_datetime(ts_clean, errors="coerce", utc=True)
    df = df.dropna(subset=[TS_COL]).sort_values(TS_COL).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    return df


def _find_positive_runs(y: np.ndarray, min_run: int, merge_gap: int) -> list[tuple[int, int]]:
    if y.size == 0:
        return []

    y = (y.astype(int) == 1).astype(int)
    padded = np.concatenate(([0], y, [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0] - 1  # inclusive
    runs = [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_run]
    if not runs:
        return []

    if merge_gap <= 0:
        return runs

    merged: list[tuple[int, int]] = []
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        if s <= cur_e + merge_gap + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _find_clean_green_runs(
    df: pd.DataFrame,
    min_run: int,
    min_green_ratio: float,
    max_red: int,
    min_return: float,
    merge_gap: int,
) -> list[tuple[int, int]]:
    if len(df) == 0:
        return []

    opens = df["open"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    greens = closes > opens

    runs: list[tuple[int, int]] = []
    start = 0
    while start < len(df):
        if not greens[start]:
            start += 1
            continue

        end = start
        red_count = 0
        green_count = 0
        while end < len(df):
            if greens[end]:
                green_count += 1
            else:
                red_count += 1
                if red_count > max_red:
                    break
            end += 1

        end = end - 1
        if end >= start:
            length = end - start + 1
            green_ratio = green_count / max(1, length)
            ret = (closes[end] - closes[start]) / max(1e-9, closes[start])
            if length >= min_run and green_ratio >= min_green_ratio and ret >= min_return:
                runs.append((int(start), int(end)))

        start = max(start + 1, end + 1)

    if not runs:
        return []

    if merge_gap <= 0:
        return runs

    merged: list[tuple[int, int]] = []
    cur_s, cur_e = runs[0]
    for s, e in runs[1:]:
        if s <= cur_e + merge_gap + 1:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _maybe_downsample_for_overview(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    idx = np.linspace(0, len(df) - 1, max_points).astype(int)
    return df.iloc[idx].reset_index(drop=True)


def _candlesticks(ax, ohlc: pd.DataFrame) -> None:
    if ohlc.empty:
        return

    # Use tz-naive numpy datetimes to avoid pandas' to_pydatetime deprecation warnings
    times = ohlc[TS_COL].dt.tz_convert(None).to_numpy()
    x = mdates.date2num(times)
    delta = float(np.median(np.diff(x))) if len(x) > 1 else 1 / (24 * 60)
    width = delta * 0.7

    opens = ohlc["open"].astype(float).to_numpy()
    highs = ohlc["high"].astype(float).to_numpy()
    lows = ohlc["low"].astype(float).to_numpy()
    closes = ohlc["close"].astype(float).to_numpy()

    up = closes >= opens
    colors = np.where(up, "tab:green", "tab:red")

    ax.vlines(x, lows, highs, color=colors, linewidth=0.8, alpha=0.9, zorder=2)

    bottoms = np.minimum(opens, closes)
    heights = np.abs(closes - opens)
    heights = np.where(heights == 0, 1e-9, heights)
    for xi, bottom, height, c in zip(x, bottoms, heights, colors):
        rect = plt.Rectangle(
            (xi - width / 2, float(bottom)),
            float(width),
            float(height),
            facecolor=c,
            edgecolor=c,
            alpha=0.6,
            zorder=3,
        )
        ax.add_patch(rect)

    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M", tz=mdates.UTC))
    ax.tick_params(axis="x", rotation=30)


def _savefig(outdir: str, name: str, show: bool) -> None:
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=170)
    if show:
        plt.show()
    plt.close()
    print(f"[saved] {path}")


def plot_overview(df: pd.DataFrame, runs: list[tuple[int, int]], cfg: Cfg) -> None:
    df_plot = _maybe_downsample_for_overview(df, cfg.overview_max_points)
    fig, ax = plt.subplots(figsize=(14, 5))
    _candlesticks(ax, df_plot[[TS_COL, "open", "high", "low", "close"]])
    ax.set_title("OHLC candlesticks over time (positive runs shaded)")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.grid(True, alpha=0.25)

    for (s, e) in runs:
        ax.axvspan(df[TS_COL].iloc[s], df[TS_COL].iloc[e], color="red", alpha=0.15)
    _savefig(cfg.outdir, "00_overview_positive_runs.png", cfg.show)


def plot_event_zoom(df: pd.DataFrame, run: tuple[int, int], cfg: Cfg, event_id: int) -> None:
    s, e = run
    start = max(0, s - cfg.pre_bars)
    end = min(len(df) - 1, e + cfg.post_bars)
    view = df.iloc[start: end + 1].copy()

    fig, ax = plt.subplots(figsize=(14, 5))
    _candlesticks(ax, view[[TS_COL, "open", "high", "low", "close"]])
    ax.grid(True, alpha=0.25)

    ax.axvspan(df[TS_COL].iloc[s], df[TS_COL].iloc[e], color="red", alpha=0.18, label="target=1 run")
    ax.axvline(df[TS_COL].iloc[s], color="red", alpha=0.35, linewidth=1.0)
    ax.axvline(df[TS_COL].iloc[e], color="red", alpha=0.35, linewidth=1.0)

    # Mark key prices
    ax.scatter(df[TS_COL].iloc[s], df["close"].iloc[s], s=30, c="red", zorder=4, label="run start")
    ax.scatter(df[TS_COL].iloc[e], df["close"].iloc[e], s=30, c="darkred", zorder=4, label="run end")

    run_len = e - s + 1
    ax.set_title(f"Event {event_id:03d}: run [{s}:{e}] len={run_len} (context -{cfg.pre_bars}/+{cfg.post_bars} bars)")
    ax.set_xlabel("time")
    ax.set_ylabel("price")
    ax.legend(loc="best")

    _savefig(cfg.outdir, f"event_{event_id:03d}_run_{s}_{e}.png", cfg.show)


def run(cfg: Cfg) -> None:
    _safe_makedirs(cfg.outdir)
    df = _read_uptrend_csv(cfg.file)

    if cfg.target_col not in df.columns:
        raise ValueError(f"Target column '{cfg.target_col}' not found. Available: {df.columns.tolist()}")
    df[cfg.target_col] = pd.to_numeric(df[cfg.target_col], errors="coerce").fillna(0).astype(int)

    target_runs = _find_positive_runs(df[cfg.target_col].values, min_run=cfg.min_run, merge_gap=cfg.merge_gap)
    clean_runs = _find_clean_green_runs(
        df,
        min_run=cfg.clean_min_run,
        min_green_ratio=cfg.clean_min_green_ratio,
        max_red=cfg.clean_max_red,
        min_return=cfg.clean_min_return,
        merge_gap=cfg.merge_gap,
    )

    def _mask_from_runs(n: int, rs: list[tuple[int, int]]) -> np.ndarray:
        m = np.zeros((n,), dtype=bool)
        for s, e in rs:
            m[s : e + 1] = True
        return m

    if cfg.run_source == "target":
        runs = target_runs
        run_label = "target"
    elif cfg.run_source == "clean":
        runs = clean_runs
        run_label = "clean"
    elif cfg.run_source == "both":
        # union for plotting events; overview will still shade using chosen label below
        runs = sorted(set(target_runs + clean_runs))
        run_label = "both"
    else:
        raise ValueError("--run-source must be one of: target, clean, both")

    print(f"File: {cfg.file}")
    print(f"Rows: {len(df):,}")
    print(
        f"Target runs ({cfg.target_col}): {len(target_runs)} (min_run={cfg.min_run}, merge_gap={cfg.merge_gap})"
    )
    print(
        "Clean runs:",
        f"{len(clean_runs)} (min_run={cfg.clean_min_run}, green_ratio>={cfg.clean_min_green_ratio}, "
        f"max_red={cfg.clean_max_red}, min_return>={cfg.clean_min_return}, merge_gap={cfg.merge_gap})",
    )
    # Quick consistency stats (bar-level overlap)
    if len(df) > 0:
        m_t = _mask_from_runs(len(df), target_runs)
        m_c = _mask_from_runs(len(df), clean_runs)
        tp = int(np.sum(m_t & m_c))
        fp = int(np.sum(~m_t & m_c))
        fn = int(np.sum(m_t & ~m_c))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        print(f"Bar overlap (clean vs target): precision={prec:.3f} recall={rec:.3f} tp={tp} fp={fp} fn={fn}")

    print(f"Plotting run_source='{cfg.run_source}' (events={len(runs)})")
    if not runs:
        print(f"No runs found for run_source='{run_label}' with current settings.")
        return

    # Overview: for 'both' we shade clean runs in green and target runs in red to reveal ambiguity.
    if cfg.run_source == "both":
        fig, ax = plt.subplots(figsize=(14, 5))
        df_plot = _maybe_downsample_for_overview(df, cfg.overview_max_points)
        _candlesticks(ax, df_plot[[TS_COL, "open", "high", "low", "close"]])
        ax.set_title("OHLC candlesticks (target runs=red, clean runs=green)")
        ax.set_xlabel("time")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.25)
        for (s, e) in clean_runs:
            ax.axvspan(df[TS_COL].iloc[s], df[TS_COL].iloc[e], color="tab:green", alpha=0.14)
        for (s, e) in target_runs:
            ax.axvspan(df[TS_COL].iloc[s], df[TS_COL].iloc[e], color="red", alpha=0.14)
        _savefig(cfg.outdir, "00_overview_target_vs_clean.png", cfg.show)
    else:
        plot_overview(df, runs, cfg)

    n = min(cfg.max_events, len(runs)) if cfg.max_events > 0 else len(runs)
    for i in range(n):
        plot_event_zoom(df, runs[i], cfg, event_id=i + 1)

    print("\nDone. Open the generated PNGs in:", cfg.outdir)


def parse_args(argv: Optional[list[str]] = None) -> Cfg:
    p = argparse.ArgumentParser(description="Visualisiert target=1 Runs + Kontextfenster im Uptrend-Datensatz")
    p.add_argument("--file", default=DEFAULT_FILE, help="CSV file at repo root (default: uptrendStocksQUBTUNAMBG.csv)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory for PNG plots")
    p.add_argument("--target-col", default=DEFAULT_TARGET_COL, help="Which target column to use (e.g. target_clean)")
    p.add_argument("--pre-bars", type=int, default=120, help="Bars before a positive run to include")
    p.add_argument("--post-bars", type=int, default=120, help="Bars after a positive run to include")
    p.add_argument("--min-run", type=int, default=1, help="Minimum consecutive target=1 length to count as an event")
    p.add_argument("--merge-gap", type=int, default=0, help="Merge runs if the gap between them is <= this many bars")
    p.add_argument("--max-events", type=int, default=30, help="Max number of events to plot (0=all)")
    p.add_argument(
        "--overview-max-points",
        type=int,
        default=3000,
        help="Downsample overview candlesticks to at most N points (0=disabled)",
    )
    p.add_argument(
        "--run-source",
        choices=["target", "clean", "both"],
        default="target",
        help="Which runs to visualize: label-based target, heuristic clean-green, or both (overlay)",
    )
    p.add_argument("--clean-min-run", type=int, default=8, help="Minimum length for a clean up-run (bars)")
    p.add_argument(
        "--clean-min-green-ratio",
        type=float,
        default=0.85,
        help="Minimum fraction of green candles within a clean run",
    )
    p.add_argument(
        "--clean-max-red",
        type=int,
        default=1,
        help="How many red candles allowed inside a clean run before it ends",
    )
    p.add_argument(
        "--clean-min-return",
        type=float,
        default=0.003,
        help="Minimum close-to-close return from start to end of run (e.g. 0.01 = +1%)",
    )
    p.add_argument("--show", action="store_true", help="Show plots interactively (also saves PNGs)")
    a = p.parse_args(argv)
    return Cfg(
        file=a.file,
        outdir=a.outdir,
        target_col=a.target_col,
        pre_bars=a.pre_bars,
        post_bars=a.post_bars,
        min_run=a.min_run,
        merge_gap=a.merge_gap,
        max_events=a.max_events,
        overview_max_points=a.overview_max_points,
        run_source=a.run_source,
        clean_min_green_ratio=a.clean_min_green_ratio,
        clean_min_run=a.clean_min_run,
        clean_max_red=a.clean_max_red,
        clean_min_return=a.clean_min_return,
        show=a.show,
    )


if __name__ == "__main__":
    run(parse_args())
