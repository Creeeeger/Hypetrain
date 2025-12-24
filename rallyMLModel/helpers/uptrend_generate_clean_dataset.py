import argparse
import itertools
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


TS_COL = "timestamp"
TARGET_COL = "target"


@dataclass(frozen=True)
class RunParams:
    min_run: int
    min_green_ratio: float
    max_red: int
    min_return: float
    merge_gap: int


@dataclass(frozen=True)
class Score:
    bars_covered: int
    runs: int
    avg_green_ratio: float
    avg_return: float
    median_return: float


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
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df


def _merge_runs(runs: list[tuple[int, int]], merge_gap: int) -> list[tuple[int, int]]:
    if not runs or merge_gap <= 0:
        return runs
    runs = sorted(runs)
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


def find_clean_green_runs(df: pd.DataFrame, p: RunParams) -> tuple[list[tuple[int, int]], pd.DataFrame]:
    opens = df["open"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    greens = closes > opens

    runs: list[tuple[int, int]] = []
    per_run = []

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
                if red_count > p.max_red:
                    break
            end += 1

        end = end - 1
        if end >= start:
            length = end - start + 1
            green_ratio = green_count / max(1, length)
            ret = (closes[end] - closes[start]) / max(1e-9, closes[start])
            if length >= p.min_run and green_ratio >= p.min_green_ratio and ret >= p.min_return:
                runs.append((int(start), int(end)))
                per_run.append(
                    {
                        "start": int(start),
                        "end": int(end),
                        "len": int(length),
                        "green_ratio": float(green_ratio),
                        "return": float(ret),
                    }
                )

        start = max(start + 1, end + 1)

    runs = _merge_runs(runs, p.merge_gap)
    per_run_df = pd.DataFrame(per_run)
    return runs, per_run_df


def mask_from_runs(n: int, runs: list[tuple[int, int]]) -> np.ndarray:
    m = np.zeros((n,), dtype=bool)
    for s, e in runs:
        m[s : e + 1] = True
    return m


def score_runs(df: pd.DataFrame, runs: list[tuple[int, int]], per_run: pd.DataFrame) -> Score:
    m = mask_from_runs(len(df), runs)
    bars_covered = int(np.sum(m))
    if per_run is None or per_run.empty:
        return Score(bars_covered=bars_covered, runs=len(runs), avg_green_ratio=0.0, avg_return=0.0, median_return=0.0)

    return Score(
        bars_covered=bars_covered,
        runs=len(runs),
        avg_green_ratio=float(per_run["green_ratio"].mean()),
        avg_return=float(per_run["return"].mean()),
        median_return=float(per_run["return"].median()),
    )


def _parse_list_int(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip() != ""]


def _parse_list_float(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip() != ""]


def _write_dataset(
    df: pd.DataFrame,
    mask: np.ndarray,
    out_path: str,
    target_name: str,
) -> None:
    out = df.copy()
    out[target_name] = mask.astype(int)
    out.to_csv(out_path, index=False)
    print(f"[wrote] {out_path}")


def run_sweep(
    df: pd.DataFrame,
    file_stem: str,
    outdir: str,
    target_name: str,
    min_run_list: list[int],
    min_green_ratio_list: list[float],
    max_red_list: list[int],
    min_return_list: list[float],
    merge_gap_list: list[int],
    max_candidates: int,
    export_best: bool,
    export_top_k: int,
    require_min_green_ratio: float,
    require_max_red: int,
    require_min_return: float,
    require_min_run: int,
) -> None:
    os.makedirs(outdir, exist_ok=True)

    rows = []
    best = None

    combos = list(itertools.product(min_run_list, min_green_ratio_list, max_red_list, min_return_list, merge_gap_list))
    if max_candidates > 0:
        combos = combos[:max_candidates]

    for (min_run, min_green_ratio, max_red, min_return, merge_gap) in combos:
        p = RunParams(
            min_run=int(min_run),
            min_green_ratio=float(min_green_ratio),
            max_red=int(max_red),
            min_return=float(min_return),
            merge_gap=int(merge_gap),
        )
        if p.min_green_ratio < require_min_green_ratio:
            continue
        if p.max_red > require_max_red:
            continue
        if p.min_return < require_min_return:
            continue
        if p.min_run < require_min_run:
            continue

        runs, per_run = find_clean_green_runs(df, p)
        sc = score_runs(df, runs, per_run)
        coverage = sc.bars_covered / max(1, len(df))
        row = {
            "min_run": p.min_run,
            "min_green_ratio": p.min_green_ratio,
            "max_red": p.max_red,
            "min_return": p.min_return,
            "merge_gap": p.merge_gap,
            "runs": sc.runs,
            "bars_covered": sc.bars_covered,
            "coverage": float(coverage),
            "avg_green_ratio": sc.avg_green_ratio,
            "avg_return": sc.avg_return,
            "median_return": sc.median_return,
        }
        rows.append(row)

        # Objective: maximize coverage first; then maximize avg_green_ratio; then maximize avg_return
        key = (row["coverage"], row["avg_green_ratio"], row["avg_return"])
        if best is None or key > best["key"]:
            best = {"key": key, "params": p, "runs": runs, "per_run": per_run, "row": row}

    report = pd.DataFrame(rows)
    if report.empty:
        print("No candidates matched the required constraints. Try relaxing --require-* flags.")
        return

    report = report.sort_values(
        by=["coverage", "avg_green_ratio", "avg_return", "bars_covered"], ascending=[False, False, False, False]
    )
    report_path = os.path.join(outdir, f"{file_stem}__clean_sweep_report.csv")
    report.to_csv(report_path, index=False)
    print(f"[wrote] {report_path}")

    top_k = export_top_k if export_top_k > 0 else 0
    if top_k:
        top_path = os.path.join(outdir, f"{file_stem}__clean_sweep_top{top_k}.csv")
        report.head(top_k).to_csv(top_path, index=False)
        print(f"[wrote] {top_path}")

    if export_best and best is not None:
        p = best["params"]
        runs = best["runs"]
        m = mask_from_runs(len(df), runs)
        out_name = (
            f"{file_stem}__{target_name}__minrun{p.min_run}_gr{p.min_green_ratio}_red{p.max_red}"
            f"_ret{p.min_return}_gap{p.merge_gap}.csv"
        )
        out_path = os.path.join(outdir, out_name)
        _write_dataset(df, m, out_path, target_name=target_name)
        print("Best params:", best["row"])


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generiert neue Labels für 'clean green' Uptrend-Runs + Parameter-Sweep")
    p.add_argument("--file", default="uptrendStocksQUBTUNAMBG.csv", help="Input CSV (repo root)")
    p.add_argument("--outdir", default="clean_dataset_out", help="Output directory")
    p.add_argument("--target-name", default="target_clean", help="Name of the generated target column")

    p.add_argument("--min-run", default="6,8,10,12", help="Comma-separated list")
    p.add_argument("--min-green-ratio", default="0.75,0.8,0.85,0.9", help="Comma-separated list")
    p.add_argument("--max-red", default="0,1,2", help="Comma-separated list")
    p.add_argument("--min-return", default="0.0,0.002,0.004,0.006", help="Comma-separated list (0.01=+1%)")
    p.add_argument("--merge-gap", default="0,1,2", help="Comma-separated list")

    p.add_argument("--max-candidates", type=int, default=0, help="Limit number of combinations (0=all)")
    p.add_argument("--export-best", action="store_true", help="Write best dataset CSV to outdir")
    p.add_argument("--export-top-k", type=int, default=25, help="Write top-k rows CSV (0=disable)")

    p.add_argument(
        "--require-min-green-ratio",
        type=float,
        default=0.85,
        help="Hard constraint: only consider configs with min_green_ratio >= this",
    )
    p.add_argument(
        "--require-max-red",
        type=int,
        default=1,
        help="Hard constraint: only consider configs with max_red <= this",
    )
    p.add_argument(
        "--require-min-return",
        type=float,
        default=0.0,
        help="Hard constraint: only consider configs with min_return >= this",
    )
    p.add_argument(
        "--require-min-run",
        type=int,
        default=8,
        help="Hard constraint: only consider configs with min_run >= this",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    df_in = _read_uptrend_csv(args.file)
    stem = os.path.splitext(os.path.basename(args.file))[0]

    run_sweep(
        df=df_in,
        file_stem=stem,
        outdir=args.outdir,
        target_name=args.target_name,
        min_run_list=_parse_list_int(args.min_run),
        min_green_ratio_list=_parse_list_float(args.min_green_ratio),
        max_red_list=_parse_list_int(args.max_red),
        min_return_list=_parse_list_float(args.min_return),
        merge_gap_list=_parse_list_int(args.merge_gap),
        max_candidates=int(args.max_candidates),
        export_best=bool(args.export_best),
        export_top_k=int(args.export_top_k),
        require_min_green_ratio=float(args.require_min_green_ratio),
        require_max_red=int(args.require_max_red),
        require_min_return=float(args.require_min_return),
        require_min_run=int(args.require_min_run),
    )
