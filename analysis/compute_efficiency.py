"""
compute_efficiency.py — Compute the efficiency table for the paper.

    Strategy        AUC-FRR    Avg context tokens    LLM calls/run
    baseline          0.92           2400                 13
    rolling_summary   0.71            380                 15
    hierarchical      0.79            520                 17
    rag               0.68            290                 13

This script derives all three columns from data that already exists:
    - AUC-FRR      : from the scored CSV (judgment column)
    - Avg ctx tokens: from the JSONL history files (context size logged at recall turns)
    - LLM calls/run : from the JSONL history files (call_count field logged per turn)

IMPORTANT: This script requires run_benchmark.py to have been updated to log
two extra fields in history JSONL entries at recall turns:
    - "context_chars": int   (length of context string passed to LLM)
    - "llm_calls_so_far": int (total generate() calls made so far in this run)

See the patch to run_benchmark.py below for how to add these fields.

Usage:
    python analysis/compute_efficiency.py \\
        --scored   results/scored_*.csv \\
        --history  results/histories/

Outputs:
    results/efficiency_table_TIMESTAMP.csv
    results/efficiency_table_TIMESTAMP.txt   (print-ready for paper)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# AUC-FRR
# ---------------------------------------------------------------------------

def compute_auc_frr(scored_csv: str) -> pd.DataFrame:
    """
    Load scored CSV and compute AUC-FRR per strategy.

    AUC-FRR = mean of FRR@5, FRR@10, FRR@15, FRR@20, FRR@25
    (uniform weighting across recall turns — equivalent to area under
    a step function with equal spacing).

    Returns DataFrame with columns: strategy, auc_frr
    """
    df = pd.read_csv(scored_csv)

    if "judgment" not in df.columns:
        raise ValueError(
            f"'judgment' column not found in {scored_csv}. "
            "Run analysis/score_results.py first."
        )

    df["hit"] = (df["judgment"] == "CORRECT").astype(int)

    # FRR per strategy × turn
    frr = (
        df.groupby(["strategy", "turn"])
        .agg(hits=("hit", "sum"), total=("hit", "count"))
        .assign(frr=lambda d: d["hits"] / d["total"])
        .reset_index()
    )

    # AUC = mean FRR across turns
    auc = (
        frr.groupby("strategy")["frr"]
        .mean()
        .reset_index()
        .rename(columns={"frr": "auc_frr"})
    )
    auc["auc_frr"] = auc["auc_frr"].round(3)
    return auc


# ---------------------------------------------------------------------------
# Context size and LLM call count from JSONL histories
# ---------------------------------------------------------------------------

def load_history_metrics(history_dir: str) -> pd.DataFrame:
    """
    Read all JSONL history files and extract two metrics per run:
        - avg_context_tokens : average context size (chars // 4) at recall turns
        - llm_calls_per_run  : total LLM generate() calls in the run

    Each JSONL file = one (script × strategy × rep) run.
    Each line = one turn, with optional fields:
        "context_chars"    : int  (present at recall turns)
        "llm_calls_so_far" : int  (present at every assistant turn)

    Returns DataFrame with columns:
        script_id, strategy, rep, avg_context_tokens, llm_calls_per_run
    """
    records = []
    pattern = os.path.join(history_dir, "history_*.jsonl")
    files = glob.glob(pattern)

    if not files:
        raise FileNotFoundError(
            f"No history files found in {history_dir}. "
            "Make sure run_benchmark.py has been run with history saving enabled."
        )

    for fpath in sorted(files):
        fname = os.path.basename(fpath).replace("history_", "").replace(".jsonl", "")

        # Parse filename: {script_id}_{strategy_name}_rep{rep}
        # Strategy names may contain underscores so we split from the right on _rep
        parts = fname.rsplit("_rep", 1)
        if len(parts) != 2:
            print(f"[warn] Skipping unrecognised filename: {fname}")
            continue

        try:
            rep = int(parts[1])
        except ValueError:
            print(f"[warn] Cannot parse rep from: {fname}")
            continue

        # Split script_id and strategy_name
        # Strategy names: baseline, rolling_summary, hierarchical, rag
        strategy_names = ["rolling_summary", "hierarchical", "baseline", "rag"]
        strategy_name = None
        script_id = None
        for sname in strategy_names:
            marker = f"_{sname}"
            if marker in parts[0]:
                idx = parts[0].rfind(marker)
                script_id = parts[0][:idx]
                strategy_name = sname
                break

        if strategy_name is None or script_id is None:
            print(f"[warn] Cannot parse strategy from: {fname}")
            continue

        # Read JSONL
        context_chars_at_recall: list[int] = []
        max_llm_calls: int = 0

        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Context size logged at recall turns (user role, is_recall=True)
                if entry.get("is_recall") and "context_chars" in entry:
                    context_chars_at_recall.append(entry["context_chars"])

                # LLM call count — take the maximum (= total at end of run)
                if "llm_calls_so_far" in entry:
                    max_llm_calls = max(max_llm_calls, entry["llm_calls_so_far"])

        avg_context_tokens = (
            int(np.mean(context_chars_at_recall) // 4)
            if context_chars_at_recall
            else None
        )
        llm_calls = max_llm_calls if max_llm_calls > 0 else None

        records.append({
            "script_id":          script_id,
            "strategy":           strategy_name,
            "rep":                rep,
            "avg_context_tokens": avg_context_tokens,
            "llm_calls_per_run":  llm_calls,
        })

    if not records:
        raise ValueError("No valid history records parsed. Check JSONL format.")

    df = pd.DataFrame(records)

    # Aggregate across scripts and reps
    summary = (
        df.groupby("strategy")
        .agg(
            avg_context_tokens=("avg_context_tokens", "mean"),
            llm_calls_per_run=("llm_calls_per_run", "mean"),
        )
        .reset_index()
    )
    summary["avg_context_tokens"] = summary["avg_context_tokens"].round(0).astype("Int64")
    summary["llm_calls_per_run"]  = summary["llm_calls_per_run"].round(1)
    return summary


# ---------------------------------------------------------------------------
# Combine into efficiency table
# ---------------------------------------------------------------------------

def build_efficiency_table(
    scored_csv: str,
    history_dir: str,
) -> pd.DataFrame:
    """
    Combine AUC-FRR, avg context tokens, and LLM calls into one table.

    Returns DataFrame with columns:
        strategy, auc_frr, avg_context_tokens, llm_calls_per_run,
        context_reduction_pct, retention_vs_baseline_pct
    """
    auc_df     = compute_auc_frr(scored_csv)
    metrics_df = load_history_metrics(history_dir)

    table = auc_df.merge(metrics_df, on="strategy", how="outer")

    # Sort by AUC descending
    table = table.sort_values("auc_frr", ascending=False).reset_index(drop=True)

    # Derived columns for paper
    baseline_auc     = table.loc[table["strategy"] == "baseline", "auc_frr"].values
    baseline_ctx     = table.loc[table["strategy"] == "baseline", "avg_context_tokens"].values

    if len(baseline_auc) > 0:
        table["retention_vs_baseline_pct"] = (
            (table["auc_frr"] / baseline_auc[0] * 100).round(1)
        )
    else:
        table["retention_vs_baseline_pct"] = None

    if len(baseline_ctx) > 0 and baseline_ctx[0] is not None:
        table["context_reduction_pct"] = (
            ((1 - table["avg_context_tokens"] / baseline_ctx[0]) * 100).round(1)
        )
    else:
        table["context_reduction_pct"] = None

    return table


# ---------------------------------------------------------------------------
# Pretty print for paper
# ---------------------------------------------------------------------------

def print_table(df: pd.DataFrame) -> str:
    """Format the efficiency table as a readable string for the paper."""
    lines = []
    lines.append(
        f"{'Strategy':<20} {'AUC-FRR':>8} {'Ctx tokens':>12} "
        f"{'Ctx reduction':>14} {'Retention vs base':>18} {'LLM calls/run':>14}"
    )
    lines.append("─" * 90)

    for _, row in df.iterrows():
        ctx_red  = f"{row['context_reduction_pct']:.1f}%" if pd.notna(row.get("context_reduction_pct")) else "—"
        ret_base = f"{row['retention_vs_baseline_pct']:.1f}%" if pd.notna(row.get("retention_vs_baseline_pct")) else "—"
        ctx_tok  = str(row["avg_context_tokens"]) if pd.notna(row["avg_context_tokens"]) else "—"
        llm_c    = str(row["llm_calls_per_run"]) if pd.notna(row["llm_calls_per_run"]) else "—"

        lines.append(
            f"{row['strategy']:<20} {row['auc_frr']:>8.3f} {ctx_tok:>12} "
            f"{ctx_red:>14} {ret_base:>18} {llm_c:>14}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute efficiency table for paper.")
    parser.add_argument("--scored",  required=True, help="Path to scored_*.csv")
    parser.add_argument("--history", required=True, help="Path to results/histories/ dir")
    args = parser.parse_args()

    results_dir = os.path.dirname(os.path.abspath(args.scored))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Computing AUC-FRR …")
    table = build_efficiency_table(
        scored_csv=args.scored,
        history_dir=args.history,
    )

    # Save CSV
    out_csv = os.path.join(results_dir, f"efficiency_table_{ts}.csv")
    table.to_csv(out_csv, index=False)

    # Save and print formatted table
    formatted = print_table(table)
    out_txt = os.path.join(results_dir, f"efficiency_table_{ts}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(formatted + "\n")

    print("\n" + formatted)
    print(f"\nSaved → {out_csv}")
    print(f"Saved → {out_txt}")


if __name__ == "__main__":
    main()