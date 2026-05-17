"""
score_results.py — Score a benchmark results CSV using the LLM judge.

Usage:
    python score_results.py results/benchmark_results_YYYYMMDD_HHMMSS.csv

Outputs:
    results/scored_YYYYMMDD_HHMMSS.csv   — original CSV + 'judgment' column
    results/frr_summary_YYYYMMDD_HHMMSS.csv  — FRR per strategy × turn
    results/frr_by_distance_YYYYMMDD_HHMMSS.csv — FRR per strategy × distance
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from harness.judge import compute_frr, compute_frr_by_distance, score_results_csv
from harness.llm_groq import SimpleGroqClient
from harness.utils import filter_none_keys, set_seed, setup_logging

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Score benchmark results CSV.")
    parser.add_argument("results_csv", help="Path to benchmark_results_*.csv")
    parser.add_argument(
        "--threshold",
        choices=["CORRECT", "PARTIAL"],
        default="CORRECT",
        help="FRR hit threshold (default: CORRECT)",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.results_csv):
        print(f"Error: file not found: {args.results_csv}", file=sys.stderr)
        sys.exit(1)

    results_dir = os.path.dirname(args.results_csv) or "results"
    setup_logging(results_dir)
    set_seed(42)

    groq_keys = filter_none_keys(
        [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ]
    )
    llm = SimpleGroqClient(groq_keys)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    scored_csv = os.path.join(results_dir, f"scored_{ts}.csv")
    frr_csv = os.path.join(results_dir, f"frr_summary_{ts}.csv")
    frr_dist_csv = os.path.join(results_dir, f"frr_by_distance_{ts}.csv")

    print(f"Scoring {args.results_csv} …")
    df = score_results_csv(
        results_csv=args.results_csv,
        llm=llm,
        output_csv=scored_csv,
    )

    frr = compute_frr(df, threshold=args.threshold)
    frr.to_csv(frr_csv, index=False)
    print(f"\nFRR summary ({args.threshold} threshold):")
    print(frr.to_string(index=False))
    print(f"\nSaved to {frr_csv}")

    frr_dist = compute_frr_by_distance(df)
    frr_dist.to_csv(frr_dist_csv, index=False)
    print(f"\nFRR by distance saved to {frr_dist_csv}")


if __name__ == "__main__":
    main()