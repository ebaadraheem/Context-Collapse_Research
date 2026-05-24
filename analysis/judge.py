from __future__ import annotations

import json
import logging
import re
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are an impartial judge for a fact-recall benchmark.
Determine whether the agent's response correctly answers the recall question.

Fact ID        : {fact_id}
Recall question: {question}
Ground truth (any one of these is acceptable):
{ground_truth_list}

Agent response: "{agent_response}"

Scoring rubric:
- CORRECT  : The agent's response conveys the same information as at least one \
ground-truth answer. Minor wording differences, unit variations, and \
abbreviations are acceptable.
- PARTIAL  : The agent recalled part of a compound fact (e.g. one of two \
required values).
- INCORRECT: The agent's response is wrong, hallucinated a different value, \
said "I don't know", or is blank.

Reply with ONLY one word: CORRECT, PARTIAL, or INCORRECT.

Judgment:\
"""

# Ordered from most specific to least specific. This order is critical for the regex matching logic in judge_response().
_VERDICT_ORDER = ["INCORRECT", "PARTIAL", "CORRECT"]

# Word-boundary regex patterns — compiled once at module load.
_VERDICT_RE = {v: re.compile(rf"\b{v}\b") for v in _VERDICT_ORDER}

# Regex for the "I don't know" family of non-answers.
_IDK_RE = re.compile(r"i\s+don'?t\s+know\.?", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------

def judge_response(
    llm: Any,
    fact_id: str,
    ground_truth: list[str],
    agent_response: str,
    question: str = "",
) -> str:

    agent_norm = agent_response.strip().lower()

    # 1. Hard INCORRECT
    if not agent_norm or _IDK_RE.fullmatch(agent_norm):
        return "INCORRECT"

    # 2. Fast path — exact substring match (case-insensitive)
    for gt in ground_truth:
        if gt.strip().lower() in agent_norm:
            logger.debug("Fast-path CORRECT for fact_id=%s", fact_id)
            return "CORRECT"

    # 3. Slow path — LLM judge
    gt_formatted = "\n".join(f"  - {gt}" for gt in ground_truth)
    prompt = _JUDGE_PROMPT.format(
        fact_id=fact_id,
        question=question or "(not provided)",
        ground_truth_list=gt_formatted,
        agent_response=agent_response.strip(),
    )

    judge_system = (
        "You are a precise, impartial evaluator. "
        "Reply with exactly one word: CORRECT, PARTIAL, or INCORRECT."
    )

    try:
        raw = llm.generate(
            prompt,
            system_prompt=judge_system,
            max_tokens=10,
            use_stop_tokens=False,
        )
    except Exception as exc:
        logger.warning("Judge LLM call failed for fact_id=%s: %s", fact_id, exc)
        return "INCORRECT"

    verdict_upper = raw.strip().upper()

    for v in _VERDICT_ORDER:
        if _VERDICT_RE[v].search(verdict_upper):
            return v

    logger.warning(
        "Unrecognised judge verdict %r for fact_id=%s. Defaulting to INCORRECT.",
        verdict_upper, fact_id,
    )
    return "INCORRECT"


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def score_results_csv(
    results_csv: str,
    llm: Any,
    output_csv: str | None = None,
) -> pd.DataFrame:
 
    df = pd.read_csv(results_csv)

    # ground_truth is stored as a JSON string — parse it back.
    df["ground_truth_parsed"] = df["ground_truth"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else []
    )

    judgments: list[str] = []
    for _, row in df.iterrows():
        verdict = judge_response(
            llm=llm,
            fact_id=str(row.get("target_fact_id", "")),
            ground_truth=row["ground_truth_parsed"],
            agent_response=str(row.get("agent_response", "")),
            question=str(row.get("question", "")),
        )
        judgments.append(verdict)
        logger.info(
            "script=%s strategy=%s turn=%s fact=%s → %s",
            row.get("script_id"), row.get("strategy"),
            row.get("turn"), row.get("target_fact_id"), verdict,
        )

    df["judgment"] = judgments

    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"[judge] Scored results saved to {output_csv}")

    return df


# ---------------------------------------------------------------------------
# FRR computation
# ---------------------------------------------------------------------------

def compute_frr(df: pd.DataFrame, threshold: str = "CORRECT") -> pd.DataFrame:
    """
    Compute Factual Retention Rate (FRR) per strategy × recall turn.
    FRR@K = hits_at_K / total_at_K
    """
    hit_values = {"CORRECT", "PARTIAL"} if threshold == "PARTIAL" else {"CORRECT"}
    df = df.copy()
    df["hit"] = df["judgment"].isin(hit_values).astype(int)
    return (
        df.groupby(["strategy", "turn"])
        .agg(hits=("hit", "sum"), total=("hit", "count"))
        .assign(frr=lambda d: d["hits"] / d["total"])
        .reset_index()
    )


def compute_frr_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FRR by distance (turns) from fact injection to recall.

    Assumes facts are always injected at turn 1, so distance = turn - 1.

    Returns:
        DataFrame with columns: strategy, distance, hits, total, frr.
    """
    df = df.copy()
    df["distance"] = df["turn"] - 1
    df["hit"] = (df["judgment"] == "CORRECT").astype(int)
    return (
        df.groupby(["strategy", "distance"])
        .agg(hits=("hit", "sum"), total=("hit", "count"))
        .assign(frr=lambda d: d["hits"] / d["total"])
        .reset_index()
    )


def compute_compression_efficiency(
    results_df: pd.DataFrame,
    history_dir: str,
) -> pd.DataFrame:
    """
    Compute average context length (chars) at each recall turn per strategy.
    Useful for the paper's efficiency table (FRR vs token cost).

    Reads JSONL history files from history_dir and measures the length of the
    assistant's prompt context by approximating from stored history size.

    Returns:
        DataFrame with columns: strategy, turn, avg_context_chars.
    """
    import os, json, glob

    rows = []
    pattern = os.path.join(history_dir, "history_*.jsonl")
    for fpath in glob.glob(pattern):
        fname = os.path.basename(fpath)
        # history_{script_id}_{strategy_name}_rep{rep}.jsonl
        parts = fname.replace("history_", "").replace(".jsonl", "").rsplit("_rep", 1)
        if len(parts) != 2:
            continue
        pass  # Filename parsing logic can be added here if needed to extract script_id, strategy_name, rep

    logger.info("compute_compression_efficiency: not fully implemented yet.")
    return pd.DataFrame(columns=["strategy", "turn", "avg_context_chars"])