"""
LLM judge for the context-collapse benchmark.

Scoring pipeline:
1. Fast path: if any ground-truth string is a substring of the agent response
   (case-insensitive), return CORRECT immediately without an LLM call.
2. Slow path: ask the LLM to judge whether the response conveys the same
   information as any of the ground-truth answers.

Verdict levels
--------------
CORRECT  - The agent recalled the fact correctly (exact or semantically equivalent).
PARTIAL  - The agent recalled part of a compound fact correctly.
INCORRECT - The agent was wrong, hallucinated, or said "I don't know".
"""

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

Fact ID      : {fact_id}
Recall question: {question}
Ground truth (any one of these is acceptable):
{ground_truth_list}

Agent response: "{agent_response}"

Scoring rubric:
- CORRECT  : The agent's response conveys the same information as at least one \
ground-truth answer. Minor wording differences, unit variations, and \
abbreviations are fine.
- PARTIAL  : The agent recalled part of a compound fact (e.g. one of two required \
values).
- INCORRECT: The agent's response is wrong, hallucinated a different value, said \
"I don't know", or left a blank.

Reply with ONLY one word: CORRECT, PARTIAL, or INCORRECT.

Judgment:\
"""

_VALID_VERDICTS = {"CORRECT", "PARTIAL", "INCORRECT"}


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
    """
    Judge a single agent response against the ground truth.

    Args:
        llm          : LLM client with ``generate(prompt, system_prompt, ...) -> str``.
        fact_id      : Identifier of the fact being tested (for the prompt).
        ground_truth : List of acceptable answer strings.
        agent_response: The agent's raw text response.
        question     : The recall question (for context in the prompt).

    Returns:
        One of "CORRECT", "PARTIAL", or "INCORRECT".
    """
    # Normalise
    agent_norm = agent_response.strip().lower()

    # Hard INCORRECT: empty or explicit "I don't know"
    if not agent_norm or re.fullmatch(r"i\s+don'?t\s+know\.?", agent_norm):
        return "INCORRECT"

    # Fast path: exact substring match (case-insensitive)
    for gt in ground_truth:
        if gt.strip().lower() in agent_norm:
            logger.debug("Fast-path CORRECT for fact_id=%s", fact_id)
            return "CORRECT"

    # Slow path: LLM judge
    gt_formatted = "\n".join(f"  - {gt}" for gt in ground_truth)
    prompt = _JUDGE_PROMPT.format(
        fact_id=fact_id,
        question=question or "(not provided)",
        ground_truth_list=gt_formatted,
        agent_response=agent_response.strip(),
    )

    # Use a neutral system prompt for judging (not the recall system prompt)
    judge_system = (
        "You are a precise, impartial evaluator. "
        "Reply with exactly one word from the allowed set."
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

    verdict = raw.strip().upper()

    # Extract the verdict word even if the model adds punctuation
    for v in _VALID_VERDICTS:
        if v in verdict:
            return v

    logger.warning(
        "Unrecognised judge verdict '%s' for fact_id=%s. Defaulting to INCORRECT.",
        verdict,
        fact_id,
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
    """
    Load a benchmark results CSV, run the judge on every row, and return a
    DataFrame with a ``judgment`` column added.

    Args:
        results_csv : Path to the CSV produced by run_benchmark.py.
        llm         : LLM client used for slow-path judging.
        output_csv  : If provided, write the scored DataFrame here.

    Returns:
        DataFrame with columns: script_id, strategy, repetition, turn,
        target_fact_id, agent_response, ground_truth, question, judgment.
    """
    df = pd.read_csv(results_csv)

    # ground_truth column is stored as a JSON string
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
            row.get("script_id"),
            row.get("strategy"),
            row.get("turn"),
            row.get("target_fact_id"),
            verdict,
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
    Compute Factual Retention Rate (FRR) per strategy × turn.

    FRR@K = (number of CORRECT recalls at turn K) / (total recalls at turn K)

    Args:
        df        : Scored DataFrame with a ``judgment`` column.
        threshold : ``"CORRECT"`` (strict) or ``"PARTIAL"`` (lenient, counts
                    CORRECT + PARTIAL as hits).

    Returns:
        DataFrame indexed by (strategy, turn) with columns: hits, total, frr.
    """
    if threshold == "PARTIAL":
        hit_values = {"CORRECT", "PARTIAL"}
    else:
        hit_values = {"CORRECT"}

    df = df.copy()
    df["hit"] = df["judgment"].isin(hit_values).astype(int)

    summary = (
        df.groupby(["strategy", "turn"])
        .agg(hits=("hit", "sum"), total=("hit", "count"))
        .assign(frr=lambda d: d["hits"] / d["total"])
        .reset_index()
    )
    return summary


def compute_frr_by_distance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute FRR by distance from fact injection (turn 1) to recall turn.
    Useful for plotting how retention decays with distance.

    Assumes facts are always injected at turn 1.
    """
    df = df.copy()
    df["distance"] = df["turn"] - 1   # turns since injection
    df["hit"] = (df["judgment"] == "CORRECT").astype(int)

    return (
        df.groupby(["strategy", "distance"])
        .agg(hits=("hit", "sum"), total=("hit", "count"))
        .assign(frr=lambda d: d["hits"] / d["total"])
        .reset_index()
    )