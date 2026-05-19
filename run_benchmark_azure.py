"""
Sequential Context‑Collapse Benchmark using Azure OpenAI.

No parallelism – runs one script × strategy × repetition at a time.
Saves results CSV and full conversation histories.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from llm.llm_azure import AzureOpenAIClient
from llm.utils import (
    load_scripts,
    save_conversation_history,
    set_seed,
    setup_logging,
    validate_script,
)
from memory_strategies import (
    BaselineMemory,
    HierarchicalMemory,
    RAGMemory,
    RollingSummaryMemory,
)

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SCRIPTS_DIR = "test_scripts"
RESULTS_DIR = "results"
HISTORY_DIR = os.path.join(RESULTS_DIR, "histories")
REPETITIONS = 1

# ---------------------------------------------------------------------------
# Prompts (same as original)
# ---------------------------------------------------------------------------

RECALL_USER_TEMPLATE = """\
Conversation history:
{context}

Recall question: {question}"""

NO_CONTEXT_USER_TEMPLATE = """\
Recall question: {question}"""

# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

STRATEGIES: list[tuple[str, type, dict]] = [
    ("baseline", BaselineMemory, {}),
    # ("rolling_summary", RollingSummaryMemory, {"token_budget": 1500}),
    # ("hierarchical", HierarchicalMemory, {"working_budget": 500, "episodic_budget": 1000}),
    # ("rag", RAGMemory, {"k": 3, "buffer_size": 4, "alpha": 0.7}),
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory factory (reused from original)
# ---------------------------------------------------------------------------

def _make_memory(strategy_class: type, kwargs: dict, llm: AzureOpenAIClient):
    """Instantiate a memory strategy, injecting llm if needed."""
    if getattr(strategy_class, "NEEDS_LLM", False):
        return strategy_class(llm, **kwargs)
    return strategy_class(**kwargs)


# ---------------------------------------------------------------------------
# Single run (identical to original, but uses Azure client)
# ---------------------------------------------------------------------------

def run_strategy(
    script: dict,
    strategy_name: str,
    strategy_class: type,
    strategy_kwargs: dict,
    llm: AzureOpenAIClient,
    script_id: str,
    rep: int,
    memory=None,
) -> tuple[list[dict], list[dict]]:
    """
    Execute one benchmark run.

    Args:
        memory: Pre-created instance (or None to create fresh).

    Returns:
        results : Recall-turn rows (for CSV).
        history : Full conversation (for JSONL).
    """
    if memory is not None:
        memory.reset()
    else:
        memory = _make_memory(strategy_class, strategy_kwargs, llm)

    results: list[dict] = []
    history: list[dict] = []

    llm_call_counter = llm.get_call_count()   # snapshot at run start

    for turn_obj in script["turns"]:
        turn_num: int   = turn_obj["turn"]
        role: str       = turn_obj["role"]
        content: str    = turn_obj["content"]
        is_recall: bool = bool(turn_obj.get("is_recall", False))

        if role == "user":
            context = memory.get_context(query=content)
            context_chars = len(context)

            user_prompt = (
                RECALL_USER_TEMPLATE.format(context=context, question=content)
                if context
                else NO_CONTEXT_USER_TEMPLATE.format(question=content)
            )

            response = llm.generate(user_prompt, use_stop_tokens=is_recall)

            memory.add_message("user", content)
            memory.add_message("assistant", response)
            calls_so_far = llm.get_call_count() - llm_call_counter

            # User turn entry
            user_entry: dict = {
                "turn": turn_num,
                "role": "user",
                "content": content,
                "is_recall": is_recall,
            }
            if is_recall:
                user_entry["context_chars"] = context_chars

            history.append(user_entry)
            history.append({
                "turn": turn_num,
                "role": "assistant",
                "content": response,
                "is_recall": False,
                "llm_calls_so_far": calls_so_far,
            })

            if is_recall:
                results.append({
                    "script_id": script_id,
                    "strategy": strategy_name,
                    "rep": rep,
                    "turn": turn_num,
                    "question": content,
                    "target_fact_id": turn_obj.get("target_fact_id"),
                    "ground_truth": turn_obj.get("ground_truth", []),
                    "agent_response": response,
                })

        else:   # scripted assistant turn
            memory.add_message("assistant", content)
            history.append({
                "turn": turn_num,
                "role": "assistant",
                "content": content,
                "is_recall": False,
            })

    return results, history


# ---------------------------------------------------------------------------
# Main sequential loop
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging(RESULTS_DIR)
    set_seed(42)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.csv")

    # ------------------------------------------------------------------
    # Load Azure credentials
    # ------------------------------------------------------------------
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key = os.getenv("AZURE_OPENAI_KEY")
    deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

    if not (azure_endpoint and azure_key and deployment_name):
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_KEY, and AZURE_DEPLOYMENT_NAME in .env"
        )

    llm = AzureOpenAIClient(azure_endpoint, azure_key, deployment_name)

    # ------------------------------------------------------------------
    # Load and validate scripts
    # ------------------------------------------------------------------
    scripts = load_scripts(SCRIPTS_DIR)
    for script in scripts:
        for w in validate_script(script):
            logger.warning("Script validation: %s", w)

    total_runs = len(scripts) * len(STRATEGIES) * REPETITIONS
    logger.info(
        "Total runs: %d scripts × %d strategies × %d reps = %d",
        len(scripts), len(STRATEGIES), REPETITIONS, total_runs
    )

    # ------------------------------------------------------------------
    # Open CSV file and write header
    # ------------------------------------------------------------------
    fieldnames = [
        "script_id", "strategy", "repetition", "turn",
        "question", "target_fact_id", "agent_response", "ground_truth",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        run_counter = 0
        for script in scripts:
            script_id = script.get("script_id", "unknown")

            for strategy_name, strategy_class, strategy_kwargs in STRATEGIES:
                # For non‑LLM strategies, we can reuse one instance per strategy
                # across repetitions (reset between reps). For RAG, we need a fresh
                # Chroma directory per repetition to avoid cross‑rep contamination.
                # We'll simply create a new memory each rep for simplicity.
                # (Reusing would require per‑rep reset, which we already do inside run_strategy.)
                for rep in range(REPETITIONS):
                    run_counter += 1
                    logger.info(
                        "[%d/%d] script=%s strategy=%s rep=%d",
                        run_counter, total_runs, script_id, strategy_name, rep
                    )

                    results, history = run_strategy(
                        script=script,
                        strategy_name=strategy_name,
                        strategy_class=strategy_class,
                        strategy_kwargs=strategy_kwargs,
                        llm=llm,
                        script_id=script_id,
                        rep=rep,
                        memory=None,   # create fresh each time (simpler)
                    )

                    # Write recall rows to CSV
                    for row in results:
                        writer.writerow({
                            "script_id": row["script_id"],
                            "strategy": row["strategy"],
                            "repetition": row["rep"],
                            "turn": row["turn"],
                            "question": row["question"],
                            "target_fact_id": row["target_fact_id"],
                            "agent_response": row["agent_response"],
                            "ground_truth": json.dumps(row["ground_truth"]),
                        })
                    f.flush()

                    # Save full conversation history
                    save_conversation_history(
                        history=history,
                        out_dir=HISTORY_DIR,
                        script_id=script_id,
                        strategy_name=strategy_name,
                        rep=rep,
                    )

    logger.info("Complete. Results saved to %s", output_csv)
    print(f"\nDone. Results → {output_csv}")


if __name__ == "__main__":
    main()