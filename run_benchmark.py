"""
Context-Collapse Benchmark — main entry point.

Usage:
    python run_benchmark.py

Environment variables (in .env):
    GROQ_API_KEY_1, GROQ_API_KEY_2, GROQ_API_KEY_3
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm


from harness.llm_groq import SimpleGroqClient
from harness.utils import (
    filter_none_keys,
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
REPETITIONS = 5
SLEEP_BETWEEN_LLM_CALLS = 0.5  # seconds between main inference calls

# ---------------------------------------------------------------------------
# Prompts
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
# Each entry: (name, class, kwargs_for_init)
# - For LLM-dependent strategies, the harness will inject `llm` as the first
#   positional argument automatically (detected via NEEDS_LLM class attribute).
# - kwargs_for_init are passed as **kwargs after llm (or as sole kwargs for
#   non-LLM strategies).
# - Add entries here to sweep hyperparameters without changing any other code.
# ---------------------------------------------------------------------------

STRATEGIES: list[tuple[str, type, dict]] = [
    ("baseline",        BaselineMemory,       {}),
    ("rolling_summary", RollingSummaryMemory,  {"token_budget": 1500}),
    ("hierarchical",    HierarchicalMemory,    {"working_budget": 500, "episodic_budget": 1000}),
    ("rag",             RAGMemory,             {"k": 3, "buffer_size": 4, "alpha": 0.7}),
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory factory
# ---------------------------------------------------------------------------

def _make_memory(strategy_class: type, kwargs: dict, llm: SimpleGroqClient):
    """
    Instantiate a memory strategy.

    Detects whether the strategy needs an LLM by checking for a NEEDS_LLM
    class attribute (avoids hard-coding class names in a tuple).
    """
    needs_llm = getattr(strategy_class, "NEEDS_LLM", False)
    if needs_llm:
        return strategy_class(llm, **kwargs)
    return strategy_class(**kwargs)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_strategy(
    script: dict,
    strategy_name: str,
    strategy_class: type,
    strategy_kwargs: dict,
    llm: SimpleGroqClient,
    script_id: str,
    rep: int,
    memory=None,          # pre-created instance (for reuse with reset())
) -> tuple[list[dict], list[dict]]:
    """
    Execute a single (script × strategy × rep) run.

    Args:
        memory : If provided (e.g. for RAGMemory reuse), reset() is called on it
                 instead of creating a new instance. This avoids reloading the
                 sentence-transformer model 200 times.

    Returns:
        results  : Recall-turn result dicts (written to CSV).
        history  : Full conversation history (written to JSONL).
    """
    if memory is not None:
        memory.reset()
    else:
        memory = _make_memory(strategy_class, strategy_kwargs, llm)

    turns = script["turns"]
    results: list[dict] = []
    history: list[dict] = []

    for turn_obj in turns:
        turn_num: int = turn_obj["turn"]
        role: str = turn_obj["role"]
        content: str = turn_obj["content"]
        is_recall: bool = bool(turn_obj.get("is_recall", False))

        if role == "user":
            # Build context — strategies self-manage compression now.
            context = memory.get_context(query=content)

            user_prompt = (
                RECALL_USER_TEMPLATE.format(context=context, question=content)
                if context
                else NO_CONTEXT_USER_TEMPLATE.format(question=content)
            )

            logger.debug("Turn %d: LLM call (prompt_chars=%d).", turn_num, len(user_prompt))
            response = llm.generate(user_prompt, use_stop_tokens=is_recall)
            logger.debug("Turn %d: response=%r", turn_num, response)
            time.sleep(SLEEP_BETWEEN_LLM_CALLS)

            # Add to memory AFTER calling LLM so the recall question itself
            # is not in the context when we ask it.
            memory.add_message("user", content)
            memory.add_message("assistant", response)

            history.append({"turn": turn_num, "role": "user",      "content": content,  "is_recall": is_recall})
            history.append({"turn": turn_num, "role": "assistant",  "content": response, "is_recall": False})

            if is_recall:
                results.append({
                    "script_id":      script_id,
                    "strategy":       strategy_name,
                    "rep":            rep,
                    "turn":           turn_num,
                    "question":       content,
                    "target_fact_id": turn_obj.get("target_fact_id"),
                    "ground_truth":   turn_obj.get("ground_truth", []),
                    "agent_response": response,
                })

        else:
            # Scripted assistant turn — add to memory only, no LLM call.
            memory.add_message("assistant", content)
            history.append({"turn": turn_num, "role": "assistant", "content": content, "is_recall": False})

    return results, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging(RESULTS_DIR)
    set_seed(42)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.csv")

    # ------------------------------------------------------------------
    # Load & validate scripts
    # ------------------------------------------------------------------
    scripts = load_scripts(SCRIPTS_DIR)
    for script in scripts:
        for w in validate_script(script):
            logger.warning("Script validation: %s", w)

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------
    groq_keys = filter_none_keys([
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ])
    llm = SimpleGroqClient(groq_keys)

    # ------------------------------------------------------------------
    # Pre-create reusable memory instances.
    # RAGMemory loads a sentence-transformer model at init — expensive.
    # We create one instance per strategy and call reset() between reps
    # instead of constructing a new object each time.
    # ------------------------------------------------------------------
    reusable_memory: dict[str, Any] = {}
    for name, cls, kwargs in STRATEGIES:
        if not getattr(cls, "NEEDS_LLM", False):
            # Only pre-create non-LLM strategies (RAGMemory, BaselineMemory).
            # LLM strategies need the llm arg and are cheap to construct.
            reusable_memory[name] = _make_memory(cls, kwargs, llm)

    # ------------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------------
    fieldnames = [
        "script_id", "strategy", "repetition", "turn",
        "question", "target_fact_id", "agent_response", "ground_truth",
    ]

    total_runs = len(scripts) * len(STRATEGIES) * REPETITIONS
    pbar = tqdm(total=total_runs, desc="Overall progress")

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for script in scripts:
            script_id = script.get("script_id", "unknown")

            for strategy_name, strategy_class, strategy_kwargs in STRATEGIES:
                for rep in range(REPETITIONS):
                    logger.info(
                        "Running: script=%s strategy=%s rep=%d",
                        script_id, strategy_name, rep,
                    )

                    # Use pre-created instance if available, else None (will be created fresh).
                    memory_instance = reusable_memory.get(strategy_name)

                    try:
                        results, history = run_strategy(
                            script=script,
                            strategy_name=strategy_name,
                            strategy_class=strategy_class,
                            strategy_kwargs=strategy_kwargs,
                            llm=llm,
                            script_id=script_id,
                            rep=rep,
                            memory=memory_instance,
                        )
                    except Exception as exc:
                        logger.error(
                            "FAILED: script=%s strategy=%s rep=%d — %s",
                            script_id, strategy_name, rep, exc,
                            exc_info=True,
                        )
                        pbar.update(1)
                        continue

                    for row in results:
                        writer.writerow({
                            "script_id":      row["script_id"],
                            "strategy":       row["strategy"],
                            "repetition":     row["rep"],
                            "turn":           row["turn"],
                            "question":       row["question"],
                            "target_fact_id": row["target_fact_id"],
                            "agent_response": row["agent_response"],
                            "ground_truth":   json.dumps(row["ground_truth"]),
                        })
                    csv_file.flush()

                    hist_path = save_conversation_history(
                        history=history,
                        out_dir=HISTORY_DIR,
                        script_id=script_id,
                        strategy_name=strategy_name,
                        rep=rep,
                    )
                    logger.debug("History saved to %s", hist_path)
                    pbar.update(1)

    pbar.close()
    logger.info("Benchmark complete. Results → %s", output_csv)
    print(f"\nDone. Results → {output_csv}")


if __name__ == "__main__":
    main()