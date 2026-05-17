from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime

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
SLEEP_BETWEEN_LLM_CALLS = 0.5   # seconds between every LLM call

# How often (every N *user* turns) to trigger memory compression.
# Recall turns are at positions 5,10,15,20,25.
# Compression fires AFTER a non-recall user turn at those positions to avoid
# compressing fact-injection turns immediately before recall.
COMPRESS_EVERY_N = 5

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

# (name, class, needs_llm)
STRATEGIES: list[tuple[str, type, bool]] = [
    ("baseline", BaselineMemory, False),
    ("rolling_summary", RollingSummaryMemory, True),
    ("hierarchical", HierarchicalMemory, True),
    ("rag", RAGMemory, False),
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_strategy(
    script: dict,
    strategy_name: str,
    strategy_class: type,
    llm: SimpleGroqClient,
    script_id: str,
    rep: int,
) -> tuple[list[dict], list[dict]]:
    """
    Execute a single (script × strategy × rep) run.

    Returns:
        results  : List of recall-turn result dicts (written to CSV).
        history  : Full conversation history (written to JSONL).
    """
    needs_llm = strategy_class in (RollingSummaryMemory, HierarchicalMemory)
    memory = strategy_class(llm) if needs_llm else strategy_class()

    turns = script["turns"]
    results: list[dict] = []
    history: list[dict] = []
    user_turn_count = 0   # counts only user turns (drives compression cadence)

    for turn_obj in turns:
        turn_num: int = turn_obj["turn"]
        role: str = turn_obj["role"]
        content: str = turn_obj["content"]
        is_recall: bool = bool(turn_obj.get("is_recall", False))

        if role == "user":
            user_turn_count += 1

            if (
                user_turn_count % COMPRESS_EVERY_N == 0
                and not is_recall
            ):
                logger.debug("Turn %d: compressing memory.", turn_num)
                memory.compress()
                time.sleep(SLEEP_BETWEEN_LLM_CALLS)

            # Build context
            context = memory.get_context(query=content)

            # Build user prompt
            if context:
                user_prompt = RECALL_USER_TEMPLATE.format(
                    context=context, question=content
                )
            else:
                user_prompt = NO_CONTEXT_USER_TEMPLATE.format(question=content)

            # LLM call
            logger.debug(
                "Turn %d: calling LLM (prompt_len=%d).", turn_num, len(user_prompt)
            )
            response = llm.generate(user_prompt, use_stop_tokens=is_recall)
            logger.debug("Turn %d: response=%r", turn_num, response)
            time.sleep(SLEEP_BETWEEN_LLM_CALLS)

            # Store exchange in memory
            memory.add_message("user", content)
            memory.add_message("assistant", response)

            # Log to history
            history.append(
                {
                    "turn": turn_num,
                    "role": "user",
                    "content": content,
                    "is_recall": is_recall,
                }
            )
            history.append(
                {
                    "turn": turn_num,
                    "role": "assistant",
                    "content": response,
                    "is_recall": False,
                }
            )

            # Record recall result
            if is_recall:
                results.append(
                    {
                        "script_id": script_id,
                        "strategy": strategy_name,
                        "rep": rep,
                        "turn": turn_num,
                        "question": content,
                        "target_fact_id": turn_obj.get("target_fact_id"),
                        "ground_truth": turn_obj.get("ground_truth", []),
                        "agent_response": response,
                    }
                )

        else:
            # Scripted assistant turn (not generated)
            memory.add_message("assistant", content)
            history.append(
                {"turn": turn_num, "role": "assistant", "content": content, "is_recall": False}
            )

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
        warnings = validate_script(script)
        for w in warnings:
            logger.warning("Script validation: %s", w)

    # ------------------------------------------------------------------
    # LLM client
    # ------------------------------------------------------------------
    groq_keys = filter_none_keys(
        [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ]
    )
    llm = SimpleGroqClient(groq_keys)

    # ------------------------------------------------------------------
    # CSV setup
    # ------------------------------------------------------------------
    fieldnames = [
        "script_id",
        "strategy",
        "repetition",
        "turn",
        "question",
        "target_fact_id",
        "agent_response",
        "ground_truth",
    ]

    total_runs = len(scripts) * len(STRATEGIES) * REPETITIONS
    pbar = tqdm(total=total_runs, desc="Overall progress")

    with open(output_csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for script in scripts:
            script_id = script.get("script_id", "unknown")

            for strategy_name, strategy_class, _ in STRATEGIES:
                for rep in range(REPETITIONS):
                    logger.info(
                        "Running: script=%s strategy=%s rep=%d",
                        script_id,
                        strategy_name,
                        rep,
                    )

                    try:
                        results, history = run_strategy(
                            script=script,
                            strategy_name=strategy_name,
                            strategy_class=strategy_class,
                            llm=llm,
                            script_id=script_id,
                            rep=rep,
                        )
                    except Exception as exc:
                        logger.error(
                            "FAILED: script=%s strategy=%s rep=%d — %s",
                            script_id,
                            strategy_name,
                            rep,
                            exc,
                            exc_info=True,
                        )
                        pbar.update(1)
                        continue

                    # Write recall results to CSV
                    for row in results:
                        writer.writerow(
                            {
                                "script_id": row["script_id"],
                                "strategy": row["strategy"],
                                "repetition": row["rep"],
                                "turn": row["turn"],
                                "question": row["question"],
                                "target_fact_id": row["target_fact_id"],
                                "agent_response": row["agent_response"],
                                "ground_truth": json.dumps(row["ground_truth"]),
                            }
                        )
                    csv_file.flush()

                    # Save full conversation history
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
    logger.info("Benchmark complete. Results saved to %s", output_csv)
    print(f"\nDone. Results → {output_csv}")


if __name__ == "__main__":
    main()