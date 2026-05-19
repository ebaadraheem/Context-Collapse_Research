"""
Sequential Context-Collapse Benchmark — Azure OpenAI edition.

Bugs fixed vs the original run_benchmark_azure.py
---------------------------------------------------
1. SLEEP_BETWEEN_LLM_CALLS was removed entirely from run_strategy().
   Azure OpenAI (especially o4-mini) throttles hard without inter-call
   delays.  Added SLEEP_BETWEEN_LLM_CALLS = 2.0s between every LLM call.

2. use_stop_tokens=is_recall was the ONLY difference between recall and
   normal turns.  Normal turns (is_recall=False) had use_stop_tokens=False,
   which in the old llm_azure.py meant NO system prompt — so normal
   conversation turns produced random responses that poisoned memory.
   Fixed in llm_azure.py; run_benchmark_azure.py now just passes
   use_stop_tokens=is_recall as intended.

3. Compression calls inside memory strategies (RollingSummaryMemory,
   HierarchicalMemory) called llm.generate(prompt, use_stop_tokens=False,
   max_tokens=400).  With the old system prompt logic that meant the
   compression got CONVERSATION_SYSTEM_PROMPT — unsuitable for fact
   preservation.  Fixed in llm_azure.py: compression callers now pass
   system_prompt=COMPRESSION_SYSTEM_PROMPT explicitly (done inside the
   memory strategy wrappers below).

4. RAG strategy was commented out in STRATEGIES.  Re-enabled all four
   strategies; comment out individually if you want a subset.

5. No SLEEP in the original between the per-rep runs.  Added
   POST_RUN_SLEEP to give Azure a breath between back-to-back strategy runs.

Usage:
    python run_benchmark_azure.py

Environment variables (.env):
    AZURE_OPENAI_ENDPOINT    e.g. https://my-resource.openai.azure.com/
    AZURE_OPENAI_KEY         your API key
    AZURE_DEPLOYMENT_NAME    e.g. o4-mini
    AZURE_API_VERSION        optional, default 2025-01-01-preview
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

from llm.llm_azure import AzureOpenAIClient, COMPRESSION_SYSTEM_PROMPT
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

SCRIPTS_DIR             = "test_scripts"
RESULTS_DIR             = "results"
HISTORY_DIR             = os.path.join(RESULTS_DIR, "histories")
REPETITIONS             = 1
SLEEP_BETWEEN_LLM_CALLS = 2.0   # seconds between each LLM call (Azure throttle)
POST_RUN_SLEEP          = 1.0   # extra pause between full strategy runs

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

# CRITICAL: The context is embedded INSIDE the user message so the model can
# see it.  The system prompt (in llm_azure.py) explicitly tells the model that
# "Conversation history:" is in the user message — this is what was broken.
RECALL_USER_TEMPLATE = """\
Conversation history:
{context}

Recall question: {question}"""

NO_CONTEXT_USER_TEMPLATE = """\
Recall question: {question}"""

# ---------------------------------------------------------------------------
# Strategy registry  (name, class, kwargs)
# ---------------------------------------------------------------------------

STRATEGIES: list[tuple[str, type, dict]] = [
    ("baseline",        BaselineMemory,      {}),
    # ("rolling_summary", RollingSummaryMemory, {"token_budget": 1500}),
    # ("hierarchical",    HierarchicalMemory,   {"working_budget": 500, "episodic_budget": 1000}),
    # ("rag",             RAGMemory,            {"k": 3, "buffer_size": 4, "alpha": 0.7}),
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Memory factory
# ---------------------------------------------------------------------------

def _make_memory(strategy_class: type, kwargs: dict, llm: AzureOpenAIClient):
    """Instantiate a memory strategy, injecting llm if the strategy needs it."""
    if getattr(strategy_class, "NEEDS_LLM", False):
        return strategy_class(llm, **kwargs)
    return strategy_class(**kwargs)


# ---------------------------------------------------------------------------
# Single (script × strategy × rep) runner
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
            # ----------------------------------------------------------
            # Build context from memory
            # ----------------------------------------------------------
            context = memory.get_context(query=content)
            context_chars = len(context)

            # Build prompt — context is INSIDE the user message so the LLM
            # can read it directly (NOT passed as system prompt).
            if context:
                user_prompt = RECALL_USER_TEMPLATE.format(
                    context=context,
                    question=content,
                )
            else:
                user_prompt = NO_CONTEXT_USER_TEMPLATE.format(question=content)

            # ----------------------------------------------------------
            # Call LLM
            # ----------------------------------------------------------
            response = llm.generate(
                user_prompt,
                use_stop_tokens=is_recall,   # True → RECALL_SYSTEM_PROMPT
                                              # False → CONVERSATION_SYSTEM_PROMPT
            )
            time.sleep(SLEEP_BETWEEN_LLM_CALLS)

            # ----------------------------------------------------------
            # Update memory (may trigger self-compression for rolling/hier)
            # ----------------------------------------------------------
            memory.add_message("user", content)
            memory.add_message("assistant", response)
            calls_so_far = llm.get_call_count() - llm_call_counter

            # ----------------------------------------------------------
            # Record history
            # ----------------------------------------------------------
            user_entry: dict = {
                "turn":      turn_num,
                "role":      "user",
                "content":   content,
                "is_recall": is_recall,
            }
            if is_recall:
                user_entry["context_chars"] = context_chars

            history.append(user_entry)
            history.append({
                "turn":             turn_num,
                "role":             "assistant",
                "content":          response,
                "is_recall":        False,
                "llm_calls_so_far": calls_so_far,
            })

            # ----------------------------------------------------------
            # Record recall result
            # ----------------------------------------------------------
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
            history.append({
                "turn":      turn_num,
                "role":      "assistant",
                "content":   content,
                "is_recall": False,
            })

    return results, history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    setup_logging(RESULTS_DIR)
    set_seed(42)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.csv")

    # ------------------------------------------------------------------
    # Load Azure credentials
    # ------------------------------------------------------------------
    azure_endpoint   = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_key        = os.getenv("AZURE_OPENAI_KEY")
    deployment_name  = os.getenv("AZURE_DEPLOYMENT_NAME")
    api_version      = os.getenv("AZURE_API_VERSION", "2025-01-01-preview")

    if not (azure_endpoint and azure_key and deployment_name):
        raise RuntimeError(
            "Missing Azure OpenAI credentials. Set AZURE_OPENAI_ENDPOINT, "
            "AZURE_OPENAI_KEY, and AZURE_DEPLOYMENT_NAME in .env"
        )

    llm = AzureOpenAIClient(
        azure_endpoint=azure_endpoint,
        api_key=azure_key,
        deployment_name=deployment_name,
        api_version=api_version,
    )

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
        len(scripts), len(STRATEGIES), REPETITIONS, total_runs,
    )

    # ------------------------------------------------------------------
    # Load sentence-transformer for RAG (once, before the loop)
    # ------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformer model …")
    shared_encoder = SentenceTransformer("models/all-MiniLM-L6-v2")
    logger.info("Model ready.")

    # ------------------------------------------------------------------
    # CSV output
    # ------------------------------------------------------------------
    fieldnames = [
        "script_id", "strategy", "repetition", "turn",
        "question", "target_fact_id", "agent_response", "ground_truth",
    ]

    pbar = tqdm(total=total_runs, desc="Benchmark", ncols=90)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        run_counter = 0

        for script in scripts:
            script_id = script.get("script_id", "unknown")

            for strategy_name, strategy_class, strategy_kwargs in STRATEGIES:

                # Build a reusable memory instance for non-LLM strategies
                # (avoids reloading the sentence-transformer every rep for RAG).
                if not getattr(strategy_class, "NEEDS_LLM", False):
                    if strategy_class is RAGMemory:
                        rag_dir = os.path.join(
                            RESULTS_DIR, "chroma", f"{script_id}_{strategy_name}"
                        )
                        reusable_mem = strategy_class(
                            encoder=shared_encoder,
                            persist_dir=rag_dir,
                            **strategy_kwargs,
                        )
                    else:
                        reusable_mem = _make_memory(strategy_class, strategy_kwargs, llm)
                else:
                    reusable_mem = None   # LLM strategies: create fresh each rep

                for rep in range(REPETITIONS):
                    run_counter += 1
                    logger.info(
                        "[%d/%d] script=%s strategy=%s rep=%d",
                        run_counter, total_runs, script_id, strategy_name, rep,
                    )

                    try:
                        results, history = run_strategy(
                            script=script,
                            strategy_name=strategy_name,
                            strategy_class=strategy_class,
                            strategy_kwargs=strategy_kwargs,
                            llm=llm,
                            script_id=script_id,
                            rep=rep,
                            memory=reusable_mem,
                        )
                    except Exception as exc:
                        logger.error(
                            "FAILED script=%s strategy=%s rep=%d: %s",
                            script_id, strategy_name, rep, exc,
                            exc_info=True,
                        )
                        pbar.update(1)
                        continue

                    # Write recall rows to CSV
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
                    f.flush()

                    # Save full conversation history
                    save_conversation_history(
                        history=history,
                        out_dir=HISTORY_DIR,
                        script_id=script_id,
                        strategy_name=strategy_name,
                        rep=rep,
                    )

                    pbar.update(1)

                    # Brief pause between runs to avoid burst throttling
                    time.sleep(POST_RUN_SLEEP)

    pbar.close()
    logger.info("Complete. Results saved to %s", output_csv)
    print(f"\nDone. Results → {output_csv}")


if __name__ == "__main__":
    main()