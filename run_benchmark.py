
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
REPETITIONS             = 5                     
SLEEP_BETWEEN_LLM_CALLS = 0.2                   # seconds between each LLM call (Azure throttle)
POST_RUN_SLEEP          = 0.0                   # extra pause between full strategy runs

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RECALL_USER_TEMPLATE = """\
Conversation history:
{context}

Recall question: {question}

INSTRUCTIONS:
- If the question asks "Is that right?" or "Should we...", answer with "Yes" or "No" first, then explain briefly.
- If the question asks for a specific fact (e.g., "Who is the lead?"), give the exact name/value.
- Use the conversation history to infer the answer when needed.
- Keep your answer under 20 words.
"""

NO_CONTEXT_USER_TEMPLATE = """\
Recall question: {question}"""

# ---------------------------------------------------------------------------
# Strategy registry (all four strategies enabled)
# ---------------------------------------------------------------------------

STRATEGIES: list[tuple[str, type, dict]] = [
    # ("baseline",        BaselineMemory,      {}),
    # ("rolling_summary", RollingSummaryMemory, {"token_budget": 400}),
    ("hierarchical",    HierarchicalMemory,   {"working_budget": 300, "episodic_budget": 600}),
    # ("rag",             RAGMemory,            {"k": 8, "buffer_size": 6, "alpha": 0.7}),
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
    if memory is not None:
        memory.reset()
    else:
        memory = _make_memory(strategy_class, strategy_kwargs, llm)

    results: list[dict] = []
    history: list[dict] = []

    llm_call_counter = llm.get_call_count()   

    for turn_obj in script["turns"]:
        turn_num: int   = turn_obj["turn"]
        role: str       = turn_obj["role"]
        content: str    = turn_obj["content"]
        is_recall: bool = bool(turn_obj.get("is_recall", False))

        if role == "user":
            # Build context from memory
            context = memory.get_context(query=content)
            context_chars = len(context)

            if context:
                user_prompt = RECALL_USER_TEMPLATE.format(
                    context=context,
                    question=content,
                )
            else:
                user_prompt = NO_CONTEXT_USER_TEMPLATE.format(question=content)

            # Call LLM
            response = llm.generate(
                user_prompt,
                use_stop_tokens=is_recall,
            )
            time.sleep(SLEEP_BETWEEN_LLM_CALLS)

            # Update memory (may trigger self-compression)
            memory.add_message("user", content)
            memory.add_message("assistant", response)
            calls_so_far = llm.get_call_count() - llm_call_counter

            # Record history
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

            # Record recall result
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

    # Load Azure credentials
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

    # Load and validate scripts
    scripts = load_scripts(SCRIPTS_DIR)
    for script in scripts:
        for w in validate_script(script):
            logger.warning("Script validation: %s", w)

    total_runs = len(scripts) * len(STRATEGIES) * REPETITIONS
    logger.info(
        "Total runs: %d scripts × %d strategies × %d reps = %d",
        len(scripts), len(STRATEGIES), REPETITIONS, total_runs,
    )

    # Load sentence-transformer for RAG (once, shared across all RAG instances)
    from sentence_transformers import SentenceTransformer
    logger.info("Loading sentence-transformer model …")
    shared_encoder = SentenceTransformer("models/all-MiniLM-L6-v2")
    logger.info("Model ready.")

    # CSV output
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
                for rep in range(REPETITIONS):
                    run_counter += 1
                    logger.info(
                        "[%d/%d] script=%s strategy=%s rep=%d",
                        run_counter, total_runs, script_id, strategy_name, rep,
                    )

                    if getattr(strategy_class, "NEEDS_LLM", False):
                        memory = strategy_class(llm, **strategy_kwargs)
                    else:
                        if strategy_class is RAGMemory:
                            rag_dir = os.path.join(
                                RESULTS_DIR, "chroma",
                                f"{script_id}_{strategy_name}_rep{rep}"
                            )
                            memory = strategy_class(
                                encoder=shared_encoder,
                                persist_dir=rag_dir,
                                **strategy_kwargs,
                            )
                        else:
                            # BaselineMemory, RollingSummaryMemory (non-LLM, no encoder)
                            memory = strategy_class(**strategy_kwargs)

                    try:
                        results, history = run_strategy(
                            script=script,
                            strategy_name=strategy_name,
                            strategy_class=strategy_class,
                            strategy_kwargs=strategy_kwargs,
                            llm=llm,
                            script_id=script_id,
                            rep=rep,
                            memory=memory,
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

                    # Optional brief pause between runs (set to 0 by default)
                    time.sleep(POST_RUN_SLEEP)

    pbar.close()
    logger.info("Complete. Results saved to %s", output_csv)
    print(f"\nDone. Results → {output_csv}")


if __name__ == "__main__":
    main()