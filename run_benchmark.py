"""
Context-Collapse Benchmark — Parallel entry point.

Architecture
------------
- 5 parallel workers, one per Groq API key.
- Scripts are distributed via a thread-safe queue.
- Each worker owns its key exclusively — no locking, no sharing.
- Results are written to CSV via a dedicated writer thread (also queue-based).
- Histories are written directly per-worker (filenames are unique per run).
- Progress bar is updated thread-safely via a lock.

Usage:
    python run_benchmark.py

Environment variables (in .env):
    GROQ_API_KEY_1 ... GROQ_API_KEY_5
"""

from __future__ import annotations

import csv
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime

from dotenv import load_dotenv
from tqdm import tqdm

from llm.llm_groq import SimpleGroqClient
from llm.utils import (
    filter_none_keys,
    load_scripts,
    save_conversation_history,
    set_seed,
    setup_logging,
    validate_script,
)
from sentence_transformers import SentenceTransformer
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
REPETITIONS             = 2
NUM_WORKERS             = 2      # one per Groq key
SLEEP_BETWEEN_LLM_CALLS = 0.1   # reduced from 0.5 — each worker has its own key

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
# Strategy registry  (name, class, kwargs)
# ---------------------------------------------------------------------------

STRATEGIES: list[tuple[str, type, dict]] = [
    ("baseline",        BaselineMemory,      {}),
    ("rolling_summary", RollingSummaryMemory, {"token_budget": 1500}),
    ("hierarchical",    HierarchicalMemory,   {"working_budget": 500, "episodic_budget": 1000}),
    ("rag",             RAGMemory,            {"k": 3, "buffer_size": 4, "alpha": 0.7}),
]

logger = logging.getLogger(__name__)

# Sentinel that tells the CSV writer thread to exit.
_WRITER_DONE = object()


# ---------------------------------------------------------------------------
# Memory factory
# ---------------------------------------------------------------------------

def _make_memory(strategy_class: type, kwargs: dict, llm: SimpleGroqClient):
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
    llm: SimpleGroqClient,
    script_id: str,
    rep: int,
    memory=None,
) -> tuple[list[dict], list[dict]]:
    """
    Execute one benchmark run.

    Args:
        memory: Pre-created instance. reset() is called on it to avoid
                reloading expensive resources (e.g. RAG sentence-transformer).
                Pass None to construct a fresh instance.

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

    # -------------------------------------------------------------------
    # Instrumentation counters
    # llm_call_counter : counts every generate() call in this run,
    #                    including compression calls fired inside add_message().
    #                    We snapshot it before and after add_message() to catch
    #                    compression calls that happen inside the memory strategy.
    # -------------------------------------------------------------------
    llm_call_counter = llm.get_call_count()   # snapshot at run start

    for turn_obj in script["turns"]:
        turn_num: int   = turn_obj["turn"]
        role: str       = turn_obj["role"]
        content: str    = turn_obj["content"]
        is_recall: bool = bool(turn_obj.get("is_recall", False))

        if role == "user":
            context = memory.get_context(query=content)
            context_chars = len(context)            # ← measure BEFORE building prompt

            user_prompt = (
                RECALL_USER_TEMPLATE.format(context=context, question=content)
                if context
                else NO_CONTEXT_USER_TEMPLATE.format(question=content)
            )

            response = llm.generate(user_prompt, use_stop_tokens=is_recall)
            time.sleep(SLEEP_BETWEEN_LLM_CALLS)

            # Snapshot call count AFTER add_message so compression calls
            # (which fire inside add_message for self-triggering strategies)
            # are included in the count.
            memory.add_message("user", content)
            memory.add_message("assistant", response)
            calls_so_far = llm.get_call_count() - llm_call_counter

            # User turn entry — log context_chars at recall turns only
            user_entry: dict = {
                "turn":     turn_num,
                "role":     "user",
                "content":  content,
                "is_recall": is_recall,
            }
            if is_recall:
                user_entry["context_chars"] = context_chars   # ← for efficiency table

            history.append(user_entry)
            history.append({
                "turn":             turn_num,
                "role":             "assistant",
                "content":          response,
                "is_recall":        False,
                "llm_calls_so_far": calls_so_far,             # ← for efficiency table
            })

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
            # Scripted assistant turn — add to memory only.
            memory.add_message("assistant", content)
            history.append({"turn": turn_num, "role": "assistant", "content": content, "is_recall": False})

    return results, history


# ---------------------------------------------------------------------------
# Worker function (one thread per key)
# ---------------------------------------------------------------------------

def worker(
    worker_id: int,
    api_key: str,
    script_queue: queue.Queue,
    result_queue: queue.Queue,
    pbar_lock: threading.Lock,
    pbar: tqdm,
    shared_encoder=None,
) -> None:
    """
    Worker thread: pulls scripts from script_queue, runs all strategies × reps,
    pushes result rows to result_queue for the CSV writer.

    Each worker owns its API key exclusively — no cross-thread sharing.
    shared_encoder: pre-loaded SentenceTransformer shared from main().
                    encode() is thread-safe (read-only inference).
    """
    thread_logger = logging.getLogger(f"worker-{worker_id}")
    thread_logger.info("Started with key …%s", api_key[-6:])

    # Private LLM client — this worker is the only one using this key.
    llm = SimpleGroqClient([api_key])

    # Pre-create reusable non-LLM memory instances per worker.
    # This avoids reloading the sentence-transformer model on every rep.
    reusable: dict[str, object] = {}
    for name, cls, kwargs in STRATEGIES:
        if not getattr(cls, "NEEDS_LLM", False):
            if cls is RAGMemory and shared_encoder is not None:
                reusable[name] = cls(encoder=shared_encoder, **kwargs)
            else:
                reusable[name] = _make_memory(cls, kwargs, llm)

    while True:
        try:
            script = script_queue.get(timeout=3)
        except queue.Empty:
            break  # no more scripts — exit cleanly

        script_id = script.get("script_id", "unknown")

        for strategy_name, strategy_class, strategy_kwargs in STRATEGIES:
            for rep in range(REPETITIONS):
                thread_logger.info(
                    "script=%s strategy=%s rep=%d", script_id, strategy_name, rep
                )

                memory_instance = reusable.get(strategy_name)  # None for LLM strategies

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
                    thread_logger.error(
                        "FAILED script=%s strategy=%s rep=%d: %s",
                        script_id, strategy_name, rep, exc,
                        exc_info=True,
                    )
                    with pbar_lock:
                        pbar.update(1)
                    continue

                # Push recall rows to the writer thread.
                result_queue.put(results)

                # History files are safe to write directly — filenames are
                # unique per (script_id, strategy, rep) so there's no conflict.
                save_conversation_history(
                    history=history,
                    out_dir=HISTORY_DIR,
                    script_id=script_id,
                    strategy_name=strategy_name,
                    rep=rep,
                )

                with pbar_lock:
                    pbar.update(1)

        script_queue.task_done()

    thread_logger.info("Worker %d done.", worker_id)


# ---------------------------------------------------------------------------
# CSV writer thread
# ---------------------------------------------------------------------------

def csv_writer_thread(
    result_queue: queue.Queue,
    output_csv: str,
    fieldnames: list[str],
) -> None:
    """
    Dedicated thread that drains result_queue and writes rows to CSV.

    A single writer thread means no file locking is needed — only one
    thread ever touches the file. Exits when it receives _WRITER_DONE.
    """
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        while True:
            item = result_queue.get()

            if item is _WRITER_DONE:
                break

            # item is list[dict] of recall rows from one (script × strategy × rep)
            for row in item:
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
            f.flush()  # flush after each run so partial results survive a crash


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
    # Load & validate scripts
    # ------------------------------------------------------------------
    scripts = load_scripts(SCRIPTS_DIR)
    for script in scripts:
        for w in validate_script(script):
            logger.warning("Script validation: %s", w)

    # ------------------------------------------------------------------
    # API keys — one per worker, strictly separated
    # ------------------------------------------------------------------
    all_keys = filter_none_keys([
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY_4"),
        os.getenv("GROQ_API_KEY_5"),
    ])

    if not all_keys:
        raise RuntimeError("No Groq API keys found. Set GROQ_API_KEY_1 … _5 in .env")

    num_workers = min(NUM_WORKERS, len(all_keys), len(scripts))
    keys = all_keys[:num_workers]

    logger.info(
        "Benchmark: %d scripts × %d strategies × %d reps = %d total runs",
        len(scripts), len(STRATEGIES), REPETITIONS,
        len(scripts) * len(STRATEGIES) * REPETITIONS,
    )
    logger.info("Parallel workers: %d", num_workers)

    # ------------------------------------------------------------------
    # Script queue — workers pull from this
    # ------------------------------------------------------------------
    script_q: queue.Queue = queue.Queue()
    for script in scripts:
        script_q.put(script)

    # ------------------------------------------------------------------
    # Result queue — workers push to this, writer drains it
    # ------------------------------------------------------------------
    result_q: queue.Queue = queue.Queue()

    # ------------------------------------------------------------------
    # Progress bar (shared, protected by a lock)
    # ------------------------------------------------------------------
    total_runs = len(scripts) * len(STRATEGIES) * REPETITIONS
    pbar       = tqdm(total=total_runs, desc="Benchmark", ncols=90)
    pbar_lock  = threading.Lock()

    # ------------------------------------------------------------------
    # Start CSV writer thread first so it's ready before results arrive
    # ------------------------------------------------------------------
    fieldnames = [
        "script_id", "strategy", "repetition", "turn",
        "question", "target_fact_id", "agent_response", "ground_truth",
    ]
    writer = threading.Thread(
        target=csv_writer_thread,
        args=(result_q, output_csv, fieldnames),
        name="csv-writer",
        daemon=False,   # non-daemon: must finish flushing before process exits
    )
    writer.start()

    # ------------------------------------------------------------------
    # Load sentence-transformer model ONCE before spawning workers.
    # Workers share this read-only encoder — encode() is thread-safe.
    # ------------------------------------------------------------------
    logger.info("Loading sentence-transformer model (one-time) …")
    shared_encoder = SentenceTransformer("models/all-MiniLM-L6-v2")
    logger.info("Model loaded and ready.")

    # ------------------------------------------------------------------
    # Start worker threads
    # ------------------------------------------------------------------
    workers: list[threading.Thread] = []
    for i, key in enumerate(keys):
        t = threading.Thread(
            target=worker,
            args=(i, key, script_q, result_q, pbar_lock, pbar, shared_encoder),
            name=f"worker-{i}",
            daemon=False,
        )
        t.start()
        workers.append(t)
        logger.info("Started worker-%d with key …%s", i, key[-6:])

    # ------------------------------------------------------------------
    # Wait for all workers to finish, then shut down the writer
    # ------------------------------------------------------------------
    for t in workers:
        t.join()

    # Signal writer to stop and wait for final flush
    result_q.put(_WRITER_DONE)
    writer.join()

    pbar.close()
    logger.info("Complete. Results → %s", output_csv)
    print(f"\nDone. Results → {output_csv}")


if __name__ == "__main__":
    main()