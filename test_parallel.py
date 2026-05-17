"""
Pre-flight check: verify all 5 Groq keys work and measure per-key latency.

Run this before starting the full benchmark to catch dead keys early.

Usage:
    python check_parallel.py
"""

from __future__ import annotations

import os
import threading
import time
from dotenv import load_dotenv

from llm.llm_groq import SimpleGroqClient
from llm.utils import filter_none_keys

load_dotenv()

TEST_PROMPT = (
    "Conversation history:\n"
    "user: The contract value is $4.2M and the deadline is March 10.\n"
    "assistant: Noted.\n\n"
    "Recall question: What is the contract value?"
)

EXPECTED = "$4.2m"

results: dict[int, dict] = {}
lock = threading.Lock()


def check_key(idx: int, key: str) -> None:
    try:
        llm = SimpleGroqClient([key])
        t0 = time.time()
        response = llm.generate(TEST_PROMPT, use_stop_tokens=True)
        elapsed = time.time() - t0
        correct = EXPECTED in response.lower()
        with lock:
            results[idx] = {
                "key_suffix": key[-6:],
                "response":   response,
                "correct":    correct,
                "latency_s":  round(elapsed, 2),
                "status":     "✅ OK" if correct else "⚠️  WRONG ANSWER",
            }
    except Exception as exc:
        with lock:
            results[idx] = {
                "key_suffix": key[-6:],
                "response":   str(exc),
                "correct":    False,
                "latency_s":  -1,
                "status":     "❌ ERROR",
            }


def main() -> None:
    keys = filter_none_keys([
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
        os.getenv("GROQ_API_KEY_4"),
        os.getenv("GROQ_API_KEY_5"),
    ])

    if not keys:
        print("❌ No keys found. Check your .env file.")
        return

    print(f"Checking {len(keys)} key(s) in parallel …\n")

    threads = [
        threading.Thread(target=check_key, args=(i, key))
        for i, key in enumerate(keys)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    print(f"{'#':<4} {'Key':<12} {'Status':<16} {'Latency':>8}  Response")
    print("─" * 70)
    all_ok = True
    for i in sorted(results):
        r = results[i]
        print(
            f"{i:<4} …{r['key_suffix']:<10} {r['status']:<16} "
            f"{r['latency_s']:>6.2f}s  {r['response']!r}"
        )
        if not r["correct"]:
            all_ok = False

    print()
    if all_ok:
        print("✅ All keys working. Safe to run the full benchmark.")
    else:
        print("⚠️  Some keys failed. Fix before running run_benchmark.py.")


if __name__ == "__main__":
    main()