import os
import json
import csv
import time
from datetime import datetime
from tqdm import tqdm

# Import memory strategies
from memory_strategies import (
    BaselineMemory,
    RollingSummaryMemory,
    HierarchicalMemory,
    RAGMemory,
)

from harness.llm_gemini import RotatingGeminiClient
from harness.llm_groq import SimpleGroqClient
from dotenv import load_dotenv
load_dotenv()

# Configuration
SCRIPTS_DIR = "test_scripts"     # for quick testing
RESULTS_DIR = "results"
REPETITIONS = 5
SLEEP_BETWEEN_CALLS = 0.5  # to avoid hitting rate limits too quickly

def get_all_scripts():
    scripts = []
    for fname in os.listdir(SCRIPTS_DIR):
        if fname.endswith(".json"):
            with open(os.path.join(SCRIPTS_DIR, fname), "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    scripts.extend(data)
                else:
                    scripts.append(data)
    return scripts

def get_user_messages(script):
    user_msgs = []
    for turn in script["turns"]:
        if turn["role"] == "user":
            user_msgs.append(turn["content"])
    return user_msgs

def is_recall_turn(script, turn_index):
    turn_num = turn_index + 1
    if turn_num not in {5,10,15,20,25}:
        return False
    for turn in script["turns"]:
        if turn.get("turn") == turn_num and turn.get("is_recall"):
            return True
    return False

def get_recall_info(script, turn_index):
    turn_num = turn_index + 1
    for turn in script["turns"]:
        if turn.get("turn") == turn_num and turn.get("is_recall"):
            return turn.get("target_fact_id"), turn.get("ground_truth")
    return None, None

def run_strategy(script, strategy_name, strategy_class, llm, script_id, rep):
    print(f"      [DEBUG] Starting {strategy_name} rep {rep}")
    if strategy_class in (RollingSummaryMemory, HierarchicalMemory):
        memory = strategy_class(llm)
    else:
        memory = strategy_class()

    turns = script["turns"]
    results = []

    for turn_obj in turns:
        turn_num = turn_obj["turn"]
        role = turn_obj["role"]
        content = turn_obj["content"]

        if role == "user":
            # Build context
            if isinstance(memory, RAGMemory):
                context = memory.get_context(content)
            else:
                context = memory.get_context()

            # Build prompt (same as before)
            if context:
                full_prompt = (
                    f"Conversation history:\n{context}\n\n"
                    f"Question: {content}\n\n"
                    f"Instructions: Answer using ONLY the facts in the conversation history above. "
                    f"Answer in one short phrase (max 6 words). "
                    f"If the exact answer is not present, respond with exactly 'I don't know'.\n\n"
                    f"Answer:"
                )
            else:
                full_prompt = (
                    f"Question: {content}\n\n"
                    f"Instructions: Answer directly. If you don't know, say 'I don't know'.\n\n"
                    f"Answer:"
                )

            print(f"      [DEBUG] Turn {turn_num} - calling llm.generate (prompt length {len(full_prompt)})")
            response = llm.generate(full_prompt)
            print(f"      [DEBUG] Turn {turn_num} - got response (length {len(response)})")

            # Store in memory
            memory.add_message("user", content)
            memory.add_message("assistant", response)

            # If this is a recall turn, record the result
            if turn_obj.get("is_recall"):
                target_id = turn_obj.get("target_fact_id")
                gt_list = turn_obj.get("ground_truth")
                results.append({
                    "turn": turn_num,
                    "target_fact_id": target_id,
                    "ground_truth": gt_list,
                    "agent_response": response,
                    "script_id": script_id,
                    "strategy": strategy_name,
                    "rep": rep
                })
        else:
            memory.add_message("assistant", content)

        # Trigger compression every 5 turns (based on actual turn number)
        if turn_num % 5 == 0 and role == "assistant":
            print(f"      [DEBUG] Turn {turn_num} - calling compress()")
            memory.compress()

        time.sleep(SLEEP_BETWEEN_CALLS)

    return results

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(RESULTS_DIR, f"benchmark_results_{timestamp}.csv")

    scripts = get_all_scripts()
    print(f"Loaded {len(scripts)} scripts.")

    groq_keys = [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ]
    llm = SimpleGroqClient(groq_keys)
    print("Groq LLM client ready.")

    strategies = [
        ("rolling_summary", RollingSummaryMemory),
        ("baseline", BaselineMemory),
        ("hierarchical", HierarchicalMemory),
        ("rag", RAGMemory),
    ]

    csv_file = open(output_csv, "w", newline="", encoding="utf-8")
    fieldnames = ["script_id", "strategy", "repetition", "turn", "target_fact_id", "agent_response", "ground_truth"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    total_runs = len(scripts) * len(strategies) * REPETITIONS
    pbar = tqdm(total=total_runs, desc="Overall progress")

    for script in scripts:
        script_id = script.get("script_id", "unknown")
        for strategy_name, strategy_class in strategies:
            for rep in range(REPETITIONS):
                print(f"  Starting {strategy_name} on script {script_id}, rep {rep}")
                results = run_strategy(script, strategy_name, strategy_class, llm, script_id, rep)
                for row in results:
                    writer.writerow({
                        "script_id": row["script_id"],
                        "strategy": row["strategy"],
                        "repetition": row["rep"],
                        "turn": row["turn"],
                        "target_fact_id": row["target_fact_id"],
                        "agent_response": row["agent_response"],
                        "ground_truth": json.dumps(row["ground_truth"]),
                    })
                csv_file.flush()
                pbar.update(1)

    csv_file.close()
    pbar.close()
    print(f"\nResults saved to {output_csv}")

if __name__ == "__main__":
    main()