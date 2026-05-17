"""
Smoke test: verify Groq client works end-to-end with a sample benchmark prompt.

Usage:
    python test_groq.py
"""

from dotenv import load_dotenv
import os
from harness.llm_groq import SimpleGroqClient
from harness.utils import filter_none_keys

load_dotenv()

keys = filter_none_keys([
    os.getenv("GROQ_API_KEY_1"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
])
llm = SimpleGroqClient(keys)

# Simulated recall prompt (turn 5 of a legal script)
context = """\
user: We are reviewing a contract between TechNova and Meridian. Deadline is March 10. Value is $4.2M. Governing law is Delaware.
assistant: Understood. I have noted the key details.
user: Section 8 sets the liability cap at 100% of contract value. Is that a concern?
assistant: Yes, a 100% liability cap is unusually high. Consider negotiating it down to 150% or capping at a fixed amount.
user: Patricia will be busy in mid-March. Do we need a buffer?
assistant: Given Patricia's schedule, I recommend building in a two-week buffer, targeting a March 24 completion instead."""

question = "What is the contract value between TechNova and Meridian?"
ground_truth = ["$4.2M", "4.2 million", "4,200,000"]

prompt = f"Conversation history:\n{context}\n\nRecall question: {question}"

print("─" * 60)
print("Prompt length:", len(prompt))
print("─" * 60)
response = llm.generate(prompt, use_stop_tokens=True)
print("Response:", repr(response))

# Simple check
hit = any(gt.lower() in response.lower() for gt in ground_truth)
print("Correct:", hit)