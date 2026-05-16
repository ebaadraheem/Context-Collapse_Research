from dotenv import load_dotenv
import os
from harness.llm_groq import SimpleGroqClient

load_dotenv()
keys = [os.getenv("GROQ_API_KEY_1"), os.getenv("GROQ_API_KEY_2"), os.getenv("GROQ_API_KEY_3")]
llm = SimpleGroqClient(keys)

# The exact prompt from turn 5 of your benchmark (from run_benchmark.py)
prompt = """Conversation history:
User: We are reviewing a contract between TechNova and Meridian. Deadline is March 10. Value is $4.2M. Law is Delaware.
Assistant: OK.
User: Section 8 says liability cap 100%. Concern?
Assistant: Change to 150%.
User: Patricia busy mid-March. Buffer needed?

Question: Patricia busy mid-March. Buffer needed?

Instructions: Answer using ONLY the facts in the conversation history above. Answer in one very short phrase (max 6 words). If the exact answer is not present, respond with exactly 'I don't know'.

Answer:"""

print("Prompt length:", len(prompt))
response = llm.generate(prompt)
print("Response:", response)