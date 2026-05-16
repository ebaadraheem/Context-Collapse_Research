from groq import Groq
import itertools
import time
import os

class SimpleGroqClient:
    def __init__(self, api_keys):
        self.keys = [k for k in api_keys if k]
        if not self.keys:
            raise ValueError("No Groq API keys provided")
        self.key_cycle = itertools.cycle(self.keys)
        self.current_key = next(self.key_cycle)
        self.client = Groq(api_key=self.current_key)
        # Use a powerful, free model that follows instructions well
        self.model = "llama-3.3-70b-versatile"
        print(f"[Groq] Initialized with {len(self.keys)} keys, model={self.model}")

    def _rotate_key(self):
        self.current_key = next(self.key_cycle)
        self.client = Groq(api_key=self.current_key)
        print("[Groq] Rotated to a new API key")

    def generate(self, prompt: str, max_attempts=3) -> str:
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,       # deterministic
                    max_tokens=30,         # force short answers
                    stop=["\n", "Question:", "User:", "Answer:"]
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    print(f"[Groq] Rate limit, rotating key (attempt {attempt+1}/{max_attempts})")
                    self._rotate_key()
                    time.sleep(1)
                    continue
                raise e
        raise Exception("All Groq keys exhausted or failed")