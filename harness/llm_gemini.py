"""Gemini LLM client with per-key cooldown tracking and exponential back-off."""

from __future__ import annotations

import time

from google import genai
from google.genai import types

RECALL_SYSTEM_PROMPT = """\
You are a fact-recall assistant operating inside a benchmark.
Your only job is to recall a specific fact from the conversation history provided.

Rules:
1. Answer with the exact value from the conversation history: name, number, date, amount, etc.
2. Do NOT explain, do NOT add context, do NOT paraphrase.
3. If the answer is not explicitly stated in the history, respond with exactly: I don't know
4. Maximum answer length: 10 words.\
"""


class RotatingGeminiClient:
    """
    Gemini API wrapper with per-key cooldown tracking.
    self.client is updated on every key rotation (original bug fixed).
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(self, api_keys: list[str], model_name: str = DEFAULT_MODEL) -> None:
        self.keys = [k for k in api_keys if k]
        if not self.keys:
            raise ValueError("No Gemini API keys provided.")
        self.model_name = model_name
        self.key_cooldown_until: dict[str, float] = {k: 0.0 for k in self.keys}
        self.current_key: str = self.keys[0]
        self.client = genai.Client(api_key=self.current_key)
        print(f"[Gemini] Initialised with {len(self.keys)} key(s), model={self.model_name}")

    def _pick_available_key(self, cooldown_on_current: float = 0.0) -> None:
        if cooldown_on_current > 0:
            self.key_cooldown_until[self.current_key] = time.time() + cooldown_on_current

        now = time.time()
        available = [k for k, t in self.key_cooldown_until.items() if t <= now]
        if available:
            others = [k for k in available if k != self.current_key]
            chosen = others[0] if others else available[0]
        else:
            earliest = min(self.key_cooldown_until, key=self.key_cooldown_until.get)
            wait = self.key_cooldown_until[earliest] - now + 0.5
            print(f"[Gemini] All keys cooling. Waiting {wait:.1f}s …")
            time.sleep(wait)
            chosen = earliest

        if chosen != self.current_key:
            self.current_key = chosen
            self.client = genai.Client(api_key=chosen)   # critical: update client
            print(f"[Gemini] Switched to key …{chosen[-6:]}")

    def generate(
        self,
        prompt: str,
        system_prompt: str = RECALL_SYSTEM_PROMPT,
        max_output_tokens: int = 100,
        max_attempts: int = 5,
    ) -> str:
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.0,
                        max_output_tokens=max_output_tokens,
                    ),
                )
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        text_parts = [p.text for p in candidate.content.parts if p.text]
                        if text_parts:
                            return "".join(text_parts).strip()
                print(f"[Gemini] Empty response on attempt {attempt + 1}.")
                self._pick_available_key(cooldown_on_current=15)

            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "429" in msg or "quota" in msg or "rate limit" in msg:
                    cooldown = 30 * (2 ** attempt)
                    print(f"[Gemini] Rate limit (attempt {attempt + 1}). Cooling {cooldown}s.")
                    self._pick_available_key(cooldown_on_current=cooldown)
                else:
                    print(f"[Gemini] Error: {exc}. Rotating key.")
                    self._pick_available_key(cooldown_on_current=10)

        raise RuntimeError(
            f"All Gemini attempts exhausted after {max_attempts} tries. Last error: {last_exc}"
        )