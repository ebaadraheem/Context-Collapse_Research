"""
Groq LLM client with per-key cooldown tracking and exponential back-off.

Key design points
-----------------
- max_tokens defaults to 30 for recall calls (short answers expected).
  Compression/summarisation callers MUST pass a larger max_tokens explicitly
  (e.g. 300–600) — failing to do so silently truncates summaries.
- use_stop_tokens=True adds stop sequences for recall calls only.
- system_prompt defaults to RECALL_SYSTEM_PROMPT when use_stop_tokens=True,
  and to "" (no system prompt) otherwise, so compression calls get a clean slate.
"""

from __future__ import annotations

import time

from groq import Groq


# ---------------------------------------------------------------------------
# System prompt for recall calls
# ---------------------------------------------------------------------------

RECALL_SYSTEM_PROMPT = """\
You are a fact-recall assistant operating inside a benchmark.
Your only job is to recall a specific fact from the conversation history provided.

Rules:
1. Answer with the exact value from the conversation history: name, number, date, amount, etc.
2. Do NOT explain, do NOT add context, do NOT paraphrase.
3. If the answer is not explicitly stated in the history, respond with exactly: I don't know
4. Maximum answer length: 10 words.\
"""


class SimpleGroqClient:
    """
    Groq API wrapper with:
    - Per-key cooldown tracking with exponential back-off on rate limits.
    - Explicit max_tokens per call type (recall vs compression).
    - system_prompt support passed as the 'system' role message.
    - seed=42 for reproducibility.
    """

    DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self, api_keys: list[str], model_name: str = DEFAULT_MODEL) -> None:
        self.keys = [k for k in api_keys if k]
        if not self.keys:
            raise ValueError("No Groq API keys provided.")
        self.model = model_name
        self.key_cooldown_until: dict[str, float] = {k: 0.0 for k in self.keys}
        self.current_key: str = self.keys[0]
        self.client: Groq = Groq(api_key=self.current_key)
        print(f"[Groq] Initialised with {len(self.keys)} key(s), model={self.model}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_available_key(self, cooldown_on_current: float = 0.0) -> None:
        """
        Apply cooldown_on_current to the current key, then select the next
        available key. If all keys are cooling, block until the earliest is ready.
        """
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
            print(f"[Groq] All keys cooling. Waiting {wait:.1f}s …")
            time.sleep(wait)
            chosen = earliest

        if chosen != self.current_key:
            self.current_key = chosen
            self.client = Groq(api_key=chosen)
            print(f"[Groq] Switched to key …{chosen[-6:]}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 30,
        max_attempts: int = 5,
        use_stop_tokens: bool = True,
    ) -> str:
        """
        Generate a completion.

        Args:
            prompt          : The user-turn content.
            system_prompt   : Optional system message.
                              Auto-set to RECALL_SYSTEM_PROMPT when
                              use_stop_tokens=True and system_prompt is empty.
            max_tokens      : Token budget for the response.
                              *** Callers for compression MUST pass a larger value
                              (e.g. 300–600). The default 30 is for recall only. ***
            max_attempts    : Total retry budget across all keys.
            use_stop_tokens : If True, adds stop sequences for recall calls.
                              Set False for compression / judge calls.

        Returns:
            Stripped text response.
        """
        # Auto-apply recall system prompt for recall calls.
        if not system_prompt and use_stop_tokens:
            system_prompt = RECALL_SYSTEM_PROMPT

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Stop tokens prevent the model running past the answer for recall calls.
        # Bare "\n" is intentionally excluded — it truncates multi-part answers.
        stop = ["\nQuestion:", "\nUser:", "\nAnswer:"] if use_stop_tokens else None

        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=max_tokens,
                    stop=stop,
                    seed=42,
                )
                return resp.choices[0].message.content.strip()

            except Exception as exc:
                last_exc = exc
                msg = str(exc).lower()
                if "rate limit" in msg or "429" in msg:
                    cooldown = 30 * (2 ** attempt)  # 30 / 60 / 120 / 240 / 480 s
                    print(
                        f"[Groq] Rate limit on key …{self.current_key[-6:]} "
                        f"(attempt {attempt + 1}/{max_attempts}). "
                        f"Cooling {cooldown}s then rotating."
                    )
                    self._pick_available_key(cooldown_on_current=cooldown)
                else:
                    print(f"[Groq] Error: {exc}. Rotating key.")
                    self._pick_available_key(cooldown_on_current=10)

        raise RuntimeError(
            f"All Groq attempts exhausted after {max_attempts} tries. "
            f"Last error: {last_exc}"
        )