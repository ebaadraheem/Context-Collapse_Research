"""
Azure OpenAI client for o4-mini with inference‑friendly system prompt.
"""

from __future__ import annotations

import time
import openai
from openai import AzureOpenAI

# New system prompt that encourages inference from conversation history
RECALL_SYSTEM_PROMPT = """\
You are a fact-recall assistant. Use the conversation history above to answer the user's question.

Rules:
- If the user asks about a deadline, date, amount, name, or percentage, extract that exact value from the history.
- If the user asks whether something is correct, compare it to the history and answer with "Yes" or "No", then provide the correct fact if needed.
- If the question is indirect (e.g., "Should we build a buffer?"), infer the answer from the relevant facts in the history.
- Answer concisely in one short sentence or phrase.
- If the answer is truly not present in the history, say "I don't know".

Examples:
Q: "Patricia has a meeting in mid-March. Should we build a buffer?" 
History: "Internal review deadline is March 10, 2025."
A: "Yes, because the deadline is March 10, which is before mid-March."

Q: "The draft says 99% uptime. Is that right?"
History: "SLA uptime requirement is 99.5% monthly."
A: "No, it should be 99.5%."

Now answer the user's question based on the conversation history above.
"""

class AzureOpenAIClient:
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2024-12-01-preview",
        max_retries: int = 5,
        base_sleep: float = 10.0,
    ) -> None:
        self.client = AzureOpenAI(
            azure_endpoint=azure_endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self.deployment_name = deployment_name
        self.max_retries = max_retries
        self.base_sleep = base_sleep
        self._call_count = 0

    def get_call_count(self) -> int:
        return self._call_count

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 150,
        use_stop_tokens: bool = True,
    ) -> str:
        # Use the inference‑friendly system prompt for recall turns
        if use_stop_tokens and not system_prompt:
            system_prompt = RECALL_SYSTEM_PROMPT

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_exception = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
                self._call_count += 1
                content = response.choices[0].message.content
                return content.strip() if content else "I don't know"
            except openai.RateLimitError as exc:
                last_exception = exc
                wait = self.base_sleep * (2 ** attempt)
                print(f"[Azure] Rate limit hit. Retry {attempt+1}/{self.max_retries} after {wait:.1f}s")
                time.sleep(wait)
            except Exception as exc:
                last_exception = exc
                print(f"[Azure] Error: {exc}. Retry {attempt+1}/{self.max_retries}")
                time.sleep(2)

        raise RuntimeError(f"Azure OpenAI failed after {self.max_retries} attempts. Last error: {last_exception}")