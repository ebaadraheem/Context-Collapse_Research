from __future__ import annotations

import time
import logging

import openai
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

RECALL_SYSTEM_PROMPT = """\
You are a fact-recall assistant that performs simple reasoning when needed.

The user's message contains:
- "Conversation history:" – prior conversation.
- "Recall question:" – a question.

Rules:
- If the question asks for a direct fact (name, number, date, amount, percentage, location), answer with ONLY that exact value.
- If the question asks for a comparison ("is that right?", "does it match?"), answer with "Yes" or "No" and the correct fact.
- If the question asks for a classification ("mid‑market or significant?"), infer the classification from the numeric value.
- If the question asks for a decision or inference ("should we...", "do you think..."), answer with a one‑sentence logical conclusion.
- If the question involves legal interpretation under a named jurisdiction, state the general principle (e.g., "Delaware allows negotiation").
- Keep your answer under 20 words.
- Only if the necessary information is completely missing, respond with exactly: I don't know
"""
# Used for non-recall (normal conversation) turns.
CONVERSATION_SYSTEM_PROMPT = """\
You are a helpful assistant participating in a professional conversation.
Use any conversation history provided to give relevant, concise responses.
"""

# Used by memory strategies when calling the LLM for compression/summarisation.
COMPRESSION_SYSTEM_PROMPT = """\
You are a conversation summariser for a fact-retention benchmark.
Your output is a bullet-point fact list.
Preserve ALL specific values EXACTLY as stated: names, numbers, dates,
dollar amounts, locations, identifiers, percentages, durations, codes.
Do NOT paraphrase or round any value.
"""


class AzureOpenAIClient:

    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        deployment_name: str,
        api_version: str = "2025-01-01-preview",
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
        self._call_count: int = 0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def get_call_count(self) -> int:
        return self._call_count

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 300,
        use_stop_tokens: bool = True,
    ) -> str:
        if not system_prompt:
            system_prompt = (
                RECALL_SYSTEM_PROMPT
                if use_stop_tokens
                else CONVERSATION_SYSTEM_PROMPT
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
        ]

        last_exception: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.deployment_name,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                )
                self._call_count += 1

                content = response.choices[0].message.content
                if content is None:
                    logger.warning(
                        "Azure returned None content (attempt %d). "
                        "Finish reason: %s",
                        attempt + 1,
                        response.choices[0].finish_reason,
                    )
                    return "I don't know"

                result = content.strip()
                return result if result else "I don't know"

            except openai.RateLimitError as exc:
                last_exception = exc
                wait = self.base_sleep * (2 ** attempt)
                logger.warning(
                    "[Azure] Rate limit. Retry %d/%d after %.1fs",
                    attempt + 1, self.max_retries, wait,
                )
                time.sleep(wait)

            except openai.BadRequestError as exc:
                logger.error("[Azure] BadRequestError (not retryable): %s", exc)
                raise

            except Exception as exc:
                last_exception = exc
                wait = 2.0 * (attempt + 1)
                logger.warning(
                    "[Azure] Error: %s. Retry %d/%d after %.1fs",
                    exc, attempt + 1, self.max_retries, wait,
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Azure OpenAI failed after {self.max_retries} attempts. "
            f"Last error: {last_exception}"
        )