"""
Azure OpenAI client for o4-mini (and compatible models).

Root causes fixed vs the original
-----------------------------------
1. RECALL_SYSTEM_PROMPT referenced "conversation history above" — but the
   history lives in the USER message, not above the system prompt.  The model
   was told to look somewhere the history wasn't, so it always fell through to
   "I don't know".  System prompt now tells the model WHERE the history is.

2. use_stop_tokens controlled both stop sequences AND system-prompt injection.
   For o4-mini (a reasoning model) stop sequences are unreliable and can cut
   the answer mid-sentence.  Stop tokens are now removed entirely; we rely on
   max_completion_tokens instead.

3. max_tokens=150 was too tight for o4-mini which emits a brief chain-of-thought
   before the answer.  Recall default raised to 300; compression callers pass
   their own value explicitly.

4. Non-recall turns (use_stop_tokens=False) received NO system prompt at all,
   so the model had no instruction context for normal conversation turns.
   Added a lightweight CONVERSATION_SYSTEM_PROMPT for those turns.

5. Added a COMPRESSION_SYSTEM_PROMPT so memory strategy LLM calls
   (rolling summary, hierarchical) get explicit fact-preservation instructions
   instead of inheriting the recall prompt.
"""

from __future__ import annotations

import time
import logging

import openai
from openai import AzureOpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

# Used for recall turns (is_recall=True).
# CRITICAL: tells the model the history is inside THIS message, not "above".
RECALL_SYSTEM_PROMPT = """\
You are a precise fact-recall assistant.

The user's message contains two sections:
  1. "Conversation history:" – a transcript of the prior conversation.
  2. "Recall question:" – a specific question about a fact from that history.

Your task:
- Read the conversation history carefully.
- Find the specific fact the recall question is asking about.
- Answer with ONLY the exact value from the history (name, number, date,
  dollar amount, percentage, location, etc.).
- Do NOT explain, paraphrase, or add context.
- If the exact answer is not present in the conversation history,
  respond with exactly: I don't know
- Maximum answer length: 15 words.
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
    """
    Thin wrapper around Azure OpenAI chat completions.

    Args:
        azure_endpoint   : Your Azure OpenAI endpoint URL.
        api_key          : Azure OpenAI API key.
        deployment_name  : Model deployment name (e.g. "o4-mini").
        api_version      : Azure OpenAI API version string.
        max_retries      : Number of retry attempts on transient errors.
        base_sleep       : Base sleep time (seconds) for exponential back-off.
    """

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
        """Return the total number of successful generate() calls so far."""
        return self._call_count

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 300,
        use_stop_tokens: bool = True,
    ) -> str:
        """
        Call Azure OpenAI and return the text response.

        Args:
            prompt          : The user-role message (includes conversation
                              history inline when relevant).
            system_prompt   : Override the system prompt.  If empty:
                                - use_stop_tokens=True  → RECALL_SYSTEM_PROMPT
                                - use_stop_tokens=False → CONVERSATION_SYSTEM_PROMPT
                              Pass an explicit value for compression calls.
            max_tokens      : Maximum completion tokens.
                              • Recall turns:      300  (default)
                              • Compression turns: 400–600 (caller sets this)
                              • Judge turns:       20
            use_stop_tokens : True  = recall turn → use RECALL_SYSTEM_PROMPT.
                              False = other turn  → use CONVERSATION_SYSTEM_PROMPT.
                              (Ignored if system_prompt is explicitly provided.)
        """
        # Resolve system prompt
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
                    # o4-mini can return None content on content-filter hits
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
                # Content policy or malformed request — not retryable
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