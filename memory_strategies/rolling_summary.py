"""
Rolling summarisation memory.

Compression is self-triggered when context size exceeds token_budget.
compress() is also a valid external entry point (called from the harness
for strategies that do not self-trigger).
"""

from __future__ import annotations

import time
from typing import Any

from .base import MemoryBase

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_COMPRESS_PROMPT = """\
You are a conversation summariser for a fact-retention benchmark.
Your job is to produce a bullet-point fact list from the conversation below.

CRITICAL RULES:
1. Preserve ALL specific values EXACTLY as stated: names, numbers, dates, dollar amounts,
   locations, identifiers, percentages, durations. Do NOT paraphrase them.
2. Write one fact per bullet line, e.g. "- Deadline: March 10"
3. Do NOT add any information not present in the conversation.
4. Do NOT omit any fact, even if it seems minor.
5. Integrate the existing summary (if provided) with the new messages.

{existing_summary_block}
New messages:
{new_messages}

Updated fact list:"""


class RollingSummaryMemory(MemoryBase):
    """
    Keeps a running bullet-point fact list.

    Raw messages accumulate since the last compression. When get_context() is
    called or when add_message() detects the context exceeds token_budget, all
    raw messages are compressed into the summary via an LLM call.

    Args:
        llm          : LLM client with generate(prompt, use_stop_tokens, max_tokens).
        token_budget : Approximate token ceiling before self-triggering compression.
        llm_sleep    : Seconds to sleep after each internal LLM compression call.
    """

    NEEDS_LLM: bool = True

    def __init__(
        self,
        llm: Any,
        token_budget: int = 1500,
        llm_sleep: float = 0.5,
    ) -> None:
        self.llm = llm
        self.token_budget = token_budget
        self.llm_sleep = llm_sleep
        self.summary: str = ""
        self.raw_messages: list[dict] = []

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def _token_count(self) -> int:
        """Approximate token count of the full context (summary + raw messages)."""
        return max(1, len(self.get_context()) // 4)

    # ------------------------------------------------------------------
    # MemoryBase interface
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        self.raw_messages.append({"role": role, "content": content})
        # Self-trigger compression when budget is exceeded.
        if self._token_count() > self.token_budget:
            self._do_compress()

    def compress(self) -> None:
        """
        External compression hook.

        Called by the harness (run_benchmark.py). Delegates to _do_compress()
        only when there are raw messages to process.
        """
        if self.raw_messages:
            self._do_compress()

    def _do_compress(self) -> None:
        """Core compression logic — shared by add_message and compress."""
        if not self.raw_messages:
            return

        new_messages_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.raw_messages
        )

        if self.summary:
            existing_block = (
                f"Existing fact list (DO NOT lose any of these facts):\n{self.summary}\n"
            )
        else:
            existing_block = ""

        prompt = _COMPRESS_PROMPT.format(
            existing_summary_block=existing_block,
            new_messages=new_messages_text,
        )

        self.summary = self.llm.generate(
            prompt,
            use_stop_tokens=False,
            max_tokens=400,
        ).strip()

        self.raw_messages = []

        # Sleep after internal LLM call to avoid rate-limit bursts.
        time.sleep(self.llm_sleep)

    def get_context(self, query: str = "") -> str:
        parts: list[str] = []
        if self.summary:
            parts.append(f"[Conversation fact list]:\n{self.summary}")
        if self.raw_messages:
            recent = "\n".join(
                f"{m['role']}: {m['content']}" for m in self.raw_messages
            )
            parts.append(f"[Recent messages (not yet summarised)]:\n{recent}")
        return "\n\n".join(parts)

    def reset(self) -> None:
        self.summary = ""
        self.raw_messages = []