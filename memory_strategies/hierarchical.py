"""
Hierarchical memory: three-level architecture (working → episodic → semantic).

Compression is self-triggered by token-budget pressure inside add_message(),
so external compress() calls from the llm are intentional no-ops.

Levels
------
Working memory  : raw recent messages, always included verbatim.
Episodic memory : structured episode summaries flushed from working memory.
Semantic memory : canonical long-term fact list distilled from episodic entries.
"""

from __future__ import annotations

import time
from typing import Any

from .base import MemoryBase

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EPISODIC_PROMPT = """\
You are compressing a fragment of a conversation into an episodic memory entry.

CRITICAL RULES:
1. Preserve ALL specific values EXACTLY as stated: names, numbers, dates, dollar \
amounts, locations, identifiers, percentages, durations, codes.
2. Do NOT paraphrase or round numbers. "$4.2M" must stay "$4.2M", not "over four million".
3. Write in this exact format:
   Topic: <what was being discussed, one short phrase>
   Facts:
   - <fact 1>
   - <fact 2>
   Outcome: <what was concluded or left open, one sentence>
4. Do NOT add any information not present in the conversation fragment.
5. Do NOT omit any fact, even if it seems minor.

Conversation fragment:
{text}

Episodic memory entry:"""


SEMANTIC_PROMPT = """\
You are updating a long-term semantic memory for a fact-retention benchmark.
You will receive an existing fact list and a set of new episodic memory entries.
Your job is to merge them into a single updated fact list.

CRITICAL RULES:
1. Preserve ALL specific values EXACTLY as stated in both sources: names, numbers, \
dates, dollar amounts, locations, identifiers, percentages, durations, codes.
2. Do NOT paraphrase or round any value.
3. If the same fact appears in both sources with DIFFERENT values, keep BOTH and \
mark the conflict: "- Contract value: $4.2M [original] / $4.5M [updated]"
4. Remove exact duplicates — if the same fact appears twice with the same value, keep one.
5. Output format: one fact per bullet line. Examples:
   - Deadline: March 10
   - Contract value: $4.2M
   - Governing law: Delaware
   - Liability cap: 100% of contract value
6. Do NOT add any information not present in either source.
7. Do NOT omit any fact, even if it seems minor.

Existing semantic fact list:
{existing}

New episodic entries to integrate:
{episodes}

Updated semantic fact list:"""


# Sentinel used when semantic memory is empty so the prompt stays valid.
_EMPTY_SENTINEL = "(none yet)"


class HierarchicalMemory(MemoryBase):
    """
    Three-level hierarchical memory.

    Compression is self-triggered by token-budget pressure, NOT by external
    turn-count logic.  compress() is kept as a public no-op to satisfy the
    MemoryBase interface — run_benchmark.py may call it harmlessly.

    Token counting uses the standard 1 token ≈ 4 chars heuristic.

    Args:
        llm             : LLM client with generate(prompt, use_stop_tokens, max_tokens).
        working_budget  : Approximate token ceiling for working memory before
                          flushing to episodic (default 500 ≈ ~2000 chars).
        episodic_budget : Approximate token ceiling for episodic store before
                          distilling to semantic (default 1000 ≈ ~4000 chars).
        llm_sleep       : Seconds to sleep after each internal LLM call to avoid
                          rate-limit bursts from self-triggered compression.
    """

    # Detected by run_benchmark to decide whether to pass llm at init.
    NEEDS_LLM: bool = True

    def __init__(
        self,
        llm: Any,
        working_budget: int = 500,
        episodic_budget: int = 1000,
        llm_sleep: float = 0.5,
    ) -> None:
        self.llm = llm
        self.working_budget = working_budget
        self.episodic_budget = episodic_budget
        self.llm_sleep = llm_sleep

        self.working: list[dict] = []   # raw recent messages
        self.episodic: list[str] = []   # structured episode strings
        self.semantic: str = ""         # canonical long-term fact list

    # ------------------------------------------------------------------
    # Token counting (all use the same heuristic for consistency)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokens_of(text: str) -> int:
        """Approximate token count from character length."""
        return max(1, len(text) // 4)

    def _tokens_working(self) -> int:
        joined = "\n".join(f"{m['role']}: {m['content']}" for m in self.working)
        return self._tokens_of(joined)

    def _tokens_episodic(self) -> int:
        # Join episodic list into a single string for consistent counting.
        return self._tokens_of("\n".join(self.episodic))

    # ------------------------------------------------------------------
    # Internal flush helpers
    # ------------------------------------------------------------------

    def _flush_working_to_episodic(self) -> None:
        """
        Compress the oldest working-memory messages into one episodic entry,
        keeping the last 2 messages (most recent exchange) in working memory.

        Guard: if working memory has ≤ 2 messages there is nothing to flush —
        the budget is too small for even a single full exchange.
        """
        if len(self.working) <= 2:
            return  # nothing to flush safely

        to_compress = self.working[:-2]   # all but the last exchange
        self.working = self.working[-2:]  # retain most recent pair

        text = "\n".join(f"{m['role']}: {m['content']}" for m in to_compress)
        prompt = EPISODIC_PROMPT.format(text=text)

        episode = self.llm.generate(
            prompt,
            use_stop_tokens=False,
            max_tokens=300,
        ).strip()

        if episode:
            self.episodic.append(episode)

        # Sleep to avoid bursting the API from self-triggered calls.
        time.sleep(self.llm_sleep)

    def _flush_episodic_to_semantic(self) -> None:
        """
        Distil all episodic entries into the semantic fact list and clear the
        episodic store.
        """
        if not self.episodic:
            return

        episodes_text = "\n\n---\n\n".join(self.episodic)
        existing = self.semantic if self.semantic else _EMPTY_SENTINEL

        prompt = SEMANTIC_PROMPT.format(
            existing=existing,
            episodes=episodes_text,
        )

        self.semantic = self.llm.generate(
            prompt,
            use_stop_tokens=False,
            max_tokens=600,
        ).strip()

        self.episodic = []

        time.sleep(self.llm_sleep)

    # ------------------------------------------------------------------
    # MemoryBase interface
    # ------------------------------------------------------------------

    def add_message(self, role: str, content: str) -> None:
        self.working.append({"role": role, "content": content})

        # Check working budget first.
        if self._tokens_working() > self.working_budget:
            self._flush_working_to_episodic()

        # Then check episodic budget (flushing working may have grown episodic).
        if self._tokens_episodic() > self.episodic_budget:
            self._flush_episodic_to_semantic()

    def compress(self) -> None:
        """
        External compression hook — intentional no-op.

        HierarchicalMemory self-triggers compression inside add_message() based
        on token-budget pressure. This method exists only to satisfy the
        MemoryBase interface contract.
        """
        pass

    def get_context(self, query: str = "") -> str:
        parts: list[str] = []

        if self.semantic:
            parts.append(f"[Long-term facts]:\n{self.semantic}")

        if self.episodic:
            joined = "\n\n---\n\n".join(self.episodic)
            parts.append(f"[Recent episodes]:\n{joined}")

        if self.working:
            raw = "\n".join(f"{m['role']}: {m['content']}" for m in self.working)
            parts.append(f"[Working memory]:\n{raw}")

        return "\n\n".join(parts)

    def reset(self) -> None:
        self.working = []
        self.episodic = []
        self.semantic = ""