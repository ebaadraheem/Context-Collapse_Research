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
3. If a value is not present in the fragment, write `[missing]` instead of inventing a number.
4. Write in this exact format:
   Topic: <what was being discussed, one short phrase>
   Facts:
   - <fact 1>
   - <fact 2>
   Outcome: <what was concluded or left open, one sentence>
5. Do NOT add any information not present in the conversation fragment.
6. Never invent facts or numbers.
7. Do NOT omit any fact, even if it seems minor.

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
3. If a value is missing from both sources, write `[unknown]` instead of inventing a number.
4. If the same fact appears in both sources with DIFFERENT values, keep BOTH and \
mark the conflict: "- Contract value: $4.2M [original] / $4.5M [updated]"
5. Remove exact duplicates — if the same fact appears twice with the same value, keep one.
6. Output format: one fact per bullet line. Examples:
   - Deadline: March 10
   - Contract value: $4.2M
   - Governing law: Delaware
   - Liability cap: 100% of contract value
7. Do NOT add any information not present in either source.
8. Never invent facts or numbers.
9. Do NOT omit any fact, even if it seems minor.

Existing semantic fact list:
{existing}

New episodic entries to integrate:
{episodes}

Updated semantic fact list:"""


# Sentinel used when semantic memory is empty so the prompt stays valid.
_EMPTY_SENTINEL = "(none yet)"


class HierarchicalMemory(MemoryBase):

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
    
        if len(self.working) <= 2:
            return  

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

        time.sleep(self.llm_sleep)

    def _flush_episodic_to_semantic(self) -> None:
  
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