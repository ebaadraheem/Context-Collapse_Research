"""Hierarchical memory: short-term buffer + long-term fact list."""

from typing import Any
from .base import MemoryBase

SEMANTIC_PROMPT = """\
You are updating a long-term semantic memory for a fact-retention benchmark.
You will receive an existing fact list and a set of new episodic memory entries.
Your job is to merge them into a single updated fact list.

CRITICAL RULES:
1. Preserve ALL specific values EXACTLY as stated in both sources: names, numbers, \
dates, dollar amounts, locations, identifiers, percentages, durations, codes.
2. Do NOT paraphrase or round any value.
3. If the same fact appears in both sources with different values, keep BOTH and \
mark the conflict: "- Contract value: $4.2M [turn 1] / $4.5M [episode 3]"
4. Remove duplicates — if the same fact appears twice with the same value, keep one.
5. Output format: one fact per bullet line, e.g.:
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


EPISODIC_PROMPT = """\
You are compressing a fragment of a conversation into an episodic memory entry.

CRITICAL RULES:
1. Preserve ALL specific values EXACTLY as stated: names, numbers, dates, dollar \
amounts, locations, identifiers, percentages, durations, codes.
2. Do NOT paraphrase or round numbers. "$4.2M" must stay "$4.2M", not "over four million".
3. Write in this format:
   Topic: <what was being discussed in one phrase>
   Facts: <bullet list of all specific values and decisions, one per line>
   Outcome: <what was concluded or left open, one sentence>
4. Do NOT add any information not present in the conversation fragment.
5. Do NOT omit any fact, even if it seems minor.

Conversation fragment:
{text}

Episodic memory entry:"""


class HierarchicalMemory(MemoryBase):
    """
    Maintains:
      - short_term_buffer: the most recent ``buffer_size`` messages (raw).
      - long_term_summary: a bullet-point fact list of older messages.

    compress() moves messages beyond ``buffer_size`` from the buffer into the
    long-term summary via an LLM call.
    """

    def __init__(self, llm, 
                 working_budget=500,    # tokens
                 episodic_budget=1000): # tokens
        self.llm = llm
        self.working_budget = working_budget
        self.episodic_budget = episodic_budget
        self.working: list[dict] = []       # raw recent messages
        self.episodic: list[str] = []       # compressed episode summaries
        self.semantic: str = ""             # long-term distilled facts

    def add_message(self, role, content):
        self.working.append({"role": role, "content": content})
        if self._tokens(self.working) > self.working_budget:
            self._flush_working_to_episodic()
        if self._tokens(self.episodic) > self.episodic_budget:
            self._flush_episodic_to_semantic()

    def _flush_working_to_episodic(self):
        # Summarise working memory into one episodic entry
        text = "\n".join(f"{m['role']}: {m['content']}" for m in self.working[:-2])
        episode = self.llm.generate(EPISODIC_PROMPT.format(text=text))
        self.episodic.append(episode)
        self.working = self.working[-2:]  # keep last exchange

    def _flush_episodic_to_semantic(self):
        # Distil episodic entries into the semantic fact list
        episodes_text = "\n".join(self.episodic)
        self.semantic = self.llm.generate(
            SEMANTIC_PROMPT.format(existing=self.semantic, episodes=episodes_text)
        )
        self.episodic = []
        
    def compress(self) -> None:
        pass  

    def get_context(self, query=""):
        parts = []
        if self.semantic:
            parts.append(f"[Long-term facts]:\n{self.semantic}")
        if self.episodic:
            parts.append(f"[Recent episodes]:\n" + "\n".join(self.episodic))
        if self.working:
            raw = "\n".join(f"{m['role']}: {m['content']}" for m in self.working)
            parts.append(f"[Working memory]:\n{raw}")
        return "\n\n".join(parts)

    def reset(self) -> None:
        self.semantic = ""
        self.episodic = []
        self.working = []