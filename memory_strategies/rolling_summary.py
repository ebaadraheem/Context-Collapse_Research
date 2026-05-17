"""Rolling summarisation memory: every N turns, summarise all history into a fact list."""

from typing import Any
from .base import MemoryBase

# Compression prompt engineered to preserve exact values verbatim.
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
    Keeps a running bullet-point fact list of the entire conversation.
    Raw messages are kept only between compression points.
    On compress(), the fact list is updated to incorporate the new raw messages.
    """

    def __init__(self, llm, token_budget: int = 1500):
        self.llm = llm
        self.token_budget = token_budget
        self.summary = ""
        self.raw_messages = []

    def add_message(self, role, content):
        self.raw_messages.append({"role": role, "content": content})
        # Compress dynamically when context exceeds budget
        if self._token_count() > self.token_budget:
            self.compress()
            
    def _token_count(self):
        # Approximate: 1 token ≈ 4 chars
        return len(self.get_context()) // 4        

    def compress(self) -> None:
        """Summarise all raw messages (plus existing summary) into a bullet fact list."""
        if not self.raw_messages:
            return

        new_messages_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in self.raw_messages
        )

        if self.summary:
            existing_block = f"Existing fact list (DO NOT lose any of these facts):\n{self.summary}\n"
        else:
            existing_block = ""

        prompt = _COMPRESS_PROMPT.format(
            existing_summary_block=existing_block,
            new_messages=new_messages_text,
        )

        self.summary = self.llm.generate(prompt).strip()
        self.raw_messages = []  # clear buffer after compression

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

 