"""Abstract base class for all memory strategies."""

from abc import ABC, abstractmethod


class MemoryBase(ABC):

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """Append a message to memory."""
        ...

    @abstractmethod
    def get_context(self, query: str = "") -> str:
        """
        Return the context string to prepend to the current prompt.
        """
        ...

    @abstractmethod
    def compress(self) -> None:
        """
        Compress or consolidate memory.

        For self-triggering strategies (HierarchicalMemory, RollingSummaryMemory),
        this is an intentional no-op — compression fires inside add_message().
        For BaselineMemory and RAGMemory this is also a no-op.
        The method exists to satisfy the interface contract.
        """
        ...

    def reset(self) -> None:
        """
        Reset all internal state between benchmark repetitions.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement reset() "
            "to prevent state bleed between benchmark repetitions."
        )