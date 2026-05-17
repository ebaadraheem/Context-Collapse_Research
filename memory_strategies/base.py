"""Abstract base class for all memory strategies."""

from abc import ABC, abstractmethod


class MemoryBase(ABC):
    """
    All memory strategies must implement this interface.

    The get_context() signature accepts an optional query string so that
    retrieval-based strategies (RAGMemory) can use it for vector search while
    non-retrieval strategies safely ignore it.
    """

    @abstractmethod
    def add_message(self, role: str, content: str) -> None:
        """Append a message to memory."""
        ...

    @abstractmethod
    def get_context(self, query: str = "") -> str:
        """
        Return the context string to prepend to the current prompt.

        Args:
            query: The current user message. Used by RAG-based strategies for
                   similarity search; ignored by others.
        Returns:
            A formatted string representing conversation history / summary.
        """
        ...

    @abstractmethod
    def compress(self) -> None:
        """
        Compress or consolidate memory.
        Called after every N turns. No-op for strategies that don't compress.
        """
        ...

    def reset(self) -> None:
        """
        Reset all internal state.
        Called between benchmark repetitions to prevent state bleed.
        Subclasses MUST override this if they hold any mutable state.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement reset() "
            "to prevent state bleed between benchmark repetitions."
        )