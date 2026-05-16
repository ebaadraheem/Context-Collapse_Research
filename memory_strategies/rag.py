"""RAG (Retrieval‑Augmented) memory: store all messages in a vector DB, retrieve top‑k relevant past turns."""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import Any, List, Dict

class RAGMemory:
    """
    Each message (user or assistant) is embedded and stored in ChromaDB.
    At query time, the current user message is embedded and the top-k most similar past messages
    are retrieved and prepended to the recent buffer.
    """

    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", k: int = 3):
        """
        Args:
            embedding_model: Name of a sentence-transformers model.
            k: Number of past messages to retrieve.
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("conversation_memory")
        self.message_buffer = []          # recent raw messages (kept for local context)
        self.counter = 0                  # unique ID for each stored message
        self.k = k

    def add_message(self, role: str, content: str) -> None:
        """
        Store a message in the vector database and also keep it in the short buffer.
        """
        self.counter += 1
        full_text = f"{role}: {content}"
        embedding = self.encoder.encode(full_text).tolist()
        self.collection.add(
            documents=[full_text],
            embeddings=[embedding],
            ids=[str(self.counter)]
        )
        # Keep a small buffer of recent messages (last 3 turns) to maintain local coherence
        self.message_buffer.append({"role": role, "content": content})
        if len(self.message_buffer) > 3:
            self.message_buffer.pop(0)

    def get_context(self, current_user_query: str) -> str:
        """
        Retrieve relevant past messages using the current user query, then build context.
        """
        # Encode the user query
        query_embedding = self.encoder.encode(current_user_query).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.k
        )
        retrieved_docs = results['documents'][0] if results['documents'] else []

        # Build context string
        parts = []
        if retrieved_docs:
            parts.append("[Most relevant past messages]:\n" + "\n".join(retrieved_docs))
        if self.message_buffer:
            parts.append("[Recent conversation]:\n" +
                         "\n".join([f"{m['role']}: {m['content']}" for m in self.message_buffer]))
        return "\n\n".join(parts) if parts else ""

    def compress(self) -> None:
        """
        RAG memory does not use explicit compression; the retrieval step replaces it.
        This method exists for API compatibility.
        """
        pass