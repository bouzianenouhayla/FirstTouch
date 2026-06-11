import hashlib
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

_COLLECTION = "user_memories"
_DB_PATH = str(Path(__file__).resolve().parents[2] / "vectorstore" / "memories")
_EMBED_MODEL = "all-MiniLM-L6-v2"

# Module-level singleton — SentenceTransformer is ~90MB; load once per process.
_embedder: SentenceTransformer | None = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL)
    return _embedder


class MemoryStore:
    """Semantic memory store backed by ChromaDB.

    Stores user facts as embeddings so relevant context can be retrieved
    for any new question, regardless of when the fact was learned.
    """

    def __init__(self) -> None:
        self._client = chromadb.PersistentClient(path=_DB_PATH)
        self._collection = self._client.get_or_create_collection(_COLLECTION)

    def store(self, fact: str, session_id: str) -> None:
        """Embed and persist a user fact.

        Args:
            fact: Short factual statement about the user, e.g. 'plays as a midfielder'.
            session_id: Session the fact was learned in. Stored as metadata for future
                        user_id filtering once auth is added.
        """
        embedding = _get_embedder().encode(fact).tolist()
        fact_id = f"{session_id}-{hashlib.md5(fact.encode()).hexdigest()}"
        self._collection.upsert(
            documents=[fact],
            embeddings=[embedding],
            metadatas=[{"session_id": session_id}],
            ids=[fact_id],
        )

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve the most relevant user facts for a given query.

        Args:
            query: The user's current question, used to find relevant context.
            k: Maximum number of facts to return.

        Returns:
            List of fact strings, most relevant first. Empty if no facts stored.
        """
        if self._collection.count() == 0:
            return []
        embedding = _get_embedder().encode(query).tolist()
        results = self._collection.query(
            query_embeddings=[embedding],
            n_results=min(k, self._collection.count()),
        )
        return results["documents"][0] if results["documents"] else []
