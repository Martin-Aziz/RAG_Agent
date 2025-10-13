from typing import List, Dict, Any
from core.agents.retriever_vector import VectorRetriever


class MemoryAgent:
    def __init__(self):
        self.episodic = []
        self.semantic = VectorRetriever()

    def memorize(self, key: str, triple: Dict[str, Any], to_semantic: bool = False):
        self.episodic.append({"key": key, "triple": triple})
        if to_semantic:
            # index into semantic store
            docs = [{"doc_id": f"mem-{i}", "text": str(e["triple"])} for i, e in enumerate(self.episodic)]
            self.semantic.index(docs)

    def recall(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        return self.semantic.retrieve(query, k=k)
