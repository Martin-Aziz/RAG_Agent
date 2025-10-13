from typing import List, Dict, Any
import math


class BM25Retriever:
    def __init__(self):
        self.docs = []

    def index(self, docs: List[Dict[str, Any]]):
        self.docs = docs

    def _score(self, query: str, text: str) -> float:
        qwords = set(query.lower().split())
        twords = text.lower().split()
        common = qwords.intersection(twords)
        return float(len(common)) / (1 + math.log(1 + len(twords)))

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        scored = []
        for i, d in enumerate(self.docs):
            s = self._score(query, d.get("text", ""))
            scored.append((s, i, d))
        scored.sort(key=lambda x: -x[0])
        results = []
        for s, i, d in scored[:k]:
            results.append({"doc_id": d.get("doc_id"), "passage_id": f"p{i}", "score": s, "text": d.get("text", "")})
        return results
