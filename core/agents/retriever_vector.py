from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


class VectorRetriever:
    def __init__(self):
        # simple in-memory corpus
        self.docs = []
        self.ids = []
        self.vectorizer = TfidfVectorizer()
        self._matrix = None

    def index(self, docs: List[Dict[str, Any]]):
        texts = [d.get("text", "") for d in docs]
        self.docs = docs
        self.ids = [d.get("doc_id") for d in docs]
        if texts:
            self._matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        if self._matrix is None or not self.docs:
            return []
        qv = self.vectorizer.transform([query])
        scores = (self._matrix @ qv.T).toarray().ravel()
        idx = np.argsort(-scores)[:k]
        results = []
        for i in idx:
            results.append({"doc_id": self.ids[i], "passage_id": f"p{i}", "score": float(scores[i]), "text": self.docs[i].get("text", "")})
        return results
