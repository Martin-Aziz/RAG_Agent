from typing import List, Dict, Any, Optional
import os

try:
    import faiss
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    faiss = None
    np = None
    TfidfVectorizer = None


class FAISSRetriever:
    """FAISS adapter supporting an optional embedder.

    If an embedder is provided (object with .embed(list[str]) -> list[list[float]]),
    it will be used to produce embeddings for docs and queries. Otherwise a TF-IDF
    vectorizer is used as a fallback.
    """

    def __init__(self, embedder: Optional[Any] = None):
        if faiss is None:
            raise RuntimeError("faiss not available; install faiss-cpu to use FAISSRetriever")
        self.docs: List[Dict[str, Any]] = []
        self.ids: List[str] = []
        self.faiss_index = None
        self.embedder = embedder
        self.vectorizer = TfidfVectorizer() if TfidfVectorizer is not None else None

    def index(self, docs: List[Dict[str, Any]]):
        texts = [d.get("text", "") for d in docs]
        self.docs = docs
        self.ids = [d.get("doc_id") for d in docs]
        if not texts:
            return

        if self.embedder is not None:
            embs = self.embedder.embed(texts)
            arr = np.array(embs, dtype='float32')
        else:
            if self.vectorizer is None:
                raise RuntimeError("No vectorizer available to index texts")
            mat = self.vectorizer.fit_transform(texts)
            arr = mat.astype('float32').toarray()

        d = arr.shape[1]
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(np.array(arr))
        # keep original array for persistence if needed
        self._last_index_array = arr

    def save(self, index_path: str, mapping_path: str):
        """Persist FAISS index and document mapping to disk."""
        if self.faiss_index is None:
            raise RuntimeError("No FAISS index to save")
        # write faiss index
        faiss.write_index(self.faiss_index, index_path)
        # save ids and docs
        import json
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump({"ids": self.ids, "docs": self.docs}, f)

    def load(self, index_path: str, mapping_path: str):
        """Load FAISS index and mapping from disk."""
        import json
        if not (os.path.exists(index_path) and os.path.exists(mapping_path)):
            raise RuntimeError("Index or mapping file not found")
        self.faiss_index = faiss.read_index(index_path)
        with open(mapping_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        self.ids = obj.get("ids", [])
        self.docs = obj.get("docs", [])

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.faiss_index is None:
            return []

        if self.embedder is not None:
            qv = np.array(self.embedder.embed([query]), dtype='float32')
        else:
            if self.vectorizer is None:
                return []
            qv = self.vectorizer.transform([query]).astype('float32').toarray()

        D, I = self.faiss_index.search(qv, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({"doc_id": self.ids[int(idx)], "passage_id": f"p{idx}", "score": float(score), "text": self.docs[int(idx)].get("text", "")})
        return results
