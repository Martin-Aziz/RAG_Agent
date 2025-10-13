from core.agents.retriever_vector import VectorRetriever
from core.agents.retriever_bm25 import BM25Retriever


def test_vector_and_bm25_basic():
    docs = [{"doc_id": "d1", "text": "Alice loves music and plays piano."}, {"doc_id": "d2", "text": "Bob is a filmmaker who directed Inception."}]
    v = VectorRetriever()
    v.index(docs)
    res = v.retrieve("director of Inception", k=1)
    assert isinstance(res, list)

    b = BM25Retriever()
    b.index(docs)
    res2 = b.retrieve("who directed Inception", k=1)
    assert isinstance(res2, list)
