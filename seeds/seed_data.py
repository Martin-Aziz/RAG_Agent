import json
from core.agents.retriever_vector import VectorRetriever
from core.agents.retriever_bm25 import BM25Retriever
from core.agents.hoprag_graph import HopRAG
import os
from core.agents.retriever_faiss import FAISSRetriever
from core.embedders.ollama_embedder import OllamaEmbedder
import os


def seed():
    here = os.path.join(os.path.dirname(__file__), "..")
    # allow richer dataset path via env var, else prefer examples/richer_dataset.json if present
    rich_path = os.getenv("RICH_DATA_PATH")
    if not rich_path:
        candidate = os.path.join(here, "examples", "richer_dataset.json")
        if os.path.exists(candidate):
            rich_path = candidate
        else:
            rich_path = os.path.join(here, "examples", "dataset_multi_hop.json")

    with open(rich_path) as f:
        data = json.load(f)
    docs = []
    # data may be list of entries each with a 'facts' list, or a flat list of docs
    if isinstance(data, list) and data and isinstance(data[0], dict) and "facts" in data[0]:
        for item in data:
            for fct in item.get("facts", []):
                docs.append({"doc_id": fct.get("doc_id"), "text": fct.get("text")})
    else:
        # assume it's a flat list of docs
        for d in data:
            docs.append({"doc_id": d.get("doc_id", d.get("id", "doc")), "text": d.get("text")})

    # also load any user-added custom docs persisted under data/custom_docs.json
    custom_path = os.path.join("data", "custom_docs.json")
    if os.path.exists(custom_path):
        try:
            with open(custom_path) as cf:
                cdocs = json.load(cf)
            for i, cd in enumerate(cdocs):
                docs.append({"doc_id": cd.get("doc_id", f"custom_{i}"), "text": cd.get("text")})
        except Exception:
            pass

    use_faiss = os.getenv("ENABLE_FAISS", "0") == "1"
    if use_faiss:
        # try to use Ollama embedder if configured; default to qwen3-embedding:latest
        embed_model = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:latest")
        if embed_model:
            embedder = OllamaEmbedder(model=embed_model)
            v = FAISSRetriever(embedder=embedder)
        else:
            v = FAISSRetriever()
    else:
        v = VectorRetriever()
    v.index(docs)
    b = BM25Retriever()
    b.index(docs)
    g = HopRAG(v, b)
    g.build_graph(docs)
    # optionally persist FAISS index
    if os.getenv("ENABLE_FAISS", "0") == "1":
        try:
            os.makedirs("data", exist_ok=True)
            index_path = os.path.join("data", "faiss_index.iv")
            mapping_path = os.path.join("data", "faiss_mapping.json")
            if hasattr(v, "save"):
                v.save(index_path, mapping_path)
                print(f"Saved FAISS index to {index_path} and mapping to {mapping_path}")
        except Exception as e:
            print("Warning: failed to persist FAISS index:", e)

    print("Seeded docs:", [d["doc_id"] for d in docs])


if __name__ == "__main__":
    seed()
