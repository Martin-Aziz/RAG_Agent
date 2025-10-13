from core.agents.verifier import EmbeddingVerifier


class DummyEmbedder:
    def __init__(self):
        pass

    def embed(self, texts):
        # simple mapping: return vector whose first element is length of text
        out = []
        for t in texts:
            v = [len(t)] + [0.0] * 7
            out.append(v)
        return out


def test_embedding_verifier_pass():
    embedder = DummyEmbedder()
    verifier = EmbeddingVerifier(embedder=embedder, threshold=0.9)
    instruction = "Find director of Inception"
    passages = [{"text": "Inception was directed by Christopher Nolan."}, {"text": "Some other text."}]
    ok, info = verifier.grade(instruction, passages)
    assert isinstance(ok, bool)
    assert "best_score" in info


def test_embedding_verifier_fail():
    class LowEmbedder(DummyEmbedder):
        def embed(self, texts):
            # return zero vectors -> zero similarity
            return [[0.0] * 8 for _ in texts]

    embedder = LowEmbedder()
    verifier = EmbeddingVerifier(embedder=embedder, threshold=0.1)
    instruction = "Find director of Inception"
    passages = [{"text": "Some unrelated text."}]
    ok, info = verifier.grade(instruction, passages)
    assert ok is False
    assert info.get("best_score", None) == 0.0
