from typing import List, Dict, Any, Tuple, Optional


class Verifier:
    def __init__(self):
        pass

    def grade(self, instruction: str, passages: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        # simple rule: pass if any passage contains at least 2 words in common with instruction
        inst_words = set(instruction.lower().split())
        best = 0
        best_pass = None
        for p in passages:
            tw = set(p.get("text", "").lower().split())
            common = inst_words.intersection(tw)
            if len(common) > best:
                best = len(common)
                best_pass = p
        ok = best >= 2
        return ok, {"best_common": best, "best_passage": best_pass}


class EmbeddingVerifier:
    """Verifier that uses an embedder to compute cosine similarity between the
    instruction and candidate passages. Returns pass if max similarity >= threshold.
    """

    def __init__(self, embedder, threshold: Optional[float] = 0.7):
        self.embedder = embedder
        # if threshold is None, compute dynamically per-instruction in grade()
        self.threshold = threshold

    def _cosine(self, a, b) -> float:
        import math
        # assume a and b are lists or numpy arrays
        try:
            import numpy as _np
            a = _np.asarray(a, dtype=float)
            b = _np.asarray(b, dtype=float)
            na = _np.linalg.norm(a)
            nb = _np.linalg.norm(b)
            if na == 0 or nb == 0:
                return 0.0
            return float((_np.dot(a, b) / (na * nb)).item())
        except Exception:
            # fallback pure python
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            if na == 0 or nb == 0:
                return 0.0
            return dot / (na * nb)

    def grade(self, instruction: str, passages: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        if not passages:
            return False, {"reason": "no_passages"}
        # compute embedding for instruction and passages
        try:
            inst_emb = self.embedder.embed([instruction])[0]
            texts = [p.get("text", "") for p in passages]
            pass_embs = self.embedder.embed(texts)
        except Exception as e:
            return False, {"reason": "embed_error", "error": str(e)}

        best_score = -1.0
        best_idx = -1
        for i, pe in enumerate(pass_embs):
            try:
                score = self._cosine(inst_emb, pe)
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best_idx = i

        # determine threshold: if self.threshold is None, compute a dynamic value
        if self.threshold is None:
            # use instruction length (words) to scale threshold modestly
            n_words = len(instruction.split()) if instruction else 0
            # base 0.45, add 0.005 per word, clamp between 0.5 and 0.9
            dyn = min(0.9, max(0.5, 0.45 + 0.005 * float(n_words)))
            threshold = dyn
        else:
            threshold = self.threshold

        ok = best_score >= threshold
        best_pass = passages[best_idx] if best_idx >= 0 else None
        return ok, {"best_score": best_score, "best_passage": best_pass, "threshold": threshold}

