from typing import Literal


class Router:
    def __init__(self):
        pass

    def route(self, instruction: str) -> Literal["vector", "bm25"]:
        # naive heuristic: if question contains 'which' or 'who' use bm25, else vector
        q = instruction.lower()
        if any(w in q for w in ["which", "who", "when", "where"]):
            return "bm25"
        return "vector"
