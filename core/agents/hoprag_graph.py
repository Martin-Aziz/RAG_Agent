from typing import List, Dict, Any, Tuple
import networkx as nx


class HopRAG:
    def __init__(self, vector_retriever, bm25_retriever):
        self.vector = vector_retriever
        self.bm25 = bm25_retriever
        self.graph = nx.DiGraph()

    def build_graph(self, docs: List[Dict[str, Any]]):
        # nodes are doc_id:passage index
        for d in docs:
            nid = d.get("doc_id")
            self.graph.add_node(nid, text=d.get("text", ""))
        # naive edges by shared words
        nodes = list(self.graph.nodes())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                a = nodes[i]
                b = nodes[j]
                ta = self.graph.nodes[a].get("text", "")
                tb = self.graph.nodes[b].get("text", "")
                if set(ta.split()).intersection(tb.split()):
                    self.graph.add_edge(a, b)
                    self.graph.add_edge(b, a)

    async def traverse(self, seed_query: str, max_hops: int = 2) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # initial seeds from vector retriever
        seeds = self.vector.retrieve(seed_query, k=2)
        trace = []
        collected = []
        for s in seeds:
            nid = s.get("doc_id")
            collected.append({"doc_id": nid, "passage_id": s.get("passage_id"), "score": s.get("score"), "text": s.get("text")})
            trace.append({"step_id": f"seed-{nid}", "agent": "hoprag", "action": "seed", "result": s})

        # BFS expand
        fringe = [c["doc_id"] for c in collected]
        visited = set(fringe)
        hops = 0
        while hops < max_hops:
            next_fringe = []
            for n in fringe:
                if not self.graph.has_node(n):
                    continue
                for nb in self.graph.neighbors(n):
                    if nb in visited:
                        continue
                    visited.add(nb)
                    text = self.graph.nodes[nb].get("text", "")
                    collected.append({"doc_id": nb, "passage_id": f"p-{nb}", "score": 0.5, "text": text})
                    trace.append({"step_id": f"expand-{n}-{nb}", "agent": "hoprag", "action": "expand", "result": {"from": n, "to": nb}})
                    next_fringe.append(nb)
            fringe = next_fringe
            hops += 1

        return collected, trace
