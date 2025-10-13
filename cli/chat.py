#!/usr/bin/env python3
"""Simple terminal chatbot that uses the Orchestrator to answer queries.

Usage: python cli/chat.py
"""
import asyncio
import os
import sys

# ensure project root is on PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import subprocess
# avoid importing pydantic models to keep the CLI lightweight; we'll pass a simple object


def ensure_seeded():
    # call the seed script to create a small dummy knowledge base
    try:
        # run seed in a subprocess so we don't import heavy deps (sklearn/faiss) at module import
        PY = sys.executable
        cmd = [PY, "-c", "from seeds.seed_data import seed; seed()"]
        env = os.environ.copy()
        # ensure project root is on PYTHONPATH for the subprocess
        env["PYTHONPATH"] = ROOT + (":" + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else "")
        subprocess.run(cmd, env=env, check=True)
    except Exception as e:
        print("Warning: seeding failed:", e)


async def repl(no_fallback: bool = False):
    ensure_seeded()
    # import orchestrator lazily to avoid heavy imports during module import
    try:
        from core.orchestrator import Orchestrator
        orch = Orchestrator()
        use_fallback = False
    except Exception as e:
        if no_fallback:
            raise
        print("Note: full project orchestrator unavailable (missing deps). Using lightweight fallback chat engine.")
        use_fallback = True

    # fallback simple in-memory engine (no external deps)
    class SimpleEngine:
        def __init__(self):
            # some dummy knowledge
            self.docs = [
                {"doc_id": "d1", "text": "Inception was directed by Christopher Nolan."},
                {"doc_id": "d2", "text": "The movie Inception received awards including Best Cinematography."},
                {"doc_id": "d3", "text": "Christopher Nolan also directed Memento and The Dark Knight."},
                {"doc_id": "d4", "text": "The cinematographer for Inception was Wally Pfister."},
                {"doc_id": "d5", "text": "Dunkirk won several technical awards in 2018."},
            ]

        async def handle_query(self, req):
            q = req.query.lower()
            q_words = set([w.strip(".,?!)('\"") for w in q.split() if w])
            scored = []
            for d in self.docs:
                tw = set([w.strip(".,?!)('\"") for w in d["text"].lower().split() if w])
                common = q_words.intersection(tw)
                score = len(common)
                if score > 0:
                    scored.append((score, d))
            scored.sort(key=lambda x: x[0], reverse=True)
            if not scored:
                return type("Resp", (), {"answer": "I don't know based on the available dummy knowledge.", "evidence": []})()
            top = [d for s, d in scored[:3]]
            answer_lines = ["Based on the knowledge:"]
            for t in top:
                answer_lines.append(t["text"])
            ans = "\n".join(answer_lines)
            # create evidence-like objects
            evidence = [type("E", (), {"doc_id": d["doc_id"], "text": d["text"], "score": 1.0})() for d in top]
            return type("Resp", (), {"answer": ans, "evidence": evidence})()

    if use_fallback:
        orch = SimpleEngine()
    print("Welcome to PAR-RAG chatbot (type 'exit' or Ctrl-C to quit)")
    while True:
        try:
            q = input("You: ")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye")
            return
        q = q.strip()
        if not q:
            continue
        # support CLI command to add a document: add:doc_id|text
        if q.lower().startswith("add:"):
            try:
                parts = q[len("add:"):].split("|", 1)
                doc_id = parts[0].strip()
                text = parts[1].strip()
                os.makedirs("data", exist_ok=True)
                path = os.path.join("data", "cli_docs.json")
                existing = []
                if os.path.exists(path):
                    import json
                    with open(path, "r", encoding="utf-8") as cf:
                        existing = json.load(cf)
                existing.append({"doc_id": doc_id, "text": text})
                with open(path, "w", encoding="utf-8") as cf:
                    json.dump(existing, cf, ensure_ascii=False, indent=2)
                print(f"Added doc {doc_id}")
            except Exception as e:
                print("Failed to add doc. Use: add:doc_id|text", e)
            continue
        if q.lower() in ("exit", "quit"):
            print("Goodbye")
            return

        # lightweight request object with required attributes
        req = type("Req", (), {"query": q, "mode": "parrag"})()
        try:
            resp = await orch.handle_query(req)
            print("Bot:")
            print(resp.answer)
            if resp.evidence:
                print("\nEvidence:")
                for e in resp.evidence:
                    print(f"- [{e.doc_id}] {e.text} (score={e.score})")
        except Exception as e:
            print("Error answering query:", e)


def main():
    asyncio.run(repl())


if __name__ == "__main__":
    main()
