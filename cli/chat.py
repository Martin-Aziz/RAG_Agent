#!/usr/bin/env python3
"""Simple terminal chatbot that uses the Orchestrator to answer queries.

Usage: python cli/chat.py
"""
import argparse
import asyncio
import json
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


async def repl():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fallback", action="store_true", help="Fail if required dependencies are missing (no fallback).")
    parser.add_argument("--stream", action="store_true", help="Use streaming LLM output when available.")
    parser.add_argument("--add-doc", type=str, help="Add a custom document (text) to the knowledge base and persist it.")
    args, _ = parser.parse_known_args()

    if args.add_doc:
        os.makedirs("data", exist_ok=True)
        custom_path = os.path.join("data", "custom_docs.json")
        existing = []
        if os.path.exists(custom_path):
            try:
                with open(custom_path) as cf:
                    existing = json.load(cf)
            except Exception:
                existing = []
        existing.append({"doc_id": f"custom_{len(existing)}", "text": args.add_doc})
        with open(custom_path, "w") as cf:
            json.dump(existing, cf, indent=2)
        print("Added custom doc and re-seeding...")

    ensure_seeded()
    # import orchestrator lazily to avoid heavy imports during module import
    try:
        from core.orchestrator import Orchestrator
        orch = Orchestrator()
        use_fallback = False
    except Exception as e:
        if args.no_fallback:
            print("Error: required project dependencies are missing and --no-fallback was specified. Exiting.")
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
        if q.lower() in ("exit", "quit"):
            print("Goodbye")
            return

        # lightweight request object with required attributes
        req = type("Req", (), {"query": q, "mode": "parrag"})()
        try:
            if args.stream and hasattr(orch, "model") and hasattr(orch.model, "generate_answer_stream"):
                # stream tokens from the model and print as they arrive
                print("Bot:", end="\n")
                # handle both async generator and sync fallback
                try:
                    async for chunk in orch.model.generate_answer_stream(req.query, getattr(req, "evidence", [])):
                        print(chunk, end="", flush=True)
                    print("\n", end="")
                except TypeError:
                    # not async generator; fallback to single call
                    resp = await orch.handle_query(req)
                    print(resp.answer)
            else:
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
