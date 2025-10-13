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

from seeds.seed_data import seed
from core.orchestrator import Orchestrator
from api.schemas import QueryRequest


def ensure_seeded():
    # call the seed script to create a small dummy knowledge base
    try:
        seed()
    except Exception as e:
        print("Warning: seeding failed:", e)


async def repl():
    ensure_seeded()
    orch = Orchestrator()
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

        req = QueryRequest(query=q, mode="parrag")
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
