import json
from api.schemas import QueryRequest
from core.orchestrator import Orchestrator
import argparse


def run(dataset_path: str):
    with open(dataset_path) as f:
        data = json.load(f)
    orch = Orchestrator()
    results = []
    for item in data:
        req = QueryRequest(user_id="eval", session_id=item.get("id"), query=item.get("query"), mode="parrag")
        resp = orch.handle_query(req)
        # if coroutine
        if hasattr(resp, "__await__"):
            import asyncio
            resp = asyncio.get_event_loop().run_until_complete(resp)
        results.append({"id": item.get("id"), "answer": resp.answer, "evidence_count": len(resp.evidence)})
    out = {"results": results}
    print(json.dumps(out, indent=2))
    with open("evaluation_report.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="examples/dataset_multi_hop.json")
    args = parser.parse_args()
    run(args.dataset)
