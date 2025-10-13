import pytest
import os
from core.orchestrator import Orchestrator
from api.schemas import QueryRequest


@pytest.mark.asyncio
async def test_parrag_plan_and_execution(tmp_path):
    orch = Orchestrator()
    # seed docs
    from seeds.seed_data import seed
    seed()
    req = QueryRequest(user_id="u1", session_id="s1", query="Which movie directed by the director of Inception won an Oscar for Best Cinematography?", mode="parrag")
    resp = await orch.handle_query(req)
    assert resp is not None
    assert len(resp.trace) >= 2


def test_tool_registry_and_calculator():
    orch = Orchestrator()
    res = orch.tools.call("calculator", {"expression": "1+2*3"})
    assert res.get("value") == 7.0


def test_memory_store_and_recall():
    orch = Orchestrator()
    orch.memory.memorize("pref", {"likes": "jazz"}, to_semantic=True)
    out = orch.memory.recall("jazz")
    assert isinstance(out, list)
