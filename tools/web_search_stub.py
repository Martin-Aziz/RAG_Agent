from pydantic import BaseModel
from typing import Dict


class WebSearchArgs(BaseModel):
    query: str


class WebSearchResult(BaseModel):
    query: str
    results: list


class WebSearch:
    @staticmethod
    def spec():
        return type("ToolSpec", (), {"name": "web_search", "arg_model": WebSearchArgs, "result_model": WebSearchResult, "executor": WebSearch.execute})

    @staticmethod
    def execute(args: Dict):
        q = args.get("query")
        # deterministic fake results
        return {"query": q, "results": [{"title": "stub", "snippet": f"Result for {q}"}]}
