from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    user_id: str
    session_id: str
    query: str
    mode: str = Field("parrag", regex="^(parrag|hoprag|modular)$")
    context_ids: List[str] = []
    prefer_low_cost: bool = True


class EvidenceItem(BaseModel):
    doc_id: str
    passage_id: str
    score: float
    text: str


class AgentStep(BaseModel):
    step_id: str
    agent: str
    action: str
    result: Optional[dict]


class QueryResponse(BaseModel):
    answer: str
    evidence: List[EvidenceItem]
    trace: List[AgentStep]
    confidence: float
