from fastapi import FastAPI, HTTPException, Request, Response
from api.schemas import QueryRequest, QueryResponse
from core.orchestrator import Orchestrator
from core.observability import get_logger, new_request_id, redact_pii, REQUEST_COUNTER, LATENCY_HIST
import uvicorn
import time

app = FastAPI(title="Agentic RAG MVP")
orch = Orchestrator()
logger = get_logger()


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: Request):
    body = await req.json()
    # validate with pydantic schema
    try:
        q = QueryRequest(**body)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    request_id = new_request_id()
    mode = q.mode
    REQUEST_COUNTER.labels(mode=mode).inc()
    start = time.time()
    try:
        logger.info(f"handling request", extra={"request_id": request_id})
        result = await orch.handle_query(q)
        elapsed = time.time() - start
        LATENCY_HIST.labels(mode=mode).observe(elapsed)
        return result
    except Exception as e:
        logger.exception("error handling request", extra={"request_id": request_id})
        raise HTTPException(status_code=500, detail="internal error")


@app.get('/metrics')
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
