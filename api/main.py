from fastapi import FastAPI, HTTPException, Request, Response
from api.schemas import QueryRequest, QueryResponse
from core.orchestrator import Orchestrator
from core.observability import get_logger, new_request_id, redact_pii, REQUEST_COUNTER, LATENCY_HIST
import uvicorn
import time
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI(title="Agentic RAG MVP")
orch = None
logger = get_logger()

# mount static web UI
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=os.path.join(static_dir)), name="static")


@app.get("/")
async def root_html():
    idx = os.path.join(static_dir, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx, media_type="text/html")
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    # seed the knowledge base so retrievers have data available
    try:
        from seeds.seed_data import seed
        print("Seeding knowledge base at startup...")
        # run seed synchronously (we're in startup)
        seed()
    except Exception as e:
        print("Warning: seeding failed at startup:", e)

    # initialize orchestrator after seeding so it can load persisted FAISS if present
    global orch
    try:
        orch = Orchestrator()
        print("Orchestrator initialized")
    except Exception as e:
        print("Failed to initialize Orchestrator:", e)


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
        # lazy initialize orchestrator if startup event didn't run
        global orch
        if orch is None:
            try:
                from seeds.seed_data import seed
                seed()
            except Exception:
                pass
            orch = Orchestrator()
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
