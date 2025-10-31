from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from api.schemas import QueryRequest, QueryResponse
from api.middleware import RateLimitMiddleware
from core.orchestrator import Orchestrator
from core.observability import get_logger, new_request_id, redact_pii, REQUEST_COUNTER, LATENCY_HIST
from core.exceptions import RAGAgentException, ValidationException
from core.validation import QueryValidator, sanitize_dict_for_logging
import uvicorn
import time
import traceback
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
import os

app = FastAPI(
    title="Agentic RAG MVP",
    description="Advanced Retrieval-Augmented Generation System",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Add rate limiting (configurable via environment)
rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
if rate_limit_enabled:
    requests_per_minute = int(os.getenv("RATE_LIMIT_RPM", "60"))
    burst_size = int(os.getenv("RATE_LIMIT_BURST", "10"))
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=requests_per_minute,
        burst_size=burst_size,
        by_ip=True
    )

orch = None
logger = get_logger()
query_validator = QueryValidator(max_length=2000, enable_xss_protection=True)

# Custom exception handler for RAG exceptions
@app.exception_handler(RAGAgentException)
async def rag_exception_handler(request: Request, exc: RAGAgentException):
    """Handle custom RAG exceptions with structured responses."""
    logger.error(
        f"RAG error: {exc.message}",
        extra={"error_code": exc.error_code, "details": exc.details}
    )
    return JSONResponse(
        status_code=400,
        content=exc.to_dict()
    )


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    request_id = request.headers.get("X-Request-ID", "unknown")
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "request_id": request_id,
            "path": request.url.path,
            "traceback": traceback.format_exc()
        }
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "request_id": request_id
        }
    )


# mount static web UI once so assets are always available
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "web")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def root_html():
    idx = os.path.join(static_dir, "index.html")
    if os.path.exists(idx):
        return FileResponse(idx, media_type="text/html")
    return {"status": "ok"}


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    # Return an empty favicon to avoid 404s when browsers request it.
    return Response(content=b"", media_type="image/x-icon")


@app.get("/styles.css", include_in_schema=False)
async def styles_redirect():
    # Redirect old cached requests to the /static mount
    return RedirectResponse(url="/static/styles.css", status_code=301)


@app.get("/app.js", include_in_schema=False)
async def app_js_redirect():
    # Redirect old cached requests to the /static mount
    return RedirectResponse(url="/static/app.js", status_code=301)


@app.get("/health")
async def health():
    """Comprehensive health check endpoint."""
    health_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "components": {}
    }
    
    # Check orchestrator
    try:
        global orch
        if orch is None:
            health_status["components"]["orchestrator"] = {
                "status": "not_initialized",
                "healthy": False
            }
            health_status["status"] = "degraded"
        else:
            health_status["components"]["orchestrator"] = {
                "status": "ready",
                "healthy": True,
                "features": {
                    "advanced_rag": orch.use_advanced if hasattr(orch, 'use_advanced') else False,
                    "faiss_enabled": isinstance(orch.vector, type) and 'FAISS' in str(type(orch.vector))
                }
            }
    except Exception as e:
        health_status["components"]["orchestrator"] = {
            "status": "error",
            "healthy": False,
            "error": str(e)
        }
        health_status["status"] = "unhealthy"
    
    # Check data directory
    try:
        data_dir = "data"
        if os.path.exists(data_dir):
            docs_path = os.path.join(data_dir, "docs.json")
            health_status["components"]["data"] = {
                "status": "ready",
                "healthy": True,
                "docs_exist": os.path.exists(docs_path)
            }
        else:
            health_status["components"]["data"] = {
                "status": "missing",
                "healthy": False
            }
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["data"] = {
            "status": "error",
            "healthy": False,
            "error": str(e)
        }
    
    # Overall health
    if health_status["status"] == "unhealthy":
        return JSONResponse(status_code=503, content=health_status)
    elif health_status["status"] == "degraded":
        return JSONResponse(status_code=200, content=health_status)
    else:
        return health_status


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
    """Handle query requests with comprehensive validation and error handling."""
    request_id = new_request_id()
    start = time.time()
    
    try:
        body = await req.json()
    except Exception as e:
        logger.error(f"Failed to parse JSON body", extra={"error": str(e), "request_id": request_id})
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    
    # Validate with pydantic schema
    try:
        q = QueryRequest(**body)
    except Exception as e:
        logger.error(f"Request validation failed", extra={"error": str(e), "request_id": request_id})
        raise HTTPException(status_code=400, detail=f"Invalid request: {str(e)}")
    
    # Validate query content
    try:
        query_validator.validate_or_raise(q.query)
    except ValidationException as e:
        logger.warning(
            f"Query validation failed",
            extra={"error": e.message, "request_id": request_id}
        )
        raise HTTPException(status_code=400, detail=e.message)
    
    # Sanitize query
    original_query = q.query
    q.query = query_validator.sanitize(q.query)
    
    if original_query != q.query:
        logger.info(
            f"Query sanitized",
            extra={
                "request_id": request_id,
                "original_length": len(original_query),
                "sanitized_length": len(q.query)
            }
        )
    
    mode = q.mode
    REQUEST_COUNTER.labels(mode=mode).inc()
    
    try:
        logger.info(
            f"Processing query",
            extra={
                "request_id": request_id,
                "mode": mode,
                "query_length": len(q.query),
                "user_id": q.user_id,
                "session_id": q.session_id
            }
        )
        
        # Lazy initialize orchestrator if startup event didn't run
        global orch
        if orch is None:
            logger.warning("Orchestrator not initialized, initializing now")
            try:
                from seeds.seed_data import seed
                seed()
            except Exception as e:
                logger.warning(f"Seeding failed: {e}")
            orch = Orchestrator()
        
        result = await orch.handle_query(q)
        
        elapsed = time.time() - start
        LATENCY_HIST.labels(mode=mode).observe(elapsed)
        
        logger.info(
            f"Query completed successfully",
            extra={
                "request_id": request_id,
                "elapsed": elapsed,
                "evidence_count": len(result.evidence),
                "confidence": result.confidence
            }
        )
        
        return result
        
    except RAGAgentException as e:
        # Re-raise RAG exceptions to be handled by exception handler
        raise
    except Exception as e:
        logger.exception(
            "Error handling request",
            extra={
                "request_id": request_id,
                "error": str(e),
                "traceback": traceback.format_exc()
            }
        )
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get('/metrics')
async def metrics():
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
