from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import retrieval
from app.agent import run_agent
from app.models import ChatRequest, ChatResponse


tags_metadata = [
    {
        "name": "Recommendations",
        "description": "Conversational assessment recommendation endpoints.",
    },
    {
        "name": "Health",
        "description": "Service monitoring and availability.",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize retrieval pipeline at startup."""
    print("[startup] Loading assessment catalog...")
    retrieval.startup()
    print("[startup] API ready.")
    yield


app = FastAPI(
    title="SHL Assessment Recommendation Agent",
    description="""
Production-ready conversational recommendation API developed for the SHL GenAI Challenge.

### Features
- Conversational assessment recommendation
- BM25 retrieval engine
- Multi-turn conversational memory
- Deterministic grounding validation
- RESTful API with OpenAPI documentation

Built using FastAPI and designed for scalable deployment.
""",
    version="1.0.0",
    lifespan=lifespan,
    openapi_tags=tags_metadata,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", summary="API Information")
def root():
    return {
        "name": "SHL Assessment Recommendation Agent",
        "version": "1.0.0",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "chat": "/chat",
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    description="Returns the operational status of the API.",
)
def health():
    return {"status": "ok"}


@app.post(
    "/chat",
    tags=["Recommendations"],
    summary="Recommend SHL Assessments",
    description="""
Returns relevant SHL assessments based on a user's hiring requirements
through a conversational retrieval pipeline.
""",
    response_model=ChatResponse,
)
async def chat_endpoint(request: ChatRequest):

    if not request.messages:
        raise HTTPException(
            status_code=422,
            detail="messages cannot be empty",
        )

    if len(request.messages) > 20:
        raise HTTPException(
            status_code=422,
            detail="too many messages",
        )

    return run_agent(request.messages)