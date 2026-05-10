from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import retrieval
from app.agent import run_agent
from app.models import ChatRequest, ChatResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load catalog and build FAISS index at startup."""
    print("[startup] Loading catalog and building index...")
    retrieval.startup()
    print("[startup] Ready.")
    yield


app = FastAPI(title="SHL Assessment Recommender", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    if not request.messages:
        raise HTTPException(status_code=422, detail="messages cannot be empty")

    # Hard cap: evaluator sends max 8 turns, just be safe
    if len(request.messages) > 20:
        raise HTTPException(status_code=422, detail="too many messages")

    return run_agent(request.messages)
