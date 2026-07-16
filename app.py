"""
TaxPayBuddy API server.

Place this file at the ROOT of the TaxPayBuddy project (same level as
the `src/` folder and `requirements.txt`), then run it from that folder:

    uvicorn app:app --reload --port 8000

This exposes the existing multi-agent RAG pipeline (RouterAgent) over a
simple HTTP endpoint so the React frontend can talk to it.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.framework.database.chroma_store import ChromaStore
from src.framework.llm.gemini_client import GeminiClient
from src.agents.router_agent.router_main import RouterAgent

# Holds the single shared RouterAgent instance for the app's lifetime.
# Building the LLM client + vector store once at startup, instead of per
# request, avoids re-opening the ChromaDB connection on every question.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = GeminiClient()
    vector_store = ChromaStore()
    state["router"] = RouterAgent(llm=llm, vector_store=vector_store)
    yield
    state.clear()


app = FastAPI(title="TaxPayBuddy API", lifespan=lifespan)

# Allow the local Vite dev server (and a same-origin production build)
# to call this API. Add your deployed frontend's URL here too if you
# host it somewhere other than localhost.
# Allow the local Vite dev server, plus whatever production frontend
# domain(s) you set in the ALLOWED_ORIGINS env var (comma-separated),
# e.g. ALLOWED_ORIGINS=https://taxpaybuddy.com,https://www.taxpaybuddy.com
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class SourceChunk(BaseModel):
    source: str
    text: str


class ChatResponse(BaseModel):
    answer: str
    routed_agent: str
    sources: list[SourceChunk]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    router: RouterAgent = state["router"]
    result = router.route_and_execute(message)

    sources = [
        SourceChunk(source=chunk.source, text=chunk.text[:300])
        for chunk in result.retrieved_chunks
    ]

    return ChatResponse(
        answer=result.answer,
        routed_agent=router.last_routed_label or "general_fallback",
        sources=sources,
    )
