import os
import sys
from fastapi import FastAPI
from contextlib import asynccontextmanager

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from schemas import QueryRequest, PolicyResponse
from graph import app_graph, get_or_create_collection

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure vectors are initialized upon startup
    get_or_create_collection()
    yield

app = FastAPI(
    title="Zepto Customer Support Assistant",
    description="Grounded GenAI Policy Assistant using LangGraph and ChromaDB",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
def root():
    return {"status": "ok", "service": "zepto-support-assistant"}

@app.post("/chat", response_model=PolicyResponse)
def chat_endpoint(payload: QueryRequest):
    initial_state = {
        "query": payload.query,
        "intent": "",
        "retrieved_chunks": [],
        "retrieved_ids": [],
        "response": None
    }
    result = app_graph.invoke(initial_state)
    return result["response"]
