import os
import sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
import chromadb
from chromadb.utils import embedding_functions

# Ensure support_assistant directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from schemas import PolicyResponse

MOCK_LLM = os.getenv("MOCK_LLM", "1") == "1"
POLICY_KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

class AgentState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[str]
    retrieved_ids: List[str]
    response: PolicyResponse

def get_or_create_collection():
    db_path = os.path.join(BASE_DIR, "chroma_db")
    docs_dir = os.path.join(BASE_DIR, "docs")
    client = chromadb.PersistentClient(path=db_path)
    embed_fn = embedding_functions.DefaultEmbeddingFunction()
    col = client.get_or_create_collection(name="zepto_policies", embedding_function=embed_fn)

    if col.count() == 0:
        docs, metas, ids = [], [], []
        for filename in sorted(os.listdir(docs_dir)):
            if filename.endswith(".txt"):
                path = os.path.join(docs_dir, filename)
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                doc_id = os.path.splitext(filename)[0]
                docs.append(content)
                metas.append({"source": doc_id})
                ids.append(doc_id)
        if docs:
            col.add(documents=docs, metadatas=metas, ids=ids)
    return col

def classify_intent(state: AgentState) -> AgentState:
    query = state["query"].lower()
    matched = any(kw in query for kw in POLICY_KEYWORDS)
    intent = "policy_question" if matched else "general_question"
    return {"intent": intent}

def retrieve_and_answer(state: AgentState) -> AgentState:
    col = get_or_create_collection()
    results = col.query(query_texts=[state["query"]], n_results=3)
    chunks = results["documents"][0]
    doc_ids = results["ids"][0]

    top_snippet = chunks[0][:200]
    answer_text = f"Based on the retrieved context: {top_snippet}"
    resp = PolicyResponse(answer=answer_text, sources=doc_ids, confidence=1.0)
    return {"retrieved_chunks": chunks, "retrieved_ids": doc_ids, "response": resp}

def direct_answer(state: AgentState) -> AgentState:
    resp = PolicyResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )
    return {"response": resp}

def route_intent(state: AgentState) -> str:
    return "retrieve_and_answer" if state["intent"] == "policy_question" else "direct_answer"

workflow = StateGraph(AgentState)
workflow.add_node("classify_intent", classify_intent)
workflow.add_node("retrieve_and_answer", retrieve_and_answer)
workflow.add_node("direct_answer", direct_answer)

workflow.set_entry_point("classify_intent")
workflow.add_conditional_edges("classify_intent", route_intent, {
    "retrieve_and_answer": "retrieve_and_answer",
    "direct_answer": "direct_answer"
})
workflow.add_edge("retrieve_and_answer", END)
workflow.add_edge("direct_answer", END)

app_graph = workflow.compile()
