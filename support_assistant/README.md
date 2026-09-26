# Module 3 — Support Assistant (/support_assistant)

## Overview
A grounded GenAI support service built on Zepto's policy documentation. Employs LangGraph for state routing, ChromaDB for vector retrieval with `all-MiniLM-L6-v2` embeddings, and a deterministic offline mock baseline (`MOCK_LLM=1`) requiring no external API keys or paid tiers.

## RAG Pipeline Architecture Walkthrough
1. **Ingestion (`ingest.py` / `graph.py`)**: Reads the 8 canonical policy documents from `support_assistant/docs/` and generates embeddings locally using `all-MiniLM-L6-v2`. Vectors and text chunks are committed to a persistent ChromaDB store in `support_assistant/chroma_db/`.
2. **Intent Classification (`classify_intent`)**:
   - `MOCK_LLM=1` (Graded Baseline): Evaluates query text against defined policy domain keywords (`delivery`, `return`, `refund`, `membership`, `tracking`, `cancel`, `gift card`, `support hours`). Routes matching queries to `policy_question` and non-matching queries to `general_question`.
   - `MOCK_LLM=0` (Optional Real LLM): Uses a structured prompt on Groq API to classify intent.
3. **Retrieval (`retrieve_and_answer`)**:
   - Queries ChromaDB using cosine distance to retrieve the top 3 most relevant policy chunks and their source IDs (`doc_01` to `doc_08`). Retrieval executes live and locally in both mock and real modes.
4. **Generation & Schema Validation**:
   - `MOCK_LLM=1`: Generates a deterministic response adhering to `Based on the retrieved context: {top_chunk_snippet}` with `confidence=1.0`.
   - `MOCK_LLM=0`: Prompts the external model with our negative-constraint structured template, validating output against `PolicyResponse`.
   - `direct_answer`: Handles general queries with a fixed fallback string and empty sources list.

## Structured Prompt Template (for MOCK_LLM=0)
- **Role**: Zepto Official Support Specialist.
- **Context**: `{context}` (retrieved ChromaDB chunks).
- **Task**: Answer the customer inquiry using strictly the provided context.
- **Format**: Valid JSON matching `PolicyResponse(answer=..., sources=[...], confidence=...)`.
- **Negative Constraint**: Do not extrapolate, infer, or provide information absent from the supplied context.
- **Few-Shot Example**: Includes sample Q&A grounded in customer support hours (`doc_08`).

## Verified Endpoint Responses (Mock Mode Baseline)

### Call 1: Policy Question (Retrieval Triggered)
**Query**: `"What is the delivery fee for orders below 149?"`
```json
{
  "answer": "Based on the retrieved context: Zepto delivers grocery and household essentials to serviceable pin codes within 10 to 30 minutes of order confirmation, depending on the customer's delivery zone and current order volume. Standard de",
  "sources": [
    "doc_01",
    "doc_05",
    "doc_03"
  ],
  "confidence": 1.0
}
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
Containerization
Configured in Dockerfile exposing port 7860.

Build locally: docker build -t zepto-support-assistant -f support_assistant/Dockerfile .

Run locally: docker run -p 7860:7860 -e MOCK_LLM=1 zepto-support-assistant