# Zepto Data & AI Platform

An end-to-end AI/ML and GenAI platform engineered for Zepto's analytics and support guild. This repository implements three cohesive modules:
1. `/data_pipeline`: Scrapes raw catalog pricing from `books.toscrape.com`, applies a fixed-rate currency conversion (1 GBP = 105.50 INR), loads a normalized two-table SQLite database, and verifies SQL vs. Pandas join equivalence.
2. `/analytics`: Comprehensive EDA, missingness handling, outlier detection, and predictive modeling on the Titanic passenger dataset with leakage-free preprocessing, hyperparameter-tuned classifiers, and a saved production pipeline.
3. `/support_assistant`: Grounded policy support assistant leveraging LangGraph, ChromaDB vector retrieval, Pydantic schema validation, and a deterministic offline mock baseline (`MOCK_LLM=1`) wrapped in FastAPI and Docker.
zepto-data-ai-platform/
├── README.md
├── requirements.txt
├── data_pipeline/
│   ├── README.md
│   ├── pipeline.py
│   ├── query_benchmark.py
│   └── zepto_catalog.db
├── analytics/
│   ├── README.md
│   ├── titanic.csv
│   ├── run_analytics.py
│   ├── test_inference.py
│   └── best_classifier_pipeline.joblib
└── support_assistant/
    ├── README.md
    ├── Dockerfile
    ├── docs/ (doc_01.txt ... doc_08.txt)
    ├── ingest.py
    ├── graph.py
    ├── schemas.py
    ├── main.py
    └── test_assistant.py
Installation & Setup
Install all dependencies using the consolidated requirements.txt:
pip install -r requirements.txt
Execution Guide
1. Data Pipeline
python data_pipeline/pipeline.py
python data_pipeline/query_benchmark.py
2. Analytics Pipeline
python analytics/run_analytics.py
python analytics/test_inference.py
3. Support Assistant
# Test LangGraph routing & schemas locally
python support_assistant/test_assistant.py

# Serve FastAPI app with Uvicorn
uvicorn support_assistant.main:app --host 127.0.0.1 --port 7860
Git Workflow Compliance
A feature branch (feature/data-pipeline) was branched from main, received multiple commits for the scraper and query benchmarks, and was merged back into main using --no-ff (verifiable via git log --graph --oneline --all).