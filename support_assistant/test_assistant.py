import json
from graph import app_graph

# Test 1: Policy Question (should trigger retrieval)
q1 = {"query": "What is the delivery fee for orders below 149?"}
res1 = app_graph.invoke(q1)["response"].model_dump()

# Test 2: General Question (should NOT trigger retrieval)
q2 = {"query": "What is the capital of France?"}
res2 = app_graph.invoke(q2)["response"].model_dump()

print("=== EXAMPLE 1: POLICY QUERY (RETRIEVAL TRIGGERED) ===")
print(json.dumps(res1, indent=2))

print("\n=== EXAMPLE 2: GENERAL QUERY (DIRECT ANSWER) ===")
print(json.dumps(res2, indent=2))
