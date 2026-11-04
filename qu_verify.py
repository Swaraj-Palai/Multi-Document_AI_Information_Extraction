import sys, os
sys.path.insert(0, r'c:\Users\swara\Downloads\pdf-information-extractor-main')
os.environ.setdefault("GROQ_API_KEY", "test-key")
os.environ.setdefault("LLM_MODEL", "openai/gpt-oss-120b")
os.environ.setdefault("QUERY_UNDERSTANDING_ENABLED", "true")
os.environ.setdefault("QUERY_UNDERSTANDING_USE_LLM", "false")
from app.componenets.query_understanding import understand_query, get_query_understanding_summary
tests = [
    "Hello, how are you?",
    "What is machine learning?",
    "Compare drip irrigation vs flood irrigation",
    "List the steps to prevent overfitting",
    "How do I calculate the total cost?",
    "What happened in 2023?",
    "Tell me about this app capabilities",
    "By how much does drip irrigation reduce evaporation?",
]
print("=== QUERY UNDERSTANDING TESTS ===")
for q in tests:
    u = understand_query(q)
    print(f"Q: {q}")
    print(f"   intent={u.intent} conf={u.confidence:.2f} complexity={u.complexity} top_k={u.suggested_top_k} docs={u.requires_documents} rewrite={u.needs_query_rewrite}")
    print(f"   entities={u.entities} keywords={u.keywords[:5]} constraints={u.constraints}")
print("\n=== RETRIEVER INTEGRATION TESTS ===")
from app.componenets import retriver as R
class FakeMsg:
    def __init__(self, c): self.content = c
class FakeLLM:
    def __init__(self, tag): self.tag = tag
    def invoke(self, prompt):
        return FakeMsg(f"[ANS:{self.tag}]")
R.load_llm = lambda *a, **k: FakeLLM("LLM1")
R.load_vector_store = lambda: None
res = R.query_rag("Hello, how are you?")
print(f"Case A (greeting): mode={res['mode']} docs_req={res['query_understanding']['requires_documents']}")
assert res["mode"] == "general"
assert res["query_understanding"]["requires_documents"] == False
print("  OK")
res = R.query_rag("What is the capital of France?")
print(f"Case B (factual, no docs): mode={res['mode']}")
assert res["mode"] == "general"
print("  OK")
class FakeDb:
    def similarity_search(self, q, k=4): return []
R.load_vector_store = lambda: FakeDb()
res = R.query_rag("By how much does drip irrigation reduce evaporation?")
print(f"Case C (docs, no match): mode={res['mode']} answer={res['answer'][:40]}")
assert res["mode"] == "rag"
assert "not found" in res["answer"].lower()
print("  OK")
res = R.query_rag("   ")
print(f"Case D (empty): mode={res['mode']}")
assert res["mode"] == "rag"
print("  OK")
print("\n=== ALL TESTS PASSED ===")
