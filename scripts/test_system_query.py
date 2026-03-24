# scripts/test_system_query.py
import sys, os
sys.path.insert(0, ".")

from src.retrieval import retrieve
from src.generation import generate_answer

query = "抑郁症的诊断标准是什么？"
print(f"提问: {query}\n")

chunks = retrieve(query, top_k=2)
for i, c in enumerate(chunks):
    page = c.metadata.get("page", "?")
    print(f"[来源: 第 {page} 页] {c.text[:120]}...")

answer = generate_answer(query, chunks)
print("\n🤖 LLM 回答:")
print(answer)