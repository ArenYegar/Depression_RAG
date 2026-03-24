# scripts/test_retrieval.py

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 👇 关键：设置环境变量为绝对路径
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(project_root, "db", "chroma")

from src.retrieval import retrieve

print("🔍 开始检索...")
query = "抑郁症如何治疗？"
print(f"提问: {query}")

results = retrieve(query, top_k=2)

print(f"✅ 检索完成，返回 {len(results)} 个结果")

if not results:
    print("⚠️ 警告：未检索到任何结果！")
else:
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (score={r.score:.3f}) ---")
        print(r.text[:100] + "...")