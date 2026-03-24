# scripts/test_generation.py

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# 设置数据库路径（确保能检索）
os.environ["CHROMA_PERSIST_DIR"] = os.path.join(project_root, "db", "chroma")

from src.retrieval import retrieve
from src.generation import generate_answer

if __name__ == "__main__":
    query = "抑郁症如何治疗？"
    print(f"提问: {query}\n")

    # 1. 检索
    chunks = retrieve(query, top_k=2)
    print(f"检索到 {len(chunks)} 个片段\n")

    # 2. 生成
    answer = generate_answer(query, chunks)
    print("🤖 LLM 回答:")
    print(answer)