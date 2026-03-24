# scripts/test_indexing.py
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ingestion import ingest_files
from src.indexing import Indexer

if __name__ == "__main__":
    # 1. 加载文档
    files = ["data/sample_docs/depression_guide.txt"]  # 确保这个文件存在！
    if not os.path.exists(files[0]):
        print("❌ 请先创建 data/sample_docs/depression_guide.txt")
        exit(1)

    chunks = ingest_files(files, chunk_size=300, chunk_overlap=30)
    print(f"📄 加载 {len(chunks)} 个文本块")

    # 2. 构建索引
    indexer = Indexer(persist_directory="./db/chroma")
    indexer.build_from_chunks(chunks)

    print("\n🔍 测试检索（直接调用 vector store）")
    from src.indexing import VectorStore
    vs = VectorStore(persist_directory="./db/chroma")
    
    # 手动生成 query embedding（模拟 retrieval 模块行为）
    from src.indexing.embedding_model import BGELargeZH
    embedder = BGELargeZH()
    query_emb = embedder.encode(["抑郁症的核心症状有哪些？"])[0]
    
    results = vs.search(query_emb, k=2)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score={r.score:.3f}) ---")
        print(f"Text: {r.text[:100]}...")