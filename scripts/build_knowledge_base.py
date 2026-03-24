"""
构建多源权威知识库：支持 data/knowledge_base/ 下所有 PDF/DOCX/TXT
同时构建 FAISS（稠密）和 BM25（稀疏）双索引
"""

import os
import sys
import shutil
import logging
from pathlib import Path
import numpy as np
from sklearn.preprocessing import normalize

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.document_loader import load_documents_from_directory
from src.ingestion.text_splitter import split_documents
from src.indexing.embedding_model import BGELargeZH
from src.indexing.faiss_store import FAISSStore
from src.schemas import RetrievedChunk
from src.retrieval.bm25_retriever import BM25Retriever  # ← 新增


def main():
    # 配置路径
    KNOWLEDGE_DIR = project_root / "data" / "knowledge_base"
    FAISS_PERSIST_DIR = project_root / "db" / "faiss"
    BM25_PERSIST_DIR = project_root / "db" / "bm25"

    # 参数
    CHUNK_SIZE = 600
    CHUNK_OVERLAP = 80

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    # 确保知识库目录存在
    if not KNOWLEDGE_DIR.exists():
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建知识库目录: {KNOWLEDGE_DIR}")
        print("👉 请将权威医学文档（PDF/DOCX/TXT）放入此目录后重新运行。")
        sys.exit(0)

    # 清理旧数据库
    for d in [FAISS_PERSIST_DIR, BM25_PERSIST_DIR]:
        if d.exists():
            logging.info(f"🗑️  清理旧知识库: {d}")
            shutil.rmtree(d)

    # 1. 加载多源文档
    logging.info(f"📥 正在从 {KNOWLEDGE_DIR} 加载多源权威指南...")
    raw_chunks = load_documents_from_directory(str(KNOWLEDGE_DIR))
    if not raw_chunks:
        logging.error("❌ 未找到任何支持的文档（.pdf/.docx/.txt），请检查目录内容")
        sys.exit(1)

    # 2. 分块
    logging.info("✂️  正在分块文本...")
    split_chunks = split_documents(
        documents=raw_chunks,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    logging.info(f"✅ 生成 {len(split_chunks)} 个文本块")

    # 转为 RetrievedChunk 格式（用于 BM25）
    retrieved_chunks = [
        RetrievedChunk(
            text=chunk.text,
            source=chunk.source,
            chunk_id=chunk.chunk_id,
            metadata=chunk.metadata,
            score=0.0
        )
        for chunk in split_chunks
    ]

    # 3. 构建 FAISS 索引
    logging.info("🧠 正在生成向量嵌入...")
    embedder = BGELargeZH()
    texts = [chunk.text for chunk in split_chunks]
    embeddings = embedder.encode(texts)

    logging.info("⚖️  正在对嵌入向量进行 L2 归一化（用于余弦相似度）...")
    embeddings_np = np.array(embeddings, dtype=np.float32)
    embeddings_normalized = normalize(embeddings_np, norm='l2').tolist()

    logging.info(f"💾 正在构建 FAISS 索引并保存至: {FAISS_PERSIST_DIR}")
    faiss_store = FAISSStore(persist_directory=str(FAISS_PERSIST_DIR))
    faiss_store.add(split_chunks, embeddings_normalized)
    faiss_store.save_local()

    # 4. 构建 BM25 索引
    logging.info(f"🔍 正在构建 BM25 稀疏索引并保存至: {BM25_PERSIST_DIR}")
    bm25_retriever = BM25Retriever(persist_directory=str(BM25_PERSIST_DIR))
    bm25_retriever.build_from_chunks(retrieved_chunks)

    print(f"✅ 嵌入维度: {len(embeddings[0])}")  # 应为 1024
    logging.info("🎉 多源知识库构建完成！")
    print("\n📌 使用提示：")
    print(f"   - 已加载文档: {[Path(c.source).name for c in raw_chunks[:3]]}{'...' if len(raw_chunks) > 3 else ''}")
    print(f"   - FAISS 向量库位置: {FAISS_PERSIST_DIR}")
    print(f"   - BM25 索引位置: {BM25_PERSIST_DIR}")


if __name__ == "__main__":
    main()