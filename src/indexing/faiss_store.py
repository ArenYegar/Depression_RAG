# src/indexing/faiss_store.py
import os
import pickle
from typing import List, Dict
import numpy as np
from src.schemas import DocumentChunk, RetrievedChunk
import faiss

class FAISSStore:
    def __init__(self, persist_directory: str = "./db/faiss"):
        self.persist_directory = persist_directory
        self.index = None          # FAISS 索引
        self.chunk_data = {}       # chunk_id -> DocumentChunk 映射
        self.chunk_ids = []        # 按顺序存储 chunk_id（与 FAISS id 对齐）
        os.makedirs(self.persist_directory, exist_ok=True)

    def add(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        # 转为 numpy array (n, d)
        embeddings_np = np.array(embeddings, dtype=np.float32)
        d = embeddings_np.shape[1]

        # 初始化 FAISS 索引（L2 距离，但我们将用余弦相似度）
        import faiss
        self.index = faiss.IndexFlatL2(d)
        self.index.add(embeddings_np)

        # 存储元数据
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunk_data = {chunk.chunk_id: chunk for chunk in chunks}

        print(f"✅ 已添加 {len(chunks)} 个 chunk 到 FAISS 索引")

    def save_local(self) -> None:
        """保存索引和元数据到磁盘"""
        import faiss

        # 保存 FAISS 索引
        index_path = os.path.join(self.persist_directory, "index.faiss")
        faiss.write_index(self.index, index_path)

        # 保存元数据（chunk_id 列表 + chunk_data 字典）
        meta_path = os.path.join(self.persist_directory, "metadata.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({
                "chunk_ids": self.chunk_ids,
                "chunk_data": self.chunk_data
            }, f)

        print(f"💾 FAISS 知识库已保存至: {self.persist_directory}")

    def load_local(self) -> bool:
        """从磁盘加载索引和元数据"""
        import faiss

        index_path = os.path.join(self.persist_directory, "index.faiss")
        meta_path = os.path.join(self.persist_directory, "metadata.pkl")

        if not (os.path.exists(index_path) and os.path.exists(meta_path)):
            return False

        # 加载索引
        self.index = faiss.read_index(index_path)

        # 加载元数据
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
            self.chunk_ids = meta["chunk_ids"]
            self.chunk_data = meta["chunk_data"]

        print(f"📂 已从 {self.persist_directory} 加载 FAISS 知识库")
        return True

    def search(self, query_embedding: List[float], k: int = 3) -> List[RetrievedChunk]:
        if self.index is None or len(self.chunk_ids) == 0:
            return []

        # 转为 numpy 并 reshape
        query_np = np.array(query_embedding, dtype=np.float32).reshape(1, -1)

        # FAISS 使用 L2 距离，但我们希望用余弦相似度
        # → 先对向量做 L2 归一化，则 L2 距离 ∝ 1 - cosine_similarity
        faiss.normalize_L2(query_np)
        faiss.normalize_L2(self.index.reconstruct_n(0, self.index.ntotal))  # ❌ 不可行！

        # ✅ 更简单方案：在构建时就归一化嵌入！
        # 所以我们在 build_knowledge_base.py 中提前归一化

        # 直接搜索（假设 embeddings 已归一化）
        distances, indices = self.index.search(query_np, k)

        results = []
        for i in range(len(indices[0])):
            idx = indices[0][i]
            if idx == -1:  # 无效结果
                continue

            chunk_id = self.chunk_ids[idx]
            chunk = self.chunk_data[chunk_id]

            # 因为使用 L2 距离 on normalized vectors: similarity = 1 - distance²/2
            distance = float(distances[0][i])
            similarity = 1.0 - (distance ** 2) / 2.0
            similarity = max(0.0, min(1.0, similarity))  # clamp to [0,1]

            results.append(RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                metadata=chunk.metadata,
                score=similarity
            ))

        return results