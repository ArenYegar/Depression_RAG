# src/retrieval/bm25_retriever.py
import os
import pickle
from typing import List, Dict
from rank_bm25 import BM25Okapi
from src.schemas import RetrievedChunk


class BM25Retriever:
    def __init__(self, persist_directory: str = "./db/bm25"):
        self.persist_directory = persist_directory
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_data: Dict[str, RetrievedChunk] = {}
        self.corpus: List[List[str]] = []
        self.chunk_ids: List[str] = []

    def build_from_chunks(self, chunks: List[RetrievedChunk]) -> None:
        """从 chunk 列表构建 BM25 索引（通常在知识库构建时调用）"""
        import jieba
        
        self.chunk_ids = [c.chunk_id for c in chunks]
        self.chunk_data = {c.chunk_id: c for c in chunks}
        
        # 中文分词
        self.corpus = [
            list(jieba.cut(chunk.text.replace("\n", " ").strip()))
            for chunk in chunks
        ]
        
        # 过滤空词
        self.corpus = [[word for word in doc if word.strip()] for doc in self.corpus]
        
        self.bm25 = BM25Okapi(self.corpus)
        self._save()

    def _save(self):
        os.makedirs(self.persist_directory, exist_ok=True)
        with open(os.path.join(self.persist_directory, "bm25.pkl"), "wb") as f:
            pickle.dump({
                "corpus": self.corpus,
                "chunk_ids": self.chunk_ids,
                "chunk_data": self.chunk_data
            }, f)

    def load_local(self) -> bool:
        path = os.path.join(self.persist_directory, "bm25.pkl")
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.corpus = data["corpus"]
            self.chunk_ids = data["chunk_ids"]
            self.chunk_data = data["chunk_data"]
            self.bm25 = BM25Okapi(self.corpus)
        return True

    def search(self, query: str, k: int = 10) -> List[RetrievedChunk]:
        import jieba
        if self.bm25 is None:
            return []
        
        tokenized_query = list(jieba.cut(query.strip()))
        scores = self.bm25.get_scores(tokenized_query)
        
        # 获取 top-k 索引
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        results = []
        for idx in top_k_indices:
            chunk_id = self.chunk_ids[idx]
            chunk = self.chunk_data[chunk_id]
            results.append(RetrievedChunk(
                text=chunk.text,
                source=chunk.source,
                chunk_id=chunk.chunk_id,
                metadata=chunk.metadata,
                score=float(scores[idx])  # BM25 分数
            ))
        return results