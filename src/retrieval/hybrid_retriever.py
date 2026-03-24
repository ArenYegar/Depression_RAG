# src/retrieval/hybrid_retriever.py
"""
混合检索器：FAISS（稠密） + BM25（稀疏） +（可选）重排序
支持加权 RRF 和 BM25 过滤，提升 reranker 输入质量
"""

import os
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import normalize

from src.schemas import RetrievedChunk
from src.indexing.faiss_store import FAISSStore
from src.retrieval.bm25_retriever import BM25Retriever
from src.indexing.embedding_model import BGELargeZH
from src.retrieval.reranker import Reranker


# 全局单例（按需初始化）
_faiss_store: Optional[FAISSStore] = None
_bm25_retriever: Optional[BM25Retriever] = None
_reranker: Optional[Reranker] = None
_embedding_model: Optional[BGELargeZH] = None


def _get_faiss_store() -> FAISSStore:
    global _faiss_store
    if _faiss_store is None:
        persist_dir = os.getenv("FAISS_PERSIST_DIR")
        if persist_dir is None:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            persist_dir = os.path.join(project_root, "db", "faiss")
        _faiss_store = FAISSStore(persist_directory=persist_dir)
        if not _faiss_store.load_local():
            raise RuntimeError(f"❌ 未找到 FAISS 知识库，请先运行 build_knowledge_base.py\n路径: {persist_dir}")
    return _faiss_store


def _get_bm25_retriever() -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever is None:
        from src.retrieval.bm25_retriever import BM25Retriever
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        persist_dir = os.path.join(project_root, "db", "bm25")
        _bm25_retriever = BM25Retriever(persist_directory=persist_dir)
        if not _bm25_retriever.load_local():
            raise RuntimeError(f"❌ 未找到 BM25 索引，请先运行 build_knowledge_base.py\n路径: {persist_dir}")
    return _bm25_retriever


def _get_embedding_model() -> BGELargeZH:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = BGELargeZH()
    return _embedding_model


def _get_reranker() -> Optional[Reranker]:
    global _reranker
    if _reranker is None:
        if os.getenv("USE_RERANKER", "false").lower() == "true":
            from src.retrieval.reranker import Reranker
            _reranker = Reranker()
    return _reranker


def _weighted_reciprocal_rank_fusion(
    dense_results: List[RetrievedChunk],
    sparse_results: List[RetrievedChunk],
    top_k: int = 20,
    k: int = 60,
    dense_weight: float = 0.8,
    sparse_weight: float = 0.2
) -> List[RetrievedChunk]:
    """
    加权 RRF 融合：允许调整稠密和稀疏检索的贡献比例
    """
    scores = {}
    chunk_map = {}

    # 稠密结果加权
    for rank, chunk in enumerate(dense_results):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0) + dense_weight * (1.0 / (k + rank + 1))
        chunk_map[cid] = chunk

    # 稀疏结果加权
    for rank, chunk in enumerate(sparse_results):
        cid = chunk.chunk_id
        scores[cid] = scores.get(cid, 0) + sparse_weight * (1.0 / (k + rank + 1))
        chunk_map[cid] = chunk

    # 排序并去重
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    seen = set()
    for chunk_id, score in sorted_items:
        if chunk_id not in seen:
            chunk = chunk_map[chunk_id]
            chunk.score = score
            result.append(chunk)
            seen.add(chunk_id)
            if len(result) >= top_k:
                break
    return result


def retrieve(
    query: str,
    top_k: int = 3
) -> List[RetrievedChunk]:
    """
    主检索入口函数：混合检索 +（可选）重排序
    使用加权 RRF 和 BM25 过滤提升 reranker 输入质量
    """
    if not query.strip():
        return []

    # 获取组件
    faiss_store = _get_faiss_store()
    bm25_retriever = _get_bm25_retriever()
    embedder = _get_embedding_model()

    # 1. FAISS 稠密检索（取更多候选）
    query_emb = embedder.encode([query])[0]
    query_emb_norm = normalize(np.array([query_emb], dtype=np.float32), norm='l2')[0].tolist()
    dense_results = faiss_store.search(query_embedding=query_emb_norm, k=top_k * 3)

    # 2. BM25 稀疏检索（取更多候选，并过滤低分结果）
    sparse_candidates = bm25_retriever.search(query, k=top_k * 5)  # 取更多
    if sparse_candidates:
        # 只保留 BM25 分数 > 中位数的结果（去除明显不相关项）
        bm25_scores = [r.score for r in sparse_candidates]
        median_score = float(np.median(bm25_scores))
        sparse_results = [r for r in sparse_candidates if r.score >= median_score][:top_k * 3]
    else:
        sparse_results = []

    # 3. 加权 RRF 融合
    fused_results = _weighted_reciprocal_rank_fusion(
        dense_results=dense_results,
        sparse_results=sparse_results,
        top_k=top_k * 3,
        dense_weight=0.8,
        sparse_weight=0.2
    )

    if not fused_results:
        return []

    # 4. （可选）重排序
    reranker = _get_reranker()
    if reranker is not None and len(fused_results) >= 1:
        try:
            rescored = reranker.rescore(query=query, chunks=fused_results)
            rescored.sort(key=lambda x: x.score, reverse=True)
            return rescored[:top_k]
        except Exception as e:
            print(f"⚠️ Reranker failed, falling back to hybrid results: {e}")

    return sorted(fused_results, key=lambda x: x.score, reverse=True)[:top_k]