# src/retrieval/faiss_only_retriever.py
import os
from typing import List, Optional
from src.schemas import RetrievedChunk
from src.indexing.faiss_store import FAISSStore
from src.indexing.embedding_model import BGELargeZH
import numpy as np
from sklearn.preprocessing import normalize

_faiss_store: Optional[FAISSStore] = None
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


def _get_embedding_model() -> BGELargeZH:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = BGELargeZH()
    return _embedding_model


def retrieve(query: str, top_k: int = 3) -> List[RetrievedChunk]:
    if not query.strip():
        return []
    
    store = _get_faiss_store()
    embedder = _get_embedding_model()
    
    query_emb = embedder.encode([query])[0]
    query_emb_norm = normalize(np.array([query_emb], dtype=np.float32), norm='l2')[0].tolist()
    
    results = store.search(query_embedding=query_emb_norm, k=top_k)
    return sorted(results, key=lambda x: x.score, reverse=True)[:top_k]