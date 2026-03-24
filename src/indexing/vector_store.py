# src/indexing/vector_store.py
from typing import List, Optional, Dict
from src.schemas import DocumentChunk

class VectorStore:
    def __init__(self, persist_directory: str = "./db/chroma"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]

        # ✅ 构建合法的 metadata：只保留非 None 且类型合法的字段
        metadatas = []
        for chunk in chunks:
            clean_meta = {}
            # 必须包含 source（假设它不为 None）
            if chunk.source is not None:
                clean_meta["source"] = chunk.source
            
            # 处理额外 metadata，跳过 None 和非法类型
            for k, v in chunk.metadata.items():
                if v is None:
                    continue  # ⚠️ 跳过 None
                if isinstance(v, (str, int, float, bool)):
                    clean_meta[k] = v
                # 可选：将 list/dict 转为 JSON 字符串（如果需要保留）
                # elif isinstance(v, (list, dict)):
                #     clean_meta[k] = json.dumps(v, ensure_ascii=False)
            
            metadatas.append(clean_meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas  # 现在全是合法值
        )

    def search(
        self,
        query_embedding: List[float],
        k: int = 3
    ) -> List['RetrievedChunk']:
        from src.schemas import RetrievedChunk
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_chunks = []
        for i in range(len(results["ids"][0])):
            chunk_id = results["ids"][0][i]
            text = results["documents"][0][i]
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            similarity = 1 - (distance / 2)

            clean_metadata = {k: v for k, v in metadata.items() if k != "source"}

            retrieved_chunks.append(RetrievedChunk(
                text=text,
                source=metadata["source"],
                chunk_id=chunk_id,
                metadata=clean_metadata,
                score=similarity
            ))
        return retrieved_chunks