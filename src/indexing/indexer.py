# src/indexing/indexer.py
from typing import List
from src.schemas import DocumentChunk
from .embedding_model import BGELargeZH
from .vector_store import VectorStore

class Indexer:
    def __init__(self, persist_directory: str = "./db/chroma"):
        self.embedding_model = BGELargeZH()
        self.vector_store = VectorStore(persist_directory=persist_directory)

    def build_from_chunks(self, chunks: List[DocumentChunk]) -> None:
        if not chunks:
            print("⚠️ 无文档块，跳过索引构建")
            return

        print(f"🧠 正在为 {len(chunks)} 个文本块生成 embedding...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.encode(texts)

        # ✅ 不再修改 metadata！保持原始 metadata 干净
        # 直接传递 texts, embeddings, metadatas 给 vector_store
        self.vector_store.add(chunks=chunks, embeddings=embeddings)  # 👈 新接口