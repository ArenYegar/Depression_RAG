# src/schemas.py
from pydantic import BaseModel
from typing import List, Dict

class DocumentChunk(BaseModel):
    text: str
    source: str
    chunk_id: str
    metadata: Dict

class RetrievedChunk(DocumentChunk):
    score: float  # 相似度分数（Chroma 返回 distance，我们转为 similarity）