# src/ingestion/text_splitter.py
import re
from typing import List
from src.schemas import DocumentChunk

def _split_text_by_length(text: str, max_length: int = 512, overlap: int = 50) -> List[str]:
    """按字符长度分块，保留句子边界（简单版）"""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_length
        # 尽量在句号、换行或空格处分割
        if end < len(text):
            # 向前找最近的句子结束符
            tail = text[start:end]
            last_period = max(
                tail.rfind("."),
                tail.rfind("。"),
                tail.rfind("\n"),
                tail.rfind(" "),
                -1
            )
            if last_period != -1:
                end = start + last_period + 1
        chunks.append(text[start:end])
        start = end - overlap if overlap < end else end
    return chunks

def split_documents(documents: List[DocumentChunk], chunk_size: int = 512, chunk_overlap: int = 50) -> List[DocumentChunk]:
    """
    将长文档切分为小块，生成新的 chunk_id 和独立元数据
    """
    new_chunks = []
    for doc in documents:
        text_chunks = _split_text_by_length(doc.text, chunk_size, chunk_overlap)
        for i, chunk_text in enumerate(text_chunks):
            if not chunk_text.strip():
                continue
            new_chunk = DocumentChunk(
                text=chunk_text.strip(),
                source=doc.source,
                chunk_id=f"{doc.chunk_id.replace('_raw_', '_')}_{i:03d}",
                metadata=doc.metadata.copy()  # 保留原始元数据
            )
            new_chunks.append(new_chunk)
    return new_chunks