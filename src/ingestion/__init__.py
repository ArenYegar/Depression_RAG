# src/ingestion/__init__.py
from typing import List
from src.schemas import DocumentChunk
from .document_loader import load_documents_from_directory
from .text_splitter import split_documents

def ingest_files(file_paths: List[str], chunk_size: int = 512, chunk_overlap: int = 50) -> List[DocumentChunk]:
    """
    统一入口：加载并分块文档
    Args:
        file_paths: 文件路径列表
        chunk_size: 分块最大长度（字符）
        chunk_overlap: 块间重叠长度
    Returns:
        List[DocumentChunk]
    """
    raw_docs = load_documents_from_directory(file_paths)
    chunks = split_documents(raw_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunks