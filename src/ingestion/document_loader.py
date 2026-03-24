# src/ingestion/document_loader.py
"""
多源文档加载器：支持 PDF / DOCX / TXT，自动提取 metadata
"""

import os
import logging
from typing import List, Optional
from pathlib import Path
from src.schemas import DocumentChunk

logger = logging.getLogger(__name__)


def load_documents_from_directory(
    directory: str,
    file_types: tuple = (".pdf", ".docx", ".txt")
) -> List[DocumentChunk]:
    """
    从指定目录递归加载所有支持的文档，返回原始（未分块）DocumentChunk 列表
    """
    all_chunks = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        raise FileNotFoundError(f"知识库目录不存在: {directory}")
    
    # 遍历所有文件
    for file_path in dir_path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in file_types:
            logger.info(f"📚 加载文档: {file_path.name}")
            try:
                chunks = _load_single_document(str(file_path))
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"❌ 加载失败 {file_path}: {e}")
    
    logger.info(f"✅ 共加载 {len(all_chunks)} 个原始文档块")
    return all_chunks


def _load_single_document(file_path: str) -> List[DocumentChunk]:
    """根据文件扩展名调用对应加载器"""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        raw_pages = _load_pdf(file_path)
    elif ext == ".docx":
        raw_pages = _load_docx(file_path)
    elif ext == ".txt":
        raw_pages = _load_txt(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    # 转换为 DocumentChunk
    chunks = []
    for i, item in enumerate(raw_pages):
        chunk = DocumentChunk(
            text=item["text"],
            source=file_path,  # 完整路径，用于后续溯源
            chunk_id=f"{os.path.basename(file_path)}_raw_{i:04d}",
            metadata={
                "page": item["page"],      # PDF 有页码，其他为 None
                "file_type": ext[1:]       # 'pdf', 'docx', 'txt'
            }
        )
        chunks.append(chunk)
    
    return chunks


def _load_pdf(file_path: str) -> List[dict]:
    """使用 PyPDF2 加载 PDF（保留页码）"""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise ImportError("请安装 PyPDF2: pip install PyPDF2")
    
    reader = PdfReader(file_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"text": text, "page": i + 1})  # 页码从1开始
    return pages


def _load_docx(file_path: str) -> List[dict]:
    """加载 DOCX（无页码）"""
    try:
        from docx import Document
    except ImportError:
        raise ImportError("请安装 python-docx: pip install python-docx")
    
    doc = Document(file_path)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append({"text": text, "page": None})
    return paragraphs


def _load_txt(file_path: str) -> List[dict]:
    """加载 TXT（整篇作为一块，无页码）"""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read().str