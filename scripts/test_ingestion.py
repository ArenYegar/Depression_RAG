# scripts/test_ingestion.py
import os
import sys

# 将项目根目录加入 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.ingestion import ingest_files
from src.schemas import DocumentChunk

if __name__ == "__main__":
    sample_dir = "data/sample_docs"
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)
        print(f"请在 {sample_dir} 中放入 PDF/DOCX/TXT 文件")
        exit(1)
    
    files = [
        os.path.join(sample_dir, f)
        for f in os.listdir(sample_dir)
        if f.lower().endswith(('.pdf', '.docx', '.txt'))
    ]
    
    if not files:
        print(f"⚠️ {sample_dir} 中无有效文档")
        exit(1)

    print(f"📥 正在处理 {len(files)} 个文件...")
    chunks = ingest_files(files, chunk_size=300, chunk_overlap=30)
    
    print(f"✅ 共生成 {len(chunks)} 个文本块")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk.chunk_id}")
        print(f"Source: {chunk.source}")
        print(f"Metadata: {chunk.metadata}")
        print(f"Text (前100字): {chunk.text[:100]}...")

    assert all(isinstance(c, DocumentChunk) for c in chunks), "类型错误！"
    print("\n✅ 所有输出均为 DocumentChunk 类型，符合接口契约！")