# scripts/debug_db.py
import sys, os
sys.path.insert(0, ".")

from src.indexing import VectorStore

vs = VectorStore(persist_directory="./db/chroma")
sample = vs.collection.peek(limit=1)
print("示例元数据:", sample['metadatas'][0])

