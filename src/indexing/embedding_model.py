# src/indexing/embedding_model.py
import os
from sentence_transformers import SentenceTransformer

class BGELargeZH:
    def __init__(self):
        # 指向本地模型快照路径（替换为你的实际路径）
        model_path = r"E:\Kaggle\LLM\Depression_RAG_OA\models\embeddings\models--BAAI--bge-large-zh-v1.5\snapshots\79e7739b6ab944e86d6171e44d24c997fc1e0116"
        
        # ✅ 关键：设置 local_files_only=True + trust_remote_code=False（安全）
        self.model = SentenceTransformer(
            model_path,
            device="cuda",  # 或 "cpu"
            local_files_only=True,  # ⚠️ 强制只用本地文件，不联网
            trust_remote_code=False  # 安全起见，除非必要
        )
    
    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)