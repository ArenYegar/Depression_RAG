"""
重排序模块：从本地 models/reranker 加载 BGE-Reranker-v2-M3
"""

import os
import torch
from typing import List, Tuple
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BatchEncoding
)
from src.schemas import RetrievedChunk


class Reranker:
    def __init__(self):
        """
        从项目本地 models/reranker 目录加载模型（完全离线）
        """
        # 自动定位项目根目录下的 models/reranker
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        model_path = os.path.join(project_root, "models", "reranker")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"❌ 未找到 reranker 模型目录: {model_path}\n"
                "请先运行 `python scripts/download_reranker.py` 下载模型。"
            )

        self.device = "cuda" if torch.cuda.is_available() else \
                      "mps" if torch.backends.mps.is_available() else "cpu"
        
        # print(f"🔄 从本地加载 Reranker 模型: {model_path}")
        # print(f"   → 设备: {self.device}")

        # 从本地路径加载，强制离线
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False
        )
        self.model.to(self.device)
        self.model.eval()

    def _build_pairs(self, query: str, texts: List[str]) -> List[Tuple[str, str]]:
        return [(query, text) for text in texts]

    def _encode(self, pairs: List[Tuple[str, str]], batch_size: int = 32) -> torch.Tensor:
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), batch_size):
                batch = pairs[i:i + batch_size]
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=8192,
                    return_tensors="pt"
                ).to(self.device)

                outputs = self.model(**inputs)
                batch_scores = outputs.logits[:, -1].cpu()
                scores.append(batch_scores)
        return torch.cat(scores, dim=0)

    def rescore(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        if not chunks:
            return chunks

        texts = [chunk.text for chunk in chunks]
        pairs = self._build_pairs(query, texts)
        new_scores = self._encode(pairs)

        for i, chunk in enumerate(chunks):
            chunk.score = float(new_scores[i].item())

        return chunks