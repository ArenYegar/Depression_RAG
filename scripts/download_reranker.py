# scripts/download_reranker.py
from huggingface_hub import snapshot_download
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RERANKER_DIR = os.path.join(PROJECT_ROOT, "models", "reranker")

print(f"📥 正在下载 BGE Reranker 到: {RERANKER_DIR}")

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir=RERANKER_DIR,
    local_dir_use_symlinks=False,  # 直接复制文件，不建软链接（Windows 友好）
    resume_download=True
)

print("✅ 下载完成！")