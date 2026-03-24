# scripts/download_qwen2.py

from modelscope import snapshot_download

# 下载 Qwen2-1.5B-Instruct（约 3GB）
model_dir = snapshot_download(
    'qwen/Qwen2-1.5B-Instruct',
    cache_dir="./models",  # 保存到项目 models 目录
    revision="master"
)

print(f"✅ 模型已下载至: {model_dir}")