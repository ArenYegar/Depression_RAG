enterprise-rag/
│
├── README.md                     # 项目介绍、Demo 链接、技术亮点、部署指南
├── requirements.txt              # Python 依赖（分 dev / prod 可选）
├── Dockerfile                    # 容器化构建文件
├── docker-compose.yml            # （可选）用于启动 Milvus / Chroma 等服务
│
├── configs/                      # 配置文件
│   ├── app_config.yaml           # 应用主配置（模型路径、chunk size 等）
│   └── vector_db_config.yaml     # 向量数据库连接参数
│
├── data/                         # 示例数据（勿提交大文件，用 .gitignore）
│   ├── sample_docs/              # 示例 PDF/DOCX（用于演示）
│   └── processed/                # （.gitignore）预处理后的 chunk JSONL
│
├── src/                          # 核心源码（按模块划分）
│   │
│   ├── ingestion/                # 文档摄入模块
│   │   ├── __init__.py
│   │   ├── document_loader.py    # 支持 PDF/DOCX/TXT 加载
│   │   └── text_splitter.py      # 智能分块逻辑
│   │
│   ├── indexing/                 # 向量索引模块
│   │   ├── __init__.py
│   │   ├── embedding_model.py    # 封装 bge-large-zh
│   │   ├── vector_store.py       # Chroma/Milvus 接口抽象
│   │   └── indexer.py            # 构建/更新索引入口
│   │
│   ├── retrieval/                # 检索模块
│   │   ├── __init__.py
│   │   ├── hybrid_retriever.py   # 向量 + BM25 混合检索
│   │   └── reranker.py           # bge-reranker 调用
│   │
│   ├── generation/               # LLM 生成模块
│   │   ├── __init__.py
│   │   ├── llm_client.py         # llama.cpp / vLLM 客户端封装
│   │   └── prompt_template.py    # Prompt 模板管理
│   │
│   ├── api/                      # 后端 API
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI 入口
│   │   ├── routes/
│   │   │   ├── upload.py         # 文档上传接口
│   │   │   ├── query.py          # 问答接口
│   │   │   └── auth.py           # 登录/角色接口
│   │   └── schemas.py            # Pydantic 模型定义
│   │
│   └── utils/                    # 工具函数
│       ├── logger.py             # 统一日志
│       ├── metrics.py            # 记录检索/生成耗时、top-k 准确率等
│       └── helpers.py            # 通用工具
│
├── frontend/                     # 前端界面（可选）
│   ├── app_gradio.py             # Gradio Demo（1 文件即可）
│   └── app_streamlit.py          # 或 Streamlit 版本
│
├── scripts/                      # 辅助脚本
│   ├── ingest_sample_data.py     # 一键导入示例文档
│   ├── evaluate_retrieval.py     # 评估检索准确率（人工构造 test set）
│   └── run_local.sh              # 本地启动脚本
│
├── tests/                        # 单元测试（加分项！）
│   ├── test_document_loader.py
│   ├── test_retriever.py
│   └── conftest.py
│
├── logs/                         # （.gitignore）运行日志
│
└── .gitignore                    # 忽略模型、大文件、日志等

