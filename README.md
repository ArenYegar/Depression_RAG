# 抑郁症科普问答助手

基于权威医学资料的 RAG 智能问答系统，提供专业、可靠的抑郁症相关知识科普。

## 📋 项目概览

本项目是一个专注于抑郁症领域的智能问答系统，采用先进的 RAG（Retrieval-Augmented Generation）技术，结合权威医学资料，为用户提供准确、可靠的抑郁症相关知识。

### 核心功能

- **智能混合检索**：融合 FAISS 向量检索和 BM25 稀疏检索，提供更准确的资料匹配
- **重排序优化**：使用 Reranker 对检索结果进行重排序，提升相关性
- **可信度评估**：基于检索结果的相关性计算置信度，确保回答质量
- **信息溯源**：提供回答的信息来源，增强可解释性
- **医疗安全兜底**：对低置信度回答提供专业医疗建议提醒
- **友好界面**：基于 Streamlit 的直观交互界面

## 🛠️ 技术架构

### 系统架构

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   用户界面层     │     │    应用逻辑层    │     │    数据处理层    │
│ Streamlit 前端  │────▶│  检索 + 生成    │────▶│  知识库管理     │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 核心模块

1. **文档摄入模块** (`src/ingestion/`)
   - 支持 PDF、DOCX、TXT 文档加载
   - 智能文本分块，优化检索效果

2. **向量索引模块** (`src/indexing/`)
   - 使用 BGE Large ZH 模型生成文本嵌入
   - FAISS 向量存储，高效相似性搜索

3. **检索模块** (`src/retrieval/`)
   - 混合检索：FAISS + BM25
   - 加权 RRF 融合策略
   - 可选 Reranker 重排序

4. **生成模块** (`src/generation/`)
   - 基于检索结果构建提示词
   - LLM 生成专业回答
   - 后处理优化，清除幻觉

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 推荐使用虚拟环境

### 安装依赖

```bash
# 克隆项目
git clone <repository-url>
cd Depression_RAG_OA

# 安装依赖
pip install -r requirements.txt
```

### 构建知识库

1. 将权威医学文档放入 `data/knowledge_base/` 目录
2. 运行构建脚本：

```bash
python scripts/build_knowledge_base.py
```

### 启动应用

```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址：`http://localhost:8501`

## 📁 项目结构

```
Depression_RAG_OA/
├── scripts/               # 辅助脚本
│   ├── build_knowledge_base.py  # 构建知识库
│   ├── download_qwen.py   # 下载 Qwen 模型
│   ├── download_reranker.py     # 下载 Reranker 模型
│   └── test_*.py          # 测试脚本
├── src/                   # 核心源码
│   ├── ingestion/         # 文档摄入模块
│   ├── indexing/          # 向量索引模块
│   ├── retrieval/         # 检索模块
│   ├── generation/        # 生成模块
│   └── schemas.py         # 数据结构定义
├── data/                  # 数据目录
│   └── knowledge_base/    # 权威医学文档
├── db/                    # 数据库目录
│   ├── faiss/             # FAISS 向量存储
│   └── bm25/              # BM25 索引
├── app.py                 # Streamlit 前端应用
├── README.md              # 项目说明
└── LICENSE                # 许可证
```

## 🔧 配置说明

### 环境变量

- `FAISS_PERSIST_DIR`：FAISS 存储目录（默认：`db/faiss`）
- `USE_RERANKER`：是否使用重排序（默认：`false`）

### 知识库构建参数

- `CHUNK_SIZE`：文本分块大小（默认：600）
- `CHUNK_OVERLAP`：分块重叠大小（默认：80）

## 📚 知识库管理

### 添加新文档

1. 将新的权威医学文档放入 `data/knowledge_base/` 目录
2. 重新运行 `build_knowledge_base.py` 脚本

### 支持的文档格式

- PDF：使用 PyPDF2 解析
- DOCX：使用 python-docx 解析
- TXT：直接读取文本

## 🧪 测试

项目提供了多个测试脚本，用于验证各模块功能：

```bash
# 测试索引构建
python scripts/test_indexing.py

# 测试检索功能
python scripts/test_retrieval.py

# 测试生成功能
python scripts/test_generation.py

# 测试系统查询
python scripts/test_system_query.py
```

## 📝 使用示例

### 基本查询

```python
from retrieval import retrieve
from generation import generate_answer

# 检索相关资料
chunks = retrieve(query="抑郁症的症状有哪些？", top_k=3)

# 生成回答
answer = generate_answer(query="抑郁症的症状有哪些？", retrieved_chunks=chunks)
print(answer)
```

### 前端界面使用

1. 在浏览器中打开应用
2. 在输入框中输入问题，例如："抑郁症如何治疗？"
3. 系统会自动检索相关资料并生成回答
4. 回答中会包含信息来源和置信度评估

## ⚠️ 注意事项

- **免责声明**：本系统仅用于教育和研究目的，不提供医疗诊断或治疗建议
- **信息来源**：系统基于权威医学资料，但仍建议咨询专业医生获取准确诊断
- **模型依赖**：使用 Qwen 模型进行生成，需要确保模型已正确下载
- **性能优化**：对于大型知识库，可能需要调整分块大小和检索参数以获得最佳性能

## 🤝 贡献

欢迎提交 Issue 和 Pull Request 来改进本项目！

## 📄 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 🌟 技术亮点

1. **混合检索策略**：结合向量检索和稀疏检索，提高召回率和准确性
2. **可信度评估**：基于检索结果的相关性计算置信度，确保回答质量
3. **信息溯源**：提供详细的信息来源，增强系统透明度
4. **医疗安全机制**：对低置信度回答提供专业医疗建议提醒
5. **模块化设计**：清晰的代码结构，易于扩展和维护

---

© 2026 抑郁症科普问答系统 | 仅用于教育与研究
