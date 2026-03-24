"""
构建 RAG Prompt：将检索结果拼接为上下文（医疗场景强化版）
"""

from typing import List
from src.schemas import RetrievedChunk

def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    """
    构建严格约束的 prompt，防止 LLM 编造来源或虚构内容
    """
    context_parts = []
    for c in chunks:
        page = c.metadata.get("page", "?")
        # 注意：source 不在 metadata 中！但此处仅用于上下文展示，实际文件名由 answer_generator 控制
        # 所以我们只显示页码，不显示文件名（避免混淆）
        clean_text = " ".join(c.text.split())  # 合并换行，提升可读性
        context_parts.append(f"[第 {page} 页]\n{clean_text}")
    
    context = "\n\n".join(context_parts)
    
    return (
        "你是一个严谨的心理健康科普助手，所有回答必须严格基于以下《精神障碍诊疗规范（2020年版）》的内容。\n"
        "【重要规则】\n"
        "- 如果资料中没有直接回答问题的信息，请明确说明：“根据《精神障碍诊疗规范（2020年版）》，未明确描述该问题。”\n"
        "- **禁止编造任何参考文献、书名、作者、出版社或出版年份**。\n"
        "- **禁止使用“相关词条”“扩展阅读”“综述”“药物治疗”等虚构栏目标题**。\n"
        "- 即使资料是表格、要点或碎片化文本，也请用自然语言总结，但不得添加外部知识。\n"
        "- 回答应简洁、专业，聚焦于规范原文内容。\n\n"
        "【参考资料】（来自《精神障碍诊疗规范（2020年版）》）\n"
        f"{context}\n\n"
        "【问题】\n"
        f"{query}\n\n"
        "【回答】\n"
    )