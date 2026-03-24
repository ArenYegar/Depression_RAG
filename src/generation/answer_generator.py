"""
主生成入口：拼接 prompt + 调用 LLM + 增强可信度与可解释性（医疗 RAG 专业版）
"""

import os
import re
from typing import List
from src.schemas import RetrievedChunk
from .prompt_template import build_prompt
from .llm_client import LLMClient


def generate_answer(query: str, retrieved_chunks: List[RetrievedChunk]) -> str:
    """
    生成带溯源和置信度评估的回答（专为医疗 RAG 场景优化）
    """
    if not retrieved_chunks:
        return "⚠️ 未检索到相关资料，无法回答该问题。"

    # === 智能过滤：保留有实质内容的 chunk ===
    meaningful_chunks = []
    for c in retrieved_chunks:
        text = c.text.strip()
        if len(text) < 30:
            continue
        if text.startswith(("表", "编码", "诊断条目", "章节", "附录", "参考文献")):
            continue
        meaningful_chunks.append(c)
    
    if not meaningful_chunks:
        meaningful_chunks = retrieved_chunks

    # === 计算置信度（基于 reranker 最高分）===
    max_score = max(c.score for c in meaningful_chunks) if meaningful_chunks else -1
    if max_score > 5.0:
        confidence = "高"
        confidence_desc = "内容高度相关"
    elif max_score > 0:
        confidence = "中"
        confidence_desc = "内容部分相关"
    else:
        confidence = "低"
        confidence_desc = "内容相关性弱"

    # === 生成核心回答 ===
    prompt = build_prompt(query, meaningful_chunks)
    llm = LLMClient()
    try:
        answer = llm.generate(prompt)
    except Exception as e:
        return f"❌ 生成失败：{str(e)}"

    # === 后处理：清除残留幻觉（兜底防护）===
    # 移除虚构的参考文献、栏目标题等
    hallucination_patterns = [
        r"(?i)参考文献\s*\[.*?\].*",
        r"(?i)(相关词条|扩展阅读|综述|药物治疗|康复训练|社会支持)\s*《.*?》",
        r"《国内外权威指南》",
        r"\[未知\]\s*\(.*?\)"
    ]
    cleaned_answer = answer.strip()
    for pattern in hallucination_patterns:
        cleaned_answer = re.sub(pattern, "", cleaned_answer, flags=re.MULTILINE)
    cleaned_answer = re.sub(r"\n\s*\n", "\n\n", cleaned_answer).strip()  # 清理多余空行

    # === 构建溯源信息（使用真实文件名）===
    source_info = []
    for c in meaningful_chunks[:2]:
        source_path = getattr(c, 'source', None)
        if source_path and isinstance(source_path, str):
            filename = os.path.basename(source_path)
            if filename.endswith(".pdf"):
                display_name = "《" + filename[:-4] + "》"
            else:
                display_name = filename
        else:
            display_name = "《精神障碍诊疗规范（2020年版）》"
        
        page = c.metadata.get("page", "?")
        source_info.append(f"{display_name} 第 {page} 页")

    sources_str = "；".join(source_info) if source_info else "未明确标注"

    # === 组合最终回答 ===
    final_answer = cleaned_answer

    # 医疗安全兜底
    if confidence == "低":
        final_answer += "\n\n⚠️ 注意：当前参考资料相关性较低，建议咨询专业医生获取准确诊断。"
    
    final_answer += f"\n\n🔍 信息来源：{sources_str}\n✅ 相关性置信度：{confidence}（{confidence_desc}）"

    return final_answer