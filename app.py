# app.py

import os
import sys
from pathlib import Path

# 将 src 加入路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

import streamlit as st
from retrieval import retrieve
from generation import generate_answer


# ======================
# 页面配置
# ======================
st.set_page_config(
    page_title="抑郁症科普问答助手",
    page_icon="🧠",
    layout="centered"
)

# ======================
# 标题与说明
# ======================
st.title("🧠 抑郁症科普问答助手")
st.markdown(
    """
    基于权威医学资料的 AI 问答系统，提供抑郁症相关知识科普。
    **注意：本系统不提供诊疗建议，如有需要请咨询专业医生。**
    """
)

# ======================
# 初始化 session state
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# 用户输入
# ======================
if prompt := st.chat_input("例如：抑郁症如何治疗？"):
    # 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 检索 + 生成
    with st.chat_message("assistant"):
        with st.spinner("正在检索资料并生成回答..."):
            try:
                # 1. 检索（无权限过滤）
                chunks = retrieve(query=prompt, top_k=2)
                
                # 2. 生成答案
                if chunks:
                    answer = generate_answer(query=prompt, retrieved_chunks=chunks)
                else:
                    answer = "未检索到相关资料，无法回答该问题。"

                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

            except Exception as e:
                error_msg = f"❌ 系统出错：{str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

# ======================
# 底部说明
# ======================
st.divider()
st.caption("© 2026 抑郁症科普问答系统 | 仅用于教育与研究")