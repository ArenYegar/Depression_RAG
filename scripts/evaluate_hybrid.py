# scripts/evaluate_hybrid.py
"""
评估混合检索 vs 仅 FAISS 检索效果
支持两种模式：
  - 默认：使用 BGE 相似度（用于无 reranker 场景）
  - 启用 USE_RERANKER=true：使用 reranker 的 score 字段作为相关性依据
"""

import os
import sys
from pathlib import Path
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.retrieval.faiss_only_retriever import retrieve as faiss_retrieve
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.indexing.embedding_model import BGELargeZH


# === 测试问题集 ===
TEST_QUERIES = [
    # （保持你原有的 50 条不变，此处省略以节省空间）
    "抑郁症能治好吗？", "舍曲林的常见副作用有哪些？", "心理治疗对轻度抑郁有效吗？",
    "抑郁症和焦虑症有什么区别？", "汉密尔顿抑郁量表怎么评分？", "SSRI 类药物有哪些？",
    "认知行为疗法具体怎么做？", "青少年抑郁症的表现和成人一样吗？", "吃抗抑郁药会上瘾吗？",
    "抑郁症复发率高吗？", "心情不好就是抑郁症吗？", "如何判断自己是否需要看医生？",
    "抑郁症会遗传吗？", "运动对抑郁症有帮助吗？", "失眠是抑郁症的症状吗？",
    "抑郁症是精神病吗？", "为什么会得抑郁症？", "抑郁症的诊断标准是什么？",
    "产后抑郁是正常的情绪波动吗？", "轻度抑郁症有哪些表现？", "重度抑郁会出现幻觉吗？",
    "长期失眠是不是抑郁症的症状？", "抑郁症患者为什么会有自杀念头？", "食欲不振是抑郁症的症状吗？",
    "抑郁症会导致记忆力下降吗？", "情绪低落多久可能是抑郁症？", "氟西汀适合哪种类型的抑郁症患者？",
    "抗抑郁药物需要服用多久才能见效？", "擅自停药会有什么后果？", "米氮平的服用注意事项是什么？",
    "抗抑郁药会成瘾吗？", "文拉法辛的剂量应该如何调整？", "服用帕罗西汀期间可以饮酒吗？",
    "不同抗抑郁药物之间可以随意切换吗？", "肝肾功能不全者能服用抗抑郁药吗？", "正念疗法适合抑郁症患者吗？",
    "家庭治疗对抑郁症恢复有帮助吗？", "团体心理治疗和个体治疗哪个效果更好？", "心理治疗需要做多少个疗程？",
    "如何选择合适的心理咨询师？", "心理治疗和药物治疗可以同时进行吗？", "抑郁症患者日常应该如何自我调节？",
    "饮食调理能改善抑郁情绪吗？", "家人应该如何陪伴抑郁症患者？", "抑郁症患者重返工作岗位需要注意什么？",
    "睡眠不好该怎么改善？", "社交活动对抑郁症恢复重要吗？", "冥想能缓解抑郁症状吗？",
    "怀疑自己有抑郁症该去哪里就诊？", "抑郁症的筛查量表有哪些？", "重度抑郁症需要住院治疗吗？",
    "电休克治疗适用于哪些抑郁症患者？", "社区心理健康服务能提供哪些帮助？", "青少年抑郁应该如何干预？",
    "抑郁症靠意志力就能克服吗？", "抑郁症患者为什么不愿意就医？"
]

# === 配置 ===
K = 3
BGE_SIM_THRESHOLD = 0.55          # BGE 相似度阈值
RERANKER_SCORE_THRESHOLD = -2.0   # BGE-Reranker-v2-M3 的典型阈值（> -2 视为相关）


def evaluate_with_bge_similarity(retrieve_func, queries, embedder, name: str):
    """使用 BGE 向量计算相似度（适用于无 reranker 或 baseline）"""
    all_similarities = []
    recalls = []
    reciprocal_ranks = []

    print(f"\n正在评估 [{name}] (BGE 相似度模式)...")

    for query in queries:
        results = retrieve_func(query, top_k=K)
        if not results:
            recalls.append(0)
            reciprocal_ranks.append(0)
            all_similarities.append(0.0)
            continue

        query_emb = embedder.encode([query])[0]
        chunk_texts = [r.text for r in results]
        chunk_embs = embedder.encode(chunk_texts)
        sim_scores = cosine_similarity([query_emb], chunk_embs)[0]
        avg_sim = float(np.mean(sim_scores[:K]))
        all_similarities.append(avg_sim)

        hit = False
        for rank, sim in enumerate(sim_scores[:K]):
            if sim >= BGE_SIM_THRESHOLD:
                recalls.append(1)
                reciprocal_ranks.append(1.0 / (rank + 1))
                hit = True
                break
        if not hit:
            recalls.append(0)
            reciprocal_ranks.append(0)

    mean_sim = np.mean(all_similarities)
    recall_at_k = np.mean(recalls)
    mrr_at_k = np.mean(reciprocal_ranks)

    print(f"\n✅ [{name}] 评估结果 (BGE, K={K}, 阈值={BGE_SIM_THRESHOLD}):")
    print(f"   • 平均语义相似度@{K}: {mean_sim:.3f}")
    print(f"   • Recall@{K}:           {recall_at_k:.1%}")
    print(f"   • MRR@{K}:              {mrr_at_k:.3f}")
    
    return {"mean_sim": mean_sim, "recall": recall_at_k, "mrr": mrr_at_k}


def evaluate_with_reranker_score(retrieve_func, queries, name: str):
    """直接使用 reranker 的 score 字段（需确保 USE_RERANKER=true）"""
    all_scores = []
    recalls = []
    reciprocal_ranks = []

    print(f"\n正在评估 [{name}] (Reranker 分数模式)...")

    for query in queries:
        results = retrieve_func(query, top_k=K)
        if not results:
            recalls.append(0)
            reciprocal_ranks.append(0)
            all_scores.append(0.0)
            continue

        scores = [r.score for r in results[:K]]
        avg_score = float(np.mean(scores))
        all_scores.append(avg_score)

        hit = False
        for rank, r in enumerate(results[:K]):
            if r.score >= RERANKER_SCORE_THRESHOLD:
                recalls.append(1)
                reciprocal_ranks.append(1.0 / (rank + 1))
                hit = True
                break
        if not hit:
            recalls.append(0)
            reciprocal_ranks.append(0)

    mean_score = np.mean(all_scores)
    recall_at_k = np.mean(recalls)
    mrr_at_k = np.mean(reciprocal_ranks)

    print(f"\n✅ [{name}] 评估结果 (Reranker, K={K}, 阈值={RERANKER_SCORE_THRESHOLD}):")
    print(f"   • 平均 Reranker 分数@{K}: {mean_score:.3f}")
    print(f"   • Recall@{K}:             {recall_at_k:.1%}")
    print(f"   • MRR@{K}:                {mrr_at_k:.3f}")
    
    return {"mean_score": mean_score, "recall": recall_at_k, "mrr": mrr_at_k}


def safe_divide(a, b):
    return a / b * 100 if b != 0 else float('nan')


def main():
    print("🔍 开始评估混合检索效果...")
    print(f"📊 测试问题数量: {len(TEST_QUERIES)}")
    use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
    print(f"⚙️  Reranker 模式: {'启用' if use_reranker else '禁用'}")

    embedder = BGELargeZH()

    if use_reranker:
        # 启用 reranker：比较 FAISS vs Hybrid，都使用 reranker 分数
        faiss_metrics = evaluate_with_reranker_score(faiss_retrieve, TEST_QUERIES, "FAISS Only + Reranker")
        hybrid_metrics = evaluate_with_reranker_score(hybrid_retrieve, TEST_QUERIES, "Hybrid + Reranker")
        
        print("\n📈 提升对比 (基于 Reranker 分数):")
        print(f"   • 平均分数: {hybrid_metrics['mean_score']:.3f} ↑ {safe_divide(hybrid_metrics['mean_score'] - faiss_metrics['mean_score'], faiss_metrics['mean_score']):.1f}%")
    else:
        # 未启用 reranker：使用 BGE 相似度
        faiss_metrics = evaluate_with_bge_similarity(faiss_retrieve, TEST_QUERIES, embedder, "FAISS Only")
        hybrid_metrics = evaluate_with_bge_similarity(hybrid_retrieve, TEST_QUERIES, embedder, "Hybrid (FAISS + BM25)")
        
        print("\n📈 提升对比 (基于 BGE 相似度):")
        print(f"   • 平均相似度: {hybrid_metrics['mean_sim']:.3f} ↑ {safe_divide(hybrid_metrics['mean_sim'] - faiss_metrics['mean_sim'], faiss_metrics['mean_sim']):.1f}%")

    # Recall 和 MRR 对比（通用）
    print(f"   • Recall@{K}:   {hybrid_metrics['recall']:.1%} ↑ {safe_divide(hybrid_metrics['recall'] - faiss_metrics['recall'], faiss_metrics['recall']):.1f}%")
    print(f"   • MRR@{K}:      {hybrid_metrics['mrr']:.3f} ↑ {safe_divide(hybrid_metrics['mrr'] - faiss_metrics['mrr'], faiss_metrics['mrr']):.1f}%")

    print("\n🎉 评估完成！")


if __name__ == "__main__":
    main()