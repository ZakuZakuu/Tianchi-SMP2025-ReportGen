import json
import logging
import random
from src.multi_provider_model import model_manager
from src.rag_system import RAGSystem

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def judge_relevance(query, document):
    """使用LLM判断文档与查询的相关性"""
    prompt = f"""请判断以下文档是否和议题有关。

议题: {query}

文档: {document}

请回答 Y 或 N（Y表示相关，N表示不相关）。只回答一个字母，不要其他内容。"""

    try:
        response = model_manager.generate_with_retry(
            prompt=prompt,
            model_key="primary",
            max_tokens=10,
            temperature=0.1
        )
        # 清理响应，只保留Y或N
        response = response.strip().upper()
        if response.startswith('Y'):
            return True
        elif response.startswith('N'):
            return False
        else:
            logger.warning(f"意外响应: {response}")
            return False
    except Exception as e:
        logger.error(f"判断相关性失败: {e}")
        return False

def calculate_metrics(retrieved_docs, relevant_docs, k_values=[1, 3, 5, 10]):
    """计算IR指标"""
    metrics = {}
    
    for k in k_values:
        if k > len(retrieved_docs):
            continue
            
        # 前k个检索结果
        top_k = retrieved_docs[:k]
        
        # 计算Precision@k
        relevant_in_top_k = sum(1 for doc in top_k if doc in relevant_docs)
        precision_at_k = relevant_in_top_k / k if k > 0 else 0
        
        # 计算Recall@k（需要知道总的相关文档数）
        total_relevant = len(relevant_docs)
        recall_at_k = relevant_in_top_k / total_relevant if total_relevant > 0 else 0
        
        # 计算F1@k
        f1_at_k = 2 * precision_at_k * recall_at_k / (precision_at_k + recall_at_k) if (precision_at_k + recall_at_k) > 0 else 0
        
        metrics[f'Precision@{k}'] = precision_at_k
        metrics[f'Recall@{k}'] = recall_at_k
        metrics[f'F1@{k}'] = f1_at_k
    
    # 计算MRR
    mrr = 0
    for relevant_doc in relevant_docs:
        for rank, doc in enumerate(retrieved_docs, 1):
            if doc == relevant_doc:
                mrr += 1 / rank
                break
    mrr /= len(relevant_docs) if relevant_docs else 1
    metrics['MRR'] = mrr
    
    return metrics

def evaluate_rag_with_llm_judge(preliminary_file, top_k=10):
    """使用LLM作为裁判评估RAG系统"""
    # 初始化RAG系统
    rag_system = RAGSystem()

    # 加载赛题数据
    with open(preliminary_file, 'r') as f:
        questions = json.load(f)

    # 随机选择10个条目进行评测
    if len(questions) > 10:
        selected_questions = random.sample(questions, 10)
        logger.info(f"从 {len(questions)} 个问题中随机选择 10 个进行评测")
    else:
        selected_questions = questions
        logger.info(f"总共 {len(questions)} 个问题，全部进行评测")

    results = []

    for item in selected_questions:
        question = item.get("question")
        category = item.get("type")
        
        if not question or not category:
            logger.warning(f"跳过无效条目: {item}")
            continue

        logger.info(f"处理问题: {question}")

        # 使用RAG系统检索相关文档
        retrieved_results = rag_system.retrieve(question, category, top_k=top_k)
        
        if not retrieved_results:
            logger.warning(f"未检索到相关文档: {question}")
            continue

        # 提取文档文本
        retrieved_docs = [result['text'] for result in retrieved_results]
        
        # 使用LLM判断每个文档的相关性
        relevant_docs = []
        for i, doc in enumerate(retrieved_docs):
            is_relevant = judge_relevance(question, doc)
            if is_relevant:
                relevant_docs.append(doc)
                logger.info(f"文档 {i+1} 被判断为相关")
            else:
                logger.debug(f"文档 {i+1} 被判断为不相关")

        # 计算指标
        metrics = calculate_metrics(retrieved_docs, relevant_docs)
        
        # 保存结果
        results.append({
            "question": question,
            "category": category,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": relevant_docs,
            "num_retrieved": len(retrieved_docs),
            "num_relevant": len(relevant_docs),
            "metrics": metrics
        })

    # 保存评估结果
    with open("rag_llm_judge_evaluation.json", "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # 计算总体指标
    overall_metrics = {}
    for result in results:
        for metric_name, value in result['metrics'].items():
            if metric_name not in overall_metrics:
                overall_metrics[metric_name] = []
            overall_metrics[metric_name].append(value)
    
    # 计算平均值
    avg_metrics = {}
    for metric_name, values in overall_metrics.items():
        avg_metrics[f'Avg_{metric_name}'] = sum(values) / len(values) if values else 0
    
    # 保存总体结果
    with open("rag_llm_judge_summary.json", "w") as f:
        json.dump({
            "total_questions": len(results),
            "sampled_from": len(questions),
            "average_metrics": avg_metrics,
            "detailed_results": results
        }, f, indent=4, ensure_ascii=False)

    logger.info("评估完成，结果已保存到 rag_llm_judge_evaluation.json 和 rag_llm_judge_summary.json")

if __name__ == "__main__":
    evaluate_rag_with_llm_judge("data/preliminary.json", top_k=10)
