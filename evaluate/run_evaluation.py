"""
化工安全RAG系统评估主运行程序
整合检索模块和生成模块的评估
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any

# 添加父目录到路径，以便导入项目模块
sys.path.append(str(Path(__file__).parent.parent))

from config import Config
from evaluate.evaluate_retrieval import RetrievalEvaluator
from evaluate.evaluate_generation import GenerationEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def run_full_evaluation(config: Config, test_data_dir: str, output_dir: str) -> Dict[str, Any]:
    """运行完整的系统评估
    
    Args:
        config: 系统配置对象
        test_data_dir: 测试数据目录
        output_dir: 输出结果目录
        
    Returns:
        包含所有评估结果的字典
    """
    # 创建输出目录
    Path(output_dir).mkdir(exist_ok=True, parents=True)
    
    # 评估检索模块
    logger.info("开始检索模块评估...")
    retrieval_evaluator = RetrievalEvaluator(config)
    retrieval_results = retrieval_evaluator.run_evaluation(
        f"{test_data_dir}/retrieval_test_data.json"
    )
    
    # 保存检索评估结果
    with open(f"{output_dir}/retrieval_results.json", "w", encoding="utf-8") as f:
        json.dump(retrieval_results, f, ensure_ascii=False, indent=2)
    logger.info(f"检索评估结果已保存至: {output_dir}/retrieval_results.json")
    
    # 评估生成模块
    logger.info("开始生成模块评估...")
    generation_evaluator = GenerationEvaluator(config)
    generation_results = generation_evaluator.run_evaluation(
        f"{test_data_dir}/generation_test_data.json"
    )
    
    # 保存生成评估结果
    with open(f"{output_dir}/generation_results.json", "w", encoding="utf-8") as f:
        json.dump(generation_results, f, ensure_ascii=False, indent=2)
    logger.info(f"生成评估结果已保存至: {output_dir}/generation_results.json")
    
    # 合并所有结果
    all_results = {
        "retrieval": retrieval_results,
        "generation": generation_results,
        "timestamp": Path(f"{output_dir}/evaluation_results.json").stat().st_mtime if Path(f"{output_dir}/evaluation_results.json").exists() else None
    }
    
    # 保存完整评估结果
    with open(f"{output_dir}/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"完整评估结果已保存至: {output_dir}/evaluation_results.json")
    
    return all_results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="化工安全RAG系统评估工具")
    
    # 添加命令行参数
    parser.add_argument(
        "--test_data_dir", 
        type=str, 
        default="evaluate/test_data",
        help="测试数据目录路径"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="evaluate/results",
        help="评估结果输出目录路径"
    )
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 运行评估
    logger.info("开始进行完整系统评估...")
    results = run_full_evaluation(config, args.test_data_dir, args.output_dir)
    
    # 打印关键指标
    print("\n" + "="*50)
    print("化工安全RAG系统评估结果")
    print("="*50)
    
    # 检索模块指标
    print("\n▶ 检索模块性能:")
    print(f"  命中率@1: {results['retrieval']['hit_rate']['hit@1']:.4f}")
    print(f"  命中率@3: {results['retrieval']['hit_rate']['hit@3']:.4f}")
    print(f"  命中率@5: {results['retrieval']['hit_rate']['hit@5']:.4f}")
    print(f"  平均倒数排名(MRR): {results['retrieval']['mrr']:.4f}")
    
    # 生成模块指标 - Ragas评估
    print("\n▶ 生成模块性能 (Ragas评估):")
    ragas_metrics = results['generation'].get('ragas_metrics', {})
    
    # 忠实度
    faithfulness = ragas_metrics.get('faithfulness', {}).get('score', 0)
    print(f"  忠实度(Faithfulness): {faithfulness:.4f}")
    
    # 答案相关性
    answer_relevancy = ragas_metrics.get('answer_relevancy', {}).get('score', 0)
    print(f"  答案相关性(Answer Relevancy): {answer_relevancy:.4f}")
    
    # 上下文精确度
    context_precision = ragas_metrics.get('context_precision', {}).get('score', 0)
    print(f"  上下文精确度(Context Precision): {context_precision:.4f}")
    
    # 上下文相关性
    context_relevancy = ragas_metrics.get('context_relevancy', {}).get('score', 0)
    print(f"  上下文相关性(Context Relevancy): {context_relevancy:.4f}")
    
    # 上下文召回率
    context_recall = ragas_metrics.get('context_recall', {}).get('score', 0)
    print(f"  上下文召回率(Context Recall): {context_recall:.4f}")
    
    # 有害性评估
    harmfulness = ragas_metrics.get('harmfulness', {}).get('score', 0)
    print(f"  无害性(Safety): {harmfulness:.4f}")
    
    # 综合得分
    overall_score = ragas_metrics.get('overall_score', 0)
    print(f"  综合评分(Overall Score): {overall_score:.4f}")
    
    # 评估样本信息
    evaluated_samples = ragas_metrics.get('meta', {}).get('evaluated_samples', 0)
    total_samples = ragas_metrics.get('meta', {}).get('total_samples', 0)
    print(f"\n  评估样本: {evaluated_samples}/{total_samples}")
    
    print("\n✅ 评估完成！详细结果保存在: " + args.output_dir)

if __name__ == "__main__":
    main() 