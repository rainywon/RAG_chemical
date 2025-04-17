"""
评估RAG系统的生成模块性能
专注于忠实度(Faithfulness)和答案相关性(Answer Relevancy)两个核心指标
使用Ragas库进行标准化评估
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# 添加父目录到路径，以便导入项目模块
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from rag_system import RAGSystem
from config import Config

# 导入ragas库用于RAG系统评估
import pandas as pd
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics.critique import harmfulness

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GenerationEvaluator:
    def __init__(self, config):
        """初始化生成评估器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        
        # 初始化ragas评估指标
        self.faithfulness_metric = faithfulness
        self.answer_relevancy_metric = answer_relevancy
        
        logger.info("生成评估器初始化完成")
    
    def evaluate_with_ragas(self, question: str, answer: str, context: str) -> Dict[str, float]:
        """使用ragas评估答案的忠实度和相关性
        
        Args:
            question: 问题
            answer: 生成的答案
            context: 参考上下文
            
        Returns:
            Dict[str, float]: 包含忠实度和相关性分数的字典
        """
        try:
            # 准备ragas评估数据
            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [[context]]
            }
            
            df = pd.DataFrame(data)
            
            # 计算忠实度分数
            faithfulness_score = self.faithfulness_metric.score(df)["faithfulness"].iloc[0]
            
            # 计算答案相关性分数
            relevancy_score = self.answer_relevancy_metric.score(df)["answer_relevancy"].iloc[0]
            
            return {
                "faithfulness_score": faithfulness_score,
                "answer_relevancy_score": relevancy_score
            }
            
        except Exception as e:
            logger.error(f"Ragas评估失败: {str(e)}")
            return {
                "faithfulness_score": 0.0,
                "answer_relevancy_score": 0.0
            }

    def run_evaluation(self, test_data_path: str) -> Dict[str, Any]:
        """运行完整评估流程
        
        Args:
            test_data_path: 测试数据文件路径
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 加载测试数据
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            logger.info(f"成功加载{len(test_data)}条测试数据")
            
            results = []
            total_faithfulness = 0.0
            total_answer_relevancy = 0.0
            
            # 逐条评估
            for i, item in enumerate(test_data):
                logger.info(f"正在处理第{i+1}条测试数据...")
                question = item.get("question", "")
                reference_context = item.get("context", "")
                reference_answer = item.get("answer", "")
                
                # RAG系统生成答案
                try:
                    generated_answer, references, _ = self.rag_system.answer_query(question)
                    
                    # 检索的上下文
                    retrieved_context = "\n".join([ref["content"] for ref in references]) if references else ""
                    
                    # 使用参考上下文进行评估（如果有）
                    context_for_eval = reference_context if reference_context else retrieved_context
                    
                    # 使用ragas评估
                    ragas_scores = self.evaluate_with_ragas(question, generated_answer, context_for_eval)
                    
                    faithfulness_score = ragas_scores["faithfulness_score"]
                    relevancy_score = ragas_scores["answer_relevancy_score"]
                    
                    logger.info(f"忠实度评分: {faithfulness_score:.4f}")
                    logger.info(f"相关性评分: {relevancy_score:.4f}")
                    
                    # 累计分数
                    total_faithfulness += faithfulness_score
                    total_answer_relevancy += relevancy_score
                    
                    # 保存结果
                    results.append({
                        "question": question,
                        "generated_answer": generated_answer,
                        "reference_answer": reference_answer,
                        "context": context_for_eval,
                        "faithfulness_score": faithfulness_score,
                        "answer_relevancy_score": relevancy_score
                    })
                    
                except Exception as e:
                    logger.error(f"处理测试数据失败: {str(e)}")
                    # 添加失败记录
                    results.append({
                        "question": question,
                        "error": str(e),
                        "faithfulness_score": 0.0,
                        "answer_relevancy_score": 0.0
                    })
            
            # 计算平均分数
            avg_faithfulness = total_faithfulness / len(test_data) if test_data else 0.0
            avg_answer_relevancy = total_answer_relevancy / len(test_data) if test_data else 0.0
            
            # 综合评分
            comprehensive_score = (avg_faithfulness + avg_answer_relevancy) / 2
            
            # 输出总体评分
            logger.info(f"评估完成！总体结果:")
            logger.info(f"平均忠实度: {avg_faithfulness:.4f}")
            logger.info(f"平均答案相关性: {avg_answer_relevancy:.4f}")
            logger.info(f"综合评分: {comprehensive_score:.4f}")
            
            # 构建完整评估结果
            evaluation_results = {
                "results": results,
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevancy": avg_answer_relevancy,
                "comprehensive_score": comprehensive_score,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 保存结果
            results_dir = Path("evaluate/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = results_dir / "generation_results.json"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"评估结果已保存至: {results_path}")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}")
            return {
                "error": str(e),
                "results": [],
                "avg_faithfulness": 0.0,
                "avg_answer_relevancy": 0.0,
                "comprehensive_score": 0.0
            }

if __name__ == "__main__":
    # 初始化配置
    config = Config()
    
    # 创建评估器
    evaluator = GenerationEvaluator(config)
    
    # 运行评估
    current_dir = Path(__file__).resolve().parent
    test_data_path = current_dir / "test_data" / "generation_test_data.json"
    evaluator.run_evaluation(str(test_data_path))