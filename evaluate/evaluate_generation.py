"""
评估RAG系统的生成模块性能
专注于忠实度(Faithfulness)和答案相关性(Answer Relevancy)两个核心指标
"""

import json
import logging
import os
import sys
import jieba
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# 添加父目录到路径，以便导入项目模块
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from rag_system import RAGSystem
from config import Config

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
        logger.info("生成评估器初始化完成")
        
    def _tokenize(self, text):
        """中文分词"""
        return [word for word in jieba.cut(text) if word.strip()]
        
    def evaluate_faithfulness(self, answer: str, context: str) -> float:
        """评估答案对上下文的忠实度
        
        使用基于词汇重叠和N-gram匹配的方法评估
        
        Args:
            answer: 生成的答案
            context: 参考上下文
            
        Returns:
            float: 忠实度分数 (0-1)
        """
        try:
            if not answer or not context:
                return 0.0
                
            # 分词处理
            answer_tokens = self._tokenize(answer)
            context_tokens = self._tokenize(context)
            
            if not answer_tokens or not context_tokens:
                return 0.0
            
            # 计算词汇重叠率
            answer_counter = Counter(answer_tokens)
            context_counter = Counter(context_tokens)
            
            # 计算交集词汇总频次
            overlap_count = sum((answer_counter & context_counter).values())
            
            # 计算答案词汇总数
            answer_count = sum(answer_counter.values())
            
            # 词汇覆盖率
            coverage_score = overlap_count / answer_count if answer_count > 0 else 0
            
            # N-gram匹配评分 (bigram)
            def get_ngrams(tokens, n=2):
                return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
            
            answer_bigrams = set(get_ngrams(answer_tokens))
            context_bigrams = set(get_ngrams(context_tokens))
            
            bigram_overlap = len(answer_bigrams.intersection(context_bigrams))
            bigram_score = bigram_overlap / len(answer_bigrams) if answer_bigrams else 0
            
            # 综合评分 (加权平均)
            faithfulness_score = 0.7 * coverage_score + 0.3 * bigram_score
            
            return min(faithfulness_score, 1.0)  # 确保分数不超过1
            
        except Exception as e:
            logger.error(f"忠实度评分失败: {str(e)}")
            return 0.0
            
    def evaluate_answer_relevancy(self, answer: str, question: str) -> float:
        """评估答案与问题的相关性
        
        Args:
            answer: 生成的答案
            question: 原始问题
            
        Returns:
            float: 相关性分数 (0-1)
        """
        try:
            if not answer or not question:
                return 0.0
                
            # 分词处理
            answer_tokens = self._tokenize(answer)
            question_tokens = self._tokenize(question)
            
            if not answer_tokens or not question_tokens:
                return 0.0
            
            # 识别问题关键词（去除停用词）
            stopwords = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', 
                         '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
                         '会', '着', '没有', '看', '好', '自己', '这', '那', '这个', '那个',
                         '怎么', '什么', '如何', '为什么', '吗', '呢', '啊', '吧'}
            
            question_keywords = [token for token in question_tokens if token not in stopwords]
            
            # 计算关键词覆盖率
            keyword_hits = sum(1 for keyword in question_keywords if keyword in answer_tokens)
            keyword_coverage = keyword_hits / len(question_keywords) if question_keywords else 0
            
            # 问题主题词在答案中的密度
            question_main_words = [token for token in question_keywords if len(token) > 1]
            if question_main_words:
                main_word_density = sum(answer_tokens.count(word) for word in question_main_words) / len(answer_tokens)
            else:
                main_word_density = 0
                
            # 综合评分
            relevancy_score = 0.7 * keyword_coverage + 0.3 * main_word_density
            
            return min(relevancy_score, 1.0)
            
        except Exception as e:
            logger.error(f"答案相关性评分失败: {str(e)}")
            return 0.0

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
                    
                    # 评估忠实度
                    faithfulness_score = self.evaluate_faithfulness(generated_answer, context_for_eval)
                    logger.info(f"忠实度评分: {faithfulness_score:.4f}")
                    
                    # 评估相关性
                    relevancy_score = self.evaluate_answer_relevancy(generated_answer, question)
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
    test_data_path = "evaluate/test_data/generation_test_data.json"
    evaluator.run_evaluation(test_data_path)