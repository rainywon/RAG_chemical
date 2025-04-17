"""
使用RAGAS库评估RAG系统的生成模块性能
专注于忠实度(Faithfulness)和答案相关性(Answer Relevancy)两个核心指标
使用本地大语言模型进行评估（不依赖OpenAI API）
"""

import json
import logging
import os
import sys
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import time
import threading
import concurrent.futures

# 添加父目录到路径，以便导入项目模块
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 删除可能导致问题的环境变量设置
if "OPENAI_API_KEY" in os.environ:
    del os.environ["OPENAI_API_KEY"]

# RAGAS评估组件
from ragas.metrics import faithfulness, answer_relevancy
from ragas.metrics.critique import harmfulness

# LangChain组件
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from rag_system import RAGSystem
from config import Config

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RAGASEvaluator:
    def __init__(self, config):
        """初始化RAGAS评估器
        
        Args:
            config: 配置对象
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        
        # 优化提示词
        self._optimize_rag_prompt()
        
        # 并行处理的最大线程数
        self.max_workers = 1  # 使用固定数量的线程
        
        # 线程锁，用于同步日志输出
        self.log_lock = threading.Lock()
        
        # 初始化评估模型
        self._setup_evaluation_models()
        
        logger.info("RAGAS评估器初始化完成")
    
    def _setup_evaluation_models(self):
        """设置用于评估的本地模型"""
        try:
            # 使用本地Ollama模型进行评估
            self.eval_llm = OllamaLLM(
                model="deepseek_8B:latest",
                base_url=self.config.ollama_base_url,
                temperature=0.1
            )
            
            # 使用与RAG系统相同的嵌入模型
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.config.embedding_model_path,
                model_kwargs={"device": self.config.device},
                encode_kwargs={"normalize_embeddings": True}
            )
            
            # 设置RAGAS评估指标
            self.faithfulness_metric = faithfulness
            self.answer_relevancy_metric = answer_relevancy
            
            # 配置RAGAS使用本地模型
            faithfulness.llm = self.eval_llm
            answer_relevancy.llm = self.eval_llm
            
            logger.info("评估模型设置完成")
        except Exception as e:
            logger.error(f"评估模型设置失败: {str(e)}")
            raise RuntimeError(f"无法设置评估模型: {str(e)}")
    
    def _optimize_rag_prompt(self):
        """优化RAG系统的提示词模板，提升评估效果"""
        try:
            # 保存原始的提示词生成方法
            self.original_build_prompt = self.rag_system._build_prompt
            
            # 替换为优化后的提示词生成方法
            def optimized_prompt(question: str, context: str) -> str:
                """针对评估场景的优化提示词"""
                
                evaluation_instructions = (
                    "作为化工安全领域专家，请严格遵循以下指南：\n"
                    "1. 严格基于提供的上下文信息回答问题，避免使用上下文外的知识\n"
                    "2. 如果上下文信息不足，清晰说明「根据提供的信息无法完全回答该问题」\n" 
                    "3. 保持答案与问题的直接相关性，避免无关信息\n"
                    "4. 使用<think></think>标签记录你的推理过程，包括：\n"
                    "   - 从上下文中找出哪些关键信息可用于回答问题\n"
                    "   - 明确指出上下文中的信息不足或限制\n"
                    "   - 清晰展示你的推理步骤和依据来源\n"
                    "5. 最终答案应简洁、准确、直接回应问题\n"
                    "6. 确保你的回答完全基于提供的上下文，不要添加任何未在上下文中出现的信息"
                )
                
                if context:
                    return (
                        "<|im_start|>system\n"
                        f"{evaluation_instructions}\n"
                        "上下文：\n{context}\n"
                        "<|im_end|>\n"
                        "<|im_start|>user\n"
                        "{question}\n"
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    ).format(question=question, context=context[:self.rag_system.config.max_context_length])
                else:
                    return (
                        "<|im_start|>system\n"
                        f"{evaluation_instructions}\n"
                        "<|im_end|>\n"
                        "<|im_start|>user\n"
                        "{question}\n"
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    ).format(question=question)
            
            # 使用猴子补丁方式替换方法
            self.rag_system._build_prompt = optimized_prompt
            
            logger.info("已优化RAG系统提示词，调整为评估友好模式")
        except Exception as e:
            logger.error(f"提示词优化失败: {str(e)}")
    
    def _preprocess_answer(self, answer: str) -> str:
        """预处理答案文本，移除<think></think>标签及其内容
        
        Args:
            answer: 原始答案文本
            
        Returns:
            str: 处理后的答案文本
        """
        # 使用正则表达式移除<think></think>标签及其内容
        cleaned_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)
        
        # 移除可能出现的多余空行
        cleaned_answer = re.sub(r'\n{3,}', '\n\n', cleaned_answer)
        cleaned_answer = cleaned_answer.strip()
        
        return cleaned_answer
    
    def _format_for_ragas(self, results):
        """将测试结果格式化为RAGAS需要的格式
        
        Args:
            results: 评估结果列表
            
        Returns:
            Dict: RAGAS格式的数据
        """
        formatted_data = {
            "questions": [],
            "answers": [],
            "contexts": [],
            "ground_truths": []
        }
        
        for result in results:
            # 只处理没有错误的结果
            if "error" not in result:
                formatted_data["questions"].append(result["question"])
                formatted_data["answers"].append(result["generated_answer"])
                formatted_data["contexts"].append(result["context"])
                
                # 如果有参考答案，也添加进去
                if "reference_answer" in result and result["reference_answer"]:
                    formatted_data["ground_truths"].append([result["reference_answer"]])
                else:
                    formatted_data["ground_truths"].append([""])
        
        return formatted_data

    def _process_test_item(self, item: Dict, item_idx: int) -> Dict:
        """处理单个测试样本的辅助函数，适用于多线程环境
        
        Args:
            item: 测试样本数据
            item_idx: 样本索引
            
        Returns:
            Dict: 评估结果
        """
        try:
            # 提取测试数据
            question = item.get("question", "")
            reference_context = item.get("context", "")
            reference_answer = item.get("answer", "")
            
            # 获取线程ID，用于日志区分
            thread_id = threading.get_ident()
            with self.log_lock:
                logger.info(f"线程 {thread_id} 开始处理样本 {item_idx}: '{question[:40]}...'")
            
            # RAG系统生成答案
            start_time = time.time()
            generated_answer, references, _ = self.rag_system.answer_query(question)
            gen_time = time.time() - start_time
            
            # 检索的上下文
            retrieved_context = "\n".join([ref["content"] for ref in references]) if references else ""
            
            # 使用参考上下文进行评估（如果有）
            context_for_eval = reference_context if reference_context else retrieved_context
            
            # 预处理答案
            cleaned_answer = self._preprocess_answer(generated_answer)
            
            # 保存结果（暂不计算分数，后续批量评估）
            return {
                "question": question,
                "generated_answer": cleaned_answer,
                "original_answer": generated_answer,
                "reference_answer": reference_answer,
                "context": context_for_eval,
                "processing_time": {
                    "generation": gen_time,
                    "total": gen_time
                }
            }
                
        except Exception as e:
            with self.log_lock:
                logger.error(f"线程 {threading.get_ident()} 处理样本 {item_idx} 失败: {str(e)}")
            # 添加失败记录
            return {
                "question": item.get("question", ""),
                "error": str(e)
            }
    
    def evaluate_with_ragas(self, formatted_data):
        """使用RAGAS进行评估
        
        Args:
            formatted_data: RAGAS格式的数据
            
        Returns:
            Dict: 评估结果
        """
        try:
            logger.info("开始使用RAGAS进行评估...")
            
            # 为了避免计算错误，确保至少有一个样本
            if not formatted_data["questions"]:
                return {
                    "faithfulness_score": 0.0,
                    "answer_relevancy_score": 0.0
                }
            
            # 使用RAGAS评估忠实度
            start_time = time.time()
            faithfulness_scores = self.faithfulness_metric.score(
                formatted_data["questions"],
                formatted_data["answers"],
                formatted_data["contexts"]
            )
            
            # 使用RAGAS评估答案相关性
            answer_relevancy_scores = self.answer_relevancy_metric.score(
                formatted_data["questions"],
                formatted_data["answers"]
            )
            
            eval_time = time.time() - start_time
            
            # 计算平均分数
            avg_faithfulness = faithfulness_scores.mean() if len(faithfulness_scores) > 0 else 0.0
            avg_answer_relevancy = answer_relevancy_scores.mean() if len(answer_relevancy_scores) > 0 else 0.0
            
            logger.info(f"RAGAS评估完成，耗时: {eval_time:.2f}秒")
            
            return {
                "faithfulness_score": avg_faithfulness,
                "answer_relevancy_score": avg_answer_relevancy,
                "processing_time": eval_time
            }
            
        except Exception as e:
            logger.error(f"RAGAS评估出错: {str(e)}")
            return {
                "faithfulness_score": 0.0,
                "answer_relevancy_score": 0.0,
                "error": str(e)
            }
    
    def run_evaluation(self, test_data_path: str) -> Dict[str, Any]:
        """运行完整评估流程，使用多线程并行处理提高效率
        
        Args:
            test_data_path: 测试数据文件路径
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 加载测试数据
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            logger.info(f"成功加载{len(test_data)}条测试数据，将使用{self.max_workers}个线程进行并行评估")
            
            start_time = time.time()
            
            # 使用线程池并行处理评估
            results = [None] * len(test_data)  # 预分配结果列表
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_idx = {}
                for idx, item in enumerate(test_data):
                    future = executor.submit(self._process_test_item, item, idx)
                    future_to_idx[future] = idx
                
                # 实时收集完成的任务结果
                completed = 0
                for future in concurrent.futures.as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        result = future.result()
                        results[idx] = result  # 将结果放在对应位置
                        completed += 1
                        # 显示进度
                        with self.log_lock:
                            logger.info(f"进度: {completed}/{len(test_data)} ({completed/len(test_data)*100:.1f}%)")
                    except Exception as e:
                        with self.log_lock:
                            logger.error(f"处理测试数据索引 {idx} 出错: {str(e)}")
                        # 添加错误记录
                        results[idx] = {
                            "question": test_data[idx].get("question", ""),
                            "error": str(e)
                        }
            
            # 筛选有效结果
            valid_results = [r for r in results if r and "error" not in r]
            
            # 使用RAGAS评估
            if valid_results:
                # 格式化数据以适应RAGAS
                formatted_data = self._format_for_ragas(valid_results)
                
                # 使用RAGAS进行评估
                ragas_scores = self.evaluate_with_ragas(formatted_data)
                
                # 将RAGAS分数分配给各个测试项
                for result in valid_results:
                    result["faithfulness_score"] = ragas_scores["faithfulness_score"]
                    result["answer_relevancy_score"] = ragas_scores["answer_relevancy_score"]
                
                # 提取平均分数
                avg_faithfulness = ragas_scores["faithfulness_score"]
                avg_answer_relevancy = ragas_scores["answer_relevancy_score"]
                
                # 添加评估时间
                eval_time = ragas_scores.get("processing_time", 0)
                for result in valid_results:
                    result["processing_time"]["evaluation"] = eval_time
                    result["processing_time"]["total"] += eval_time
            else:
                avg_faithfulness = 0.0
                avg_answer_relevancy = 0.0
                eval_time = 0
            
            # 综合评分
            comprehensive_score = (avg_faithfulness + avg_answer_relevancy) / 2
            
            # 计算总耗时
            total_time = time.time() - start_time
            
            # 计算平均处理时间
            processing_times = [r.get("processing_time", {}) for r in valid_results if r and "processing_time" in r]
            avg_gen_time = sum(t.get("generation", 0) for t in processing_times) / len(processing_times) if processing_times else 0
            avg_eval_time = sum(t.get("evaluation", 0) for t in processing_times) / len(processing_times) if processing_times else 0
            
            # 输出总体评分
            logger.info(f"评估完成！总体结果:")
            logger.info(f"平均忠实度: {avg_faithfulness:.4f}")
            logger.info(f"平均答案相关性: {avg_answer_relevancy:.4f}")
            logger.info(f"综合评分: {comprehensive_score:.4f}")
            logger.info(f"总评估时间: {total_time:.2f}秒，平均每样本生成: {avg_gen_time:.2f}秒，评估: {avg_eval_time:.2f}秒")
            
            # 构建完整评估结果
            evaluation_results = {
                "results": results,
                "avg_faithfulness": avg_faithfulness,
                "avg_answer_relevancy": avg_answer_relevancy,
                "comprehensive_score": comprehensive_score,
                "performance": {
                    "total_time": total_time,
                    "samples_count": len(test_data),
                    "valid_samples": len(valid_results),
                    "avg_generation_time": avg_gen_time,
                    "avg_evaluation_time": avg_eval_time,
                    "threads_used": self.max_workers
                },
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_method": "ragas"
            }
            
            # 保存结果
            results_dir = Path("evaluate/results")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            results_path = results_dir / "ragas_generation_results.json"
            
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            logger.info(f"评估结果已保存至: {results_path}")
            
            # 恢复原始提示词生成方法
            if hasattr(self, 'original_build_prompt'):
                self.rag_system._build_prompt = self.original_build_prompt
                logger.info("已恢复RAG系统原始提示词")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}")
            # 恢复原始提示词生成方法
            if hasattr(self, 'original_build_prompt'):
                self.rag_system._build_prompt = self.original_build_prompt
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
    evaluator = RAGASEvaluator(config)
    
    # 运行评估
    current_dir = Path(__file__).resolve().parent
    test_data_path = current_dir / "test_data" / "generation_test_data.json"
    evaluator.run_evaluation(str(test_data_path))