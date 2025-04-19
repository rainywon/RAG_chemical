"""
评估RAG系统的生成模块性能
实现了两种核心评估指标：
1. 忠实度(Faithfulness)：生成答案与检索上下文的一致性
2. 答案相关性(Answer Relevancy)：生成答案与用户问题的相关性
"""

import json
import logging
import os
import sys
import numpy as np
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# 导入RAGAS评估框架相关模块
from ragas.llms.base import BaseRagasLLM
from ragas.llms.prompt import PromptValue
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_core.callbacks import Callbacks
from langchain_core.outputs import LLMResult
from datasets import Dataset
import typing as t
import pandas as pd
import asyncio
from langchain_core.outputs.generation import Generation
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaLLM

# 添加父目录到路径，以便导入项目模块
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from config import Config
from rag_system import RAGSystem
from build_vector_store import VectorDBBuilder

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 配置文件路径
TEST_DATA_PATH = r"C:\wu\ArtifexAI\chemical_rag\evaluate\test_data\generation_test_data.json"
RESULT_PATH = r"C:\wu\ArtifexAI\chemical_rag\evaluate\results\generation_results.json"
# 详细记录目录
DETAIL_DIR = r"C:\wu\ArtifexAI\chemical_rag\evaluate\results\generation_details"

# 确保详细记录目录存在
Path(DETAIL_DIR).mkdir(parents=True, exist_ok=True)

# 定义JSON序列化的辅助函数
class NumpyEncoder(json.JSONEncoder):
    """处理NumPy数组的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

def numpy_safe_dump(obj, fp, **kwargs):
    """安全将可能包含NumPy对象的数据写入文件"""
    return json.dump(obj, fp, cls=NumpyEncoder, **kwargs)

class MyLLM(BaseRagasLLM):
    """
    自定义RAGAS评估用LLM类，基于ollama模型
    """
    def __init__(self, config: Config):
        """
        初始化评估用LLM模型
        
        Args:
            config: 配置对象
        """
        # 使用与RAG系统相同的OllamaLLM实例
        self.ollama_llm = OllamaLLM(
            model="deepseek_8B:latest",  # 模型名称
            base_url=config.ollama_base_url,  # Ollama服务地址
            temperature=config.llm_temperature,  # 温度参数
            num_predict=config.llm_max_tokens,  # 最大生成token数
            stop=["<|im_end|>"]
        )
        logger.info("✅ Ollama模型初始化完成")

    @property
    def llm(self):
        return self.ollama_llm

    def generate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks: Callbacks = None,
    ) -> LLMResult:
        """生成文本"""
        if callbacks is None:
            callbacks = []
            
        content = prompt.to_string()
        try:
            # 直接调用ollama_llm的invoke方法
            text = self.ollama_llm.invoke(content)
            
            # 构造LLMResult
            generations = [[Generation(text=text)]]
            token_total = len(text)
            llm_output = {'token_total': token_total}
            
            return LLMResult(generations=generations, llm_output=llm_output)
        except Exception as e:
            logger.error(f"生成文本失败: {str(e)}")
            # 返回空结果
            return LLMResult(generations=[[Generation(text="")]], llm_output={'token_total': 0})

    async def agenerate_text(
            self,
            prompt: PromptValue,
            n: int = 1,
            temperature: float = 1e-8,
            stop: t.Optional[t.List[str]] = None,
            callbacks: Callbacks = None,
    ) -> LLMResult:
        """异步生成文本"""
        if callbacks is None:
            callbacks = []
            
        content = prompt.to_string()
        try:
            # 异步调用ollama_llm
            text = await self.ollama_llm.ainvoke(content)
            
            # 构造LLMResult
            generations = [[Generation(text=text)]]
            token_total = len(text)
            llm_output = {'token_total': token_total}
            
            return LLMResult(generations=generations, llm_output=llm_output)
        except Exception as e:
            logger.error(f"异步生成文本失败: {str(e)}")
            # 返回空结果
            return LLMResult(generations=[[Generation(text="")]], llm_output={'token_total': 0})

class CustomFaithfulness:
    """自定义忠实度评估实现"""
    
    def __init__(self, llm):
        self.llm = llm
        self.evaluation_details = []  # 存储评估细节
        
    def extract_statements(self, answer):
        """提取答案中的事实性声明"""
        prompt = f"""
        请将以下答案分解为独立的、简短的事实性声明。每个声明应该是一个简单的事实。
        将每个声明分别列在新行，以数字序号开头。不要添加额外的解释。

        答案: {answer}

        声明列表:
        """
        
        try:
            result = self.llm.invoke(prompt)
            statements = []
            
            # 简单解析序号开头的行
            for line in result.strip().split("\n"):
                line = line.strip()
                if line and (line[0].isdigit() or (len(line) > 2 and line[0:2] in ["1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."])):
                    # 移除序号和点
                    statement = line
                    for i in range(1, 10):
                        if statement.startswith(f"{i}."):
                            statement = statement[2:].strip()
                            break
                        elif statement.startswith(f"{i} "):
                            statement = statement[2:].strip()
                            break
                    statements.append(statement)
            
            if not statements:
                # 尝试其他方式分割
                statements = [s.strip() for s in result.split("\n") if s.strip()]
                
            logger.debug(f"提取了 {len(statements)} 个声明")
            return statements
        except Exception as e:
            logger.error(f"提取声明失败: {str(e)}")
            return []
    
    def verify_statement(self, statement, contexts):
        """验证声明是否被上下文支持"""
        context_text = "\n\n".join(contexts)
        prompt = f"""
        给定以下上下文和声明，判断声明是否完全由上下文支持。
        如果声明中的所有信息都能从上下文中直接找到，回答"是"。
        如果声明包含上下文中没有的信息，或与上下文矛盾，回答"否"。
        只回答"是"或"否"，不要解释。

        上下文: {context_text}

        声明: {statement}

        判断结果:
        """
        
        try:
            result = self.llm.invoke(prompt).strip().lower()
            
            # 改进的响应解析，处理中文否定回答
            # 首先检查最后一行（模型可能会输出一段思考过程然后下结论）
            if "\n" in result:
                last_line = result.split("\n")[-1].strip().lower()
                if last_line:
                    result = last_line
            
            # 明确检查"否"和"不"等中文否定词
            if "否" in result or "不" in result or "no" in result or "not" in result or "unsupported" in result:
                is_supported = False
            # 明确检查肯定回答
            elif "是" in result or "yes" in result or "support" in result or "supported" in result:
                is_supported = True
            # 默认情况，基于更严格的匹配
            else:
                exact_match = result == "是" or result == "yes" or result.endswith("是") or result.endswith("yes")
                is_supported = exact_match
            
            # 保存判断结果和原始响应
            verification_detail = {
                "statement": statement,
                "is_supported": is_supported,
                "model_response": result
            }
            self.evaluation_details.append(verification_detail)
            
            return is_supported
            
        except Exception as e:
            logger.error(f"验证声明失败: {str(e)}")
            self.evaluation_details.append({
                "statement": statement,
                "is_supported": False,
                "model_response": f"错误: {str(e)}"
            })
            return False
    
    def calculate_faithfulness(self, answer, contexts):
        """计算答案的忠实度分数"""
        # 清空之前的评估细节
        self.evaluation_details = []
        
        statements = self.extract_statements(answer)
        if not statements:
            logger.warning("没有提取到任何声明，返回默认忠实度0")
            return 0.0
        
        supported_count = 0
        total_count = len(statements)
        
        for statement in statements:
            is_supported = self.verify_statement(statement, contexts)
            if is_supported:
                supported_count += 1
        
        # 计算忠实度分数
        if total_count == 0:
            return 0.0
        
        faithfulness_score = supported_count / total_count
        logger.info(f"忠实度分数: {faithfulness_score:.4f} ({supported_count}/{total_count})")
        return faithfulness_score
    
    def get_evaluation_details(self):
        """获取评估详细信息"""
        return self.evaluation_details

def clear_directory(dir_path):
    """清空指定目录下的所有文件"""
    try:
        dir_path = Path(dir_path)
        if dir_path.exists():
            for file_path in dir_path.glob("*"):
                if file_path.is_file():
                    file_path.unlink()
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
            logger.info(f"已清空目录: {dir_path}")
        else:
            logger.warning(f"目录不存在，无需清空: {dir_path}")
    except Exception as e:
        logger.error(f"清空目录失败: {str(e)}")

class GenerationEvaluator:
    """生成模块评估器，评估忠实度和答案相关性"""
    
    def __init__(self, config: Config):
        """
        初始化评估器
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        self.result_dir = Path(RESULT_PATH).parent
        self.result_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化评估用模型
        logger.info("初始化评估用LLM模型...")
        self.my_llm = MyLLM(config)
        
        logger.info("初始化评估用嵌入模型...")
        # 直接使用RAG系统中的嵌入模型
        self.embedding_model = self.rag_system.embeddings
        
        # 配置评估指标
        logger.info("配置评估指标...")
        faithfulness.llm = self.my_llm
        answer_relevancy.llm = self.my_llm
        answer_relevancy.embeddings = self.embedding_model
        
        # 初始化自定义忠实度评估
        self.custom_faithfulness = CustomFaithfulness(self.my_llm.ollama_llm)
        
        logger.info("生成评估器初始化完成")
    
    def save_query_details(self, idx: int, question: str, answer: str, contexts: List[str], 
                           faith_score: float, rel_score: float, evaluation_details: List[Dict]):
        """
        保存单个查询的详细信息到文件
        
        Args:
            idx: 查询索引
            question: 问题
            answer: 答案
            contexts: 上下文列表
            faith_score: 忠实度分数
            rel_score: 答案相关性分数
            evaluation_details: 评估细节
        """
        # 构建文件名，包含索引和时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_{idx+1:03d}_{timestamp}.txt"
        filepath = Path(DETAIL_DIR) / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入问题和分数
                f.write(f"查询索引: {idx+1}\n")
                f.write(f"时间戳: {timestamp}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"问题: {question}\n\n")
                f.write(f"忠实度(Faithfulness): {faith_score:.4f}\n")
                f.write(f"答案相关性(Answer Relevancy): {rel_score:.4f}\n\n")
                f.write(f"{'='*80}\n\n")
                
                # 写入答案
                f.write(f"生成答案:\n")
                f.write(f"{'-'*80}\n")
                f.write(f"{answer}\n\n")
                f.write(f"{'='*80}\n\n")
                
                # 写入上下文
                f.write(f"检索上下文 ({len(contexts)}个):\n")
                for i, ctx in enumerate(contexts):
                    f.write(f"{'-'*80}\n")
                    f.write(f"上下文 {i+1}:\n")
                    f.write(f"{ctx}\n\n")
                f.write(f"{'='*80}\n\n")
                
                # 写入声明评估详情
                f.write(f"声明评估详情:\n")
                f.write(f"{'-'*80}\n")
                for i, detail in enumerate(evaluation_details):
                    f.write(f"声明 {i+1}: {detail['statement']}\n")
                    f.write(f"支持: {'是' if detail['is_supported'] else '否'}\n")
                    f.write(f"模型回复: {detail['model_response']}\n\n")
                
            logger.info(f"查询 {idx+1} 的详细信息已保存至: {filepath}")
        except Exception as e:
            logger.error(f"保存查询详情失败: {str(e)}")
    
    def evaluate_single_query(self, question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
        """
        评估单个查询的忠实度和答案相关性
        
        Args:
            question: 问题
            answer: 生成的答案
            contexts: 上下文列表
            
        Returns:
            包含评估分数的字典
        """
        try:
            # 检查输入数据有效性
            if not answer.strip():
                logger.warning("评估失败: 答案为空")
                return {"faithfulness": 0.0, "answer_relevancy": 0.0}
            
            if not contexts or all(not ctx.strip() for ctx in contexts):
                logger.warning("评估失败: 上下文为空")
                return {"faithfulness": 0.0, "answer_relevancy": 0.0}
            
            # 创建单个问题的数据集
            data_dict = {
                "question": [question],
                "contexts": [contexts],
                "answer": [answer],
                "ground_truth": [""]  # 占位，评估时不使用
            }
            
            single_dataset = Dataset.from_dict(data_dict)
            
            # 使用RAGAS评估答案相关性
            try:
                result = evaluate(
                    dataset=single_dataset,
                    metrics=[answer_relevancy],
                    llm=self.my_llm,
                    embeddings=self.embedding_model,
                    is_async=False,
                    raise_exceptions=False,
                )
                result_df = result.to_pandas()
                rel_score = float(result_df["answer_relevancy"].iloc[0]) if not pd.isna(result_df["answer_relevancy"].iloc[0]) else 0.0
            except Exception as e:
                logger.error(f"RAGAS答案相关性评估失败: {str(e)}")
                rel_score = 0.0
            
            # 使用自定义逻辑评估忠实度
            faith_score = self.custom_faithfulness.calculate_faithfulness(answer, contexts)
            
            return {
                "faithfulness": faith_score,
                "answer_relevancy": rel_score,
                "evaluation_details": self.custom_faithfulness.get_evaluation_details()
            }
        except Exception as e:
            logger.error(f"单一问题评估失败: {str(e)}")
            return {
                "faithfulness": 0.0,
                "answer_relevancy": 0.0,
                "evaluation_details": []
            }
    
    def run_evaluation(self, test_data_path: str = TEST_DATA_PATH) -> Dict[str, Any]:
        """
        运行评估流程
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            评估结果
        """
        logger.info("开始评估生成模块性能...")
        
        # 清空详细记录目录
        logger.info("清空详细记录目录...")
        clear_directory(DETAIL_DIR)
        
        try:
            # 加载测试数据
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            # 初始化结果列表
            all_results = []
            
            # 逐个处理测试数据
            for idx, item in enumerate(test_data):
                question = item["question"]
                logger.info(f"【查询 {idx+1}/{len(test_data)}】: {question}")
                
                # 使用RAG系统获取答案和上下文
                answer, references, metadata = self.rag_system.answer_query(question)
                
                if metadata["status"] == "success":
                    # 合并上下文文本
                    contexts = [ref["content"] for ref in references]
                    
                    # 立即评估当前问题
                    scores = self.evaluate_single_query(question, answer, contexts)
                    
                    # 只打印问题和评估分数
                    logger.info(f"  • 忠实度(Faithfulness): {scores['faithfulness']:.4f}")
                    logger.info(f"  • 答案相关性(Answer Relevancy): {scores['answer_relevancy']:.4f}")
                    
                    # 立即保存详细信息到文件
                    self.save_query_details(
                        idx, 
                        question, 
                        answer, 
                        contexts, 
                        scores['faithfulness'], 
                        scores['answer_relevancy'],
                        scores['evaluation_details']
                    )
                    
                    # 保存结果
                    result_item = {
                        "question": question,
                        "answer": answer,
                        "faithfulness": scores["faithfulness"],
                        "answer_relevancy": scores["answer_relevancy"]
                    }
                    all_results.append(result_item)
                else:
                    logger.warning(f"  • 查询失败: {metadata.get('error', '未知错误')}")
            
            # 创建结果DataFrame
            result_df = pd.DataFrame(all_results)
            
            # 计算平均指标
            idx = len(result_df)
            result_df.loc[idx, "question"] = "平均值"
            result_df.loc[idx, "faithfulness"] = result_df["faithfulness"].mean()
            result_df.loc[idx, "answer_relevancy"] = result_df["answer_relevancy"].mean()
            
            # 保存评估结果
            self._save_results(result_df)
            
            # 返回评估结果
            return {
                "faithfulness": float(result_df["faithfulness"].mean()),
                "answer_relevancy": float(result_df["answer_relevancy"].mean()),
                "details": all_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            raise
    
    def _save_results(self, result_df: pd.DataFrame) -> None:
        """
        保存评估结果
        
        Args:
            result_df: 评估结果DataFrame
        """
        # 保存为JSON，使用安全的序列化方法
        result_json = {
            "faithfulness_mean": float(result_df["faithfulness"].mean()),
            "answer_relevancy_mean": float(result_df["answer_relevancy"].mean()),
            "details": result_df.to_dict(orient="records"),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(RESULT_PATH, 'w', encoding='utf-8') as f:
            numpy_safe_dump(result_json, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存至: {RESULT_PATH}")
        
        # 打印评估结果概要
        logger.info(f"评估结果概要:")
        logger.info(f"- 忠实度(Faithfulness): {result_json['faithfulness_mean']:.4f}")
        logger.info(f"- 答案相关性(Answer Relevancy): {result_json['answer_relevancy_mean']:.4f}")

if __name__ == "__main__":
    try:
        # 加载配置
        config_path = Path(parent_dir) / "config.py"
        if not config_path.exists():
            logger.error(f"未找到配置文件: {config_path}")
            sys.exit(1)
            
        config = Config()
        
        # 运行评估
        evaluator = GenerationEvaluator(config)
        evaluator.run_evaluation(TEST_DATA_PATH)
        
    except Exception as e:
        logger.exception(f"评估过程发生错误: {str(e)}")
        sys.exit(1)
