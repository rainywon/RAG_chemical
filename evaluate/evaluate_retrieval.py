"""
评估RAG系统的检索模块性能
实现多种传统核心评估指标：
1. 命中率(Hit Rate@k)：检索结果的前k个中是否包含相关文档，包括Hit@1, Hit@3, Hit@5, Hit@10
2. 平均倒数排名(Mean Reciprocal Rank, MRR)：相关文档在结果中的排名评估
"""

import sys
import json
import logging
import os
import numpy as np
import pandas as pd
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union

# 添加父目录到路径，以便导入项目模块
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from config import Config
from rag_system import RAGSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 获取项目根目录
root_dir = Path(__file__).resolve().parent.parent

# 配置文件路径
TEST_DATA_PATH = root_dir / "evaluate" / "test_data" / "retrieval_test_data.json"
RESULT_PATH = root_dir / "evaluate" / "results" / "retrieval_results.json"
# 详细记录目录
DETAIL_DIR = root_dir / "evaluate" / "results" / "retrieval_details"

# 确保详细记录目录存在
Path(DETAIL_DIR).mkdir(parents=True, exist_ok=True)


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


class PathUtils:
    """路径处理工具类"""
    
    @staticmethod
    def normalize_path(path):
        """标准化路径格式，便于比较
        
        Args:
            path: 原始路径
            
        Returns:
            标准化后的路径
        """
        # 确保路径分隔符一致
        norm_path = os.path.normpath(path).replace('\\', '/')
        # 确保小写比较（Windows不区分大小写）
        return norm_path.lower()
    
    @staticmethod
    def check_path_match(path1, path2):
        """检查两个路径是否匹配
        
        Args:
            path1: 第一个路径
            path2: 第二个路径
            
        Returns:
            是否匹配
        """
        norm1 = PathUtils.normalize_path(path1)
        norm2 = PathUtils.normalize_path(path2)
        
        # 完全匹配
        if norm1 == norm2:
            return True
        
        # 文件名匹配
        file1 = os.path.basename(norm1)
        file2 = os.path.basename(norm2)
        if file1 and file2 and file1 == file2:
            return True
        
        # 路径末尾匹配
        if norm1.endswith(norm2) or norm2.endswith(norm1):
            return True
        
        return False
    

class RetrievalMetrics:
    """检索评估指标计算类"""
    
    @staticmethod
    def calculate_hit_rate(retrieved_paths: List[str], relevant_docs: List[str], k: int) -> float:
        """计算命中率 - 检索结果中是否包含至少一个相关文档
        
        Args:
            retrieved_paths: 检索得到的文档路径列表
            relevant_docs: 真实相关文档列表
            k: 考虑的检索结果数量
            
        Returns:
            命中率分数(0或1)
        """
        # 获取前K个结果
        top_k_docs = retrieved_paths[:k] if len(retrieved_paths) >= k else retrieved_paths
        
        # 检查是否命中
                    for ref_doc in relevant_docs:
                        for ret_doc in top_k_docs:
                if PathUtils.check_path_match(ref_doc, ret_doc):
                    return 1.0
        
        return 0.0
    
    @staticmethod
    def calculate_mrr(retrieved_paths: List[str], relevant_docs: List[str]) -> float:
        """计算平均倒数排名(MRR) - 相关文档首次出现位置的倒数
        
        Args:
            retrieved_paths: 检索得到的文档路径列表
            relevant_docs: 真实相关文档列表
            
        Returns:
            MRR分数(0到1之间)
        """
        # 计算倒数排名
                    for i, doc_path in enumerate(retrieved_paths):
                    for ref_doc in relevant_docs:
                if PathUtils.check_path_match(doc_path, ref_doc):
                            # 找到第一个相关文档的位置(从1开始计数)
                    return 1.0 / (i + 1)
        
        # 如果没有找到相关文档
        return 0.0


class ReportGenerator:
    """评估报告生成器"""
    
    @staticmethod
    def generate_markdown_report(results: Dict[str, Any], records: List[Dict[str, Any]], k_values: List[int]) -> str:
        """生成Markdown格式的详细评估报告
        
        Args:
            results: 评估结果
            records: 评估记录
            k_values: k值列表
            
        Returns:
            Markdown格式报告文本
        """
        now = datetime.now()
        report = []
        
        # 报告标题
        report.append("# RAG系统检索模块评估报告")
        report.append(f"生成时间：{now.strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 总体评估结果
        report.append("## 1. 总体评估结果")
        report.append("### 检索评估指标")
        for k in k_values:
            report.append(f"- **hit@{k}**: {results['hit_rate'].get(f'hit@{k}', 0):.4f}")
        report.append(f"- **MRR**: {results['mrr']:.4f}\n")
        
        # 各查询评估详情
        report.append("## 2. 各查询评估详情")
        
        for record in records:
            report.append(f"### 查询 {record['query_index']}")
            report.append(f"**问题**: {record['query']}")
            
            # 相关文档
            report.append("\n**相关文档**:")
            for doc in record['relevant_docs']:
                report.append(f"- `{doc}`")
            
            # 检索结果
            report.append("\n**检索结果**:")
            report.append("| 排名 | 文档路径 | 是否相关 |")
            report.append("|------|---------|----------|")
            
            for doc in record.get('retrieved_docs', [])[:10]:  # 只显示前10个结果
                is_relevant = "✓" if doc.get('is_relevant', False) else "✗"
                report.append(f"| {doc['rank']} | `{doc['path']}` | {is_relevant} |")
            
            # 评估结果
            report.append("\n**评估结果**:")
            for k in k_values:
                hit_status = "✓ 命中" if record.get(f'hit@{k}_result', False) else "✗ 未命中"
                report.append(f"- **hit@{k}**: {hit_status}")
            report.append(f"- **MRR**: {record.get('mrr_result', 0):.4f}")
            
            report.append("\n---\n")
        
        return "\n".join(report)
    

class RetrievalEvaluator:
    """检索模块评估器，评估命中率和MRR"""
    
    def __init__(self, config: Config):
        """初始化评估器
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        
        # 获取项目根目录
        self.result_dir = Path(RESULT_PATH).parent
        self.result_dir.mkdir(exist_ok=True, parents=True)
        
        # 定义多个k值进行评估
        self.k_values = [1, 3, 5, 10]
        # 初始化评估记录
        self.evaluation_records = []
        
        logger.info("检索评估器初始化完成")
    
    def save_query_details(self, idx: int, question: str, 
                          retrieved_paths: List[str], relevant_docs: List[str],
                          hit_scores: Dict[str, float], mrr_score: float):
        """
        保存单个查询的详细信息到文件
        
        Args:
            idx: 查询索引
            question: 问题
            retrieved_paths: 检索到的文档路径
            relevant_docs: 真实相关文档列表
            hit_scores: 各k值的命中率分数
            mrr_score: MRR分数
        """
        # 构建文件名，包含索引和时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"retrieval_{idx+1:03d}_{timestamp}.txt"
        filepath = Path(DETAIL_DIR) / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 写入问题和分数
                f.write(f"查询索引: {idx+1}\n")
                f.write(f"时间戳: {timestamp}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write(f"问题: {question}\n\n")
                # 写入各k值的hit@k指标
                for k in self.k_values:
                    f.write(f"命中率(Hit@{k}): {hit_scores[f'hit@{k}']:.4f}\n")
                f.write(f"平均倒数排名(MRR): {mrr_score:.4f}\n\n")
                f.write(f"{'='*80}\n\n")
                
                # 写入真实相关文档
                f.write(f"真实相关文档:\n")
                f.write(f"{'-'*80}\n")
                for i, doc in enumerate(relevant_docs):
                    f.write(f"{i+1}. {doc}\n")
                f.write(f"\n{'='*80}\n\n")
                
                # 写入检索到的文档
                f.write(f"检索到的文档 (前{min(len(retrieved_paths), 10)}个):\n")
                f.write(f"{'-'*80}\n")
                for i, path in enumerate(retrieved_paths[:10]):
                    is_relevant = any(PathUtils.check_path_match(path, doc) for doc in relevant_docs)
                    relevance_mark = "✓" if is_relevant else "✗"
                    f.write(f"{i+1}. [{relevance_mark}] {path}\n")
                f.write(f"\n{'='*80}\n\n")
            
            logger.info(f"查询 {idx+1} 的详细信息已保存至: {filepath}")
        except Exception as e:
            logger.error(f"保存查询详情失败: {str(e)}")
    
    def evaluate_single_query(self, question: str, relevant_docs: List[str]) -> Dict[str, Any]:
        """
        评估单个查询的检索性能
        
        Args:
            question: 问题
            relevant_docs: 真实相关文档列表
            
        Returns:
            包含评估分数的字典
        """
        try:
            # 使用RAG系统进行检索
            retrieved_docs, _ = self.rag_system._retrieve_documents(question)
            
            # 获取检索文档路径
            retrieved_paths = [doc.metadata.get("source", "") for doc in retrieved_docs]
            
            # 初始化结果
            results = {"retrieved_paths": retrieved_paths}
            
            # 评估各k值的命中率
            for k in self.k_values:
                hit_score = RetrievalMetrics.calculate_hit_rate(retrieved_paths, relevant_docs, k)
                results[f"hit@{k}"] = hit_score
            
            # 评估MRR
            mrr_score = RetrievalMetrics.calculate_mrr(retrieved_paths, relevant_docs)
            results["mrr"] = mrr_score
            
            # 返回评估结果
            return results
            
        except Exception as e:
            logger.error(f"单一问题评估失败: {str(e)}")
            results = {"retrieved_paths": []}
            for k in self.k_values:
                results[f"hit@{k}"] = 0.0
            results["mrr"] = 0.0
            return results
    
    def _process_single_query(self, idx: int, item: Dict[str, Any], total_queries: int) -> Dict[str, Any]:
        """处理单个查询，包括评估和记录结果
        
        Args:
            idx: 查询索引
            item: 查询数据项
            total_queries: 总查询数
            
        Returns:
            查询结果
        """
        question = item["question"]
        relevant_docs = item["relevant_docs"]
        
        logger.info(f"【查询 {idx+1}/{total_queries}】: {question}")
        
        # 评估当前问题
        scores = self.evaluate_single_query(question, relevant_docs)
        
        # 打印评估分数
        for k in self.k_values:
            logger.info(f"  • 命中率(Hit@{k}): {scores[f'hit@{k}']:.4f}")
        logger.info(f"  • 平均倒数排名(MRR): {scores['mrr']:.4f}")
        
        # 保存详细信息
        hit_scores = {f"hit@{k}": scores[f"hit@{k}"] for k in self.k_values}
        self.save_query_details(
            idx,
            question,
            scores['retrieved_paths'],
            relevant_docs,
            hit_scores,
            scores['mrr']
        )
        
        # 创建结果记录
        result_item = {
            "question": question,
            "relevant_docs": relevant_docs,
            "mrr": scores["mrr"]
        }
        # 添加各k值的hit@k
        for k in self.k_values:
            result_item[f"hit@{k}"] = scores[f"hit@{k}"]
        
        # 创建评估记录（用于生成报告）
        record = {
            "query_index": idx + 1,
            "query": question,
            "relevant_docs": relevant_docs,
            "retrieved_docs": [
                {
                    "rank": i+1, 
                    "path": path, 
                    "is_relevant": any(PathUtils.check_path_match(path, doc) for doc in relevant_docs)
                } 
                for i, path in enumerate(scores["retrieved_paths"])
            ],
            "mrr_result": scores["mrr"]
        }
        # 添加各k值的hit结果
        for k in self.k_values:
            record[f"hit@{k}_result"] = scores[f"hit@{k}"] > 0
        
        # 记录结果
        self.evaluation_records.append(record)
        
        return result_item
    
    def _load_test_data(self, test_data_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """加载测试数据
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            测试数据列表
        """
        # 检查文件是否存在
        if not os.path.exists(test_data_path):
            error_msg = f"测试数据文件不存在: {test_data_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
            
            # 加载测试数据
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            return test_data
        except json.JSONDecodeError as e:
            error_msg = f"测试数据文件格式错误: {str(e)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"加载测试数据失败: {str(e)}"
            logger.error(error_msg)
            raise
    
    def _save_results(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """保存评估结果
        
        Args:
            all_results: 所有查询结果
            
        Returns:
            汇总评估结果
        """
        # 创建结果DataFrame
        result_df = pd.DataFrame(all_results)
        
        # 计算平均指标
        hit_rates = {}
        for k in self.k_values:
            hit_rates[f"hit@{k}"] = float(result_df[f"hit@{k}"].mean())
        avg_mrr = float(result_df["mrr"].mean())
        
        # 组织结果
        results = {
            "hit_rate": hit_rates,
            "mrr": avg_mrr,
            "details": all_results,
            "timestamp": datetime.now().isoformat()
        }
        
        # 保存结果到文件
        with open(RESULT_PATH, 'w', encoding='utf-8') as f:
            numpy_safe_dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"评估结果已保存至: {RESULT_PATH}")
        
        # 生成并保存报告
        report = ReportGenerator.generate_markdown_report(results, self.evaluation_records, self.k_values)
        report_path = self.result_dir / "retrieval_evaluation_details.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"详细评估报告已保存至: {report_path}")
        
        # 打印评估结果概要
        logger.info(f"评估结果概要:")
        for k in self.k_values:
            logger.info(f"- 命中率(Hit@{k}): {hit_rates[f'hit@{k}']:.4f}")
        logger.info(f"- 平均倒数排名(MRR): {avg_mrr:.4f}")
        
        return results
    
    def run_evaluation(self, test_data_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
        """
        运行评估流程
        
        Args:
            test_data_path: 测试数据路径
            
        Returns:
            评估结果
        """
        if test_data_path is None:
            test_data_path = TEST_DATA_PATH
            
        logger.info("开始评估检索模块性能...")
        
        # 清空详细记录目录
        logger.info("清空详细记录目录...")
        clear_directory(DETAIL_DIR)
        
        try:
            # 加载测试数据
            test_data = self._load_test_data(test_data_path)
            
            # 重置评估记录
            self.evaluation_records = []
            
            # 逐个处理测试数据
            all_results = []
            for idx, item in enumerate(test_data):
                result_item = self._process_single_query(idx, item, len(test_data))
                all_results.append(result_item)
            
            # 保存和返回结果
            return self._save_results(all_results)
            
        except Exception as e:
            logger.error(f"评估失败: {str(e)}")
            import traceback
            trace = traceback.format_exc()
            logger.error(trace)
            
            # 初始化失败结果
            hit_rates = {}
            for k in self.k_values:
                hit_rates[f"hit@{k}"] = 0.0
            
            return {
                "error": str(e),
                "hit_rate": hit_rates,
                "mrr": 0.0
            }


if __name__ == "__main__":
    try:
    # 加载配置
        config_path = Path(parent_dir) / "config.py"
        if not config_path.exists():
            logger.error(f"未找到配置文件: {config_path}")
            sys.exit(1)
            
    config = Config()
        
        # 运行评估
        evaluator = RetrievalEvaluator(config)
        evaluator.run_evaluation()
        
    except Exception as e:
        logger.exception(f"评估过程发生错误: {str(e)}")
        sys.exit(1) 