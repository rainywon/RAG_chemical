"""
评估RAG系统的检索模块性能
实现了两种核心评估指标：
1. 命中率(Hit Rate@5)：检索结果的前5个中是否包含相关文档
2. 平均倒数排名(Mean Reciprocal Rank, MRR)：相关文档在结果中的排名评估
"""

import sys
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

# 添加父目录到路径，以便导入项目模块
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from config import Config
from rag_system import RAGSystem


# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)




class RetrievalEvaluator:
    """检索模块评估器，专注于命中率和MRR评估"""
    
    def __init__(self, config: Config):
        """初始化评估器
        
        Args:
            config: 系统配置对象
        """
        self.config = config
        self.rag_system = RAGSystem(config)
        # 只保留K=5的评估
        self.k_value = 5
        # 初始化评估记录，用于生成Markdown报告
        self.evaluation_records = []
        
        # 获取项目根目录
        root_dir = Path(__file__).resolve().parent.parent
        # 创建结果目录
        self.result_dir = root_dir / "evaluate" / "results"
        self.result_dir.mkdir(exist_ok=True, parents=True)
        
        # 初始化查询结果文件
        self.query_results_file = self.result_dir / "query_results.json"
        logger.info(f"查询结果将保存至: {self.query_results_file}")
        
        # 如果文件已存在，先清空它
        if self.query_results_file.exists():
            logger.info("发现已存在的查询结果文件，将覆盖它")
            with open(self.query_results_file, 'w', encoding='utf-8') as f:
                f.write('[\n')  # 开始JSON数组
        else:
            logger.info("创建新的查询结果文件")
            with open(self.query_results_file, 'w', encoding='utf-8') as f:
                f.write('[\n')  # 开始JSON数组
        self.is_first_query = True
        logger.info("检索评估器初始化完成")
    
    def _normalize_path(self, path):
        """标准化路径格式，便于比较
        
        Args:
            path: 原始路径
            
        Returns:
            标准化后的路径
        """
        # 确保路径分隔符一致
        norm_path = os.path.normpath(path).replace('\\', '/')
        # 确保小写比较（Windows不区分大小写）
        norm_path = norm_path.lower()
        return norm_path
    
    def _check_path_match(self, path1, path2):
        """检查两个路径是否匹配
        
        Args:
            path1: 第一个路径
            path2: 第二个路径
            
        Returns:
            是否匹配
        """
        norm1 = self._normalize_path(path1)
        norm2 = self._normalize_path(path2)
        
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
    
    def _save_query_results(self, query_index: int, query: str, retrieved_docs: List[Dict], relevant_docs: List[str]):
        """保存查询结果到文件
        
        Args:
            query_index: 查询索引
            query: 查询问题
            retrieved_docs: 检索到的文档列表
            relevant_docs: 相关文档列表
        """
        result_data = {
            "query_index": query_index,
            "query": query,
            "retrieved_docs": retrieved_docs,
            "relevant_docs": relevant_docs,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # 追加结果到文件
            with open(self.query_results_file, 'a', encoding='utf-8') as f:
                if not self.is_first_query:
                    f.write(',\n')  # 如果不是第一个查询，添加逗号分隔符
                json.dump(result_data, f, ensure_ascii=False, indent=2)
                self.is_first_query = False
            
            logger.info(f"查询 {query_index} 的结果已保存至文件: {self.query_results_file}")
        except Exception as e:
            logger.error(f"保存查询 {query_index} 结果时出错: {str(e)}")
            logger.error(f"文件路径: {self.query_results_file}")
            raise
    
    def __del__(self):
        """析构函数，确保JSON文件正确关闭"""
        try:
            if hasattr(self, 'query_results_file') and self.query_results_file.exists():
                with open(self.query_results_file, 'a', encoding='utf-8') as f:
                    f.write('\n]')  # 结束JSON数组
                logger.info(f"查询结果文件已正确关闭: {self.query_results_file}")
        except Exception as e:
            logger.error(f"关闭查询结果文件时出错: {str(e)}")
    
    def evaluate_hit_rate(self, test_data: List[Dict]) -> Dict[str, float]:
        """评估命中率 - 检索结果中是否包含至少一个相关文档
        
        Args:
            test_data: 包含问题和相关文档的测试数据
            
        Returns:
            K=5时的命中率结果
        """
        logger.info("开始评估命中率...")
        hits = 0
        total = len(test_data)
        
        for idx, item in enumerate(test_data):
            query = item["question"]
            relevant_docs = item["relevant_docs"]  # 真实相关文档
            
            logger.info(f"处理查询 [{idx+1}/{total}]: {query}")
            
            record = {
                "query_index": idx + 1,
                "query": query,
                "relevant_docs": relevant_docs,
                "retrieved_docs": [],
                "hit_result": False,
                "mrr_result": 0
            }
            
            try:
                # 使用RAG系统进行检索
                retrieved_docs, score_info = self.rag_system._retrieve_documents(query)
                
                # 获取检索文档路径
                retrieved_paths = [doc.metadata.get("source", "") for doc in retrieved_docs]
                retrieved_scores = [info.get("final_score", 0) for info in score_info] if score_info else [0] * len(retrieved_docs)
                
                # 记录检索到的文档
                retrieved_docs_info = []
                for i, (doc_path, score) in enumerate(zip(retrieved_paths, retrieved_scores)):
                    is_relevant = any(self._check_path_match(doc_path, ref_doc) for ref_doc in relevant_docs)
                    doc_info = {
                        "rank": i + 1,
                        "path": doc_path,
                        "score": score,
                        "is_relevant": is_relevant
                    }
                    record["retrieved_docs"].append(doc_info)
                    retrieved_docs_info.append(doc_info)
                
                # 保存查询结果
                self._save_query_results(
                    query_index=idx + 1,
                    query=query,
                    retrieved_docs=retrieved_docs_info,
                    relevant_docs=relevant_docs
                )
                
                # 计算K=5的命中情况
                if len(retrieved_paths) >= self.k_value:
                    top_k_docs = retrieved_paths[:self.k_value]
                    hit = False
                    for ref_doc in relevant_docs:
                        for ret_doc in top_k_docs:
                            if self._check_path_match(ref_doc, ret_doc):
                                hits += 1
                                hit = True
                                break
                        if hit:
                            break
                    record["hit_result"] = hit
                else:
                    # 如果检索结果少于K个，检查所有结果
                    hit = False
                    for ref_doc in relevant_docs:
                        for ret_doc in retrieved_paths:
                            if self._check_path_match(ref_doc, ret_doc):
                                hits += 1
                                hit = True
                                break
                        if hit:
                            break
                    record["hit_result"] = hit
            
            except Exception as e:
                logger.error(f"处理查询时出错: {query}, 错误: {str(e)}")
                record["error"] = str(e)
            
            self.evaluation_records.append(record)
            
            # 打印当前查询结果
            logger.info(f"查询 {record['query_index']} 结果: {'命中' if record['hit_result'] else '未命中'}")
        
        # 计算命中率
        hit_rate = hits / total if total > 0 else 0
        
        logger.info(f"命中率评估结果: hit@{self.k_value} = {hit_rate:.4f}")
        
        return {"hit@5": hit_rate}
    
    def evaluate_mrr(self, test_data: List[Dict]) -> float:
        """评估平均倒数排名(MRR) - 相关文档首次出现位置的倒数平均值
        
        Args:
            test_data: 包含问题和相关文档的测试数据
            
        Returns:
            MRR得分
        """
        logger.info("开始评估MRR...")
        reciprocal_ranks = []
        
        for idx, item in enumerate(test_data):
            # 找到对应的评估记录
            record = self.evaluation_records[idx]
            query = item["question"]
            relevant_docs = item["relevant_docs"]  # 真实相关文档
            
            try:
                # 检查是否已经有检索结果
                if not record.get("retrieved_docs"):
                    # 使用RAG系统进行检索
                    retrieved_docs, _ = self.rag_system._retrieve_documents(query)
                    
                    # 获取检索文档路径
                    retrieved_paths = [doc.metadata.get("source", "") for doc in retrieved_docs]
                    
                    # 更新记录
                    for i, doc_path in enumerate(retrieved_paths):
                        is_relevant = any(self._check_path_match(doc_path, ref_doc) for ref_doc in relevant_docs)
                        record["retrieved_docs"].append({
                            "rank": i + 1,
                            "path": doc_path,
                            "is_relevant": is_relevant
                        })
                
                # 获取检索文档
                retrieved_docs = [doc["path"] for doc in record["retrieved_docs"]]
                
                # 计算倒数排名
                rank = 0
                for i, doc_path in enumerate(retrieved_docs):
                    for ref_doc in relevant_docs:
                        if self._check_path_match(doc_path, ref_doc):
                            # 找到第一个相关文档的位置(从1开始计数)
                            rank = 1 / (i + 1)
                            break
                    if rank > 0:
                        break
                
                record["mrr_result"] = rank
                reciprocal_ranks.append(rank)
                
                # 打印当前查询MRR结果
                logger.info(f"查询 {record['query_index']} MRR: {rank:.4f}")
            
            except Exception as e:
                logger.error(f"处理查询时出错: {query}, 错误: {str(e)}")
                record["error"] = str(e)
                reciprocal_ranks.append(0)
        
        # 计算MRR
        mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
        logger.info(f"MRR评估结果: {mrr:.4f}")
        
        return mrr
    
    def generate_markdown_report(self, results):
        """生成Markdown格式的详细评估报告
        
        Args:
            results: 评估结果
            
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
        report.append("### 命中率 (Hit Rate)")
        report.append(f"- **hit@{self.k_value}**: {results['hit_rate']['hit@5']:.4f}")
        
        report.append("\n### 平均倒数排名 (MRR)")
        report.append(f"- **MRR**: {results['mrr']:.4f}\n")
        
        # 各查询评估详情
        report.append("## 2. 各查询评估详情")
        
        for idx, record in enumerate(self.evaluation_records):
            report.append(f"### 查询 {record['query_index']}")
            report.append(f"**问题**: {record['query']}")
            
            # 相关文档
            report.append("\n**相关文档**:")
            for doc in record['relevant_docs']:
                report.append(f"- `{doc}`")
            
            # 检索结果
            report.append("\n**检索结果**:")
            report.append("| 排名 | 文档路径 | 分数 | 是否相关 |")
            report.append("|------|---------|------|----------|")
            
            for doc in record.get('retrieved_docs', [])[:10]:  # 只显示前10个结果
                is_relevant = "✓" if doc.get('is_relevant', False) else "✗"
                score = f"{doc.get('score', 0):.4f}" if 'score' in doc else "N/A"
                report.append(f"| {doc['rank']} | `{doc['path']}` | {score} | {is_relevant} |")
            
            # 命中结果
            report.append("\n**命中结果**:")
            status = "✓ 命中" if record.get('hit_result', False) else "✗ 未命中"
            report.append(f"- **hit@{self.k_value}**: {status}")
            
            # MRR结果
            report.append(f"\n**MRR**: {record.get('mrr_result', 0):.4f}")
            
            # 错误信息（如果有）
            if 'error' in record:
                report.append(f"\n**错误**: {record['error']}")
            
            report.append("\n---\n")
        
        return "\n".join(report)
    
    def run_evaluation(self, test_data_path: str) -> Dict[str, Any]:
        """运行完整评估流程
        
        Args:
            test_data_path: 测试数据文件路径
            
        Returns:
            Dict[str, Any]: 评估结果
        """
        try:
            # 将路径转换为Path对象，确保处理相对路径
            test_data_path = Path(test_data_path)
            
            # 检查文件是否存在
            if not test_data_path.exists():
                error_msg = f"测试数据文件不存在: {test_data_path}"
                logger.error(error_msg)
                
                return {
                    "error": error_msg,
                    "hit_rate": {"hit@5": 0.0},
                    "mrr": 0.0
                }
            
            # 加载测试数据
            logger.info(f"从文件加载测试数据: {test_data_path}")
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            logger.info(f"成功加载了 {len(test_data)} 条测试数据")
            
            # 运行评估
            hit_rates = self.evaluate_hit_rate(test_data)
            mrr = self.evaluate_mrr(test_data)
            
            # 合并结果
            results = {
                "hit_rate": hit_rates,
                "mrr": mrr
            }
            
            return results
            
        except json.JSONDecodeError as je:
            error_msg = f"测试数据文件格式错误: {str(je)}"
            logger.error(error_msg)
            
            return {
                "error": error_msg,
                "hit_rate": {"hit@5": 0.0},
                "mrr": 0.0
            }
        except Exception as e:
            logger.error(f"评估过程出错: {str(e)}")
            import traceback
            trace = traceback.format_exc()
            logger.error(trace)
            
            return {
                "error": str(e),
                "hit_rate": {"hit@5": 0.0},
                "mrr": 0.0
            }

if __name__ == "__main__":
    # 加载配置
    config = Config()
    
    # 创建评估器
    evaluator = RetrievalEvaluator(config)
    
    # 获取项目根目录的绝对路径
    root_dir = Path(__file__).resolve().parent.parent
    
    # 运行评估 - 使用绝对路径
    test_data_path = str(root_dir / "evaluate" / "test_data" / "retrieval_test_data.json")
    logger.info(f"使用测试数据路径: {test_data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(test_data_path):
        logger.error(f"测试数据文件不存在: {test_data_path}")
        logger.info("尝试创建示例测试数据文件...")
        
        # 确保目录存在
        os.makedirs(os.path.dirname(test_data_path), exist_ok=True)
        
        # 创建一个简单的测试数据文件
        example_data = [
            {
                "question": "化工厂爆炸应急预案",
                "relevant_docs": [
                    "docs/emergency/explosion.pdf",
                    "docs/safety/chemical_plant_emergency.docx"
                ]
            },
            {
                "question": "化学品安全操作规程",
                "relevant_docs": [
                    "docs/safety/chemical_handling.pdf",
                    "docs/manual/safety_procedures.docx"
                ]
            }
        ]
        
        with open(test_data_path, 'w', encoding='utf-8') as f:
            json.dump(example_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已创建示例测试数据文件: {test_data_path}")
    
    # 运行评估
    results = evaluator.run_evaluation(test_data_path)
    
    # 保存结果
    result_dir = root_dir / "evaluate" / "results"
    result_dir.mkdir(exist_ok=True, parents=True)
    
    # 保存JSON结果
    result_path = result_dir / "retrieval_results.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 生成并保存Markdown报告
    report = evaluator.generate_markdown_report(results)
    report_path = result_dir / "retrieval_evaluation_details.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"评估结果已保存至: {result_path}")
    logger.info(f"详细评估报告已保存至: {report_path}") 