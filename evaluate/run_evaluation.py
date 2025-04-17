#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行RAGAS和自定义评估方法的脚本
支持不同类型的评估选项
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加父目录到路径，以便导入项目模块
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from config import Config
from evaluate.evaluate_generation import GenerationEvaluator
from evaluate.evaluate_generation_ragas import RAGASEvaluator

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """主函数：解析命令行参数并运行评估"""
    parser = argparse.ArgumentParser(description="评估RAG系统生成模块")
    
    # 添加命令行参数
    parser.add_argument("--method", "-m", type=str, default="custom", 
                        choices=["custom", "ragas", "both"],
                        help="评估方法：custom=自定义指标评估, ragas=使用RAGAS库评估, both=两种方法都使用")
    
    parser.add_argument("--test_data", "-t", type=str, default=None,
                        help="测试数据文件路径 (默认: evaluate/test_data/generation_test_data.json)")
    
    parser.add_argument("--threads", type=int, default=1,
                        help="并行处理线程数 (默认: 1)")
    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 初始化配置
    config = Config()
    
    # 确定测试数据路径
    if args.test_data:
        test_data_path = Path(args.test_data)
    else:
        test_data_path = Path(__file__).resolve().parent / "test_data" / "generation_test_data.json"
    
    # 运行评估
    if args.method == "custom" or args.method == "both":
        logger.info("运行自定义评估方法...")
        custom_evaluator = GenerationEvaluator(config)
        custom_evaluator.max_workers = args.threads
        custom_results = custom_evaluator.run_evaluation(str(test_data_path))
        logger.info(f"自定义评估完成，综合评分: {custom_results.get('comprehensive_score', 0):.4f}")
    
    if args.method == "ragas" or args.method == "both":
        logger.info("运行RAGAS评估方法...")
        ragas_evaluator = RAGASEvaluator(config)
        ragas_evaluator.max_workers = args.threads
        ragas_results = ragas_evaluator.run_evaluation(str(test_data_path))
        logger.info(f"RAGAS评估完成，综合评分: {ragas_results.get('comprehensive_score', 0):.4f}")
    
    logger.info("所有评估完成！")

if __name__ == "__main__":
    main() 