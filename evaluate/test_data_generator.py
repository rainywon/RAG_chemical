"""
测试数据生成工具
基于实际data目录下的文档生成用于检索评估的测试数据集
增强版：提高测试数据质量，确保查询与相关文档的精确匹配
"""

import os
import json
import random
import jieba
import re
from pathlib import Path
from collections import defaultdict
import logging
from typing import List, Dict, Set, Tuple

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

class TestDataGenerator:
    """测试数据生成器类，用于生成检索评估所需的测试数据集"""
    
    def __init__(self, data_dir="data", output_path="evaluate/test_data/retrieval_test_data.json"):
        """初始化测试数据生成器
        
        Args:
            data_dir: 数据文件目录
            output_path: 输出的测试数据文件路径
        """
        self.data_dir = Path(data_dir)
        self.output_path = Path(output_path)
        self.document_contents = {}  # 存储文档内容: {文件路径: 文件内容}
        self.document_topics = defaultdict(list)  # 按主题分类的文档: {主题: [文件路径列表]}
        self.document_keywords = {}  # 文档关键词: {文件路径: [关键词列表]}
        self.topic_keywords = defaultdict(set)  # 主题关键词: {主题: {关键词集合}}
        
        # 扩展支持的文件类型
        self.file_extensions = ['.txt', '.md', '.docx', '.pdf', '.html', '.doc', '.rtf']
        
        # 停用词
        self.stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', 
                      '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
                      '会', '着', '没有', '看', '好', '自己', '这'}
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        
        logger.info("测试数据生成器初始化完成")
    
    def _extract_content_from_filename(self, filename: str) -> List[str]:
        """从文件名提取关键词和主题
        
        Args:
            filename: 文件名
            
        Returns:
            提取的关键词列表
        """
        # 去除扩展名
        base_name = os.path.splitext(filename)[0]
        
        # 从文件名中提取中括号内的标识
        brackets_content = re.findall(r'\[(.*?)\]', base_name)
        main_title = re.sub(r'\[.*?\]', '', base_name).strip()
        
        # 分词
        words = list(jieba.cut(main_title))
        words = [w for w in words if w not in self.stop_words and len(w) > 1]
        
        # 添加中括号内容作为特殊关键词
        for bracket in brackets_content:
            if len(bracket) > 1:
                words.append(bracket)
                
                # 进一步提取中括号内的关键信息
                bracket_words = list(jieba.cut(bracket))
                words.extend([w for w in bracket_words if len(w) > 1 and w not in self.stop_words])
        
        return words
    
    def scan_documents(self, sample_content=True):
        """扫描数据目录，加载文档信息
        
        Args:
            sample_content: 是否尝试读取文件内容样本
            
        Returns:
            扫描到的文件总数
        """
        logger.info(f"开始扫描数据目录: {self.data_dir}")
        
        # 遍历数据目录下的所有文件
        total_files = 0
        for root, _, files in os.walk(self.data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                
                # 构建相对于项目根目录的路径
                rel_path = file_path
                
                # 检查文件类型
                file_ext = os.path.splitext(file)[1].lower()
                
                try:
                    # 提取文件名中的关键词
                    keywords = self._extract_content_from_filename(file)
                    self.document_keywords[rel_path] = keywords
                    
                    # 记录文档路径
                    self.document_contents[rel_path] = rel_path
                    
                    # 从路径结构推断主题
                    path_parts = rel_path.split(os.sep)
                    if len(path_parts) > 1:
                        main_topic = path_parts[1]  # data/主题/...
                        self.document_topics[main_topic].append(rel_path)
                        
                        # 添加关键词到主题
                        self.topic_keywords[main_topic].update(keywords)
                    
                    total_files += 1
                    
                    # 定期记录进度
                    if total_files % 100 == 0:
                        logger.info(f"已扫描 {total_files} 个文件...")
                        
                except Exception as e:
                    logger.warning(f"处理文件时出错: {rel_path}, 错误: {str(e)}")
                    continue
        
        logger.info(f"扫描完成，共找到{total_files}个文件，{len(self.document_topics)}个主题分类")
        
        # 输出主题统计信息
        for topic, docs in self.document_topics.items():
            logger.info(f"主题 '{topic}' 包含 {len(docs)} 个文档")
            
        return total_files

    def _find_related_documents(self, query: str, topic: str, num_docs=3) -> List[str]:
        """查找与查询最相关的文档
        
        Args:
            query: 查询文本
            topic: 主题
            num_docs: 返回的文档数量
            
        Returns:
            相关文档路径列表
        """
        if topic not in self.document_topics:
            # 如果指定主题不存在，从所有文档中选择
            candidate_docs = list(self.document_contents.keys())
        else:
            candidate_docs = self.document_topics[topic]
        
        if not candidate_docs:
            return []
            
        # 查询分词
        query_words = set(jieba.cut(query))
        query_words = {w for w in query_words if w not in self.stop_words and len(w) > 1}
        
        # 评分每个文档与查询的相关性
        doc_scores = []
        for doc_path in candidate_docs:
            # 文档关键词
            doc_keywords = set(self.document_keywords.get(doc_path, []))
            
            if not doc_keywords:
                continue
                
            # 计算关键词重叠
            overlap = len(query_words.intersection(doc_keywords))
            
            # 文件名中包含查询关键词的加分
            filename = os.path.basename(doc_path)
            filename_score = sum(1 for word in query_words if word in filename) * 2
            
            # 最终分数
            score = overlap + filename_score
            
            doc_scores.append((doc_path, score))
        
        # 按分数排序
        doc_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top N
        top_docs = [doc for doc, score in doc_scores[:num_docs] if score > 0]
        
        # 如果没有找到相关文档，随机选择一些
        if not top_docs and candidate_docs:
            num_to_select = min(num_docs, len(candidate_docs))
            top_docs = random.sample(candidate_docs, num_to_select)
            logger.warning(f"未找到与查询'{query}'匹配的文档，随机选择了{len(top_docs)}个文档")
        
        return top_docs

    def generate_test_queries(self, num_queries=20):
        """生成测试查询和相关文档
        
        Args:
            num_queries: 要生成的测试查询数量
            
        Returns:
            测试数据列表
        """
        logger.info(f"开始生成{num_queries}个测试查询...")
        
        # 预定义的化工安全相关查询模板
        query_templates = [
            "{}的安全生产要求是什么？",
            "{}有哪些危险特性？",
            "{}的应急处置方案",
            "如何预防{}事故？",
            "{}的安全标准有哪些？",
            "{}的使用注意事项",
            "{}的储存要求",
            "{}的防护措施包括哪些？",
            "{}泄漏后如何处理？",
            "{}操作规程"
        ]
        
        # 从主题中提取关键词作为主题词
        topic_terms = []
        for topic, keywords in self.topic_keywords.items():
            topic_terms.append(topic)
            # 添加该主题下的高频关键词
            top_keywords = sorted(list(keywords), key=len, reverse=True)[:5]
            topic_terms.extend(top_keywords)
        
        # 如果没有足够的主题词，添加预定义的化工领域词汇
        if len(topic_terms) < 20:
            predefined_terms = [
                "化学品", "危险物", "易燃易爆", "有毒气体", "腐蚀性物质",
                "安全生产", "应急预案", "防护设备", "监测系统", "泄漏处理",
                "消防安全", "防爆技术", "安全标准", "操作规程", "职业危害"
            ]
            topic_terms.extend(predefined_terms)
        
        # 去重
        topic_terms = list(set(topic_terms))
        
        # 创建测试数据
        test_data = []
        created_queries = set()  # 避免重复查询
        
        while len(test_data) < num_queries and len(created_queries) < 100:  # 设置最大尝试次数
            # 随机选择主题和模板
            selected_topic = random.choice(topic_terms)
            selected_template = random.choice(query_templates)
            
            # 生成查询
            query = selected_template.format(selected_topic)
            
            # 检查查询是否已存在
            if query in created_queries:
                continue
                
            created_queries.add(query)
            
            # 查找相关主题
            topic_match = None
            for topic in self.document_topics.keys():
                if selected_topic.lower() in topic.lower() or topic.lower() in selected_topic.lower():
                    topic_match = topic
                    break
            
            # 确定相关文档 
            relevant_docs = self._find_related_documents(
                query=query, 
                topic=topic_match if topic_match else random.choice(list(self.document_topics.keys())),
                num_docs=random.randint(2, 3)  # 每个查询2-3个相关文档
            )
            
            # 如果找到了相关文档，添加到测试数据
            if relevant_docs:
                test_data.append({
                    "question": query,
                    "relevant_docs": relevant_docs
                })
                
                logger.info(f"生成查询: '{query}' 关联 {len(relevant_docs)} 个文档")
        
        logger.info(f"已生成{len(test_data)}个测试查询")
        return test_data
    
    def generate_and_save(self, num_queries=20):
        """生成测试数据并保存到文件
        
        Args:
            num_queries: 要生成的测试查询数量
            
        Returns:
            保存的文件路径
        """
        # 扫描文档
        self.scan_documents()
        
        # 生成测试查询
        test_data = self.generate_test_queries(num_queries)
        
        # 保存到文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"测试数据已保存至: {self.output_path}")
        return self.output_path
    
    def manually_create_test_data(self, manual_data):
        """手动创建高质量测试数据
        
        Args:
            manual_data: 手动定义的测试数据
        
        Returns:
            保存的文件路径
        """
        # 扫描文档
        self.scan_documents()
        
        # 验证手动数据中的文档路径
        validated_data = []
        for item in manual_data:
            query = item["question"]
            docs = item["relevant_docs"]
            
            # 验证文档路径
            valid_docs = []
            for doc in docs:
                norm_doc = os.path.normpath(doc)
                # 检查是否存在于已扫描文档中
                if doc in self.document_contents or norm_doc in self.document_contents:
                    valid_docs.append(doc)
                else:
                    logger.warning(f"文档路径无效: {doc}")
            
            if valid_docs:
                validated_data.append({
                    "question": query,
                    "relevant_docs": valid_docs
                })
                logger.info(f"添加手动测试数据: '{query}' 关联 {len(valid_docs)} 个文档")
        
        # 保存到文件
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(validated_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"手动测试数据已保存至: {self.output_path}")
        return self.output_path

if __name__ == "__main__":
    # 创建测试数据生成器
    generator = TestDataGenerator()
    
    # 定义一些高质量的手动测试数据
    manual_test_data = [
        {
            "question": "应急预案的安全标准有哪些？",
            "relevant_docs": [
                "data\\规范性文件\\[安监总应急〔2015〕100号]生产安全事故应急预案管理办法.docx",
                "data\\规范性文件\\[安监总厅应急〔2009〕73号]生产经营单位生产安全事故应急预案评审指南（试行）.docx"
            ]
        },
        {
            "question": "危险化学品储存有哪些安全要求？",
            "relevant_docs": [
                "data\\标准性文件\\[GB 15603-1995]常用化学危险品贮存通则.pdf",
                "data\\规范性文件\\[安监总管三〔2011〕43号]危险化学品经营企业安全技术基本要求.docx"
            ]
        },
        {
            "question": "煤矿安全生产有哪些规定？",
            "relevant_docs": [
                "data\\规范性文件\\[安监总煤装〔2011〕141号]煤矿安全监控系统及检测仪器使用管理规范.docx",
                "data\\法律\\[主席令第70号]中华人民共和国安全生产法.docx"
            ]
        }
    ]
    
    # 使用手动测试数据
    generator.manually_create_test_data(manual_test_data)
    
    # 或者使用自动生成
    # generator.generate_and_save(10)  # 生成10个测试查询 