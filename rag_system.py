# 导入必要的库和模块
import json
import logging  # 日志记录模块
from pathlib import Path  # 路径处理库
from typing import Generator, Optional, List, Tuple, Dict, Any  # 类型提示支持
import warnings  # 警告处理
import torch  # PyTorch深度学习框架
from langchain_community.vectorstores import FAISS  # FAISS向量数据库集成
from langchain_core.documents import Document  # 文档对象定义
from langchain_core.embeddings import Embeddings  # 嵌入模型接口
from langchain_ollama import OllamaLLM  # Ollama语言模型集成
from rank_bm25 import BM25Okapi  # BM25检索算法
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # Transformer模型
from config import Config  # 自定义配置文件
from build_vector_store import VectorDBBuilder  # 向量数据库构建器
import numpy as np  # 数值计算库
import pickle  # 用于序列化对象
import hashlib  # 用于生成哈希值
import time  # 添加time模块用于时间测量
import re  # 用于正则表达式处理

# 提前初始化jieba，加快后续启动速度
import os
import jieba  # 中文分词库

# 设置jieba日志级别，减少输出
jieba.setLogLevel(logging.INFO)

# 预加载jieba分词器
jieba.initialize()

# 禁用不必要的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 配置日志记录器
logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class RAGSystem:
    """RAG问答系统，支持文档检索和生成式问答

    特性：
    - 自动管理向量数据库生命周期
    - 支持流式生成和同步生成
    - 可配置的检索策略
    - 完善的错误处理
    """

    def __init__(self, config: Config):
        """初始化RAG系统

        :param config: 包含所有配置参数的Config对象
        """
        self.config = config  # 保存配置对象
        self.vector_store: Optional[FAISS] = None  # FAISS向量数据库实例
        self.llm: Optional[OllamaLLM] = None  # Ollama语言模型实例
        self.embeddings: Optional[Embeddings] = None  # 嵌入模型实例
        self.rerank_model = None  # 重排序模型
        self.vector_db_build = VectorDBBuilder(config)  # 向量数据库构建器实例
        self._tokenize_cache = {}  # 添加分词缓存字典

        # 初始化各个组件
        self._init_logging()  # 初始化日志配置
        self._init_embeddings()  # 初始化嵌入模型
        self._init_vector_store()  # 初始化向量数据库
        self._init_bm25_retriever()  # 初始化BM25检索器
        self._init_llm()  # 初始化大语言模型
        self._init_rerank_model()  # 初始化重排序模型

    def _tokenize(self, text: str) -> List[str]:
        """专业中文分词处理，使用缓存提高性能
        :param text: 待分词的文本
        :return: 分词后的词项列表
        """
        # 检查缓存中是否已有结果
        if text in self._tokenize_cache:
            return self._tokenize_cache[text]
        
        # 如果文本过长，只缓存前2000个字符的分词结果
        cache_key = text[:2000] if len(text) > 2000 else text
        
        # 分词处理
        result = [word for word in jieba.cut(text) if word.strip()]
        
        # 只在缓存不超过10000个条目时进行缓存
        if len(self._tokenize_cache) < 10000:
            self._tokenize_cache[cache_key] = result
            
        return result

    def _init_logging(self):
        """初始化日志配置"""
        logging.basicConfig(
            level=logging.INFO,  # 日志级别设为INFO
            format="%(asctime)s - %(levelname)s - %(message)s",  # 日志格式
            handlers=[logging.StreamHandler()]  # 输出到控制台
        )

    def _init_embeddings(self):
        """初始化嵌入模型"""
        try:
            logger.info("🔧 正在初始化嵌入模型...")
            # 通过构建器创建嵌入模型实例
            self.embeddings = self.vector_db_build.create_embeddings()
            logger.info("✅ 嵌入模型初始化完成")
        except Exception as e:
            logger.error("❌ 嵌入模型初始化失败")
            raise RuntimeError(f"无法初始化嵌入模型: {str(e)}")

    def _init_vector_store(self):
        """初始化向量数据库"""
        try:
            vector_path = Path(self.config.vector_db_path)  # 获取向量库路径

            # 检查现有向量数据库是否存在
            if vector_path.exists():
                logger.info("🔍 正在加载现有向量数据库...")
                if not self.embeddings:
                    raise ValueError("嵌入模型未初始化")

                # 加载本地FAISS数据库
                self.vector_store = FAISS.load_local(
                    folder_path=str(vector_path),
                    embeddings=self.embeddings,
                    allow_dangerous_deserialization=True  # 允许加载旧版本序列化数据
                )
                logger.info(f"✅ 已加载向量数据库：{vector_path}")
            else:
                # 构建新向量数据库
                logger.warning("⚠️ 未找到现有向量数据库，正在构建新数据库...")
                self.vector_store = self.vector_db_build.build_vector_store()
                logger.info(f"✅ 新建向量数据库已保存至：{vector_path}")
        except Exception as e:
            logger.error("❌ 向量数据库初始化失败")
            raise RuntimeError(f"无法初始化向量数据库: {str(e)}")

    def _init_rerank_model(self):
        """初始化重排序模型"""
        try:
            logger.info("🔧 正在初始化rerank模型...")
            # 从HuggingFace加载预训练模型和分词器
            self.rerank_model = AutoModelForSequenceClassification.from_pretrained(
                self.config.rerank_model_path
            )
            self.rerank_tokenizer = AutoTokenizer.from_pretrained(self.config.rerank_model_path)
            
            # 尝试将模型移至GPU，如果可用
            if torch.cuda.is_available():
                logger.info("🚀 将重排序模型移至GPU加速")
                self.rerank_model = self.rerank_model.to("cuda")
                self.using_gpu = True
            else:
                self.using_gpu = False
                
            # 设置为评估模式，提高推理速度
            self.rerank_model.eval()
            
            # 初始化重排序缓存
            self.rerank_cache = {}
            
            logger.info("✅ rerank模型初始化完成")
        except Exception as e:
            logger.error(f"❌ rerank模型初始化失败: {str(e)}")
            raise RuntimeError(f"无法初始化rerank模型: {str(e)}")

    def _init_llm(self):
        """初始化Ollama大语言模型"""
        try:
            logger.info("🚀 正在初始化Ollama模型...")
            # 创建OllamaLLM实例
            self.llm = OllamaLLM(
                model="deepseek_8B:latest",  # 模型名称
                #deepseek_8B:latest   1513b8b198dc    8.5 GB    59 seconds ago
                # deepseek-r1:8b             2deepseek_8B:latest GB    46 minutes ago
                # deepseek-r1:14b            ea35dfe18182    9.0 GB    29 hours ago
                base_url=self.config.ollama_base_url,  # Ollama服务地址
                temperature=self.config.llm_temperature,  # 温度参数控制随机性
                num_predict=self.config.llm_max_tokens,  # 最大生成token数
                stop=["<|im_end|>"]
            )

            # 测试模型连接
            logger.info("✅ Ollama模型初始化完成")
        except Exception as e:
            logger.error(f"❌ Ollama模型初始化失败: {str(e)}")
            raise RuntimeError(f"无法初始化Ollama模型: {str(e)}")

    def _init_bm25_retriever(self):
        """初始化BM25检索器（持久化缓存版）"""
        try:
            logger.info("🔧 正在初始化BM25检索器...")

            # 验证向量库是否包含文档
            if not self.vector_store.docstore._dict:
                raise ValueError("向量库中无可用文档")

            # 从向量库加载所有文档内容
            all_docs = list(self.vector_store.docstore._dict.values())
            self.bm25_docs = [doc.page_content for doc in all_docs]
            self.doc_metadata = [doc.metadata for doc in all_docs]
            
            # 计算文档集合的哈希值，用于缓存标识
            docs_hash = hashlib.md5(str([d[:100] for d in self.bm25_docs]).encode()).hexdigest()
            cache_path = Path(self.config.vector_db_path).parent / f"bm25_tokenized_cache_{docs_hash}.pkl"
            
            # 尝试加载缓存的分词结果
            if cache_path.exists():
                try:
                    logger.info(f"发现BM25分词缓存，正在加载：{cache_path}")
                    with open(cache_path, 'rb') as f:
                        cached_data = pickle.load(f)
                        tokenized_docs = cached_data.get('tokenized_docs')
                        
                    if tokenized_docs and len(tokenized_docs) == len(self.bm25_docs):
                        logger.info(f"成功加载缓存的分词结果，共 {len(tokenized_docs)} 篇文档")
                    else:
                        logger.warning("缓存数据不匹配，将重新处理分词")
                        tokenized_docs = None
                except Exception as e:
                    logger.warning(f"加载缓存失败: {str(e)}，将重新处理分词")
                    tokenized_docs = None
            else:
                tokenized_docs = None
            
            # 如果没有有效的缓存，重新分词处理
            if tokenized_docs is None:
                logger.info(f"开始处理 {len(self.bm25_docs)} 篇文档进行BM25索引...")
                
                # 批处理分词以减少内存压力
                batch_size = 100  # 每批处理的文档数
                tokenized_docs = []
                
                for i in range(0, len(self.bm25_docs), batch_size):
                    batch = self.bm25_docs[i:i+batch_size]
                    batch_tokenized = [self._tokenize(doc) for doc in batch]
                    tokenized_docs.extend(batch_tokenized)
                    
                    if (i + batch_size) % 500 == 0 or (i + batch_size) >= len(self.bm25_docs):
                        logger.info(f"已处理 {min(i + batch_size, len(self.bm25_docs))}/{len(self.bm25_docs)} 篇文档")
                
                # 保存分词结果到缓存
                try:
                    logger.info(f"保存分词结果到缓存：{cache_path}")
                    with open(cache_path, 'wb') as f:
                        pickle.dump({'tokenized_docs': tokenized_docs}, f)
                except Exception as e:
                    logger.warning(f"保存缓存失败: {str(e)}")

            # 验证分词结果有效性
            if len(tokenized_docs) == 0 or all(len(d) == 0 for d in tokenized_docs):
                raise ValueError("文档分词后为空，请检查分词逻辑")

            # 初始化BM25模型
            logger.info("开始构建BM25索引...")
            self.bm25 = BM25Okapi(tokenized_docs)

            logger.info(f"✅ BM25初始化完成，文档数：{len(self.bm25_docs)}")
        except Exception as e:
            logger.error(f"❌ BM25初始化失败: {str(e)}")
            raise RuntimeError(f"BM25初始化失败: {str(e)}")

    def _hybrid_retrieve(self, question: str) -> List[Dict[str, Any]]:
        """混合检索流程（向量+BM25）

        :param question: 用户问题
        :return: 包含文档和检索信息的字典列表
        """
        # 动态确定检索策略权重
        vector_weight, bm25_weight = self._determine_retrieval_weights(question)
        logger.info(f"动态权重: 向量检索={vector_weight:.2f}, BM25检索={bm25_weight:.2f}")

        # 一、向量检索部分
        # 1. 执行向量检索
        vector_results = self.vector_store.similarity_search_with_score(
            question, k=self.config.vector_top_k
        )
        
        # 2. 对向量检索结果去重并标准化分数
        unique_vector_results = {}
        for doc, score in vector_results:
            doc_id = doc.metadata.get("source", "") + str(hash(doc.page_content))
            norm_score = (score + 1) / 2  # 转换为标准余弦值（0~1范围）
            
            # 如果文档已存在且新分数更高，则更新
            if doc_id not in unique_vector_results or norm_score > unique_vector_results[doc_id][1]:
                unique_vector_results[doc_id] = (doc, norm_score)
        
        # 3. 过滤低分结果并应用权重
        filtered_vector_results = []
        for doc, score in unique_vector_results.values():
            if score >= self.config.vector_similarity_threshold:
                # 应用向量检索权重
                weighted_score = score * vector_weight
                filtered_vector_results.append({
                    "doc": doc,
                    "score": weighted_score,  # 应用权重后的分数
                    "raw_score": score,       # 原始分数
                    "type": "vector",
                    "source": doc.metadata.get("source", "unknown")
                })

        # 二、BM25检索部分
        # 1. 问题分词并计算BM25分数
        tokenized_query = self._tokenize(question)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # 2. 获取top k的BM25结果
        top_bm25_indices = np.argsort(bm25_scores)[-self.config.bm25_top_k:][::-1]
        top_bm25_scores = [bm25_scores[idx] for idx in top_bm25_indices]
        
        # 3. 对BM25分数进行归一化处理
        normalized_bm25_scores = []
        if top_bm25_scores:
            # 计算均值和标准差
            mean_score = np.mean(top_bm25_scores)
            std_score = np.std(top_bm25_scores) + 1e-9  # 避免除以0
            
            # 使用Logistic归一化
            for score in top_bm25_scores:
                z_score = (score - mean_score) / std_score  # Z-score标准化
                logistic_score = 1 / (1 + np.exp(-z_score))  # Sigmoid函数
                normalized_bm25_scores.append(logistic_score)
        
        # 4. 构建BM25检索结果并应用权重
        filtered_bm25_results = []
        for idx, norm_score in zip(top_bm25_indices, normalized_bm25_scores):
            if norm_score >= self.config.bm25_similarity_threshold:
                doc = Document(
                    page_content=self.bm25_docs[idx],
                    metadata=self.doc_metadata[idx]
                )
                # 应用BM25检索权重
                weighted_score = norm_score * bm25_weight
                filtered_bm25_results.append({
                    "doc": doc,
                    "score": weighted_score,  # 应用权重后的分数
                    "raw_score": norm_score,  # 原始分数
                    "type": "bm25",
                    "source": doc.metadata.get("source", "unknown")
                })

        # 合并两种检索的结果
        results = filtered_vector_results + filtered_bm25_results
        logger.info(f"📚 混合检索后得到{len(results)}篇文档")
        return results
    
    def _determine_retrieval_weights(self, question: str) -> Tuple[float, float]:
        """动态确定检索策略权重
        
        根据问题的特征和领域知识动态调整向量检索和BM25检索的权重，
        提高混合检索的适用性和准确性
        
        :param question: 用户问题
        :return: (向量检索权重, BM25检索权重)
        """
        # 默认权重
        default_vector = 0.5
        default_bm25 = 0.5
        
        try:
            # 1. 问题类型特征分析
            # 事实型问题特征词（偏向BM25）- 精确匹配更有效
            factual_indicators = [
                '什么是', '定义', '如何', '怎么', '哪些', '谁', '何时', '为什么', 
                '多少', '数据', '标准是', '要求是', '列举', '步骤', '方法',
                '流程', '规定', '地点', '时间', '哪里', '规范', '条例'
            ]
            
            # 概念型问题特征词（偏向向量检索）- 语义理解更有效
            conceptual_indicators = [
                '解释', '分析', '评价', '比较', '区别', '关系', '影响', '原理', 
                '机制', '思考', '可能', '建议', '预测', '推测', '评估', '优缺点',
                '意义', '价值', '联系', '看法', '观点', '理解', '认为'
            ]
            
            # 化工安全特定术语（增加领域特异性）
            chemical_safety_terms = [
                '化学品', '易燃', '易爆', '有毒', '腐蚀', '危险', '安全', '防护', 
                '事故', '泄漏', '爆炸', '火灾', '中毒', '应急', '处置', '风险',
                '危害', '防范', '措施', '操作', '反应', '物质', '气体', '液体', 
                '固体', '浓度', '温度', '压力', '储存', '运输'
            ]
            
            # 2. 多维度特征提取
            # 计算问题类型特征出现次数和强度
            factual_count = sum(1 for term in factual_indicators if term in question)
            conceptual_count = sum(1 for term in conceptual_indicators if term in question)
            domain_term_count = sum(1 for term in chemical_safety_terms if term in question)
            
            # 问题长度因素（较长问题通常偏向语义理解）
            query_length = len(question)
            length_factor = min(1.0, query_length / 50)  # 标准化长度因素
            
            # 问题复杂度因素（句子结构复杂度）
            sentence_count = len([s for s in re.split(r'[。？！.?!]', question) if s.strip()])
            complexity_factor = min(1.0, sentence_count / 3)  # 标准化复杂度因素
            
            # 数字和符号数量（更多数字通常偏向精确匹配）
            digit_count = sum(1 for c in question if c.isdigit())
            digit_factor = min(1.0, digit_count / 5)
            
            # 3. 特征整合与权重计算
            vector_weight = default_vector
            bm25_weight = default_bm25
            
            # 基础权重调整
            if factual_count > conceptual_count:
                # 事实型问题：增加BM25权重
                bm25_base = 0.6 + 0.1 * min(factual_count, 3)
            elif conceptual_count > factual_count:
                # 概念型问题：增加向量权重
                vector_base = 0.6 + 0.1 * min(conceptual_count, 3)
                bm25_base = 1.0 - vector_base
            else:
                # 混合类型问题：根据长度微调
                vector_base = default_vector
                bm25_base = default_bm25
            
            # 应用修正因子
            vector_modifiers = [
                0.1 * length_factor,         # 问题长度修正
                0.1 * complexity_factor,     # 复杂度修正
                -0.1 * digit_factor,         # 数字因素修正(减少向量权重)
                0.05 * min(domain_term_count, 4) / 4  # 领域术语修正
            ]
            
            # 计算最终权重
            if factual_count > conceptual_count:
                # 事实型问题
                bm25_weight = bm25_base
                # 对BM25权重应用小幅修正
                for modifier in vector_modifiers:
                    bm25_weight -= modifier / 2  # 减小修正因子影响
                vector_weight = 1.0 - bm25_weight
            else:
                # 概念型问题或混合型问题
                vector_weight = vector_base
                # 应用完整修正因子
                for modifier in vector_modifiers:
                    vector_weight += modifier
                bm25_weight = 1.0 - vector_weight
            
            # 边界约束
            vector_weight = max(0.2, min(0.8, vector_weight))  # 限制在0.2-0.8范围内
            bm25_weight = 1.0 - vector_weight
            
            # 确保权重和为1
            total = vector_weight + bm25_weight
            normalized_vector = vector_weight / total
            normalized_bm25 = bm25_weight / total
            
            logger.debug(f"问题特征分析: 事实型={factual_count}, 概念型={conceptual_count}, 领域术语={domain_term_count}, 长度={query_length}, 复杂度={sentence_count}, 数字={digit_count}")
            
            return normalized_vector, normalized_bm25
            
        except Exception as e:
            logger.warning(f"⚠️ 动态权重计算失败: {str(e)}")
            return default_vector, default_bm25

    def _rerank_documents(self, results: List[Dict], question: str) -> List[Dict]:
        """使用重排序模型优化检索结果

        :param results: 检索结果列表
        :param question: 原始问题
        :return: 重排序后的结果列表
        """
        try:
            if not results:
                return results

            # 批处理逻辑，每次处理少量文档
            batch_size = 8  # 减小批处理大小以避免张量维度不匹配
            batched_rerank_scores = []
            
            # 限制文档长度，避免过长文档
            max_doc_length = 5000  # 设置最大文档长度
            for res in results:
                if len(res["doc"].page_content) > max_doc_length:
                    res["doc"].page_content = res["doc"].page_content[:max_doc_length]
            
            # 分批处理文档
            for i in range(0, len(results), batch_size):
                batch_results = results[i:i+batch_size]
                batch_pairs = [(question, res["doc"].page_content) for res in batch_results]
                
                try:
                    # 对输入进行tokenize和批处理
                    batch_inputs = self.rerank_tokenizer(
                        batch_pairs,
                        padding=True,
                        truncation=True,
                        max_length=512,  # 限制统一的最大长度
                        return_tensors="pt"
                    )
                    
                    # 确保张量在正确的设备上
                    if self.using_gpu and torch.cuda.is_available():
                        batch_inputs = {k: v.to("cuda") for k, v in batch_inputs.items()}
                    
                    # 模型推理
                    with torch.no_grad():
                        batch_outputs = self.rerank_model(**batch_inputs)
                        # 使用sigmoid转换分数
                        batch_scores = torch.sigmoid(batch_outputs.logits).squeeze().cpu().tolist()
                        
                        # 确保batch_scores是列表
                        if not isinstance(batch_scores, list):
                            batch_scores = [batch_scores]
                        
                        batched_rerank_scores.extend(batch_scores)
                except Exception as e:
                    # 批处理失败时，使用原始分数
                    logger.warning(f"文档批次 {i//batch_size+1} 重排序失败: {str(e)}")
                    for res in batch_results:
                        batched_rerank_scores.append(res["score"])

            # 更新结果分数
            for res, rerank_score in zip(results, batched_rerank_scores):
                # 直接使用重排序分数作为最终分数
                res.update({
                    "original_score": res["score"],  # 保存原始检索分数
                    "rerank_score": rerank_score,
                    "final_score": rerank_score  # 直接使用重排序分数作为最终分数
                })
                
                # 记录日志
                logger.debug(f"文档重排序: {res['source']} - 原始分数: {res['original_score']:.4f} - 重排序分数: {rerank_score:.4f}")

            # 按最终分数降序排列
            sorted_results = sorted(results, key=lambda x: x["final_score"], reverse=True)
            
            # 应用多样性增强策略
            return self._diversify_results(sorted_results)
            
        except Exception as e:
            logger.error(f"重排序整体失败: {str(e)}")
            # 确保每个结果都有必要的字段
            for res in results:
                if "final_score" not in res:
                    res["final_score"] = res["score"]
                if "rerank_score" not in res:
                    res["rerank_score"] = res["score"]
                if "original_score" not in res:
                    res["original_score"] = res["score"]
            
            # 返回原始排序的结果
            return sorted(results, key=lambda x: x["score"], reverse=True)
    
    def _diversify_results(self, ranked_results: List[Dict]) -> List[Dict]:
        """增强检索结果的多样性
        
        使用MMR(Maximum Marginal Relevance)算法平衡相关性和多样性
        
        :param ranked_results: 按分数排序的检索结果
        :return: 多样性增强后的结果
        """
        if len(ranked_results) <= 2:
            return ranked_results  # 结果太少不需要多样性优化
        
        try:
            # MMR参数
            lambda_param = 0.7  # 控制相关性vs多样性的平衡，越大越偏向相关性
            
            # 初始化已选择和候选文档
            selected = [ranked_results[0]]  # 最高分文档直接选入
            candidates = ranked_results[1:]
            
            # 处理top 20文档
            while len(selected) < min(len(ranked_results), self.config.final_top_k):
                # 计算每个候选文档的MMR分数
                mmr_scores = []
                
                for candidate in candidates:
                    # 计算相似度分数（相关性部分）
                    relevance = candidate["final_score"]
                    
                    # 计算与已选文档的最大相似度（多样性部分）
                    max_sim = 0
                    for selected_doc in selected:
                        # 使用文本内容的词重叠计算相似度
                        sim = self._compute_document_similarity(
                            candidate["doc"].page_content,
                            selected_doc["doc"].page_content
                        )
                        max_sim = max(max_sim, sim)
                    
                    # 计算MMR分数
                    mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                    mmr_scores.append(mmr)
                
                # 选择MMR分数最高的文档
                best_idx = mmr_scores.index(max(mmr_scores))
                selected.append(candidates.pop(best_idx))
            
            # 返回多样性增强后的文档
            return selected
            
        except Exception as e:
            logger.error(f"多样性增强失败: {str(e)}")
            # 失败时返回原始排序的前20个文档
            return ranked_results[:self.config.final_top_k]
    
    def _compute_document_similarity(self, doc1: str, doc2: str) -> float:
        """计算两个文档之间的相似度
        
        :param doc1: 第一个文档内容
        :param doc2: 第二个文档内容
        :return: 相似度分数（0-1）
        """
        try:
            # 使用基于词集合的Jaccard相似度
            tokens1 = set(self._tokenize(doc1))
            tokens2 = set(self._tokenize(doc2))
            
            # 计算Jaccard系数
            if not tokens1 or not tokens2:
                return 0.0
                
            intersection = tokens1.intersection(tokens2)
            union = tokens1.union(tokens2)
            
            # 如果文档长度相差太大，给予惩罚
            len_ratio = min(len(doc1), len(doc2)) / max(len(doc1), len(doc2))
            
            # 加权相似度
            return (len(intersection) / len(union)) * len_ratio
            
        except Exception as e:
            logger.warning(f"文档相似度计算失败: {str(e)}")
            return 0.0

    def _retrieve_documents(self, question: str, use_rerank: bool = True) -> Tuple[List[Document], List[Dict]]:
        """完整检索流程

        :param question: 用户问题
        :param use_rerank: 是否使用重排序，默认为True
        :return: (文档列表, 分数信息列表)
        """
        try:
            # 开始计时
            start_time = time.time()
            
            # 混合检索
            hybrid_start = time.time()
            raw_results = self._hybrid_retrieve(question)
            hybrid_time = time.time() - hybrid_start
            
            if not raw_results:
                logger.warning("混合检索未返回任何结果")
                return [], []

            # 重排序(可跳过)
            rerank_time = 0
            if use_rerank:
                rerank_start = time.time()
                try:
                    reranked = self._rerank_documents(raw_results, question)
                except Exception as e:
                    logger.error(f"重排序完全失败，使用原始结果: {str(e)}")
                    # 确保每个结果都有必要的字段
                    for res in raw_results:
                        if "final_score" not in res:
                            res["final_score"] = res["score"]
                        if "rerank_score" not in res:
                            res["rerank_score"] = res["score"]
                    reranked = sorted(raw_results, key=lambda x: x["score"], reverse=True)
                rerank_time = time.time() - rerank_start
            else:
                # 如果不使用重排序，直接使用混合检索结果
                logger.info("⏩ 跳过重排序步骤，直接使用混合检索结果")
                # 确保results有所需字段
                for res in raw_results:
                    res["final_score"] = res["score"]  # 使用原始分数作为最终分数
                    res["rerank_score"] = res["score"]  # 设置相同的rerank_score值
                    res["original_score"] = res["score"]  # 保存原始分数
                reranked = sorted(raw_results, key=lambda x: x["score"], reverse=True)

            # 根据阈值过滤结果
            filter_start = time.time()
            try:
                final_results = [
                    res for res in reranked
                    if res["final_score"] >= self.config.similarity_threshold
                    and len(res["doc"].page_content.strip()) >= 12  # 添加长度检查
                ]
                final_results = sorted(
                    final_results,
                    key=lambda x: x["final_score"],
                    reverse=True
                )[:self.config.final_top_k]  # 限制返回数量
            except Exception as e:
                logger.error(f"结果过滤失败，使用前N个结果: {str(e)}")
                final_results = reranked[:min(len(reranked), self.config.final_top_k)]
            filter_time = time.time() - filter_start

            # 输出最终分数信息和时间统计
            total_time = time.time() - start_time
            logger.info(f"📊 最终文档数目:{len(final_results)}篇")
            
            # 根据是否使用重排序输出不同的日志
            if use_rerank:
                logger.info(f"⏱️ 检索耗时统计: 总计 {total_time:.3f}秒 | 混合检索: {hybrid_time:.3f}秒 | 重排序: {rerank_time:.3f}秒 | 过滤: {filter_time:.3f}秒")
            else:
                logger.info(f"⏱️ 检索耗时统计: 总计 {total_time:.3f}秒 | 混合检索: {hybrid_time:.3f}秒 | 过滤: {filter_time:.3f}秒 (跳过重排序)")

            # 提取文档和分数信息
            docs = []
            score_info = []
            
            for res in final_results:
                try:
                    doc = res["doc"]
                    info = {
                        "source": res["source"],
                        "type": res.get("type", "unknown"),
                        "vector_score": res.get("score", 0),
                        "bm25_score": res.get("score", 0),
                        "rerank_score": res.get("rerank_score", res.get("score", 0)),
                        "final_score": res.get("final_score", res.get("score", 0))
                    }
                    docs.append(doc)
                    score_info.append(info)
                except Exception as e:
                    logger.warning(f"处理单个结果时出错，已跳过: {str(e)}")
                    continue

            return docs, score_info
        except Exception as e:
            logger.error(f"文档检索严重失败: {str(e)}", exc_info=True)
            # 紧急情况下返回空结果而不是抛出异常
            return [], []

    def _build_prompt(self, question: str, context: str) -> str:
        """构建提示词模板"""
        # 系统角色定义
        system_role = (
            "你是一位经验丰富的化工安全领域专家，具有深厚的专业知识和实践经验。"
            "你需要基于提供的参考资料，给出准确、专业且易于理解的回答。"
        )
        
        # 详细工作指南
        instruction = (
            "工作指南：\n"
            "1. 引用知识：回答必须基于检索到的参考资料内容，不要编造或臆测信息\n"
            "2. 专业性：使用适当的化工安全术语，保持专业性\n"
            "3. 可读性：将复杂概念解释得清晰易懂，避免过度使用专业术语\n"
            "4. 结构性：回答应有清晰的结构，先概述要点，再详细展开\n"
            "5. 引用来源：在回答中适当引用参考资料来源，可使用「根据XX文档」的形式\n"
            "6. 知识边界：如参考资料不包含问题的答案，坦诚表明「参考资料不包含此信息」\n"
        )
        
        # 思考过程指导
        thinking_guide = (
            "思考过程：\n"
            "1. 仔细阅读参考资料，识别与问题相关的关键信息\n"
            "2. 分析问题需求，确定回答框架\n"
            "3. 组织相关信息，形成系统性回答\n"
            "4. 确保回答准确、全面且符合化工安全领域专业标准\n"
        )
        
        if context:
            return (
                "<|im_start|>system\n"
                f"{system_role}\n"
                f"{instruction}\n"
                f"{thinking_guide}\n"
                "参考资料：\n{context}\n"
                "请根据以上参考资料回答用户问题。如果参考资料不足以回答，请明确指出。\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ).format(question=question, context=context[:self.config.max_context_length])
        else:
            return (
                "<|im_start|>system\n"
                f"{system_role}\n"
                f"{instruction}\n"
                "注意：未找到与问题相关的参考资料，请基于化工安全领域的专业知识谨慎回答。\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ).format(question=question)

    def _build_chat_prompt(self, current_question: str, chat_history: List[Dict], context: str = "") -> str:
        """构建多轮对话的提示词模板
        
        :param current_question: 当前用户问题
        :param chat_history: 聊天历史记录列表，包含message_type和content
        :param context: 相关文档上下文
        :return: 完整的提示词
        """
        # 系统角色定义
        system_role = (
            "你是一位经验丰富的化工安全领域专家，具有深厚的专业知识和实践经验。"
            "你需要基于提供的参考资料和聊天历史，给出准确、专业且易于理解的回答。"
        )
        
        # 详细工作指南
        instruction = (
            "工作指南：\n"
            "1. 引用知识：回答必须基于检索到的参考资料内容，不要编造或臆测信息\n"
            "2. 专业性：使用适当的化工安全术语，保持专业性\n"
            "3. 可读性：将复杂概念解释得清晰易懂，避免过度使用专业术语使回答难以理解\n"
            "4. 连贯性：考虑对话历史，保持回答的一致性\n"
            "5. 引用来源：在回答中适当引用参考资料来源，可使用「根据XX文档」的形式\n"
            "6. 知识边界：如参考资料不包含问题的答案，坦诚表明「参考资料不包含此信息」\n"
        )
        
        # 构建系统提示部分
        prompt = "<|im_start|>system\n" + system_role + "\n" + instruction + "\n"
        
        # 添加参考资料（如果有）
        if context:
            prompt += "参考资料：\n" + context[:self.config.max_context_length] + "\n"
            prompt += "请根据以上参考资料回答用户问题。如果参考资料不足以回答，请明确指出。\n"
        else:
            prompt += "注意：未找到与问题相关的参考资料，请基于化工安全领域的专业知识谨慎回答。\n"
        
        prompt += "<|im_end|>\n"
        
        # 添加聊天历史
        for message in chat_history:
            role = "user" if message["message_type"] == "user" else "assistant"
            content = message.get("content", "")
            if content:  # 确保消息内容不为空
                prompt += f"<|im_start|>{role}\n{content}\n<|im_end|>\n"
        
        # 添加当前问题和助手角色
        prompt += f"<|im_start|>user\n{current_question}\n<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        
        return prompt
        
    def _format_references(self, docs: List[Document], score_info: List[Dict]) -> List[Dict]:
        """格式化参考文档信息"""
        return [
            {
                "file": str(Path(info["source"]).name),  # 文件名
                "content": doc.page_content,  # 截取前500字符
                "score": info["final_score"],  # 综合评分
                "type": info["type"],  # 检索类型
                "full_path": info["source"]  # 完整文件路径
            }
            for doc, info in zip(docs, score_info)
        ]


    def stream_query_with_history(self, session_id: str, current_question: str, 
                               chat_history: List[Dict] = None) -> Generator[str, None, None]:
        """带聊天历史的流式RAG查询
        
        :param session_id: 会话ID
        :param current_question: 当前用户问题
        :param chat_history: 聊天历史列表
        :return: 生成器，流式输出结果
        """
        logger.info(f"🔄 多轮对话处理 | 会话ID: {session_id} | 问题: {current_question[:50]}...")
        
        if not current_question.strip():
            yield json.dumps({
                "type": "error",
                "data": "请输入有效问题"
            }) + "\n"
            return
        
        # 初始化聊天历史
        if chat_history is None:
            chat_history = []
        
        try:
            # 阶段1：文档检索
            try:
                docs, score_info = self._retrieve_documents(current_question)
                if not docs:
                    logger.warning(f"查询 '{current_question[:50]}...' 未找到相关文档")
                    # 当没有文档时，仍然使用历史记录，但无上下文
                    context = ""
                else:
                    # 格式化参考文档信息并发送
                    references = self._format_references(docs, score_info)
                    yield json.dumps({
                        "type": "references",
                        "data": references
                    }) + "\n"
                    
                    # 构建上下文
                    context = "\n\n".join([
                        f"【参考文档{i + 1}】{doc.page_content}\n"
                        f"- 来源: {Path(info['source']).name}\n"
                        f"- 综合置信度: {info['final_score'] * 100:.1f}%"
                        for i, (doc, info) in enumerate(zip(docs, score_info))
                    ])
            except Exception as e:
                logger.error(f"文档检索失败: {str(e)}", exc_info=True)
                # 检索失败时使用空上下文
                context = ""
                yield json.dumps({
                    "type": "error", 
                    "data": "文档检索服务暂时不可用，将使用聊天历史回答..."
                }) + "\n"
            
            # 阶段2：构建多轮对话提示
            prompt = self._build_chat_prompt(current_question, chat_history, context)
            
            # 阶段3：流式生成
            try:
                for chunk in self.llm.stream(prompt):
                    cleaned_chunk = chunk.replace("<|im_end|>", "")
                    if cleaned_chunk:
                        # 发送生成内容
                        yield json.dumps({
                            "type": "content",
                            "data": cleaned_chunk
                        }) + "\n"
            except Exception as e:
                logger.error(f"流式生成中断: {str(e)}")
                yield json.dumps({
                    "type": "error",
                    "data": "\n生成过程发生意外中断，请刷新页面重试"
                }) + "\n"
                
        except Exception as e:
            logger.exception(f"多轮对话处理错误: {str(e)}")
            yield json.dumps({
                "type": "error",
                "data": " 系统处理请求时发生严重错误，请联系管理员"
            }) + "\n"
            
    def stream_query_model_with_history(self, session_id: str, current_question: str, 
                                 chat_history: List[Dict] = None) -> Generator[str, None, None]:
        """直接大模型的多轮对话流式生成（不使用知识库）
        
        :param session_id: 会话ID
        :param current_question: 当前用户问题
        :param chat_history: 聊天历史列表
        :return: 生成器，流式输出结果
        """
        logger.info(f"🔄 直接多轮对话 | 会话ID: {session_id} | 问题: {current_question[:50]}...")
        
        if not current_question.strip():
            yield json.dumps({
                "type": "error",
                "data": "⚠️ 请输入有效问题"
            }) + "\n"
            return
        
        # 初始化聊天历史
        if chat_history is None:
            chat_history = []
        
        try:
            # 构建多轮对话提示（无知识库上下文）
            prompt = self._build_chat_prompt(current_question, chat_history)
            
            # 流式生成
            try:
                for chunk in self.llm.stream(prompt):
                    cleaned_chunk = chunk.replace("<|im_end|>", "")
                    if cleaned_chunk:
                        # 发送生成内容
                        yield json.dumps({
                            "type": "content",
                            "data": cleaned_chunk
                        }) + "\n"
            except Exception as e:
                logger.error(f"直接多轮对话生成中断: {str(e)}")
                yield json.dumps({
                    "type": "error",
                    "data": "\n⚠️ 生成过程发生意外中断，请刷新页面重试"
                }) + "\n"
                
        except Exception as e:
            logger.exception(f"直接多轮对话处理错误: {str(e)}")
            yield json.dumps({
                "type": "error",
                "data": "⚠️ 系统处理请求时发生严重错误，请联系管理员"
            }) + "\n"

    def answer_query(self, question: str) -> Tuple[str, List[Dict], Dict]:
        """非流式RAG生成，适用于评估模块
        
        Args:
            question: 用户问题
            
        Returns:
            Tuple(生成的回答, 检索的文档列表, 元数据)
        """
        logger.info(f"🔍 非流式处理查询(用于评估): {question[:50]}...")
        
        try:
            # 阶段1：文档检索
            try:
                docs, score_info = self._retrieve_documents(question)
                if not docs:
                    logger.warning(f"评估查询 '{question[:50]}...' 未找到相关文档")
                    return "未找到相关文档，无法回答该问题。", [], {"status": "no_docs"}
            except Exception as e:
                logger.error(f"评估模式下文档检索失败: {str(e)}", exc_info=True)
                return f"文档检索失败: {str(e)}", [], {"status": "retrieval_error", "error": str(e)}
            
            # 格式化参考文档信息
            try:
                references = self._format_references(docs, score_info)
            except Exception as e:
                logger.error(f"格式化参考文档失败: {str(e)}")
                # 创建简化版参考信息
                references = [{"file": f"文档{i+1}", "content": doc.page_content[:200] + "..."} 
                             for i, doc in enumerate(docs)]
            
            # 阶段2：构建上下文
            try:
                context = "\n\n".join([
                    f"【参考文档{i + 1}】{doc.page_content}\n"
                    f"- 来源: {Path(info['source']).name}\n"
                    f"- 综合置信度: {info['final_score'] * 100:.1f}%"
                    for i, (doc, info) in enumerate(zip(docs, score_info))
                ])
            except Exception as e:
                logger.error(f"构建上下文失败: {str(e)}")
                # 如果构建上下文失败，使用简化版本
                context = "\n\n".join([f"【参考文档{i + 1}】{doc.page_content}" 
                                     for i, doc in enumerate(docs)])
            
            # 阶段3：构建提示模板
            prompt = self._build_prompt(question, context)
            
            # 阶段4：一次性生成（非流式）
            try:
                answer = self.llm.invoke(prompt)
                cleaned_answer = answer.replace("<|im_end|>", "").strip()
                
                return cleaned_answer, references, {"status": "success"}
            except Exception as e:
                logger.error(f"生成回答失败: {str(e)}")
                # 尝试使用简化提示
                try:
                    simple_prompt = (
                        "<|im_start|>system\n"
                        "你是一位经验丰富的化工安全领域专家，请尽量回答用户问题。\n"
                        "<|im_end|>\n"
                        "<|im_start|>user\n"
                        f"{question}\n"
                        "<|im_end|>\n"
                        "<|im_start|>assistant\n"
                    )
                    fallback_answer = self.llm.invoke(simple_prompt)
                    cleaned_fallback = fallback_answer.replace("<|im_end|>", "").strip()
                    return cleaned_fallback, references, {"status": "partial_success", "error": str(e)}
                except:
                    return f"生成回答失败: {str(e)}", references, {"status": "generation_error", "error": str(e)}
            
        except Exception as e:
            logger.exception(f"非流式处理严重错误: {str(e)}")
            return f"处理请求时发生错误: {str(e)}", [], {"status": "error", "error": str(e)}

