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
                #deepseek_8b_lora:latest    1513b8b198dc    8.5 GB    59 seconds ago
                # deepseek-r1:8b             28f8fd6cdc67    4.9 GB    46 minutes ago
                # deepseek-r1:14b            ea35dfe18182    9.0 GB    29 hours ago
                base_url=self.config.ollama_base_url,  # Ollama服务地址
                temperature=self.config.llm_temperature,  # 温度参数控制随机性
                num_predict=self.config.llm_max_tokens,  # 最大生成token数
                stop=["<|im_end|>", "【答案】"]
            )

            # 测试模型连接
            test_prompt = "测试连接"
            self.llm.invoke(test_prompt)
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
                    logger.info(f"发现BM25分词缓存，正在加载: {cache_path}")
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
                    logger.info(f"保存分词结果到缓存: {cache_path}")
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

    def _enhance_query(self, original_query: str) -> List[str]:
        """查询增强与扩展
        
        :param original_query: 原始查询
        :return: 增强后的查询列表
        """
        # 基础查询始终包含原始查询
        queries = [original_query]
        
        try:
            # 1. 移除停用词的简化查询
            stop_words = {'的', '了', '是', '在', '我', '有', '和', '就', '不', '人', '都', 
                         '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', 
                         '会', '着', '没有', '看', '好', '自己', '这'}
            
            words = self._tokenize(original_query)
            simplified_query = ' '.join([w for w in words if w not in stop_words])
            
            if simplified_query and simplified_query != original_query:
                queries.append(simplified_query)
            
            # 2. 专业术语提取和重点关注
            # 化工安全领域的专业术语及其权重
            chemical_terms = {
                '化学品': 2.0, '易燃': 2.0, '易爆': 2.0, '有毒': 2.0, '腐蚀': 2.0, 
                '危险': 1.5, '安全': 1.5, '防护': 1.5, '事故': 1.5, '泄漏': 2.0,
                '爆炸': 2.0, '火灾': 2.0, '中毒': 2.0, '应急': 1.5, '处置': 1.5,
                '风险': 1.5, '危害': 1.5, '防范': 1.5, '措施': 1.0, '操作': 1.0,
                '反应': 1.8, '物质': 1.8, '气体': 1.8, '液体': 1.8, '固体': 1.8,
                '浓度': 1.8, '温度': 1.8, '压力': 1.8, '储存': 1.8, '运输': 1.8
            }
            
            # 提取查询中的专业术语
            matched_terms = []
            term_weights = {}
            
            for term, weight in chemical_terms.items():
                if term in original_query:
                    matched_terms.append(term)
                    term_weights[term] = weight
            
            if matched_terms:
                # 构建专业术语增强的查询
                terms_query = ' '.join(matched_terms)
                if terms_query != original_query and len(matched_terms) >= 2:
                    queries.append(terms_query)
                
                # 构建加权查询，复制重要术语
                weighted_query_parts = []
                for word in words:
                    if word in term_weights:
                        # 根据权重重复术语
                        repeat = max(1, int(term_weights[word]))
                        weighted_query_parts.extend([word] * repeat)
                    else:
                        weighted_query_parts.append(word)
                
                weighted_query = ' '.join(weighted_query_parts)
                if weighted_query != original_query:
                    queries.append(weighted_query)
            
            logger.info(f"📝 查询增强: 从原始查询'{original_query}'生成了{len(queries)}个变体")
            return queries
            
        except Exception as e:
            logger.warning(f"⚠️ 查询增强失败: {str(e)}")
            return [original_query]  # 返回原始查询

    def _hybrid_retrieve(self, question: str) -> List[Dict[str, Any]]:
        """混合检索流程（向量+BM25）

        :param question: 用户问题
        :return: 包含文档和检索信息的字典列表
        """
        results = []
        
        # 查询增强处理
        enhanced_queries = self._enhance_query(question)
        
        # 动态确定检索策略权重
        vector_weight, bm25_weight = self._determine_retrieval_weights(question)

        # 向量检索部分
        all_vector_results = []
        for query in enhanced_queries:
            vector_results = self.vector_store.similarity_search_with_score(
                query, k=self.config.vector_top_k  # 获取top k结果
            )
            all_vector_results.extend(vector_results)
        
        # 去重并保留最高分数
        unique_vector_results = {}
        for doc, score in all_vector_results:
            doc_id = doc.metadata.get("source", "") + str(hash(doc.page_content))
            norm_score = (score + 1) / 2  # 转换为标准余弦值（0~1范围）
            
            # 如果文档已存在且新分数更高，则更新
            if doc_id not in unique_vector_results or norm_score > unique_vector_results[doc_id][1]:
                unique_vector_results[doc_id] = (doc, norm_score)
        
        # 对向量检索结果进行阈值过滤
        filtered_vector_results = []
        for doc, score in unique_vector_results.values():
            if score >= self.config.vector_similarity_threshold:  # 使用统一的相似度阈值
                filtered_vector_results.append({
                    "doc": doc,
                    "score": score,  # 应用动态权重
                    "raw_score": score,
                    "type": "vector",
                    "source": doc.metadata.get("source", "unknown")
                })
                # logger.info(f"🔍 向量检索结果: {doc.metadata['source']} - 分数: {score:.4f}")

        # BM25检索部分
        all_bm25_scores = {}
        for query in enhanced_queries:
            tokenized_query = self._tokenize(query)  # 问题分词
            bm25_scores = self.bm25.get_scores(tokenized_query)  # 计算BM25分数
            
            # 更新最高分数
            for idx, score in enumerate(bm25_scores):
                if idx not in all_bm25_scores or score > all_bm25_scores[idx]:
                    all_bm25_scores[idx] = score
        
        # 获取top k的索引（倒序排列）
        top_bm25_indices = np.argsort(list(all_bm25_scores.values()))[-self.config.bm25_top_k:][::-1]
        top_bm25_indices = [list(all_bm25_scores.keys())[i] for i in top_bm25_indices]

        # 对BM25分数进行归一化处理
        bm25_scores = [all_bm25_scores[idx] for idx in top_bm25_indices]
        if bm25_scores:  # 确保有分数可以归一化
            # 计算均值和标准差
            mean_score = np.mean(bm25_scores)
            std_score = np.std(bm25_scores) + 1e-9  # 避免除以0
            
            # 使用Logistic归一化
            normalized_bm25_scores = []
            for score in bm25_scores:
                # 先进行Z-score标准化
                z_score = (score - mean_score) / std_score
                # 然后应用Sigmoid函数
                logistic_score = 1 / (1 + np.exp(-z_score))
                normalized_bm25_scores.append(logistic_score)
        else:
            normalized_bm25_scores = []

        # 对BM25检索结果进行阈值过滤
        filtered_bm25_results = []
        for idx, norm_score in zip(top_bm25_indices, normalized_bm25_scores):
            if norm_score >= self.config.bm25_similarity_threshold:  # 使用统一的相似度阈值
                doc = Document(
                    page_content=self.bm25_docs[idx],
                    metadata=self.doc_metadata[idx]
                )
                filtered_bm25_results.append({
                    "doc": doc,
                    "score": norm_score,  # 使用归一化后的分数
                    "raw_score": norm_score,
                    "type": "bm25",
                    "source": doc.metadata.get("source", "unknown")
                })
                # logger.info(f"🔍 BM25检索结果: {doc.metadata['source']} - 原始分数: {all_bm25_scores[idx]:.4f} - 归一化分数: {norm_score:.4f}")

        # 合并过滤后的结果
        results = filtered_vector_results + filtered_bm25_results

        logger.info(f"📚 混合检索后得到{len(results)}篇文档")
        return results
    
    def _determine_retrieval_weights(self, question: str) -> Tuple[float, float]:
        """动态确定检索策略权重
        
        :param question: 用户问题
        :return: (向量检索权重, BM25检索权重)
        """
        # 默认权重
        default_vector = 0.5
        default_bm25 = 0.5
        
        try:
            # 1. 检测问题类型特征
            
            # 事实型问题特征词（偏向BM25）
            factual_indicators = ['什么是', '定义', '如何', '怎么', '哪些', 
                               '谁', '何时', '为什么', '多少', '数据',
                               '标准是', '要求是']
            
            # 概念型问题特征词（偏向向量检索）
            conceptual_indicators = ['解释', '分析', '评价', '比较', '区别',
                                  '关系', '影响', '原理', '机制', '思考',
                                  '可能', '建议', '预测', '推测']
                               
            # 计算各类特征出现次数
            factual_count = sum(1 for term in factual_indicators if term in question)
            conceptual_count = sum(1 for term in conceptual_indicators if term in question)
            
            # 2. 考虑问题长度因素
            # 较短问题通常是直接查询，适合关键词匹配
            # 较长问题可能是复杂概念，适合语义匹配
            query_length = len(question)
            length_factor = min(1.0, query_length / 50)  # 标准化长度因素
            
            # 3. 确定最终权重
            if factual_count > conceptual_count:
                # 事实型问题：增加BM25权重
                bm25_weight = 0.6 + 0.1 * min(factual_count, 3)
                vector_weight = 1.0 - bm25_weight
            elif conceptual_count > factual_count:
                # 概念型问题：增加向量权重
                vector_weight = 0.6 + 0.1 * min(conceptual_count, 3) + 0.1 * length_factor
                bm25_weight = 1.0 - vector_weight
            else:
                # 混合类型问题：根据长度微调
                vector_weight = default_vector + 0.1 * length_factor
                bm25_weight = 1.0 - vector_weight
                
            # 确保权重相加为1
            total = vector_weight + bm25_weight
            return vector_weight/total, bm25_weight/total
            
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
            # 准备模型输入对（问题-文档）
            pairs = [(question, res["doc"].page_content) for res in results]

            # 对输入进行tokenize和批处理
            inputs = self.rerank_tokenizer(
                pairs,
                padding=True,  # 自动填充
                truncation=True,  # 自动截断
                max_length=2048,  # 最大长度限制
                return_tensors="pt"  # 返回PyTorch张量
            )

            # 模型推理
            with torch.no_grad():
                outputs = self.rerank_model(**inputs)
                # 使用sigmoid转换分数
                rerank_scores = torch.sigmoid(outputs.logits).squeeze().tolist()
                
                # 确保rerank_scores是列表
                if not isinstance(rerank_scores, list):
                    rerank_scores = [rerank_scores]

            # 合并分数
            for res, rerank_score in zip(results, rerank_scores):
                # 打印各个分数
                
                # 加权平均策略
                final_score = (
                        self.config.retrieval_weight * res["score"] +
                        self.config.rerank_weight * rerank_score
                )
                
                res.update({
                    "rerank_score": rerank_score,
                    "final_score": final_score
                })

            # 按最终分数降序排列
            sorted_results = sorted(results, key=lambda x: x["final_score"], reverse=True)
            
            # 应用多样性增强策略
            return self._diversify_results(sorted_results)
            
        except Exception as e:
            logger.error(f"重排序失败: {str(e)}")
            return results  # 失败时返回原始排序
    
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

    def _retrieve_documents(self, question: str) -> Tuple[List[Document], List[Dict]]:
        """完整检索流程

        :param question: 用户问题
        :return: (文档列表, 分数信息列表)
        """
        try:
            # 混合检索
            raw_results = self._hybrid_retrieve(question)
            if not raw_results:
                return [], []

            # 直接重排序
            reranked = self._rerank_documents(raw_results, question)

            # 根据阈值过滤结果
            final_results = [
                res for res in reranked
                if res["final_score"] >= self.config.similarity_threshold
                and len(res["doc"].page_content.strip()) >= 12  # 添加长度检查
            ]
            final_results = sorted(
                final_results,
                key=lambda x: x["final_score"],
                reverse=True
            )

            # 输出最终分数信息
            logger.info(f"📊 最终文档数目:{len(final_results)}篇")
            logger.info("📊 最终检索结果:")
            for i, res in enumerate(final_results, 1):
                logger.info(
                    f"文档 {i}: {res['source']}\n"
                    # f"- 检索类型: {res['type']}\n"
                    # f"- 原始分数: {res['raw_score']:.4f}\n"
                    # f"- 重排序分数: {res['rerank_score']:.4f}\n"
                    # f"- 最终分数: {res['final_score']:.4f}\n"
                )

            # 提取文档和分数信息
            docs = [res["doc"] for res in final_results]
            score_info = [{
                "source": res["source"],
                "type": res["type"],
                "vector_score": res.get("score", 0),  # 兼容不同检索类型
                "bm25_score": res.get("score", 0),
                "rerank_score": res["rerank_score"],
                "final_score": res["final_score"]
            } for res in final_results]

            return docs, score_info
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            raise

    def _build_prompt(self, question: str, context: str) -> str:
        """添加 CoT 触发指令"""
        cot_instruction = (
            "请逐步推理并解释你的思考过程，将思考过程放在<think></think>标签中"
        )
        
        if context:
            return (
                "<|im_start|>system\n"
                f"你是一位经验丰富的化工安全领域专家，{cot_instruction}\n"
                "上下文：\n{context}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ).format(question=question, context=context[:self.config.max_context_length])
        else:
            return (
                "<|im_start|>system\n"
                f"你是一位经验丰富的化工安全领域专家，{cot_instruction}\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ).format(question=question)

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
    def stream_query_model(self, question: str) -> Generator[str, None, None]:
        """纯模型流式生成（不经过RAG）"""
        logger.info(f"🌀 正在直接流式生成: {question[:50]}...")
        try:
            if not question.strip():
                yield "⚠️ 请输入有效问题"
                return

            # 构建基础提示模板（不包含上下文）
            prompt = (
                "<|im_start|>system\n"
                "你是一位经验丰富的化工安全领域专家，请专业且准确地回答问题。\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                f"{question}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

            try:
                full_response = ""
                for chunk in self.llm.stream(prompt):
                    cleaned_chunk = chunk.replace("<|im_end|>", "")
                    if cleaned_chunk:
                        # 发送生成内容（作为普通文本）
                        yield json.dumps({
                            "type": "content",
                            "data": cleaned_chunk
                        }) + "\n"

            except Exception as e:
                logger.error(f"直接生成中断: {str(e)}")
                yield "\n⚠️ 生成过程发生意外中断，请稍后重试"

        except Exception as e:
            logger.exception("直接流式生成错误")
            yield "⚠️ 系统处理请求时发生严重错误，请联系管理员"

    def stream_query_rag_with_kb(self, question: str) -> Generator[str, None, None]:
        """结合知识库的流式RAG生成"""
        logger.info(f"🌊 正在流式处理查询: {question[:50]}...")
        if not question.strip():
            yield "⚠️ 请输入有效问题"
            return

        try:
            # 阶段1：文档检索
            try:
                docs, score_info = self._retrieve_documents(question)
                if not docs:
                    yield "⚠️ 未找到相关文档..."
                    return
            except Exception as e:
                logger.error(f"文档检索失败: {str(e)}")
                yield "⚠️ 文档检索服务暂时不可用"
                return

            # 格式化参考文档信息
            references = self._format_references(docs, score_info)

            # 发送参考文档信息
            yield json.dumps({
                "type": "references",
                "data": references
            }) + "\n"  # 添加换行符作为结束标记

            # 阶段2：构建上下文
            context = "\n\n".join([
                f"【参考文档{i + 1}】{doc.page_content}\n"
                f"- 来源: {Path(info['source']).name}\n"
                f"- 综合置信度: {info['final_score'] * 100:.1f}%"
                for i, (doc, info) in enumerate(zip(docs, score_info))
            ])

            # 阶段3：构建提示模板
            prompt = self._build_prompt(question, context)

            # 阶段4：流式生成
            try:
                full_response = ""
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
                    "data": "\n⚠️ 生成过程发生意外中断"
                }) + "\n"

        except Exception as e:
            logger.exception("流式处理严重错误")
            yield json.dumps({
                "type": "error",
                "data": "⚠️ 系统处理请求时发生严重错误"
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
            docs, score_info = self._retrieve_documents(question)
            if not docs:
                return "未找到相关文档，无法回答该问题。", [], {"status": "no_docs"}
            
            # 格式化参考文档信息
            references = self._format_references(docs, score_info)
            
            # 阶段2：构建上下文
            context = "\n\n".join([
                f"【参考文档{i + 1}】{doc.page_content}\n"
                f"- 来源: {Path(info['source']).name}\n"
                f"- 综合置信度: {info['final_score'] * 100:.1f}%"
                for i, (doc, info) in enumerate(zip(docs, score_info))
            ])
            
            # 阶段3：构建提示模板
            prompt = self._build_prompt(question, context)
            
            # 阶段4：一次性生成（非流式）
            answer = self.llm.invoke(prompt)
            cleaned_answer = answer.replace("<|im_end|>", "").strip()
            
            return cleaned_answer, references, {"status": "success"}
            
        except Exception as e:
            logger.exception(f"非流式处理严重错误: {str(e)}")
            return f"处理请求时发生错误: {str(e)}", [], {"status": "error", "error": str(e)}

