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
import re  # 用于正则表达式

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

    def _hybrid_retrieve(self, question: str) -> List[Dict[str, Any]]:
        """混合检索流程（向量+BM25），根据动态权重融合结果

        :param question: 用户问题
        :return: 包含文档和检索信息的字典列表
        """
        results = []
        
        # 动态确定检索策略权重
        vector_weight, bm25_weight = self._determine_retrieval_weights(question)
        logger.info(f"查询权重 - 向量检索: {vector_weight:.2f}, BM25检索: {bm25_weight:.2f}")

        # 向量检索部分
        vector_results = self.vector_store.similarity_search_with_score(
            question, k=self.config.vector_top_k  # 获取top k结果
        )
        
        # 处理向量检索结果
        filtered_vector_results = []
        for doc, score in vector_results:
            # 转换为标准余弦值（0~1范围）
            norm_score = (score + 1) / 2
            
            if norm_score >= self.config.vector_similarity_threshold:  # 使用相似度阈值过滤
                filtered_vector_results.append({
                    "doc": doc,
                    "score": norm_score,  # 原始分数
                    "weighted_score": norm_score * vector_weight,  # 应用权重后的分数
                    "raw_score": norm_score,
                    "type": "vector",
                    "source": doc.metadata.get("source", "unknown")
                })

        # BM25检索部分
        tokenized_query = self._tokenize(question)  # 问题分词
        bm25_scores = self.bm25.get_scores(tokenized_query)  # 计算BM25分数
        
        # 获取top k的索引（倒序排列）
        top_bm25_indices = np.argsort(bm25_scores)[-self.config.bm25_top_k:][::-1]
        
        # 对BM25分数进行归一化处理
        selected_bm25_scores = [bm25_scores[idx] for idx in top_bm25_indices]
        if selected_bm25_scores:  # 确保有分数可以归一化
            # 计算均值和标准差
            mean_score = np.mean(selected_bm25_scores)
            std_score = np.std(selected_bm25_scores) + 1e-9  # 避免除以0
            
            # 使用Logistic归一化
            normalized_bm25_scores = []
            for score in selected_bm25_scores:
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
            if norm_score >= self.config.bm25_similarity_threshold:  # 使用相似度阈值过滤
                doc = Document(
                    page_content=self.bm25_docs[idx],
                    metadata=self.doc_metadata[idx]
                )
                filtered_bm25_results.append({
                    "doc": doc,
                    "score": norm_score,  # 原始分数
                    "weighted_score": norm_score * bm25_weight,  # 应用权重后的分数
                    "raw_score": norm_score,
                    "type": "bm25",
                    "source": doc.metadata.get("source", "unknown")
                })

        # 合并过滤后的结果
        results = filtered_vector_results + filtered_bm25_results
        
        # 根据加权后的分数进行排序
        results = sorted(results, key=lambda x: x["weighted_score"], reverse=True)
        
        # 文档去重（可能同一文档同时被向量和BM25检索到）
        seen_docs = {}
        unique_results = []
        
        for res in results:
            doc_id = res["source"] + str(hash(res["doc"].page_content[:100]))
            
            if doc_id not in seen_docs:
                # 第一次看到这个文档，直接添加
                unique_results.append(res)
                seen_docs[doc_id] = len(unique_results) - 1
            else:
                # 文档已存在，保留得分更高的版本
                existing_idx = seen_docs[doc_id]
                if res["weighted_score"] > unique_results[existing_idx]["weighted_score"]:
                    # 更新为更高分的版本
                    unique_results[existing_idx] = res

        logger.info(f"📚 混合检索得到{len(unique_results)}篇文档，应用权重 [向量:{vector_weight:.2f}, BM25:{bm25_weight:.2f}]")
        return unique_results
    
    def _determine_retrieval_weights(self, question: str) -> Tuple[float, float]:
        """动态确定检索策略权重，自适应优化不同类型的查询
        
        :param question: 用户问题
        :return: (向量检索权重, BM25检索权重)
        """
        # 默认权重
        default_vector = 0.5
        default_bm25 = 0.5
        
        try:
            # 1. 基础特征词识别
            
            # 事实型问题特征词（偏向BM25）
            factual_indicators = [
                '什么是', '定义', '如何', '怎么', '哪些', '谁', '何时', '为什么', 
                '多少', '数据', '标准是', '要求是', '规定', '条例', '步骤',
                '方法', '操作规程', '限值', '类型', '种类', '分类', '有哪些',
                '列出', '枚举', '标准值', '参数是', '数值', '公式', '计算',
                '如何做', '怎样做', '操作方式', '使用方法', '使用步骤', '如何处理',
                '需要什么', '包括哪些', '组成部分', '执行标准', '法规要求', '技术规范',
                '应该怎么', '需要注意', '注意事项', '检查项目', '保存条件', '储存要求',
                '有效期', '失效日期', '适用范围', '使用范围', '常见问题', '故障原因'
            ]
            
            # 概念型问题特征词（偏向向量检索）
            conceptual_indicators = [
                '解释', '分析', '评价', '比较', '区别', '关系', '影响', '原理', 
                '机制', '思考', '可能', '建议', '预测', '推测', '综合', '总结',
                '联系', '详细描述', '深入', '复杂', '全面', '为什么会', '如何理解',
                '论述', '阐述', '探讨', '研究', '观点', '看法', '理论', '学说',
                '推断', '假设', '猜想', '前景', '趋势', '发展方向', '未来可能',
                '利弊', '优缺点', '合理性', '可行性', '有效性', '科学依据',
                '深层原因', '本质', '实质', '核心问题', '关键因素', '重要性',
                '系统性', '整体性', '结构性', '辩证关系', '互动机制', '协同效应',
                '理论基础', '哲学思考', '创新思路', '突破点', '解决思路'
            ]
            
            # 2. 化工特定领域特征词
            chemical_specific_terms = {
                # 化学反应类（精确匹配重要，BM25优势）
                'bm25_favor': [
                    '反应条件', '反应物', '产物', '催化剂', '化学式', 'ph值', '摩尔比',
                    '温度范围', '压力要求', '反应时间', '产率', '转化率', '选择性',
                    'MSDS', '危险品编号', 'CAS号', '熔点', '沸点', '闪点', '密度',
                    '溶解度', '挥发性', '粘度', '比重', '折射率', '分子量', '分子式',
                    '结构式', '异构体', '同分异构', '晶体结构', '光学活性', '旋光度',
                    '酸值', '碱值', '氧化还原电位', '离解常数', '电导率', '热导率',
                    '比热容', '热膨胀系数', '蒸汽压', '临界温度', '临界压力', '临界体积',
                    '爆炸极限', '自燃点', '引燃温度', '燃烧热', '燃点', '着火点',
                    '禁忌物', '聚合危害', '毒性分级', 'LD50', 'LC50', '致癌性',
                    '腐蚀性', '刺激性', '致敏性', '生物半衰期', '蓄积性', '降解性',
                    '危险货物编号', '联合国编号', '危规号', 'EINECS号', 'RTECS号',
                    # 化工工艺参数
                    '工艺参数', '工艺流程', '单元操作', '装置构成', '设备参数', '管道规格',
                    '阀门类型', '仪表型号', '控制参数', '进料速率', '出料速率', '循环比',
                    '停留时间', '空速', '液位高度', '雷诺数', '普朗特数', '传热系数',
                    '传质系数', '流体阻力', '搅拌功率', '混合度', '分离度', '提纯度',
                    # 化学品安全参数
                    '危险化学品目录', '重大危险源', '临界量', '危险等级', '危害程度',
                    '危害识别码', 'GHS标识', '危险性说明', '预防措施说明', '象形图',
                    '信号词', '毒性级别', '急性毒性', '慢性毒性', '特定靶器官毒性',
                    '水危害等级', '土壤危害', '大气危害', '生物累积性', '持久性',
                    # 化工设备安全
                    '设备安全间距', '防爆等级', '防火等级', '防腐等级', '保护等级',
                    '过压保护', '超温保护', '泄压装置', '安全阀', '爆破片',
                    '紧急切断阀', '阻火器', '防雷设施', '接地装置', '静电消除',
                    '安全联锁', '连锁保护', '双重保险', '安全联锁', '失效保护',
                    # 检测检验
                    '检测方法', '检测标准', '检测周期', '检验项目', '检验标准',
                    '取样点位', '取样方法', '样品保存', '分析方法', '仪器精度',
                    '检测限', '定量限', '测量不确定度', '校准周期', '标样浓度',
                    '标定曲线', '溯源性', '校验周期', '检测周期', '检验报告'
                ],
                # 安全理论类（语义理解重要，向量优势）
                'vector_favor': [
                    '安全管理', '风险评估', '预防措施', '应急预案', '事故分析',
                    '安全文化', '本质安全', '安全系统', '危害识别', '风险控制',
                    '连锁反应', '扩散模型', '临界点', '稳定性', '相容性',
                    '安全生产', '职业健康', '作业环境', '安全责任制', '安全教育',
                    '安全检查', '隐患排查', '危险源辨识', '风险分级', '安全审核',
                    '双重预防', '安全投入', '安全标准化', '安全绩效', '安全目标',
                    '安全技术', '安全评价', '安全防护', '安全监测', '安全信息',
                    '事故调查', '事故责任', '安全改进', '安全承诺', '安全愿景',
                    '安全领导力', '安全参与', '合规管理', '应急响应', '应急处置',
                    '应急救援', '疏散程序', '救援装备', '警戒区域', '安全疏散',
                    '伤员救护', '危险源控制', '泄漏处理', '火灾扑救', '爆炸防护',
                    '事故教训', '经验总结', '改进措施', '系统优化', '过程安全',
                    # 安全管理体系
                    '安全生产法', '法律法规', '国家标准', '行业标准', '企业标准',
                    '安全方针', '安全愿景', '安全战略', '安全规划', '职责划分',
                    'PDCA循环', '持续改进', '闭环管理', '体系审核', '符合性评价',
                    '管理评审', '自我评价', '安全认证', '体系建设', '组织机构',
                    '安全委员会', '安全管理部门', '安全总监', '安全履职', '安全问责',
                    # 新险管理与控制
                    '安全风险', '风险源', '风险矩阵', '风险接受度', '风险决策',
                    '风险沟通', '脆弱性分析', '失效模式', '作业危害分析', '危害与可操作性分析',
                    '故障树分析', '事件树分析', '安全完整性等级', '功能安全', '层次保护',
                    '本质安全设计', '非侵入式安全', '固有安全', '安全裕度', '容错设计',
                    '防误操作', '人因工程', '冗余设计', '多样性设计', '纵深防御',
                    # 应急管理
                    '应急管理体系', '应急预案体系', '应急能力评估', '应急演练', '应急培训',
                    '应急指挥', '应急决策', '应急沟通', '应急协调', '区域联动',
                    '专家组', '应急资源', '应急物资', '警报系统', '报警联动',
                    '情景构建', '情景模拟', '情景应对', '事件升级', '事件降级',
                    '恢复重建', '事后评估', '心理疏导', '社会稳定', '环境修复',
                    # 安全文化
                    '安全氛围', '安全意识', '安全行为', '安全态度', '安全习惯',
                    '安全心理', '安全感知', '安全认知', '安全决策', '安全价值观',
                    '安全信念', '安全动机', '安全承诺', '主动安全', '被动安全',
                    '安全激励', '安全沟通', '安全对话', '安全学习', '标杆管理',
                    '最佳实践', '经验分享', '安全警示', '安全警句', '安全宣传'
                ]
            }
                               
            # 计算各类特征出现次数（权重计数）
            factual_count = sum(1 for term in factual_indicators if term in question)
            conceptual_count = sum(1 for term in conceptual_indicators if term in question)
            
            # 化工特定术语权重（额外加权）
            chemical_bm25_count = sum(1.5 for term in chemical_specific_terms['bm25_favor'] if term in question)
            chemical_vector_count = sum(1.5 for term in chemical_specific_terms['vector_favor'] if term in question)
            
            # 累加领域特征权重
            factual_count += chemical_bm25_count
            conceptual_count += chemical_vector_count
            
            # 3. 数值型查询特征识别（数值查询通常是精确匹配，偏向BM25）
            number_pattern = r'\d+\.?\d*'
            unit_pattern = r'(度|克|千克|吨|升|毫升|ppm|mg|kg|℃|mol|Pa|MPa|atm)'
            
            # 判断是否包含数字+单位组合
            has_numeric_query = bool(re.search(number_pattern + r'.*?' + unit_pattern, question) or 
                                     re.search(unit_pattern + r'.*?' + number_pattern, question))
            
            if has_numeric_query:
                factual_count += 2  # 数值类查询显著增加BM25权重
            
            # 4. 专有名词识别（化学品名称、设备名称等专有名词偏向BM25精确匹配）
            # 简单启发式：连续的非常见词可能是专有名词
            words = self._tokenize(question)
            for i in range(len(words)-1):
                if len(words[i]) >= 2 and len(words[i+1]) >= 2:  # 连续两个长词
                    # 假设这可能是专有名词
                    factual_count += 0.5
            
            # 5. 考虑问题长度和复杂度
            query_length = len(question)
            length_factor = min(1.0, query_length / 50)  # 标准化长度因素
            
            # 句子复杂度（以逗号、句号等标点符号数量为参考）
            punctuation_count = len(re.findall(r'[，。？！；：、]', question))
            complexity_factor = min(1.0, punctuation_count / 3)
            
            # 6. 计算偏向系数
            # 特征词比例，决定基础偏向方向
            feature_bias = 0
            if factual_count > 0 or conceptual_count > 0:
                feature_bias = (factual_count - conceptual_count) / (factual_count + conceptual_count)
                # feature_bias范围为[-1, 1]，正值偏向BM25，负值偏向向量
            
            # 7. 确定最终权重
            if feature_bias > 0.1:  # 明显偏向事实型/BM25
                # 事实型问题：增加BM25权重
                base_bm25 = 0.6 + 0.2 * min(abs(feature_bias), 0.4)  # 最高到0.8
                # 数值查询额外加权
                if has_numeric_query:
                    base_bm25 = min(0.85, base_bm25 + 0.1)
                bm25_weight = base_bm25
                vector_weight = 1.0 - bm25_weight
            elif feature_bias < -0.1:  # 明显偏向概念型/向量
                # 概念型问题：增加向量权重
                base_vector = 0.6 + 0.2 * min(abs(feature_bias), 0.4)  # 最高到0.8
                # 长句和复杂句子加权
                vector_weight = base_vector + 0.1 * (length_factor + complexity_factor) / 2
                vector_weight = min(0.85, vector_weight)  # 限制最大值
                bm25_weight = 1.0 - vector_weight
            else:  # 混合类型（-0.1到0.1之间）
                # 混合类型：保持平衡，微调
                vector_weight = default_vector + 0.1 * length_factor
                bm25_weight = 1.0 - vector_weight
            
            # 8. 结果日志记录（便于调试和改进）
            logger.debug(f"查询权重分析 - 问题: {question[:30]}...")
            logger.debug(f"  • 事实特征得分: {factual_count:.2f}, 概念特征得分: {conceptual_count:.2f}")
            logger.debug(f"  • 偏向系数: {feature_bias:.2f}, 长度因子: {length_factor:.2f}")
            logger.debug(f"  • 最终权重 - 向量: {vector_weight:.2f}, BM25: {bm25_weight:.2f}")
                
            # 确保权重相加为1
            total = vector_weight + bm25_weight
            return vector_weight/total, bm25_weight/total
            
        except Exception as e:
            logger.warning(f"⚠️ 动态权重计算失败: {str(e)}，使用默认权重")
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
                    
                    # 模型推理
                    with torch.no_grad():
                        batch_outputs = self.rerank_model(**batch_inputs)
                        # 使用sigmoid转换分数
                        batch_scores = torch.sigmoid(batch_outputs.logits).squeeze().tolist()
                        
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

    def _retrieve_documents(self, question: str) -> Tuple[List[Document], List[Dict]]:
        """完整检索流程

        :param question: 用户问题
        :return: (文档列表, 分数信息列表)
        """
        try:
            # 混合检索
            raw_results = self._hybrid_retrieve(question)
            if not raw_results:
                logger.warning("混合检索未返回任何结果")
                return [], []

            # 直接重排序
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

            # 根据阈值过滤结果
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

            # 输出最终分数信息
            logger.info(f"📊 最终文档数目:{len(final_results)}篇")

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
        
        # 思考过程指令
        reasoning_instruction = (
            "请按照以下步骤回答问题：\n"
            "1. 仔细阅读并理解提供的参考资料\n"
            "2. 分析问题中的关键信息和要求\n"
            "3. 从参考资料中提取相关信息\n"
            "4. 给出详细的推理过程\n"
            "5. 总结并给出最终答案\n\n"
            "如果参考资料不足以回答问题，请直接说明无法回答。"
        )
        
        if context:
            return (
                "<|im_start|>system\n"
                f"{system_role}\n"
                f"{reasoning_instruction}\n"
                "参考资料：\n{context}\n"
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

    def _build_chat_prompt(self, current_question: str, chat_history: List[Dict], context: str = "") -> str:
        """构建多轮对话的提示词模板"""
        # 系统角色定义
        system_role = (
            "你是一位拥有20年经验的化工安全领域权威专家，精通危险化学品管理、安全生产、工艺安全、应急响应和风险评估。"
            "你掌握国内外化工安全法规标准，熟悉HAZOP、LOPA、JSA等安全分析方法，了解最新的安全技术和管理实践。"
            "你需要基于提供的参考资料和聊天历史，给出准确、专业且易于理解的回答。"
            "你始终坚持'安全第一'原则，在回答中优先考虑人员安全和环境保护。"
            "你的回答应保持连贯性和一致性，考虑之前的对话内容，并针对化工安全领域特定情境提供实用建议。"
            
            "\n在回答涉及法规标准时，请明确引用相关依据。"
            "\n在回答风险评估问题时，请运用系统性思维，考虑多种危害因素。"
            "\n在回答工艺安全问题时，请结合化学反应机理和工程控制措施。"
            "\n在回答设备安全问题时，请结合材料科学和机械完整性原则。"
            "\n在回答应急响应问题时，请提供清晰的程序步骤和注意事项。"
            
            "\n请确保用易于理解的方式表达专业内容，避免过度使用术语而不解释。"
            "\n如遇紧急情况类问题，请强调立即采取行动的重要性并提供具体指导。"
        )
        
        # 回答步骤指导
        reasoning_steps = (
            "请按照以下步骤回答问题：\n"
            "1. 仔细分析问题的关键点和化工安全领域背景\n"
            "2. 全面审视提供的参考资料，找出相关信息\n"
            "3. 根据化工安全领域专业知识评估信息的适用性\n"
            "4. 从多角度(工艺、设备、人员、管理)考虑问题\n"
            "5. 构建清晰的技术分析和推理过程\n"
            "6. 确保回答实用、可操作且符合安全规范\n"
            "7. 总结核心观点并给出明确建议\n\n"
            "如果参考资料不足以回答问题，请明确指出信息的局限性，并基于化工安全原理提供一般性指导。"
            "如果问题涉及紧急危险情况，优先强调人员安全和应急措施。"
        )
        
        # 构建系统提示部分
        prompt = "<|im_start|>system\n" + system_role + "\n\n" + reasoning_steps + "\n"
        
        # 添加参考资料（如果有）
        if context:
            prompt += "\n参考资料：\n" + context[:self.config.max_context_length] + "\n"
        
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
                "data": "⚠️ 请输入有效问题"
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
                    "data": "⚠️ 文档检索服务暂时不可用，将使用聊天历史回答..."
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
                    "data": "\n⚠️ 生成过程发生意外中断，请刷新页面重试"
                }) + "\n"
                
        except Exception as e:
            logger.exception(f"多轮对话处理错误: {str(e)}")
            yield json.dumps({
                "type": "error",
                "data": "⚠️ 系统处理请求时发生严重错误，请联系管理员"
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

