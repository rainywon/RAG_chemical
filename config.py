import torch
from pathlib import Path
from dataclasses import dataclass
from transformers import BitsAndBytesConfig


class Config:
    """RAG系统全局配置类，包含路径、模型参数、硬件设置等配置项"""

    def __init__(self, data_dir: str = "data"):
        # ████████ 路径配置 ████████
        self.data_dir = Path(data_dir)  # 数据存储根目录（自动转换为Path对象）
        #self.embedding_model_path = r"C:\Users\coins\Desktop\models\bge-large-zh-v1.5"  # 文本嵌入模型存储路径
        #self.vector_db_path = "vector_store/data"  # FAISS向量数据库存储目录
        #self.rerank_model_path = r"C:\Users\coins\Desktop\models\bge-reranker-large"  # 重排序模型路径
        self.embedding_model_path = r"C:\wu\models\bge-large-zh-v1.5"  # 文本嵌入模型存储路径
        self.vector_db_path = r"C:\wu\RAG\vector_store\data"  # FAISS向量数据库存储目录
        self.rerank_model_path = r"C:\wu\models\bge-reranker-large"  # 重排序模型路径
        self.excel_dir = r"C:\Users\coinrainy\Desktop\毕设\data\chunks"  # Excel文件目录

        self.cache_dir = "cache"  # 缓存目录
        self.max_backups = 5  # 保留的最大备份数量

        # ████████ 内容管理路径配置 ████████
        # 知识库、安全资料库和应急预案文件路径
        self.knowledge_base_path = r"C:\Users\coinrainy\Desktop\毕设\data\chunks"  # 知识库文件目录，存放Excel文件
        self.safety_document_path = r"C:\Users\coinrainy\Desktop\毕设\data\安全资料库"  # 安全资料库目录，存放PDF、Word文档
        self.emergency_plan_path = r"C:\Users\coinrainy\Desktop\毕设\data\案例库"  # 应急预案目录，存放PDF、Word文档

        # self.knowledge_base_path = r"C:\wu\RAG\data\chunks"  # 知识库文件目录，存放Excel文件
        # self.safety_document_path = r"C:\wu\RAG\data\safey_document"  # 安全资料库目录，存放PDF、Word文档
        # self.emergency_plan_path = r"C:\wu\RAG\data\emergency_document"  # 应急预案目录，存放PDF、Word文档
        # ████████ 硬件配置 ████████
        self.cuda_lazy_init = True  # 延迟CUDA初始化（避免显存立即被占用）
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # 自动检测设备

        # ████████ PDF OCR配置 ████████
        # PDF OCR参数说明:
        # - batch_size: 批处理大小，决定一次处理多少页。值越大速度越快但显存占用越多
        # - min_pages_for_batch: 启用批处理的最小页数，低于此值将逐页处理
        # - det_limit_side_len: 检测分辨率，影响识别精度和速度，值越小速度越快
        # - rec_batch_num: 识别批处理量，值越大速度越快但显存占用越多
        # - det_batch_num: 检测批处理量，值越大速度越快但显存占用越多
        # - use_tensorrt: 是否使用TensorRT加速，需要安装对应版本的TensorRT
        self.pdf_ocr_params = {
            'batch_size': 2,              # 批处理大小
            'min_pages_for_batch': 2,     # 启用批处理的最小页数
            'det_limit_side_len': 640,    # 检测分辨率
            'rec_batch_num': 4,           # 识别批处理量
            'det_batch_num': 2,           # 检测批处理量
            'use_tensorrt': False         # 是否使用TensorRT加速
        }
        
        # 针对1050Ti特别优化的参数（如果需要可以使用）
        self.pdf_ocr_1050ti_params = {
            'batch_size': 2,              # 批处理大小
            'min_pages_for_batch': 2,     # 启用批处理的最小页数
            'det_limit_side_len': 640,    # 检测分辨率
            'rec_batch_num': 4,           # 识别批处理量
            'det_batch_num': 2,           # 检测批处理量
            'use_tensorrt': False         # 1050Ti通常不支持高版本TensorRT
        }
        
        # 针对大文档的优化参数
        self.pdf_ocr_large_doc_params = {
            'batch_size': 1,              # 单页批处理以节省内存
            'min_pages_for_batch': 5,     # 仍然启用批处理，但每批仅处理一页
            'det_limit_side_len': 640,    # 降低检测分辨率
            'rec_batch_num': 2,           # 降低识别批处理量
            'det_batch_num': 1            # 降低检测批处理量
        }

        # ████████ 批处理配置 ████████
        self.batch_size = 32 if torch.cuda.is_available() else 8  # 根据GPU显存自动调整批次大小
        self.normalize_embeddings = True  # 是否对嵌入向量做L2归一化处理

        # ████████ 文本分块配置 ████████
        self.chunk_size = 512  # 文档分块长度（字符数）
        self.chunk_overlap = 128  # 分块重叠区域长度（增强上下文连续性）

        # ████████ Ollama大模型配置 ████████
        self.ollama_base_url = "http://localhost:11434"  # Ollama服务地址
        self.llm_max_tokens = 16384  # 生成文本的最大token数限制
        self.llm_temperature = 0.1  # 温度参数（0-1，控制生成随机性）
        
        # ████████ VLLM大模型配置 ████████
        self.vllm_model_path = r"C:\wu\models\Qwen-7B-Chat"  # VLLM模型路径
        self.vllm_tensor_parallel_size = 1  # 张量并行大小，根据GPU数量设置
        self.vllm_gpu_memory_utilization = 0.9  # GPU显存使用率
        self.vllm_swap_space = 4  # 交换空间大小，单位为GB
        
        # ████████ RAG检索配置 ████████
        self.max_context_length = 12000  # 输入LLM的上下文最大长度（避免过长导致性能下降）
        self.bm25_top_k = 30  # BM25检索返回的候选文档数
        self.vector_top_k = 30  # 向量检索返回的候选文档数
        self.vector_similarity_threshold = 0.6  # 向量检索的相似度阈值
        self.bm25_similarity_threshold = 0.5  # BM25检索的相似度阈值
        self.similarity_threshold = 0.5  # 相似度过滤阈值（低于此值的文档被丢弃）
        self.final_top_k = 10  # 最终返回给大模型的最相关文档数量

        # ████████ 详细输出配置 ████████
        self.print_detailed_chunks = False  # 启用详细输出
        self.max_chunk_preview_length = 512  # 设置最大预览长度


# ████████ 短信服务配置 ████████
URL = "https://dfsns.market.alicloudapi.com/data/send_sms"  # 阿里云短信API端点
APPCODE = 'f9b3648618f849409d2bdd5c0f07f67a'  # 用户身份验证码（需替换为实际值）
APPKEY="204805252"
APPSECRET="dRQ1HZmsKLlBteooUBSrF7ij6CH9xaoh"
SMS_SIGN_ID = "90362f6500af46bb9dadd26ac6e31e11"  # 短信签名ID（控制台获取）
TEMPLATE_ID = "CST_ptdie100"  # 短信模板ID（对应具体短信内容格式）
SERVER_URL = 'http://localhost:8000'  # 后端服务地址（生产环境需改为公网域名）