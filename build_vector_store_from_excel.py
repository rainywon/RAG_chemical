import os
import pandas as pd
from pathlib import Path
import logging
import sys
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from config import Config
import shutil
from datetime import datetime
import jieba
from typing import List, Dict, Any
import hashlib

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

class ExcelVectorDBBuilder:
    def __init__(self, config: Config):
        """
        初始化Excel向量数据库构建器
        Args:
            config (Config): 配置类，包含必要的配置
        """
        self.config = config
        
        # 设置向量数据库路径
        self.vector_dir = Path(config.vector_db_path)
        self.vector_backup_dir = self.vector_dir / "backups"
        
        # 设置Excel文件目录
        self.excel_dir = Path(self.config.excel_dir)
        
        # 确保目录存在
        self.vector_dir.mkdir(parents=True, exist_ok=True)
        self.excel_dir.mkdir(parents=True, exist_ok=True)

    def _print_chunk_preview(self, content: str, source: str, index: int, total: int):
        """打印文本块预览
        Args:
            content: 文本内容
            source: 来源文件名
            index: 当前块索引
            total: 总块数
        """

            
        logger.info("\n" + "="*50)
        logger.info(f"📄 文件: {source}")
        logger.info(f"📑 块 {index + 1}/{total}")
        logger.info("-"*50)
        logger.info(f"内容预览 ({len(content)} 字符):")
        logger.info(content)
        logger.info("="*50 + "\n")

    def load_chunks_from_excel(self) -> List[Document]:
        """从Excel文件中加载文本块"""
        logger.info("开始从Excel文件加载文本块...")
        
        all_chunks = []
        excel_files = list(self.excel_dir.glob("*.xlsx"))
        
        if not excel_files:
            logger.warning(f"在 {self.excel_dir} 目录下未找到Excel文件")
            return []
            
        for excel_file in excel_files:
            try:
                # 读取Excel文件
                df = pd.read_excel(excel_file)
                
                # 检查是否包含必要的列
                if "入库内容" not in df.columns:
                    logger.warning(f"Excel文件 {excel_file.name} 中缺少'入库内容'列")
                    continue
                
                # 获取文件名作为来源
                source = excel_file.name
                logger.info(f"\n📂 正在处理文件: {source}")
                
                # 处理每个文本块
                for idx, row in df.iterrows():
                    content = str(row["入库内容"]).strip()
                    if not content:  # 跳过空内容
                        continue
                        
                    # 对内容进行分词处理
                    tokenized_content = self._tokenize(content)
                    
                    # 创建文档对象
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(excel_file),
                            "file_name": source,
                            "chunk_index": idx,
                            "total_chunks": len(df),
                            "tokens": tokenized_content,
                            "token_count": len(tokenized_content),
                            "content_hash": hashlib.md5(content.encode()).hexdigest(),
                            "processed_at": datetime.now().isoformat()
                        }
                    )
                    all_chunks.append(doc)
                    
                    # 打印文本块预览
                    self._print_chunk_preview(content, source, idx, len(df))
                    
                logger.info(f"✅ 从 {source} 成功加载 {len(df)} 个文本块")
                
            except Exception as e:
                logger.error(f"处理Excel文件 {excel_file.name} 时出错: {str(e)}")
                continue
                
        logger.info(f"✅ 总共加载了 {len(all_chunks)} 个文本块")
        return all_chunks

    def _tokenize(self, text: str) -> List[str]:
        """专业中文分词处理
        :param text: 待分词的文本
        :return: 分词后的词项列表
        """
        return [word for word in jieba.cut(text) if word.strip()]

    def backup_vector_db(self):
        """备份现有向量数据库"""
        if not self.vector_dir.exists():
            return False
            
        try:
            # 创建备份目录
            backup_dir = self.vector_dir.parent / f"{self.vector_dir.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制所有文件到备份目录
            for item in self.vector_dir.glob('*'):
                if item.is_file():
                    shutil.copy2(item, backup_dir)
                elif item.is_dir():
                    shutil.copytree(item, backup_dir / item.name)
                    
            logger.info(f"✅ 向量数据库已备份至 {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"备份向量数据库失败: {str(e)}")
            return False

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """创建嵌入模型实例"""
        logger.info("初始化嵌入模型...")
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_path,
            model_kwargs={"device": self.config.device},
            encode_kwargs={
                "batch_size": self.config.batch_size,
                "normalize_embeddings": self.config.normalize_embeddings
            },
        )

    def build_vector_store(self):
        """构建向量数据库"""
        logger.info("开始构建向量数据库")
        
        # 加载文本块
        chunks = self.load_chunks_from_excel()
        
        if not chunks:
            logger.warning("没有文本块可以处理，跳过向量存储构建")
            return
            
        # 如果向量数据库已存在，先备份
        if self.vector_dir.exists() and any(self.vector_dir.glob('*')):
            self.backup_vector_db()
            
        # 生成嵌入模型
        embeddings = self.create_embeddings()
        
        # 构建向量存储
        logger.info("生成向量...")
        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE
        )
        
        # 保存向量数据库
        vector_store.save_local(str(self.vector_dir))
        logger.info(f"✅ 向量数据库已保存至 {self.vector_dir}")

if __name__ == "__main__":
    try:
        # 初始化配置
        config = Config()
        
        # 构建向量数据库
        builder = ExcelVectorDBBuilder(config)
        builder.build_vector_store()
        
    except Exception as e:
        logger.exception("程序运行出错")
    finally:
        logger.info("程序运行结束") 