import hashlib
import sys
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 导入文档分割工具
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings  # 导入HuggingFace嵌入模型
from langchain_community.vectorstores import FAISS  # 导入FAISS用于构建向量数据库
from langchain_community.document_loaders import UnstructuredPDFLoader  # 新增导入
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
import os
import json
from pathlib import Path  # 导入Path，用于路径处理
from datetime import datetime  # 导入datetime，用于记录时间戳
from typing import List, Dict, Optional, Set, Tuple  # 导入类型提示
import logging  # 导入日志模块，用于记录运行日志
from concurrent.futures import ThreadPoolExecutor, as_completed  # 导入线程池模块，支持并行加载PDF文件
from tqdm import tqdm  # 导入进度条模块，用于显示加载进度
from config import Config  # 导入配置类，用于加载配置参数
import shutil  # 用于文件操作

from pdf_cor_extractor.pdf_ocr_extractor import PDFProcessor


# 配置日志格式
# 配置日志格式，指定输出到stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(stream=sys.stdout)],  # 明确输出到stdout
    force=True  # 关键：强制覆盖现有配置
)
logger = logging.getLogger(__name__)


class VectorDBBuilder:
    def __init__(self, config: Config):
        """
        初始化向量数据库构建器
        Args:
            config (Config): 配置类，包含必要的配置
        """
        self.config = config
        
        # 设置缓存目录路径
        self.cache_dir = Path(config.cache_dir)
        
        # 设置向量数据库路径
        self.vector_dir = Path(config.vector_db_path)
        self.vector_backup_dir = self.vector_dir / "backups"
        
        # 文本块缓存路径
        self.chunk_cache_path = self.cache_dir / "chunks_cache.json"
        
        # 添加处理状态文件路径定义
        self.state_file = self.cache_dir / "processing_state.json"
        
        # 将源文件目录定义放在初始化方法中
        self.subfolders = ['标准']  # '标准性文件','法律', '规范性文件'
        
        # 检查文件匹配模式
        if not hasattr(config, 'files') or not config.files:
            # 如果config中没有files参数，使用默认值
            self.config.files = ["data/**/*.pdf", "data/**/*.txt", "data/**/*.md", "data/**/*.docx"]
        
        # 添加GPU使用配置
        self.use_gpu_for_ocr = "cuda" in self.config.device
        
        # 已处理文件状态
        self.processed_files = {}
        self.failed_files_count = 0
        self.need_rebuild_index = False
        
        # 是否输出详细的分块内容
        self.print_detailed_chunks = getattr(config, 'print_detailed_chunks', False)
        # 详细输出时每个文本块显示的最大字符数
        self.max_chunk_preview_length = getattr(config, 'max_chunk_preview_length', 200)

    def _load_processing_state(self) -> Dict:
        """加载之前的文件处理状态"""
        if self.state_file.exists():  # 如果状态文件存在，则加载
            with open(self.state_file, "r", encoding='utf-8') as f:
                return json.load(f)  # 返回加载的状态文件内容
        return {}  # 如果文件不存在，返回空字典

    def _save_processing_state(self):
        """保存当前文件处理状态"""
        try:
            # 确保目录存在
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存状态
            with open(self.state_file, "w", encoding='utf-8') as f:
                json.dump(self.processed_files, f, indent=2, ensure_ascii=False)
                
            logger.debug(f"✅ 已保存处理状态，共 {len(self.processed_files)} 个文件记录")
        except Exception as e:
            logger.warning(f"⚠️ 保存处理状态失败: {str(e)}")
            # 这里不抛出异常，避免中断主流程

    def _should_process(self, file_path: Path) -> bool:
        """判断文件是否需要处理"""
        file_stat = file_path.stat()  # 获取文件状态
        file_info = {
            "path": str(file_path),  # 文件路径
            "size": file_stat.st_size,  # 文件大小
            "mtime": file_stat.st_mtime  # 文件修改时间
        }

        existing = self.processed_files.get(str(file_path))  # 查找已处理的文件记录
        if not existing:  # 如果文件未处理过，则需要处理
            return True
        return not (existing["size"] == file_info["size"]
                    and existing["mtime"] == file_info["mtime"])  # 如果文件修改时间或大小不一致，则需要重新处理

    def _check_file_changes(self, current_files: List[Path]) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        检查文件变化，返回需要添加、更新和删除的文件
        
        Args:
            current_files: 当前文件系统中的文件列表
            
        Returns:
            Tuple[Set[str], Set[str], Set[str]]: 新增文件、更新文件和删除文件的路径集合
        """
        # 获取当前文件系统中的所有文件路径
        current_file_paths = {str(f) for f in current_files}
        
        # 获取已处理文件的路径
        processed_file_paths = set(self.processed_files.keys())
        
        # 需要新增的文件（当前存在但未处理过）
        new_files = current_file_paths - processed_file_paths
        
        # 需要更新的文件（已处理过但需要重新处理）
        update_files = {str(f) for f in current_files if self._should_process(f)}
        
        # 需要删除的文件（已处理过但当前不存在）
        deleted_files = processed_file_paths - current_file_paths
        
        return new_files, update_files, deleted_files

    def _load_single_document(self, file_path: Path) -> Optional[List[Document]]:
        """多线程加载单个文档文件（支持 PDF、DOCX、DOC）"""
        try:
            if not self._should_process(file_path):
                logger.info(f"[文档加载] 跳过未修改文件: {file_path.name}")
                return None

            file_extension = file_path.suffix.lower()
            docs = []

            if file_extension == ".pdf":
                try:
                    # 检查PDF页数
                    import fitz
                    with fitz.open(str(file_path)) as doc:
                        page_count = doc.page_count
                        logger.info(f"[文档加载] PDF文件 '{file_path.name}' 共有 {page_count} 页")
                        
                    # 使用配置中的参数初始化处理器
                    processor = PDFProcessor(
                        file_path=str(file_path), 
                        lang='ch', 
                        use_gpu=self.use_gpu_for_ocr
                    )
                    
                    # 根据页数选择合适的GPU参数配置
                    if page_count > 30:
                        logger.info(f"[文档加载] PDF页数较多({page_count}页)，应用大文档优化配置")
                        processor.configure_gpu(**self.config.pdf_ocr_large_doc_params)
                    else:
                        # 使用标准参数配置
                        processor.configure_gpu(**self.config.pdf_ocr_params)
                    
                    # 处理PDF
                    docs = processor.process()
                    
                    # 检查处理结果
                    if docs and len(docs) < page_count * 0.5:
                        logger.warning(f"[文档加载] 警告: 只识别出 {len(docs)}/{page_count} 页，低于50%，可能有问题")
                    elif docs:
                        logger.info(f"[文档加载] 成功识别 {len(docs)}/{page_count} 页")
                    
                    # 处理后清理内存
                    import gc
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                        
                except Exception as e:
                    logger.error(f"[文档加载] 处理PDF文件 '{file_path.name}' 失败: {str(e)}")
                    self.failed_files_count += 1
                    return None
                    
            elif file_extension in [".docx", ".doc"]:
                try:
                    # 首先尝试导入依赖模块
                    try:
                        import docx2txt
                    except ImportError:
                        logger.error(f"缺少处理Word文档所需的依赖包，请运行: pip install docx2txt")
                        # 记录错误但继续执行，以便处理其他文件类型
                        self.failed_files_count += 1
                        return None
                        
                    from langchain_community.document_loaders import Docx2txtLoader
                    loader = Docx2txtLoader(str(file_path))
                    docs = loader.load()
                except Exception as e:
                    logger.error(f"[文档加载] 处理DOCX文件 '{file_path.name}' 失败: {str(e)}")
                    
                    # 尝试使用替代方法
                    try:
                        logger.info(f"[文档加载] 尝试使用替代方法加载Word文档...")
                        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
                        loader = UnstructuredWordDocumentLoader(str(file_path))
                        docs = loader.load()
                        logger.info(f"[文档加载] 成功使用替代方法加载Word文档: {file_path.name}")
                    except Exception as e2:
                        logger.error(f"[文档加载] 替代方法也失败: {str(e2)}")
                        self.failed_files_count += 1
                        return None
            else:
                logger.warning(f"[文档加载] 不支持的文件格式: {file_path.name}")
                return None

            if docs:
                # 记录文件处理信息
                self.processed_files[str(file_path)] = {
                    "size": file_path.stat().st_size,
                    "mtime": file_path.stat().st_mtime,
                    "processed_at": datetime.now().isoformat(),
                    "pages": len(docs),
                    "file_name": file_path.name  # 添加文件名，便于后续查找
                }
                # 统一添加元数据
                for doc in docs:
                    doc.metadata["source"] = str(file_path)
                    doc.metadata["file_name"] = file_path.name
                return docs
            return None

        except Exception as e:
            logger.error(f"[文档加载] 加载 {file_path} 失败: {str(e)}")
            self.failed_files_count += 1
            return None

    def _cleanup_deleted_files(self):
        """
        清理已删除文件的缓存信息
        """
        if not self.chunk_cache_path.exists():
            return
            
        try:
            # 获取当前所有文件路径
            current_files = []
            # 使用self.subfolders代替硬编码的子文件夹列表
            for subfolder in self.subfolders:
                folder_path = self.config.data_dir / subfolder
                if folder_path.exists() and folder_path.is_dir():
                    current_files.extend([f for f in folder_path.rglob("*") 
                                       if f.suffix.lower() in ['.pdf', '.docx', '.doc']])
                    
            # 分析文件变化
            new_files, update_files, deleted_files = self._check_file_changes(current_files)
            
            if not deleted_files and not update_files:
                logger.info("没有检测到文件变化，跳过清理")
                return
                
            # 如果有文件被删除或更新，需要清理缓存
            with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # 清理已删除文件和更新文件的数据
            files_to_clean = deleted_files.union(update_files)
            if files_to_clean:
                logger.info(f"检测到 {len(deleted_files)} 个已删除文件, {len(update_files)} 个已更新文件")
                
                # 从处理状态中删除已删除的文件
                for file_path in deleted_files:
                    if file_path in self.processed_files:
                        del self.processed_files[file_path]
                        logger.info(f"从处理状态中移除已删除文件: {Path(file_path).name}")
                
                # 从缓存中过滤掉已删除和已更新的文件块
                clean_chunks = []
                removed_count = 0
                
                for chunk in cache_data.get("chunks", []):
                    chunk_source = chunk.get("metadata", {}).get("source", "")
                    if chunk_source in files_to_clean:
                        removed_count += 1
                        continue
                    clean_chunks.append(chunk)
                
                # 更新缓存数据
                if removed_count > 0:
                    cache_data["chunks"] = clean_chunks
                    cache_data["file_state"] = self.processed_files
                    with open(self.chunk_cache_path, 'w', encoding='utf-8') as f:
                        json.dump(cache_data, f, ensure_ascii=False, indent=2)
                    logger.info(f"✅ 已从缓存中移除 {removed_count} 个过时的文本块")
                    self.need_rebuild_index = True
                    
                    # 保存更新后的处理状态
                    self._save_processing_state()
                    
        except Exception as e:
            logger.error(f"清理缓存失败: {str(e)}")
            # 遇到错误，为安全起见标记需要重建索引
            self.need_rebuild_index = True

    def load_documents(self) -> List:
        """并行加载所有文档"""
        logger.info("⌛ 开始加载文档...")
        
        # 首先清理已删除文件的缓存
        self._cleanup_deleted_files()

        # 获取 data_dir 下的所有子文件夹
        # 使用self.subfolders代替硬编码的子文件夹列表
        document_files = []

        # 遍历每个子文件夹，获取其中的文档文件
        for subfolder in self.subfolders:
            folder_path = self.config.data_dir / subfolder
            if folder_path.exists() and folder_path.is_dir():
                document_files.extend([f for f in folder_path.rglob("*") 
                                    if f.suffix.lower() in ['.pdf', '.docx', '.doc']])
            else:
                logger.warning(f"子文件夹 {subfolder} 不存在或不是目录: {folder_path}")
                
        # 过滤并排序文件（先处理较小的文件，避免大文件占用显存）
        document_files = sorted(document_files, key=lambda x: x.stat().st_size)
        logger.info(f"发现 {len(document_files)} 个待处理文件")
        
        # 过滤出需要处理的文件
        files_to_process = [f for f in document_files if self._should_process(f)]
        logger.info(f"其中 {len(files_to_process)} 个文件需要处理")
        
        if not files_to_process:
            logger.info("没有新文件需要处理，使用现有缓存")
            # 从缓存加载文档
            return self._load_documents_from_cache()

        # 限制线程池大小以避免资源争用
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = [executor.submit(self._load_single_document, file) for file in files_to_process]
            results = []
            with tqdm(total=len(futures), desc="加载文档", unit="files") as pbar:
                for future in as_completed(futures):
                    res = future.result()
                    if res:
                        results.extend(res)
                        pbar.update(1)
                        pbar.set_postfix_str(f"已加载 {len(res)} 页")
                    else:
                        pbar.update(1)
        
        # 在处理完成后清理GPU缓存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
            
        logger.info(f"✅ 成功加载 {len(results)} 页文档")
        logger.info(f"❌ 未成功加载 {self.failed_files_count} 个文件")
        self._save_processing_state()
        
        # 处理文件变化后，将新处理的文档与现有缓存合并
        if results and self.chunk_cache_path.exists():
            self._merge_with_cache(results)
            # 设置标记，表示需要重建索引
            self.need_rebuild_index = True
            return self._load_documents_from_cache()
            
        return results

    def _load_documents_from_cache(self) -> List[Document]:
        """从缓存加载文档"""
        try:
            if self.chunk_cache_path.exists():
                with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    chunks = [
                        Document(
                            page_content=chunk["content"],
                            metadata=chunk["metadata"]
                        )
                        for chunk in cache_data.get("chunks", [])
                    ]
                    logger.info(f"从缓存加载了 {len(chunks)} 个文本块")
                    return chunks
        except Exception as e:
            logger.error(f"加载缓存失败: {str(e)}")
        
        return []

    def _merge_with_cache(self, new_docs: List[Document]):
        """将新处理的文档与现有缓存合并"""
        try:
            # 加载现有缓存
            if not self.chunk_cache_path.exists():
                logger.info("缓存不存在，跳过合并")
                return
                
            with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                
            # 使用与process_files相同的分块逻辑处理新文档
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
                length_function=len,
                add_start_index=True,
                is_separator_regex=False
            )
            
            # 处理新文档
            new_chunks = []
            for doc in new_docs:
                metadata = doc.metadata.copy()
                split_texts = text_splitter.split_text(doc.page_content)
                for text in split_texts:
                    # 生成内容哈希
                    content_hash = hashlib.md5(text.encode()).hexdigest()
                    enhanced_metadata = metadata.copy()
                    enhanced_metadata["content_hash"] = content_hash
                    new_chunks.append({
                        "content": text,
                        "metadata": enhanced_metadata
                    })
                    
            # 合并缓存
            existing_chunks = cache_data.get("chunks", [])
            
            # 从现有缓存中过滤掉与新文档相同来源的块
            filtered_chunks = []
            new_sources = {doc.metadata.get("source") for doc in new_docs}
            for chunk in existing_chunks:
                chunk_source = chunk.get("metadata", {}).get("source", "")
                if chunk_source not in new_sources:
                    filtered_chunks.append(chunk)
                    
            # 合并过滤后的现有块和新块
            merged_chunks = filtered_chunks + new_chunks
            
            # 更新缓存
            cache_data["chunks"] = merged_chunks
            cache_data["file_state"] = self.processed_files
            
            with open(self.chunk_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
                
            logger.info(f"✅ 已将 {len(new_chunks)} 个新块合并到缓存中，总计 {len(merged_chunks)} 个块")
            
        except Exception as e:
            logger.error(f"合并缓存失败: {str(e)}")

    def process_files(self) -> List:
        """优化的文件处理流程"""
        logger.info("开始文件处理流程")
        
        # 首先检查是否有现有缓存可用
        if self.chunk_cache_path.exists() and not self.need_rebuild_index:
            try:
                with open(self.chunk_cache_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    
                    # 比较缓存中的文件状态与当前处理状态
                    if cache_data.get("file_state", {}) == self.processed_files:
                        chunks = [
                            Document(
                                page_content=chunk["content"],
                                metadata=chunk["metadata"]
                            )
                            for chunk in cache_data.get("chunks", [])
                        ]
                        logger.info(f"✅ 从缓存加载 {len(chunks)} 个分块")
                        
                        # 打印分块结果概览
                        self._print_chunks_summary(chunks)
                        
                        return chunks
            except Exception as e:
                logger.error(f"缓存加载失败: {str(e)}")

        # 如果没有合适的缓存，处理所有文档
        all_docs = self.load_documents()

        if not all_docs:
            logger.warning("没有可处理的文件内容")
            return []

        # 只有在没有从load_documents中使用缓存时才需要进行分块处理
        if not self.need_rebuild_index:
            # 首先按文件合并页面内容，避免跨页分块断裂
            logger.info("合并文件页面内容，准备进行整体分块...")
            
            # 按文件分组整理文档
            file_docs = {}
            for doc in all_docs:
                source = doc.metadata.get("source", "")
                if source not in file_docs:
                    file_docs[source] = []
                file_docs[source].append(doc)
            
            # 对每个文件的页面进行排序和合并
            whole_docs = []
            for source, docs in file_docs.items():
                # 按页码排序
                sorted_docs = sorted(docs, key=lambda x: x.metadata.get("page", 0))
                
                # 合并文件所有页面的内容
                full_content = "\n\n".join([doc.page_content for doc in sorted_docs])
                
                # 创建完整文档对象
                file_doc = Document(
                    page_content=full_content,
                    metadata={
                        "source": source,
                        "file_name": sorted_docs[0].metadata.get("file_name", ""),
                        "page_count": len(sorted_docs),
                        "is_merged_doc": True  # 标记为合并后的完整文档
                    }
                )
                whole_docs.append(file_doc)
                
            logger.info(f"已将 {len(all_docs)} 页内容合并为 {len(whole_docs)} 个完整文档")
            
            # 对合并后的完整文档进行分块
            # 优化后的文本分割配置
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,  # 根据中文平均长度调整
                chunk_overlap=self.config.chunk_overlap,  # 适当增加重叠量
                separators=[
                    "\n\n",  # 优先按空行分割
                    "\n",  # 其次按换行符分割
                    "。", "；",  # 中文句末标点
                    "！", "？",  # 中文感叹号和问号
                    "，",  # 中文逗号（谨慎使用）
                    " ",  # 空格
                    ""  # 最后按字符分割
                ],
                length_function=len,
                add_start_index=True,
                is_separator_regex=False  # 明确使用字面分隔符
            )

            logger.info("开始智能分块处理...")
            chunks = []
            # 新增哈希生成和元数据增强
            with tqdm(total=len(whole_docs), desc="处理文档分块") as pbar:
                for doc in whole_docs:
                    metadata = doc.metadata.copy()
                    # 移除分块后不再适用的元数据
                    if "is_merged_doc" in metadata:
                        del metadata["is_merged_doc"]
                    
                    # 对完整文档进行分块
                    split_texts = text_splitter.split_text(doc.page_content)
                    
                    # 处理每个文本块
                    for i, text in enumerate(split_texts):
                        # 应用智能边界处理，确保完整句子
                        text = self._ensure_complete_sentences(text)
                        if not text.strip():  # 跳过空文本块
                            continue
                            
                        # 生成内容哈希
                        content_hash = hashlib.md5(text.encode()).hexdigest()
                        enhanced_metadata = metadata.copy()
                        enhanced_metadata["content_hash"] = content_hash
                        enhanced_metadata["chunk_index"] = i
                        enhanced_metadata["total_chunks"] = len(split_texts)
                        
                        # 添加块位置标记
                        if i == 0:
                            enhanced_metadata["position"] = "document_start"
                        elif i == len(split_texts) - 1:
                            enhanced_metadata["position"] = "document_end"
                        else:
                            enhanced_metadata["position"] = "document_middle"

                        chunks.append(Document(
                            page_content=text,
                            metadata=enhanced_metadata
                        ))
                    pbar.update(1)
            
            # 应用后处理，确保文本块的完整性和连贯性
            chunks = self._post_process_chunks(chunks)
            
            logger.info(f"生成 {len(chunks)} 个语义连贯的文本块")
            
            # 打印分块结果概览
            self._print_chunks_summary(chunks)

            # 保存处理结果到缓存
            cache_data = {
                "file_state": self.processed_files,
                "chunks": [{
                    "content": chunk.page_content,
                    "metadata": chunk.metadata
                } for chunk in chunks]
            }

            with open(self.chunk_cache_path, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)

            logger.info(f"✅ 分块缓存已保存至 {self.chunk_cache_path}")
            return chunks
        else:
            # 如果已经通过load_documents加载并更新了缓存，直接返回文档
            return all_docs
            
    def _ensure_complete_sentences(self, text: str) -> str:
        """确保文本块以完整句子开始和结束
        
        Args:
            text: 原始文本块
            
        Returns:
            处理后的文本块，确保以完整句子开始和结束
        """
        if not text or len(text) < 10:  # 文本过短则直接返回
            return text
            
        # 中文句子结束标记
        sentence_end_marks = ['。', '！', '？', '；', '\n']
        # 句子开始的可能标记（中文段落开头、章节标题等）
        sentence_start_patterns = ['\n', '第.{1,3}章', '第.{1,3}节']
        
        # 处理文本块开头
        text_stripped = text.lstrip()
        # 如果不是以句末标点开头，也不是以大写字母或数字开头（可能是新段落），则可能是不完整句子
        is_incomplete_start = True
        
        # 检查是否以完整句子或段落开始的标记
        for pattern in sentence_start_patterns:
            if text.startswith(pattern) or text_stripped[0].isupper() or text_stripped[0].isdigit():
                is_incomplete_start = False
                break
        
        if is_incomplete_start:
            # 查找第一个完整句子的开始
            for mark in sentence_end_marks:
                pos = text.find(mark)
                if pos > 0:
                    # 找到句末标记后的内容作为起点
                    try:
                        # 确保句末标记后还有内容
                        if pos + 1 < len(text):
                            text = text[pos+1:].lstrip()
                            break
                    except:
                        # 出错则保持原样
                        pass
        
        # 处理文本块结尾
        is_incomplete_end = True
        # 检查是否以完整句子结束
        for mark in sentence_end_marks:
            if text.endswith(mark):
                is_incomplete_end = False
                break
        
        if is_incomplete_end:
            # 找最后一个完整句子的结束位置
            last_pos = -1
            for mark in sentence_end_marks:
                pos = text.rfind(mark)
                if pos > last_pos:
                    last_pos = pos
                    
            if last_pos > 0:
                # 截取到最后一个完整句子结束
                text = text[:last_pos+1]
        
        return text.strip()
    
    def _post_process_chunks(self, chunks: List[Document]) -> List[Document]:
        """对分块后的文本进行后处理，优化块的质量
        
        Args:
            chunks: 原始分块列表
            
        Returns:
            处理后的分块列表
        """
        if not chunks:
            return []
            
        logger.info("对文本块进行后处理优化...")
        processed_chunks = []
        
        # 按文档源分组处理
        doc_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "")
            if source not in doc_chunks:
                doc_chunks[source] = []
            doc_chunks[source].append(chunk)
        
        total_merged = 0
        
        # 处理每个文档的块
        for source, source_chunks in doc_chunks.items():
            # 按块索引排序
            sorted_chunks = sorted(source_chunks, 
                                   key=lambda x: x.metadata.get("chunk_index", 0))
            
            # 检查和处理相邻块
            for i, chunk in enumerate(sorted_chunks):
                # 确保完整句子
                chunk.page_content = self._ensure_complete_sentences(chunk.page_content)
                
                # 跳过空块
                if not chunk.page_content.strip():
                    continue
                
                # 过滤掉过短的块（例如只有几个字的块）
                if len(chunk.page_content) < 50:  # 设置最小块长度阈值
                    continue
                
                # 检查与前一个块的重叠度
                if i > 0 and processed_chunks:
                    prev_chunk = processed_chunks[-1]
                    if prev_chunk.metadata.get("source") == source:
                        # 计算重叠度
                        overlap_ratio = self._calculate_overlap_ratio(
                            prev_chunk.page_content, chunk.page_content)
                        
                        # 如果重叠度过高（超过70%），考虑合并或跳过
                        if overlap_ratio > 0.7:
                            # 如果当前块比前一个块短，跳过当前块
                            if len(chunk.page_content) <= len(prev_chunk.page_content):
                                continue
                            # 否则用当前块替换前一个块
                            else:
                                processed_chunks[-1] = chunk
                                continue
                
                processed_chunks.append(chunk)
        
        logger.info(f"后处理完成，优化后的块数: {len(processed_chunks)}")
        return processed_chunks
        
    def _calculate_overlap_ratio(self, text1: str, text2: str) -> float:
        """计算两个文本之间的重叠比例
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            重叠比例（0-1之间的浮点数）
        """
        # 使用较短文本的长度作为分母
        min_len = min(len(text1), len(text2))
        if min_len == 0:
            return 0.0
            
        # 寻找最长的共同子串
        for i in range(min_len, 0, -1):
            if text1.endswith(text2[:i]):
                return i / min_len
            if text2.endswith(text1[:i]):
                return i / min_len
                
        return 0.0

    def _print_chunks_summary(self, chunks: List[Document]):
        """打印文本分块结果概览"""
        if not chunks:
            logger.info("没有文本块可供显示")
            return
            
        # 统计信息
        total_chunks = len(chunks)
        avg_chunk_length = sum(len(chunk.page_content) for chunk in chunks) / total_chunks
        files_count = len(set(chunk.metadata.get("source", "") for chunk in chunks))
        
        logger.info("\n" + "="*50)
        logger.info("📊 文本分块处理概览")
        logger.info("="*50)
        logger.info(f"📄 总块数: {total_chunks}")
        logger.info(f"📊 平均块长度: {avg_chunk_length:.1f} 字符")
        logger.info(f"📂 涉及文件数: {files_count}")
        
        # 文件级统计
        file_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "未知来源")
            if source not in file_chunks:
                file_chunks[source] = []
            file_chunks[source].append(chunk)
        
        logger.info("\n📂 文件级分块统计:")
        for file_path, file_chunks_list in sorted(file_chunks.items(), key=lambda x: len(x[1]), reverse=True):
            file_name = Path(file_path).name if isinstance(file_path, str) else "未知文件"
            logger.info(f"  • {file_name}: {len(file_chunks_list)} 块")
        
        
        # 输出详细分块内容 (如果开启)
        if self.print_detailed_chunks:
            self._print_detailed_chunks(chunks)
            
        logger.info("="*50)

    def _print_detailed_chunks(self, chunks: List[Document]):
        """输出详细的分块内容"""
        logger.info("\n" + "="*50)
        logger.info("📑 详细文本块内容")
        logger.info("="*50)
        
        # 将分块按文件分组
        file_chunks = {}
        for chunk in chunks:
            source = chunk.metadata.get("source", "未知来源")
            if source not in file_chunks:
                file_chunks[source] = []
            file_chunks[source].append(chunk)
        
        # 为了更有组织地输出，先按文件输出
        for file_path, file_chunks_list in sorted(file_chunks.items(), key=lambda x: len(x[1]), reverse=True):
            file_name = Path(file_path).name if isinstance(file_path, str) else "未知文件"
            logger.info(f"\n📄 文件: {file_name} (共{len(file_chunks_list)}块)")
            
            # 输出该文件的前3个块
            for i, chunk in enumerate(file_chunks_list[:3]):
                page_num = chunk.metadata.get("page", "未知页码")
                chunk_size = len(chunk.page_content)
                
                # 获取预览内容
                content_preview = chunk.page_content
                if len(content_preview) > self.max_chunk_preview_length:
                    content_preview = content_preview[:self.max_chunk_preview_length] + "..."
                
                # 替换换行符以便于控制台显示
                content_preview = content_preview.replace("\n", "\\n")
                
                logger.info(f"\n  块 {i+1}/{len(file_chunks_list[:3])} [第{page_num}页, {chunk_size}字符]:")
                logger.info(f"  {content_preview}")
            
            # 如果文件中的块数超过3个，显示省略信息
            if len(file_chunks_list) > 3:
                logger.info(f"  ... 还有 {len(file_chunks_list) - 3} 个块未显示 ...")
                
        # 输出保存完整分块内容的提示
        chunks_detail_file = self.cache_dir / "chunks_detail.txt"
        try:
            with open(chunks_detail_file, "w", encoding="utf-8") as f:
                for i, chunk in enumerate(chunks):
                    source = chunk.metadata.get("source", "未知来源")
                    file_name = Path(source).name if isinstance(source, str) else "未知文件"
                    page_num = chunk.metadata.get("page", "未知页码")
                    
                    f.write(f"=== 块 {i+1}/{len(chunks)} [{file_name} - 第{page_num}页] ===\n")
                    f.write(chunk.page_content)
                    f.write("\n\n")
            
            logger.info(f"\n✅ 所有文本块的详细内容已保存至: {chunks_detail_file}")
        except Exception as e:
            logger.error(f"保存详细块内容失败: {str(e)}")
        
        logger.info("="*50)

    def create_embeddings(self) -> HuggingFaceEmbeddings:
        """创建嵌入模型实例"""
        logger.info("初始化嵌入模型...")
        return HuggingFaceEmbeddings(
            model_name=self.config.embedding_model_path,  # 嵌入模型的路径
            model_kwargs={"device": self.config.device},  # 设置设备为CPU或GPU
            encode_kwargs={
                "batch_size": self.config.batch_size,  # 批处理大小
                "normalize_embeddings": self.config.normalize_embeddings  # 是否归一化嵌入
            },
        )

    def backup_vector_db(self):
        """备份现有向量数据库"""
        vector_db_path = Path(self.config.vector_db_path)
        if not vector_db_path.exists():
            return False
            
        try:
            # 创建备份目录
            backup_dir = vector_db_path.parent / f"{vector_db_path.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # 复制所有文件到备份目录
            for item in vector_db_path.glob('*'):
                if item.is_file():
                    shutil.copy2(item, backup_dir)
                elif item.is_dir():
                    shutil.copytree(item, backup_dir / item.name)
                    
            logger.info(f"✅ 向量数据库已备份至 {backup_dir}")
            return True
        except Exception as e:
            logger.error(f"备份向量数据库失败: {str(e)}")
            return False

    def build_vector_store(self):
        """构建向量数据库"""
        logger.info("开始构建向量数据库")

        # 创建必要目录
        Path(self.config.vector_db_path).mkdir(parents=True, exist_ok=True)

        # 处理文档
        chunks = self.process_files()  # 处理文档并分块
        
        if not chunks:
            logger.warning("没有文档块可以处理，跳过向量存储构建")
            return

        # 检查是否需要重建索引
        if self.need_rebuild_index:
            logger.info("检测到文件变化，需要重建向量索引")
            if Path(self.config.vector_db_path).exists() and any(Path(self.config.vector_db_path).glob('*')):
                self.backup_vector_db()

        # 生成嵌入模型
        embeddings = self.create_embeddings()

        # 构建向量存储
        logger.info("生成向量...")
        # 构建向量存储时显式指定
        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
            distance_strategy=DistanceStrategy.COSINE  # 明确指定余弦相似度
        )

        # 保存向量数据库
        vector_store.save_local(str(self.config.vector_db_path))  # 保存向量存储到指定路径
        logger.info(f"向量数据库已保存至 {self.config.vector_db_path}")  # 输出保存路径


if __name__ == "__main__":
    try:
        # 初始化配置
        config = Config()
        
        # 添加: 解析命令行参数，允许用户指定是否打印详细分块内容
        import argparse
        parser = argparse.ArgumentParser(description='构建化工安全领域向量数据库')
        parser.add_argument('--detailed-chunks', action='store_true', 
                           help='是否输出详细的分块内容')
        parser.add_argument('--max-preview', type=int, default=510,
                           help='详细输出时每个文本块显示的最大字符数')
        args = parser.parse_args()
        
        # 更新配置
        if args.detailed_chunks:
            config.print_detailed_chunks = True
            config.max_chunk_preview_length = args.max_preview
            print(f"将输出详细分块内容，每块最多显示 {args.max_preview} 字符")

        # 构建向量数据库
        builder = VectorDBBuilder(config)
        builder.build_vector_store()

    except Exception as e:
        logger.exception("程序运行出错")  # 记录程序异常
    finally:
        logger.info("程序运行结束")  # 程序结束日志
