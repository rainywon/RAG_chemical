import time
from typing import List, Dict, Optional, Tuple
import fitz
import cv2
import numpy as np
from langchain_core.documents import Document
from paddleocr import PaddleOCR
from multiprocessing import Pool, cpu_count
import os
import logging
import torch  # 添加torch导入以检测GPU
import paddle  # 直接导入paddle检查环境
from pathlib import Path
from config import Config
import json
from datetime import datetime

"""
使用示例:

# 基础使用方式（从Config加载参数）
from config import Config
config = Config()
processor = PDFProcessor(
    file_path='example.pdf',
    lang='ch', 
    use_gpu=True,
    gpu_params=config.pdf_ocr_params
)
docs = processor.process()

# 手动配置GPU参数
processor = PDFProcessor(file_path='example.pdf', lang='ch', use_gpu=True)
processor.configure_gpu(
    batch_size=4,              # 批处理大小
    min_pages_for_batch=3,     # 最小启用批处理的页数
    det_limit_side_len=1280,   # 检测分辨率（更高分辨率可能提高准确性但降低速度）
    rec_batch_num=15,          # 识别批处理量
    det_batch_num=8,           # 检测批处理量
    use_tensorrt=True          # 使用TensorRT加速（需要安装TensorRT）
)
docs = processor.process()

# 使用大文档优化参数
processor = PDFProcessor(
    file_path='large_document.pdf',
    lang='ch', 
    use_gpu=True,
    gpu_params=config.pdf_ocr_large_doc_params
)
docs = processor.process()

# 使用1050Ti优化参数
processor = PDFProcessor(
    file_path='example.pdf',
    lang='ch', 
    use_gpu=True,
    gpu_params=config.pdf_ocr_1050ti_params
)
docs = processor.process()

# 不使用GPU
processor = PDFProcessor(file_path='example.pdf', lang='ch', use_gpu=False)
docs = processor.process()
"""

# 使用彩色日志格式和更简洁的输出
class ColoredFormatter(logging.Formatter):
    """自定义彩色日志格式器"""
    COLORS = {
        'INFO': '\033[92m',      # 绿色
        'WARNING': '\033[93m',   # 黄色
        'ERROR': '\033[91m',     # 红色
        'CRITICAL': '\033[91m',  # 红色
        'DEBUG': '\033[94m',     # 蓝色
        'RESET': '\033[0m'       # 重置颜色
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # 简化时间格式，只显示时:分:秒
        record.asctime = self.formatTime(record, datefmt='%H:%M:%S')
        
        # 使用图标代替日志级别，增强可读性
        if record.levelname == 'INFO':
            level_icon = 'ℹ️'
        elif record.levelname == 'WARNING':
            level_icon = '⚠️'
        elif record.levelname == 'ERROR':
            level_icon = '❌'
        elif record.levelname == 'CRITICAL':
            level_icon = '🔥'
        else:
            level_icon = '🔍'
            
        # 替换原始消息中的多余标签
        message = record.getMessage()
        message = message.replace('[文档加载]', '📄').replace('[PDF转换]', '🔄')
        message = message.replace('[PDF处理]', '📊').replace('[OCR处理]', '👁️')
        
        # 组装最终日志格式
        log_fmt = f"{log_color}{record.asctime} {level_icon} {message}{reset_color}"
        record.msg = log_fmt
        return super(logging.Formatter, self).format(record)

# 更新日志配置，替换build_vector_store.py中的相应代码
def setup_logging():
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(ColoredFormatter())
    
    # 获取根日志记录器
    root_logger = logging.getLogger()
    root_logger.handlers = []  # 清除现有处理程序
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    
    # 设置一些第三方库的日志级别更高，减少干扰
    logging.getLogger('paddleocr').setLevel(logging.WARNING)
    logging.getLogger('paddle').setLevel(logging.WARNING)

class PDFProcessor:
    def __init__(self, file_path: str = None, lang: str = 'ch', use_gpu: bool = True, gpu_params: dict = None):
        self.file_path = file_path
        self.lang = lang
        self.use_gpu = use_gpu
        self.base_zoom = 1.2
        self._ocr_engine = None
        
        # GPU相关参数 - 默认值
        self.gpu_params = {
            'batch_size': 3,          # 批处理大小
            'min_pages_for_batch': 3, # 启用批处理的最小页数
            'det_limit_side_len': 960,# 检测分辨率
            'rec_batch_num': 8,       # 识别批处理量
            'det_batch_num': 4,       # 检测批处理量
            'use_tensorrt': False     # 是否使用TensorRT加速
        }
        
        # 如果提供了GPU参数，则使用提供的参数
        if gpu_params is not None:
            for key, value in gpu_params.items():
                if key in self.gpu_params:
                    self.gpu_params[key] = value
        
        # 检查GPU可用性
        self.gpu_available = False
        self._check_gpu_availability()
        
    def _check_gpu_availability(self):
        """检查GPU可用性"""
        try:
            # 首先检查环境变量
            if 'CUDA_VISIBLE_DEVICES' not in os.environ:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                
            # 尝试使用torch检测CUDA可用性
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                self.gpu_available = True
                logging.info(f"✅ GPU可用: {gpu_name}, 显存: {gpu_mem:.1f}GB")
                
                # 针对1050Ti特别优化参数
                if "1050 Ti" in gpu_name:
                    logging.info(f"⚡ 检测到1050Ti GPU，应用优化参数")
                    # 1050Ti有4GB显存，针对性设置参数
                    self.gpu_params = {
                        'batch_size': 2,          # 较小批处理大小避免显存溢出
                        'min_pages_for_batch': 2, # 更低的批处理启用阈值
                        'det_limit_side_len': 640,# 降低检测分辨率以减少显存占用
                        'rec_batch_num': 4,       # 较小识别批处理量
                        'det_batch_num': 2,       # 较小检测批处理量
                        'use_tensorrt': False     # 1050Ti通常不支持高版本TensorRT
                    }
            else:
                logging.warning("⚠️ 未检测到可用GPU，将使用CPU模式")
                self.gpu_available = False
                self.use_gpu = False
                
            # 设置paddle环境（无论是否有GPU）
            try:
                if self.use_gpu and self.gpu_available:
                    paddle.set_device('gpu:0')
                    logging.info("✅ Paddle已设置为GPU模式")
                else:
                    paddle.set_device('cpu')
                    logging.info("✅ Paddle已设置为CPU模式")
            except Exception as e:
                logging.warning(f"⚠️ Paddle设备设置失败: {str(e)}")
                if self.use_gpu:
                    self.use_gpu = False
                    logging.warning("⚠️ 已自动切换到CPU模式")
                
        except Exception as e:
            logging.error(f"⚠️ GPU检测失败: {str(e)}")
            self.gpu_available = False
            self.use_gpu = False

    def configure_gpu(self, **kwargs):
        """配置GPU相关参数
        
        参数:
            batch_size (int): 批处理大小
            min_pages_for_batch (int): 启用批处理的最小页数
            det_limit_side_len (int): 检测分辨率
            rec_batch_num (int): 识别批处理量
            det_batch_num (int): 检测批处理量
            use_tensorrt (bool): 是否使用TensorRT加速
        """
        # 更新GPU参数
        for key, value in kwargs.items():
            if key in self.gpu_params:
                self.gpu_params[key] = value
                logging.info(f"更新GPU参数: {key} = {value}")
        
        # 如果已经初始化了OCR引擎，则需要重新初始化
        if self._ocr_engine is not None:
            logging.info("参数已更改，重新初始化OCR引擎...")
            self._ocr_engine = None

    @property
    def ocr_engine(self):
        if self._ocr_engine is None:
            # 根据实际GPU可用性设置use_gpu
            actual_use_gpu = self.use_gpu and self.gpu_available
            
            try:
                # 初始化OCR引擎
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=False,
                    lang=self.lang,
                    use_gpu=actual_use_gpu,  # 使用实际GPU可用性
                    show_log=False,  # 减少日志输出
                    rec_image_shape="3, 48, 320",
                    drop_score=0.6,
                    det_limit_side_len=self.gpu_params['det_limit_side_len'],  # 检测分辨率
                    det_db_unclip_ratio=1.5,  # 优化检测参数
                    rec_algorithm='SVTR_LCNet',
                    rec_batch_num=self.gpu_params['rec_batch_num'],  # 识别批处理量
                    det_batch_num=self.gpu_params['det_batch_num'],  # 检测批处理量
                    use_tensorrt=self.gpu_params['use_tensorrt']  # TensorRT加速
                )
                
                # 日志输出OCR引擎配置情况
                mode_str = "GPU" if actual_use_gpu else "CPU"
                logging.info(f"🔧 OCR引擎初始化完成，使用{mode_str}模式")
            except Exception as e:
                logging.error(f"❌ OCR引擎初始化失败: {str(e)}")
                logging.info("⚠️ 尝试使用CPU模式初始化OCR引擎")
                try:
                    # 确保使用CPU设备
                    paddle.set_device('cpu')
                    self._ocr_engine = PaddleOCR(
                        use_angle_cls=False,
                        lang=self.lang,
                        use_gpu=False,  # 强制使用CPU
                        show_log=False
                    )
                    logging.info("🔧 OCR引擎(CPU模式)初始化完成")
                except Exception as e2:
                    logging.error(f"❌ CPU模式初始化也失败: {str(e2)}")
                    raise e2
            
        return self._ocr_engine

    def _convert_pages(self, pdf_path: str) -> List[Tuple[int, np.ndarray]]:
        try:
            with fitz.open(pdf_path) as doc:
                page_count = doc.page_count
                logging.info(f"[PDF处理] 开始转换 '{Path(pdf_path).name}' ({page_count}页)")
                
                # 直接处理每一页，不使用进程池
                converted = []
                for pg in range(page_count):
                    try:
                        page = doc[pg]
                        matrix = fitz.Matrix(self.base_zoom, self.base_zoom)
                        pix = page.get_pixmap(matrix=matrix, alpha=False)
                        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                        converted.append((pg, img_array))
                    except Exception as e:
                        logging.warning(f"[PDF转换] 页面{pg+1}失败: {str(e)}")
                
                if not converted:
                    logging.error("[PDF转换] 失败: 没有页面成功转换")
                else:
                    logging.info(f"[PDF转换] 完成: 成功转换{len(converted)}/{page_count}页 ({int(len(converted)/page_count*100)}%)")
                
                # 确保页码顺序正确
                converted.sort(key=lambda x: x[0])
                return converted
                
        except Exception as e:
            logging.error(f"[PDF转换] 失败: {str(e)}")
            return []

    @staticmethod
    def _convert_page(args: tuple) -> Optional[Tuple[int, np.ndarray]]:
        """
        此方法保留但不再使用，为保持API兼容性
        """
        pdf_path, pg, zoom = args
        try:
            with fitz.open(pdf_path) as doc:
                page = doc[pg]
                matrix = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=matrix, alpha=False)
                img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
                return (pg, img_array)  # 直接返回RGB格式
        except Exception as e:
            logging.warning(f"Page {pg} conversion failed: {e}")
            return None

    def _parse_ocr_result(self, result: list) -> str:
        return "\n".join(
            line[1][0].strip() for line in result[0]
            if line[1][0].strip()
        ) if result and result[0] else ""

    def _print_progress_bar(self, progress: float, current_page: int):
        """控制台进度条显示，改为单行固定格式"""
        bar_length = 20
        filled = int(progress / 100 * bar_length)
        bar = '█' * filled + '░' * (bar_length - filled)
        # 使用\r确保每个文件只占用一行
        print(f"\r[OCR识别] 进度: {bar} {progress:.1f}% | 页面: {current_page}", end='', flush=True)

    def _print_summary(self, total: int, success: int, failed: list, duration: float):
        """输出处理结果摘要"""
        print("\n" + "─" * 40)
        logging.info(f"[OCR处理] 摘要:")
        logging.info(f"  • 总页数    : {total} 页")
        logging.info(f"  • 成功识别  : {success} 页 ({success / total * 100:.1f}%)")
        if failed:
            logging.info(f"  • 失败页面  : {len(failed)} 页 ({', '.join(map(str, failed[:5]))}" + 
                         (f"...等{len(failed)-5}页" if len(failed) > 5 else "") + ")")
        logging.info(f"  • 总耗时    : {duration:.1f} 秒")
        logging.info(f"  • 平均速度  : {duration / total:.1f} 秒/页" if total > 0 else "")
        print("─" * 40 + "\n")

    def _batch_process_pages(self, converted_pages: List[Tuple[int, np.ndarray]]) -> List[Document]:
        """使用批处理方式处理页面，提高GPU利用率"""
        documents = []
        batch_size = self.gpu_params['batch_size']
        total_pages = len(converted_pages)
        success_count = 0
        fail_pages = []
        stage_start = time.time()
        
        logging.info(f"[OCR处理] 使用GPU批处理模式 (批次大小: {batch_size})")
        
        # 确保GPU模式激活
        if not (self.use_gpu and self.gpu_available):
            logging.warning("[OCR处理] GPU不可用，切换到单页处理模式")
            # 单页处理模式
            return self.process_pdf_single_page(converted_pages)
        
        # 首先清理GPU缓存
        try:
            torch.cuda.empty_cache()
        except:
            pass
            
        # 分批处理
        for batch_idx in range(0, total_pages, batch_size):
            batch_end = min(batch_idx + batch_size, total_pages)
            batch_pages = converted_pages[batch_idx:batch_end]
            
            
            # 批量处理前释放内存
            if self.gpu_available and batch_idx > 0:
                # 优化显存使用
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
            
            # 批量处理
            batch_start = time.time()
            
            for pg, img in batch_pages:
                page_num = pg + 1
                try:
                    # 处理前调整图像大小以节省显存
                    if max(img.shape[0], img.shape[1]) > 1600:
                        scale = 1600 / max(img.shape[0], img.shape[1])
                        new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                    # OCR处理
                    result = self.ocr_engine.ocr(img, cls=False)
                    page_text = self._parse_ocr_result(result)
                    
                    # 记录文档
                    documents.append(
                        Document(
                            page_content=page_text,
                            metadata={
                                "page": page_num,
                                "image_size": img.shape,
                                "batch_process": True
                            }
                        )
                    )
                    success_count += 1
                except Exception as e:
                    fail_pages.append(page_num)
                    logging.warning(f"[OCR处理] 页面{page_num}失败: {str(e)}")
            
            
            # 每批次后清理GPU内存
            if self.gpu_available:
                try:
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                except:
                    pass
        
        # 最终统计
        total_time = time.time() - stage_start
        # 清空进度条
        print()
        self._print_summary(total_pages, success_count, fail_pages, total_time)
        
        # 按页码排序返回
        documents.sort(key=lambda doc: doc.metadata["page"])
        return documents
        
    def process_pdf_single_page(self, converted_pages: List[Tuple[int, np.ndarray]]) -> List[Document]:
        """使用单页处理模式处理PDF"""
        documents = []
        total_pages = len(converted_pages)
        success_count = 0
        fail_pages = []
        stage_start = time.time()
        
        logging.info("[OCR处理] 使用单页处理模式")
        
        # 存储上次进度更新的时间，避免日志过多
        last_log_time = time.time()
        
        for idx, (pg, img) in enumerate(converted_pages):
            page_num = pg + 1
            page_start = time.time()

            try:                   
                # 处理前调整图像大小以节省显存
                if max(img.shape[0], img.shape[1]) > 1600:
                    scale = 1600 / max(img.shape[0], img.shape[1])
                    new_h, new_w = int(img.shape[0] * scale), int(img.shape[1] * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # OCR处理
                result = self.ocr_engine.ocr(img, cls=False)
                page_text = self._parse_ocr_result(result)

                # 记录文档
                documents.append(
                    Document(
                        page_content=page_text,
                        metadata={
                            "page": page_num,
                            "image_size": img.shape,
                            "process_time": time.time() - page_start
                        }
                    )
                )
                success_count += 1
            except Exception as e:
                fail_pages.append(page_num)
                logging.warning(f"[OCR处理] 页面{page_num}失败: {str(e)}")

        # 最终统计
        total_time = time.time() - stage_start
        # 清空进度条
        print()
        self._print_summary(total_pages, success_count, fail_pages, total_time)
        return documents
        
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """处理PDF并显示实时进度"""
        documents = []
        stage_start = time.time()

        try:            
            # 阶段1：PDF转图像
            logging.info("[PDF处理] 阶段1/2: 页面转换中...")
            converted_pages = self._convert_pages(pdf_path)
            if not converted_pages:
                logging.warning("[PDF处理] 没有可处理页面")
                return []

            total_pages = len(converted_pages)
            parse_time = time.time() - stage_start
            logging.info(f"[PDF处理] 页面转换完成，共{total_pages}页 (耗时{parse_time:.1f}s)")

            # 阶段2：OCR处理
            logging.info("[PDF处理] 阶段2/2: OCR文字识别中...")

            # GPU模式下使用批处理提高性能
            if self.use_gpu and self.gpu_available and total_pages > self.gpu_params['min_pages_for_batch']:
                return self._batch_process_pages(converted_pages)
            else:
                # 单页处理模式
                return self.process_pdf_single_page(converted_pages)

        except Exception as e:
            logging.error(f"[PDF处理] 处理异常终止: {str(e)}")
            return []

    def process(self) -> List[Document]:
        """处理PDF文件，返回文档对象列表"""
        if not self.file_path:
            raise ValueError("请提供PDF文件路径")
        return self.process_pdf(self.file_path)

    def _generate_processing_report(self):
        """生成文档处理报告，提供更丰富的统计信息"""
        report = {
            "总文件数": len(self.processed_files),
            "总页数": sum(info.get("pages", 0) for info in self.processed_files.values()),
            "平均每文件页数": sum(info.get("pages", 0) for info in self.processed_files.values()) / max(len(self.processed_files), 1),
            "处理失败文件数": self.failed_files_count,
            "成功率": 1 - (self.failed_files_count / max(len(self.processed_files), 1)),
            "文件类型统计": {},
            "处理时间": datetime.now().isoformat()
        }
        
        # 统计文件类型
        for file_path in self.processed_files:
            ext = Path(file_path).suffix.lower()
            if ext in report["文件类型统计"]:
                report["文件类型统计"][ext] += 1
            else:
                report["文件类型统计"][ext] = 1
        
        # 保存报告
        with open(self.cache_dir / "processing_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        # 输出简要报告
        logger.info("\n📊 文档处理报告")
        logger.info(f"📑 总文件数: {report['总文件数']} 个")
        logger.info(f"📄 总页数: {report['总页数']} 页")
        logger.info(f"📊 平均每文件: {report['平均每文件页数']:.1f} 页")
        logger.info(f"✅ 成功率: {report['成功率']:.1%}")
        
        # 文件类型统计
        logger.info("📂 文件类型分布:")
        for ext, count in report["文件类型统计"].items():
            logger.info(f"   - {ext}: {count} 个")