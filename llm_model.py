import logging
from typing import Generator, Optional, Tuple, List
import torch
from transformers import (
    AutoModelForCausalLM,  # 导入自回归语言模型类
    AutoTokenizer,         # 导入自动分词器类
    StoppingCriteria,     # 导入停止条件类
    StoppingCriteriaList  # 导入停止条件列表类
)
from config import Config  # 导入自定义配置类

# 设置日志记录器
logger = logging.getLogger(__name__)

class QwenLLM:
    """大语言模型封装类，支持同步生成和流式生成"""

    def __init__(self, config: Config):
        """
        初始化QwenLLM实例。
        :param config: 配置对象，包含了模型的配置参数
        """
        self.config = config  # 配置类实例
        self.device = config.device  # 获取配置中的设备（CPU或GPU）
        self._load_components()  # 加载模型和分词器等组件
        logger.info("✅ 模型组件初始化完成")

    def _load_components(self) -> None:
        """加载模型和分词器"""
        try:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(levelname)s - %(message)s",  # 配置日志格式
                handlers=[logging.StreamHandler()]  # 输出日志到控制台
            )
            logger.info("🔧 正在加载分词器...")
            # 加载模型所需的分词器
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.llm_model_path,  # 从配置中获取模型路径
                trust_remote_code=True,  # 允许从远程下载代码
                use_fast=True  # 使用快速分词器（可以加速分词过程）
            )

            logger.info("🚀 正在加载大模型...")
            # 加载模型
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.llm_model_path,  # 从配置中获取模型路径
                device_map=self.config.device,  # 将模型加载到配置指定的设备上（CPU/GPU）
                torch_dtype=self.config.torch_dtype,  # 模型权重的数据类型（如 float16, float32）
                trust_remote_code=True,  # 允许从远程下载代码
                # attn_implementation="flash_attention_2" if self.config.use_flash_attn else None,
                # quantization_config=self.config.quantization_config,
            ).eval()  # 将模型设置为评估模式（关闭 dropout 等）

            if self.config.compile_model:
                logger.info("⚡ 正在编译模型...")
                # 使用 PyTorch 编译器优化模型（加速推理过程）
                self.model = torch.compile(self.model)

        except Exception as e:
            logger.error(f"❌ 模型加载失败: {str(e)}")
            raise
    def _prepare_inputs(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """预处理输入"""
        try:
            # 使用分词器将输入的文本（prompt）转换为模型能够接受的格式
            inputs = self.tokenizer(
                prompt,  # 输入的提示文本
                return_tensors="pt",  # 返回PyTorch张量格式
                max_length=2048,  # 设置最大输入长度，防止超出模型的最大长度限制
                truncation=True  # 超过最大长度时进行截断
            ).to(self.model.device)  # 将输入移动到与模型相同的设备上
            return inputs.input_ids, inputs.attention_mask  # 返回输入的token id 和 attention mask

        except Exception as e:
            logger.error(f"输入处理失败: {str(e)}")  # 如果出现错误，记录日志并抛出异常
            raise

    def generate(self, prompt: str) -> str:
        """同步生成完整回答"""
        try:
            logger.info("🧠 开始同步生成...")
            input_ids, attention_mask = self._prepare_inputs(prompt)  # 获取输入的token id 和 attention mask

            # 调用模型的 generate 方法进行文本生成
            outputs = self.model.generate(
                input_ids=input_ids,  # 输入的token ids
                attention_mask=attention_mask,  # 输入的attention mask
                max_new_tokens=self.config.max_new_tokens,  # 最大生成token数
                temperature=self.config.temperature,  # 生成的多样性控制参数
                top_p=self.config.top_p,  # 核采样参数
                do_sample=self.config.do_sample,  # 是否启用采样
                pad_token_id=self.tokenizer.eos_token_id  # 使用模型的eos token作为填充token
            )

            # 解码生成的token为文本，并清理掉特殊token
            generated_text = self.tokenizer.decode(
                outputs[0][input_ids.shape[1]:],  # 跳过输入的部分，只解码生成的部分
                skip_special_tokens=True,  # 跳过特殊token
                clean_up_tokenization_spaces=True  # 清理多余的空格
            ).strip()  # 去掉两端的空格

            logger.info("✅ 同步生成完成")
            logger.info(generated_text)
            return generated_text  # 返回生成的文本

        except torch.cuda.OutOfMemoryError:
            logger.error("⚠️ CUDA内存不足，尝试减小max_new_tokens")  # 如果GPU内存不足，记录错误
            raise
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")  # 如果生成过程中出现其他错误，记录日志并抛出异常
            raise
    def __enter__(self):
        """进入上下文管理器"""
        return self  # 返回当前实例

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文管理器，清理资源"""
        if hasattr(self, "model"):
            del self.model  # 删除模型实例
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # 清空CUDA缓存
        logger.info("♻️ 模型资源已释放")  # 记录资源释放日志
