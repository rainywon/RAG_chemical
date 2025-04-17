# VLLM 模式使用说明

本项目提供了使用VLLM替代Ollama的实现方案，可以更高效地利用GPU资源，提升大模型推理速度。

## 主要特性

- **高性能并行推理**：VLLM采用PagedAttention技术，大幅提高了大模型推理吞吐量
- **兼容现有接口**：实现了与原有系统相同的接口，无缝对接
- **内存优化**：更有效的显存利用，支持较大模型在单卡运行
- **支持流式生成**：保留了流式回复功能

## 使用前提

- NVIDIA GPU，建议至少6GB显存
- CUDA环境已正确配置
- Python 3.8+

## 安装依赖

```bash
pip install vllm langchain-community transformers torch rank_bm25
```

## 配置说明

在`config.py`中添加了VLLM相关配置项：

```python
# VLLM大模型配置
self.vllm_model_path = r"C:\wu\models\Qwen-7B-Chat"  # VLLM模型路径
self.vllm_tensor_parallel_size = 1  # 张量并行大小，根据GPU数量设置
self.vllm_gpu_memory_utilization = 0.9  # GPU显存使用率
self.vllm_swap_space = 4  # 交换空间大小，单位为GB
```

请根据实际情况修改模型路径和参数：

1. `vllm_model_path`：设置为本地模型路径或Hugging Face模型名称
2. `vllm_tensor_parallel_size`：多GPU环境下可调整并行度
3. `vllm_gpu_memory_utilization`：控制GPU显存使用率，值范围0-1
4. `vllm_swap_space`：设置交换空间大小，可支持超出显存的大模型

## 使用方法

### 方法一：直接运行测试脚本

```bash
python rag_system_vllm.py
```

这将执行内置的测试用例，包括：
- 流式生成测试
- 非流式生成测试
- 纯模型生成测试（不使用知识库）

### 方法二：作为模块导入

```python
from config import Config
from rag_system_vllm import RAGSystem

# 初始化配置和系统
config = Config()
rag_system = RAGSystem(config)

# 使用RAG流式生成（结合知识库）
for chunk in rag_system.stream_query_rag_with_kb("危险化学品如何分类？"):
    print(chunk, end="", flush=True)

# 非流式RAG生成
answer, references, metadata = rag_system.query_model_with_kb("氯气泄漏应急处置方法？")
print(f"回答: {answer}")

# 纯模型生成（不使用知识库）
model_answer = rag_system.query_model_without_kb("介绍一下爆炸极限的概念")
print(f"回答: {model_answer}")
```

## VLLM与Ollama对比

| 特性 | VLLM | Ollama |
|-----|------|--------|
| 推理速度 | 更快（特别是批处理） | 较慢 |
| 显存优化 | 更高效 | 一般 |
| 模型支持 | 主流开源模型 | 更广泛 |
| 安装复杂度 | 中等 | 简单 |
| CPU模式 | 支持但不推荐 | 支持 |
| 多GPU支持 | 原生支持 | 有限支持 |

## 故障排除

1. **显存不足错误**
   - 尝试降低`vllm_gpu_memory_utilization`值
   - 增加`vllm_swap_space`大小
   - 使用更小的模型，如7B或4B版本

2. **模型加载失败**
   - 确认模型路径正确
   - 检查模型格式是否兼容VLLM
   - 确保安装了最新版本的VLLM

3. **生成速度慢**
   - 检查是否启用了正确的CUDA环境
   - 考虑增加`vllm_tensor_parallel_size`（多GPU环境）

## 开发注意事项

如果要进一步开发或修改VLLM集成，请注意以下几点：

1. VLLM的API与Hugging Face transformers略有不同
2. 流式生成需要特别处理，确保正确拼接片段
3. 停止词处理需要在采样参数中正确设置

更多信息请参考[VLLM官方文档](https://github.com/vllm-project/vllm)。 