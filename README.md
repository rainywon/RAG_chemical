# 化工安全领域RAG系统

本项目是一个基于检索增强生成(RAG)技术的化工安全知识问答系统，用于解答用户关于化工安全领域的专业问题。系统结合了先进的向量检索和大型语言模型，以提供准确、专业的化工安全信息。

## 系统特点

- **专业领域优化**：针对化工安全领域进行特殊优化，支持专业术语和特定场景
- **混合检索策略**：结合向量检索(FAISS)和关键词检索(BM25)，提高检索精度
- **基于Ollama的大模型**：使用深度定制的Deepseek-r1模型进行回答生成
- **动态查询增强**：自动扩展和优化用户查询，提高相关文档召回率
- **智能文档重排序**：使用重排序模型对检索结果进行优化排序
- **多样性增强算法**：应用MMR算法确保检索结果的多样性
- **完善的评估体系**：支持检索和生成模块的独立评估与优化

## 系统架构

![RAG系统架构](assets/rag-architecture.md)

系统主要由以下组件构成：

1. **文档处理模块**：处理PDF等格式文档，提取文本并分块
2. **向量数据库**：使用FAISS存储文档嵌入向量
3. **检索模块**：
   - 向量检索：基于语义相似度检索
   - BM25检索：基于关键词匹配检索
   - 重排序：优化检索结果排序
4. **生成模块**：基于Ollama的语言模型，结合检索结果生成回答
5. **评估模块**：独立评估检索和生成性能
6. **Web服务**：提供REST API接口

## 安装说明

### 前提条件

- Python 3.8+
- CUDA支持的GPU（推荐用于向量嵌入和模型推理）
- 至少8GB RAM
- 50GB磁盘空间（用于存储模型和向量数据库）

### 安装步骤

1. 克隆代码库

```bash
git clone https://github.com/rainywon/chemical_rag
cd chemical_rag
```

2. 创建并激活虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. 安装依赖

```bash
pip install -r requirements.txt
```

4. 安装Ollama

请参考[Ollama官方安装指南](https://ollama.ai/download)

5. 下载所需模型

```bash
# 下载deepseek-r1模型
ollama pull deepseek-r1:1.5b

# 下载嵌入模型（如果需要）
# 编辑config.py中的embedding_model_path指向BAAI/bge-large-zh-v1.5模型位置
```

6. 配置系统

编辑`config.py`文件，根据需要调整参数：
- 修改模型路径
- 调整检索参数
- 配置硬件设置

## 使用指南

### 构建向量数据库

首次使用前，需要构建向量数据库：

```bash
python build_vector_store.py
```

### 启动Web服务

```bash
python main.py
```

服务默认在http://localhost:8000运行

### API接口

主要接口：

- `POST /api/query`：文本问答接口
- `POST /api/query_stream`：流式文本问答接口

#### 问答示例

```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "苯对人体的危害有哪些？"}'
```

## 评估方法

本系统提供了完善的评估框架，用于测试和优化RAG性能。

### 运行评估

```bash
python evaluate/run_evaluation.py
```

### 评估指标

#### 检索模块评估
- **命中率(Hit Rate)**: 检索结果中包含相关文档的比例
- **平均倒数排名(MRR)**: 衡量相关文档排序质量

#### 生成模块评估
- **忠实度(Faithfulness)**: 生成内容与检索文档一致性
- **答案相关性(Answer Relevance)**: 生成内容与问题相关度

详细评估方法请参考[评估模块文档](evaluate/README.md)

## 项目结构

```
chemical_rag/
├── config.py                  # 系统配置
├── rag_system.py              # RAG系统核心实现
├── build_vector_store.py      # 向量数据库构建工具
├── main.py                    # Web服务入口
├── llm_model.py               # 语言模型接口
├── database.py                # 数据库操作
├── data/                      # 知识库文档
├── vector_store/              # 向量数据库存储
├── cache/                     # 缓存目录
├── pdf_cor_extractor/         # PDF处理工具
├── routers/                   # API路由
│   ├── query.py               # 查询接口
│   ├── login.py               # 登录接口
│   └── sms.py                 # 短信验证接口
└── evaluate/                  # 评估模块
    ├── run_evaluation.py      # 评估主程序
    ├── evaluate_retrieval.py  # 检索评估
    ├── evaluate_generation.py # 生成评估
    └── test_data/             # 测试数据
```

## 技术细节

### 检索增强

- **查询增强**：自动扩展原始查询，提高召回率
- **动态权重调整**：根据问题类型动态调整检索算法权重
- **分布感知归一化**：根据分数分布特性选择归一化策略

### 重排序与多样性

- **重排序模型**：使用BGE-Reranker优化检索结果排序
- **MMR算法**：在相关性和多样性之间寻找最佳平衡

## 贡献指南

欢迎对本项目做出贡献。请遵循以下步骤：

1. Fork本仓库
2. 创建功能分支
3. 提交变更
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件

## 联系方式

如有问题，请通过以下方式联系：

- 电子邮件：1028418330@qq.com
- GitHub Issues：请在本仓库中创建issue 