# 化工安全RAG系统评估模块

本目录包含用于评估化工安全RAG系统性能的工具和测试数据。

## 目录结构

```
evaluate/
├── README.md                   # 本说明文档
├── run_evaluation.py           # 主评估运行程序
├── evaluate_retrieval.py       # 检索模块评估工具
├── evaluate_generation.py      # 生成模块评估工具（使用Ragas）
├── test_data/                  # 测试数据目录
│   ├── retrieval_test_data.json   # 检索模块测试数据
│   └── generation_test_data.json  # 生成模块测试数据
└── results/                    # 评估结果保存目录（自动创建）
```

## 评估指标

### 检索模块评估

1. **命中率 (Hit Rate)**
   - Hit@1: 首位检索结果中包含相关文档的比例
   - Hit@3: 前三位检索结果中包含相关文档的比例
   - Hit@5: 前五位检索结果中包含相关文档的比例
   - Hit@10: 前十位检索结果中包含相关文档的比例

2. **平均倒数排名 (Mean Reciprocal Rank, MRR)**
   - 对于每个查询，计算第一个相关文档排名的倒数，然后取平均值
   - 值越接近1表示相关文档排名越靠前

### 生成模块评估 (使用Ragas)

[Ragas](https://github.com/explodinggradients/ragas) 是一个专门用于评估RAG系统的开源框架，提供多种指标衡量生成内容的质量。本项目使用以下Ragas指标：

1. **忠实度 (Faithfulness)**
   - 评估生成内容是否与检索文档内容一致，不包含虚构或矛盾信息
   - 分数范围从0到1，1表示完全忠实

2. **答案相关性 (Answer Relevancy)**
   - 评估生成内容与问题的相关程度
   - 分数范围从0到1，1表示完全相关

3. **上下文精确度 (Context Precision)**
   - 评估检索到的上下文中与问题相关的信息比例
   - 高分表示检索到的上下文高度相关

4. **上下文相关性 (Context Relevancy)**
   - 评估上下文与问题之间的语义相关性
   - 高分表示检索到的上下文与问题语义相关

5. **上下文召回率 (Context Recall)**
   - 评估上下文是否包含回答问题所需的所有信息
   - 高分表示上下文提供了完整的信息

6. **无害性 (Harmfulness)**
   - 评估生成内容是否包含有害信息
   - 高分表示生成内容无害且安全

7. **综合得分 (Overall Score)**
   - 基于上述指标的加权平均计算
   - 提供对系统整体性能的单一量化指标

## 使用方法

### 运行完整评估

```bash
python evaluate/run_evaluation.py
```

### 参数说明

- `--test_data_dir`: 测试数据目录路径，默认为 "evaluate/test_data"
- `--output_dir`: 结果输出目录路径，默认为 "evaluate/results"

例如：
```bash
python evaluate/run_evaluation.py --test_data_dir custom_data --output_dir custom_results
```

## 测试数据格式

### 检索测试数据格式

```json
[
  {
    "question": "问题文本",
    "relevant_docs": ["文档路径1", "文档路径2"]
  },
  ...
]
```

### 生成测试数据格式

```json
[
  {
    "question": "问题文本",
    "gold_answer": "标准答案文本（可选）"
  },
  ...
]
```

## 评估结果示例

```json
{
  "retrieval": {
    "hit_rate": {
      "hit@1": 0.75,
      "hit@3": 0.85,
      "hit@5": 0.95,
      "hit@10": 1.0
    },
    "mrr": 0.82
  },
  "generation": {
    "faithfulness": {
      "faithfulness_score": 0.85
    },
    "relevance": {
      "question_overlap": 0.32,
      "context_relevance": 0.75,
      "relevance_score": 0.78
    },
    "ragas_metrics": {
      "faithfulness": {
        "score": 0.85,
        "raw_scores": [...]
      },
      "answer_relevancy": {
        "score": 0.82,
        "raw_scores": [...]
      },
      "context_precision": {
        "score": 0.75,
        "raw_scores": [...]
      },
      "context_relevancy": {
        "score": 0.78,
        "raw_scores": [...]
      },
      "context_recall": {
        "score": 0.72,
        "raw_scores": [...]
      },
      "harmfulness": {
        "score": 0.95,
        "raw_scores": [...]
      },
      "overall_score": 0.81,
      "meta": {
        "total_samples": 10,
        "evaluated_samples": 10
      }
    }
  }
}
```

## Ragas配置

Ragas评估使用以下权重计算综合得分：

```python
weights = {
    "faithfulness": 0.3,          # 忠实度权重
    "answer_relevancy": 0.25,     # 答案相关性权重
    "context_precision": 0.15,    # 上下文精确度权重
    "context_relevancy": 0.15,    # 上下文相关性权重
    "context_recall": 0.15        # 上下文召回率权重
}
```

可以通过修改`evaluate_generation.py`中的权重配置来定制评估重点。 