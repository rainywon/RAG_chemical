import json
import os
from sklearn.model_selection import train_test_split
import random

# 设置随机种子以确保结果可重现
random.seed(42)

# 定义输入和输出文件路径
input_file = "build_lora_data/chemical_safety_deepseek_10k.json"
train_file = "build_lora_data/train_data.json"
test_file = "build_lora_data/test_data.json"

print(f"正在读取文件: {input_file}")

# 检查输入文件是否存在
if not os.path.exists(input_file):
    print(f"错误: 文件 {input_file} 不存在")
else:
    try:
        # 打开并读取文件
        with open(input_file, "r", encoding="utf-8") as f:
            # 加载 JSON 数据
            data = json.load(f)
            
            # 打印数据总量
            print(f"数据总量: {len(data)} 条")

            # 首先将数据分为 训练+验证+一部分测试 (90%) 和 测试 (10%)
            train_data, test_data = train_test_split(data, test_size=0.1, random_state=42)
            
            # 然后将 训练+验证 数据分为 训练 (70/90 = 77.8%) 和 验证 (20/90 = 22.2%)
            
            # 打印各集合大小及比例
            print(f"训练集: {len(train_data)} 条 ({len(train_data)/len(data)*100:.2f}%)")
            print(f"测试集: {len(test_data)} 条 ({len(test_data)/len(data)*100:.2f}%)")

            # 将数据保存到 JSON 文件
            with open(train_file, "w", encoding="utf-8") as f:
                json.dump(train_data, f, ensure_ascii=False, indent=2)
            
            with open(test_file, "w", encoding="utf-8") as f:
                json.dump(test_data, f, ensure_ascii=False, indent=2)
            

            print(f"数据已成功分割并保存:")
            print(f"- 训练集: {train_file}")
            print(f"- 测试集: {test_file}")
            
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")