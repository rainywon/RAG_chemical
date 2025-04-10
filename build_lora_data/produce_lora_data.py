import json
import os
import re
import time
import concurrent.futures
from functools import partial
from zhipuai import ZhipuAI

class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'

class CotValidator:
    @staticmethod
    def validate(answer):
        """DeepSeek风格的CoT格式验证"""
        # 检查标签完整性
        think_blocks = re.findall(r'<think>(.*?)</think>', answer, re.DOTALL)
        if not think_blocks:
            # 如果没有找到标签，尝试识别思考部分
            parts = answer.split('\n\n', 1)
            if len(parts) > 1:
                # 将第一部分视为思考，重新格式化
                answer = f"<think>{parts[0]}</think>\n\n{parts[1]}"
                think_blocks = [parts[0]]
            else:
                # 无法分割，使用前2/3作为思考
                split_point = int(len(answer) * 2/3)
                thinking = answer[:split_point]
                response = answer[split_point:]
                answer = f"<think>{thinking}</think>\n\n{response}"
                think_blocks = [thinking]
        
        # 验证思考内容质量
        for think in think_blocks:
            # 极简验证：只检查长度
            if len(think.strip()) < 30:
                raise ValueError("思考内容过短（至少30字符）")
        
        # 验证实际回答
        clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        if not clean_answer or len(clean_answer) < 20:
            raise ValueError("实际回答内容过简")
        
        return True, answer  # 返回验证结果和可能修改过的答案

def load_questions(py_path):
    """从Python文件加载问题列表"""
    try:
        with open(py_path, "r", encoding="utf-8") as f:
            namespace = {}
            exec(f.read(), namespace)
            return namespace.get("questions", [])
    except Exception as e:
        raise RuntimeError(f"解析问题文件失败: {str(e)}")

def load_existing_data(json_path):
    """加载已有数据并建立问题索引"""
    processed = set()
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    processed.add(entry["instruction"].strip())
            print(f"{Colors.GREEN}✅ 已加载{len(processed)}条已处理数据{Colors.END}")
        except Exception as e:
            os.rename(json_path, f"{json_path}.bak")
            print(f"{Colors.YELLOW}⚠ 数据文件损坏，已备份: {str(e)}{Colors.END}")
    return processed

def generate_deepseek_entry(question, answer):
    """增强数据格式生成"""
    # 格式化输出，保持简单直接的提示语
    instruction = f"{question}\n\n请先详细思考，再给出专业解答。"
    

    # 确保答案结构完整
    output = answer
    
    return {
        "instruction": instruction,
        "input": "",
        "output": output
    }

def save_with_backup(data, path):
    """带备份的安全保存（JSON数组格式）"""
    temp_path = f"{path}.tmp"
    try:
        # 读取现有数据
        existing = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)

        # 合并数据
        combined = existing + data

        # 写入临时文件
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(combined, f, ensure_ascii=False, indent=2)

        # 原子替换
        if os.path.exists(path):
            os.replace(path, f"{path}.bak")
        os.rename(temp_path, path)
    except Exception as e:
        print(f"{Colors.RED}保存失败: {str(e)}{Colors.END}")
        if os.path.exists(temp_path):
            os.remove(temp_path)

def process_question(client, system_prompt, question, error_log, retry=3):
    """改进的思考链生成逻辑"""
    for attempt in range(retry):
        try:
            # 增强提示工程
            user_prompt = f"{question}\n\n请先在<think>标签内进行全面思考分析，然后给出根据思考的内容给出答案。"
            
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.6,  # 增加一些多样性
                max_tokens=3000
            )
            answer = response.choices[0].message.content
            
            # 增强格式处理
            answer = re.sub(r'(?i)<think>', '<think>', answer)
            answer = re.sub(r'(?i)</think>', '</think>', answer)
            answer = re.sub(r'（([^）]+)）', r'（\1）', answer)  # 统一括号
            
            # 移除所有可能的部分标题
            answer = re.sub(r'【[^】]+】', '', answer)
            
            # 使用更宽松的验证器
            is_valid, formatted_answer = CotValidator.validate(answer)
            
            return generate_deepseek_entry(question, formatted_answer)
            
        except Exception as e:
            if attempt < retry - 1:
                wait_time = 2 ** (attempt + 1)
                print(f"{Colors.YELLOW}⚠ 第{attempt+1}次重试，等待{wait_time}秒...{Colors.END}")
                time.sleep(wait_time)
            else:
                # 格式化错误记录
                error_msg = str(e)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(error_log, "a", encoding="utf-8") as f:
                    f.write(f"[{timestamp}] 问题: {question}\n错误: {error_msg}\n{'='*50}\n")
                return None

def process_question_wrapper(client, system_prompt, error_log, question):
    """增加进度提示"""
    try:
        print(f"{Colors.BLUE}● 处理中: {question[:35]}...{Colors.END}")
        start_time = time.time()
        result = process_question(client, system_prompt, question, error_log)
        elapsed = time.time() - start_time
        
        if result:
            # 尝试从输出提取思考部分长度
            think_match = re.search(r'<think>(.*?)</think>', result['output'], re.DOTALL)
            think_len = len(think_match.group(1)) if think_match else 0
            ans_len = len(result['output']) - think_len if think_len > 0 else len(result['output'])
            
            print(f"{Colors.GREEN}✅ 成功 | 耗时:{elapsed:.1f}s | 思考:{think_len}字 | 回答:{ans_len}字{Colors.END}")
            return result
        else:
            print(f"{Colors.YELLOW}⚠️ 空响应: {question[:30]}...{Colors.END}")
        return None
    except Exception as e:
        print(f"{Colors.RED}❌ 失败: {str(e)[:50]}...{Colors.END}")
        return None

def process_batch(client, system_prompt, error_log, batch):
    """带统计的批次处理"""
    print(f"\n{Colors.BLUE}▶ 开始批次处理 ({len(batch)}个问题) {Colors.END}")
    with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
        process_fn = partial(process_question_wrapper, client, system_prompt, error_log)
        results = list(executor.map(process_fn, batch))

    success = sum(1 for r in results if r)
    failed = len(results) - success
    print(f"{Colors.GREEN}✔ 成功: {success} {Colors.YELLOW}⚠ 失败: {failed}{Colors.END}")
    return [r for r in results if r]

def save_progress(processed_questions, progress_file):
    """保存处理进度"""
    try:
        with open(progress_file, "w", encoding="utf-8") as f:
            json.dump(list(processed_questions), f, ensure_ascii=False)
    except Exception as e:
        print(f"{Colors.RED}❌ 保存进度失败: {str(e)}{Colors.END}")

def load_progress(progress_file):
    """加载处理进度"""
    processed = set()
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r", encoding="utf-8") as f:
                processed = set(json.load(f))
            print(f"{Colors.GREEN}✅ 已加载{len(processed)}条进度数据{Colors.END}")
        except Exception as e:
            print(f"{Colors.YELLOW}⚠ 加载进度失败: {str(e)}{Colors.END}")
    return processed

def main():
    client = ZhipuAI(api_key="4e0779dc66414dc4afe0872680957d40.HnKsmRuaJjYQHEUL")
    
    # 修改后的系统提示（关键改进）
    system_prompt = """
作为化工安全与工艺专家，请按照以下格式生成回答：

<think>
请进行系统性的专业思考，从以下维度展开分析（根据问题相关性选择重点维度）：

1. 技术背景分析
   - 问题涉及的具体化工工艺或设备
   - 相关化学反应原理和热力学特性
   - 关键工艺参数和操作条件

2. 安全风险评估
   - 物质危险性（毒性、易燃性、反应性等）
   - 工艺过程风险点识别
   - 潜在事故场景分析
   - 后果严重程度评估

3. 法规标准要求
   - 适用的国家标准和行业规范
   - 安全生产相关法规要求
   - 职业健康与环境保护标准

4. 工程实践考量
   - 设备选型和工艺设计要点
   - 安全防护措施和工程控制
   - 监测预警系统配置
   - 应急响应设施要求

5. 管理控制措施
   - 操作规程和作业指导
   - 人员培训和资质要求
   - 日常检查和维护制度
   - 变更管理流程

6. 应急预案设计
   - 事故分级响应机制
   - 应急处置流程
   - 救援资源配置
   - 恢复重建方案

请在思考过程中：
- 引用具体的技术参数和标准要求
- 考虑实际工程实施的可行性
- 分析不同方案的优缺点
- 评估控制措施的有效性
- 回答的内容尽量具体，不要使用标题，直接以自然语言呈现关键点，确保回答专业、实用、全面。
</think>

基于上述分析，给出专业、实用、可操作的解决方案，确保：
1. 回答结构清晰，回答内容尽量具体且重点突出
2. 建议具体可行，有数据支撑
3. 安全措施全面，符合规范
4. 考虑实际应用场景
"""

    # 文件配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    question_file = os.path.join(base_dir, "extracted_10000_questions.py")
    output_file = os.path.join(base_dir, "chemical_safety_deepseek_10k.json")
    error_log = os.path.join(base_dir, "deepseek_errors_10k.log")
    progress_file = os.path.join(base_dir, "progress_10k.json")

    # 检查是否需要清理错误日志
    if os.path.exists(error_log) and os.path.getsize(error_log) > 0:
        # 创建备份
        backup_error_log = f"{error_log}.{time.strftime('%Y%m%d%H%M%S')}.bak"
        os.rename(error_log, backup_error_log)
        print(f"{Colors.YELLOW}⚠ 已备份旧错误日志到 {backup_error_log}{Colors.END}")
        # 创建新的空日志文件
        with open(error_log, "w", encoding="utf-8") as f:
            f.write(f"# 错误日志 - 创建于 {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    # 加载数据
    processed = load_existing_data(output_file)
    progress = load_progress(progress_file)
    processed.update(progress)  # 合并已处理的问题
    
    all_questions = load_questions(question_file)
    todo_questions = [q for q in all_questions if q not in processed]
    
    print(f"{Colors.BLUE}📊 待处理问题：{len(todo_questions)}/{len(all_questions)}{Colors.END}")
    
    if not todo_questions:
        print(f"{Colors.GREEN}✅ 所有问题已处理完成{Colors.END}")
        return

    # 分批处理
    batch_size = 200
    for idx in range(0, len(todo_questions), batch_size):
        batch = todo_questions[idx:idx+batch_size]
        print(f"\n{Colors.BLUE}🔷 处理批次 {idx//batch_size + 1} [数量：{len(batch)}]{Colors.END}")
        
        results = process_batch(client, system_prompt, error_log, batch)
        
        if results:
            save_with_backup(results, output_file)
            # 更新进度
            processed.update(batch)
            save_progress(processed, progress_file)
            print(f"{Colors.GREEN}✅ 已保存{len(results)}条数据{Colors.END}")
            
            # 打印进度
            progress = len(processed) / len(all_questions) * 100
            print(f"{Colors.BLUE}📈 总进度: {progress:.1f}%{Colors.END}")

if __name__ == "__main__":
    main()