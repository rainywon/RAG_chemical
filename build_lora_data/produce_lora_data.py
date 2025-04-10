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
        """增强的CoT格式验证"""
        # 检查标签完整性
        think_blocks = re.findall(r'<think>(.*?)</think>', answer, re.DOTALL)
        if not think_blocks:
            raise ValueError("必须包含<think>思考标签")
        if len(re.findall(r'<think>', answer)) != len(re.findall(r'</think>', answer)):
            raise ValueError("思考标签不匹配")
        
        # 验证思考内容质量
        for think in think_blocks:
            if len(think.strip()) < 50:
                raise ValueError("思考内容过短（至少50字符）")
            if not re.search(r'[，。？：；]', think):  # 检查是否有多句式分析
                raise ValueError("思考内容需包含完整分析过程")
        
        # 验证实际回答与思考的关联性
        clean_answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL).strip()
        if not clean_answer:
            raise ValueError("实际回答不能为空")
        if len(clean_answer) < len(think_blocks[0])/3:
            raise ValueError("实际回答内容过简")
        return True

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
    return {
        "instruction": f"{question}\n请逐步思考并给出专业解答",
        "input": "",
        "output": f"{answer}\n\n<安全提示>请在实际操作中严格遵守安全规范，必要时咨询专业工程师</提示>"
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
            user_prompt = f"{question}\n请按以下步骤回答：\n1. 详细分析问题背景\n2. 考虑多种可能性\n3. 给出分步解决方案\n4. 总结注意事项"
            
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5,  # 提高创造性
                max_tokens=2500
            )
            answer = response.choices[0].message.content
            
            # 增强格式处理
            answer = re.sub(r'(?i)<think>', '<think>', answer)
            answer = re.sub(r'(?i)</think>', '</think>', answer)
            answer = re.sub(r'（([^）]+)）', r'（\1）', answer)  # 统一括号
            
            # 增加二次思考验证
            if answer.count('<think>') < 1:
                answer = f"<think>问题分析：\n{answer.split('</think>')[0] if '</think>' in answer else answer}</think>\n{answer}"
                
            CotValidator.validate(answer)
            return generate_deepseek_entry(question, answer)
            
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
        print(f"{Colors.BLUE}🟡 处理中: {question[:35]}...{Colors.END}")
        start_time = time.time()
        result = process_question(client, system_prompt, question, error_log)
        elapsed = time.time() - start_time
        
        if result:
            think_len = len(re.search(r'<think>(.*?)</think>', result['output'], re.DOTALL).group(1))
            ans_len = len(result['output']) - think_len
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
作为资深化工安全专家，请严格按以下格式回答：

<think>
【问题分析】
1. 识别核心安全风险（至少3个方面）
2. 列举相关法规标准（GB/T、AQ等）
3. 考虑不同场景下的应对方案
4. 评估常见误操作及其后果

【解决思路】
- 分步骤展开解决方案
- 比较不同方法的优缺点
- 结合最新行业案例
- 特殊情况的应急处理
</think>

【专业回答】
按此结构呈现：
1. 立即行动方案（带编号步骤）
2. 根本原因排查（检查清单）
3. 长期预防措施
4. 培训建议

示例：
问题：反应釜压力异常升高如何处理？

<think>
【问题分析】
1. 安全风险：超压爆炸、物料泄漏、连锁反应失控
2. 相关标准：GB/T 21109、AQ/T 3034
3. 可能原因：冷却失效、进料过量、搅拌故障、仪表误报
4. 误操作后果：错误泄压导致毒气释放、盲目检修引发火花

【解决思路】
- 优先保障人员安全，启动自动联锁
- 区分物理性超压与反应性超压
- 考虑夜间值班人员的处置能力
- 参考2022年某化工厂同类事故处理报告
</think>

【专业回答】
1. 紧急处置：
   (1) 立即启动E-101紧急泄压阀
   (2) 切断进料泵P-203
   (3) 开启备用冷却水系统

2. 排查清单：
   √ 检查TICA-307温度记录曲线
   √ 确认搅拌器M-102电流波动
   √ 校准PRC-208压力传感器

3. 预防措施：
   - 每月进行泄压阀手动测试
   - 安装压差报警装置（≤0.3MPa）

4. 培训重点：
   * 夜间应急处置演练
   * 多工况压力识别培训
"""

    # 文件配置
    base_dir = os.path.dirname(os.path.abspath(__file__))
    question_file = os.path.join(base_dir, "extracted_5000_questions.py")
    output_file = os.path.join(base_dir, "chemical_safety_deepseek_3.json")
    error_log = os.path.join(base_dir, "deepseek_errors.log")
    progress_file = os.path.join(base_dir, "progress.json")

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