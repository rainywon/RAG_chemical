import threading
import requests
import json
import random
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import hashlib

# === 全局配置 ===
API_CONFIG = {
    "ENDPOINT": "https://open.bigmodel.cn/api/paas/v4/chat/completions",
    "KEY": "4e0779dc66414dc4afe0872680957d40.HnKsmRuaJjYQHEUL",
    "MAX_TOKENS": 300,  # 增加token限制
    "TEMP": 0.5,  # 降低温度提高稳定性
    "BATCH_SIZE": 30,  # 减小批次大小提高质量
    "OUTPUT_FILE": "build_lora_data/random_22000_questions.py",
    "CONCURRENCY": 12,  # 降低并发数
    "API_TIMEOUT": 15,  # 增加超时时间
    "API_RETRY": 2,  # 增加重试次数
    "RATE_LIMIT_THRESHOLD": 15,  # 提高速率限制阈值
    "TOPICS": [
        ("化工现场实操", [
            "装置开停车操作", "检修安全隔离", "临时作业许可", 
            "设备密封点检查", "管线吹扫置换", "罐区安全切水",
            "阀门维修调试", "仪表零点校准", "取样标准动作", 
            "现场应急演练", "消防设施测试", "警示标识张贴",
            "工具归类存放", "安全带悬挂点", "有毒气体检测",
            "动设备润滑管理", "静设备防腐措施", "电气设备巡检",
            "工艺参数调整", "DCS操作规范", "现场交接班管理",
            "安全设施维护", "应急物资检查", "受限空间管理",
            "高处作业监护", "动火作业管控", "临时用电管理",
            "起重吊装指挥", "脚手架搭拆", "射线探伤防护"
        ]),

        ("基层员工安全", [
            "岗位安全培训", "风险辨识能力", "个人防护用品", 
            "劳动保护措施", "身心健康防护", "安全奖惩制度",
            "班组安全文化", "事故案例学习", "安全生产责任", 
            "安全隐患举报", "危险化学品知识", "班前班后会",
            "职业病防护", "安全作业证", "工作票管理",
            "新员工三级教育", "转岗培训要求", "特种作业取证",
            "安全技能竞赛", "应急能力评估", "安全行为观察",
            "安全经验分享", "安全合理化建议", "安全绩效考评",
            "安全知识竞赛", "安全技能比武", "安全文化建设",
            "安全责任落实", "安全奖惩兑现", "安全考核评价"
        ]),

        ("问题设备处理", [
            "泵轴承过热", "阀门泄漏处理", "换热器堵塞", 
            "压缩机振动", "反应釜结焦", "储罐底部沉积",
            "管道冻裂预防", "搅拌器机封", "塔板损坏", 
            "过滤器压差大", "脱硫塔腐蚀", "安全阀粘连",
            "仪表失灵判断", "控制阀卡死", "电机过载保护",
            "离心泵气蚀", "往复泵脉动", "螺杆泵磨损",
            "换热器结垢", "冷凝器泄漏", "蒸发器结焦",
            "反应器飞温", "搅拌器断轴", "塔器偏流",
            "管道振动", "法兰泄漏", "垫片失效",
            "安全阀起跳", "爆破片破裂", "呼吸阀堵塞"
        ]),

        ("化工过程安全", [
            "HAZOP偏差矩阵", "LOPA场景切片", "SIL验算蒙特卡洛",
            "泄放系统两相流", "联锁旁路时间", "本质安全层独立",
            "人为失误THERP", "多米诺概率计算", "安全仪表共因",
            "保护层分析", "安全阀口径", "泄爆面设计",
            "工艺危害分析", "变更管理评估", "开车前安全检查",
            "工艺安全信息", "机械完整性", "操作程序管理",
            "承包商管理", "事故调查分析", "应急响应计划",
            "工艺安全审计", "安全绩效指标", "安全文化建设",
            "工艺安全培训", "安全经验分享", "安全奖惩制度",
            "工艺安全标准", "安全操作规程", "安全管理制度"
        ]),

        ("危化品管理", [
            "MSDS合规性", "禁忌物料识别", "相容性管理",
            "微量分装安全", "过期化学品处理", "剧毒物管理",
            "纳米材料控制", "自反应物质存储", "临界量监测",
            "管输化学品控制", "电化学腐蚀防护", "聚合抑制剂监测",
            "危险化学品分类", "安全标签管理", "安全技术说明书",
            "储存条件控制", "使用量控制", "废弃处理规范",
            "泄漏应急处置", "火灾扑救方法", "中毒急救措施",
            "个人防护装备", "通风系统管理", "监测报警系统",
            "安全操作规程", "应急预案制定", "安全培训教育",
            "安全检查制度", "安全责任落实", "安全奖惩制度"
        ]),

        ("化工应急管理", [
            "应急预案编制", "应急演练组织", "应急物资管理",
            "事故初期处置", "人员疏散指挥", "泄漏应急处置",
            "火灾扑救方法", "中毒急救措施", "环境污染控制",
            "应急通讯保障", "应急照明系统", "应急电源管理",
            "应急指挥体系", "应急响应程序", "应急资源调配",
            "应急演练评估", "应急培训教育", "应急能力建设",
            "应急物资储备", "应急装备维护", "应急设施管理",
            "应急通讯系统", "应急照明系统", "应急电源系统",
            "应急疏散通道", "应急避难场所", "应急医疗救护",
            "应急环境监测", "应急污染控制", "应急恢复重建"
        ]),
        
        ("典型化工工艺", [
            "加氢反应控制", "聚合反应安全", "氧化工艺管理", 
            "精馏操作规程", "蒸馏系统维护", "结晶工艺参数",
            "干燥设备除尘", "萃取工艺优化", "气固反应控制", 
            "中和反应pH值", "催化剂再生", "硝化工艺冷却",
            "氯化反应抑制", "吸收塔效率", "MVR蒸发器",
            "磺化工艺控制", "胺化反应安全", "酯化工艺优化",
            "水解反应控制", "缩合反应安全", "重氮化工艺管理",
            "烷基化工艺", "酰化反应控制", "环化反应安全",
            "异构化工艺", "脱水反应控制", "脱氢反应安全",
            "羰基化工艺", "氢化反应控制", "氧化还原反应"
        ]),
        
        ("特殊作业安全", [
            "有限空间作业", "高处作业防护", "动火作业管控", 
            "吊装作业规范", "挖掘作业标准", "带电作业措施",
            "盲板抽堵作业", "射线探伤防护", "高压水清洗", 
            "低温作业防冻", "保温拆除作业", "脚手架搭建",
            "检维修作业票", "临时用电管理", "交叉作业协调",
            "受限空间监护", "高处作业监护", "动火作业监护",
            "吊装作业指挥", "挖掘作业监护", "带电作业监护",
            "盲板抽堵监护", "射线探伤监护", "高压水清洗监护",
            "低温作业监护", "保温拆除监护", "脚手架搭拆监护",
            "检维修作业监护", "临时用电监护", "交叉作业监护"
        ]),
        
        ("设备完整性", [
            "压力容器检验", "管道完整性管理", "法兰密封选型", 
            "阀门泄漏等级", "焊缝无损检测", "设备寿命评估",
            "腐蚀监测点位", "在线检测技术", "管线防腐措施", 
            "热交换器检查", "储罐底板检测", "垫片材质选择",
            "机泵振动分析", "压力表校验", "安全附件测试",
            "设备台账管理", "设备档案管理", "设备状态监测",
            "设备维护保养", "设备检修管理", "设备更新改造",
            "设备故障分析", "设备寿命预测", "设备风险评估",
            "设备安全评估", "设备可靠性分析", "设备可用性分析",
            "设备维护策略", "设备检修策略", "设备更新策略"
        ])
    ],
    "TEMPLATES": [  # 专业化的提示模板
        # 技术问题
        "在{topic}过程中，{aspect}出现异常，请分析可能的原因及处理方案。",
        "针对{topic}中的{aspect}问题，请提供专业的技术指导。",
        "关于{topic}的{aspect}控制，请说明关键参数和注意事项。",
        
        # 安全评估
        "请评估{topic}中{aspect}的安全风险等级。",
        "分析{topic}的{aspect}环节可能存在的安全隐患。",
        "请说明{topic}的{aspect}安全控制要点。",
        
        # 标准规范
        "请说明{topic}的{aspect}相关标准规范要求。",
        "分析{topic}中{aspect}的合规性要求。",
        "请提供{topic}的{aspect}技术标准参考。",
        
        # 工艺优化
        "请分析{topic}中{aspect}的优化方案。",
        "针对{topic}的{aspect}问题，请提供改进建议。",
        "请说明{topic}的{aspect}工艺参数优化方法。",
        
        # 设备管理
        "请分析{topic}设备{aspect}故障的原因及预防措施。",
        "说明{topic}的{aspect}设备维护保养要点。",
        "请提供{topic}的{aspect}设备选型建议。",
        
        # 应急管理
        "请说明{topic}中{aspect}的应急处置方案。",
        "分析{topic}的{aspect}应急预案要点。",
        "请提供{topic}的{aspect}应急响应流程。",
        
        # 培训教育
        "请说明{topic}的{aspect}培训重点内容。",
        "分析{topic}中{aspect}的培训需求。",
        "请提供{topic}的{aspect}培训方案建议。",
        
        # 工艺控制
        "请分析{topic}的{aspect}控制要点。",
        "说明{topic}中{aspect}的关键控制参数。",
        "请提供{topic}的{aspect}工艺控制方案。",
        
        # 安全管理
        "请说明{topic}的{aspect}安全管理要求。",
        "分析{topic}中{aspect}的安全风险控制措施。",
        "请提供{topic}的{aspect}安全管理建议。",
        
        # 技术评估
        "请评估{topic}的{aspect}技术可行性。",
        "分析{topic}中{aspect}的技术难点。",
        "请提供{topic}的{aspect}技术方案建议。"
    ],
    "INVALID_PATTERNS": [
        r"请用代码|示意图|定义|包括以下|例如|\d\.|选择",
        r"总结|要点|步骤|流程|注意事项|建议|可分为",
        r"[\(（][^\)）]{10,}[\)）]",  # 过滤长括号内容
        r".{100,}的[吗呢吧]",  # 修改为允许更长的问题
        r"^请问|^请告诉我|^我想知道",  # 过滤太过正式的开头
        r"感谢|谢谢",  # 过滤结尾客套语
        r"兄弟|哥们|咱们|咋整|闹心|糟了",  # 过滤口语化表达
        r"啊|呀|呢|吧|嘛|哈",  # 过滤语气词
        r"赶紧|马上|立刻|赶快",  # 过滤过于急切的表达
        r"不是闹着玩的|可不是小事|这事儿得重视"  # 过滤夸张表达
    ]
}

# === 预编译正则表达式 ===
INVALID_REGEX = [re.compile(p) for p in API_CONFIG["INVALID_PATTERNS"]]
QUESTION_END_REGEX = re.compile(r'[?？]$')


# === 核心引擎 ===
class HighSpeedGenerator:
    def __init__(self):
        self.lock = threading.Lock()
        self.seen = set()
        self.questions = []
        self.stats = {"total": 0, "valid": 0, "retries": 0, "incomplete": 0}
        self.save_interval = 50  # 每50个问题保存一次
        self.last_save_count = 0

        # 主题缓存优化
        self.topic_pool = []
        for topic, aspects in API_CONFIG["TOPICS"]:
            self.topic_pool.extend([(topic, a) for a in aspects])

        # 模板缓存
        self.templates = API_CONFIG["TEMPLATES"]

    def generate_prompt(self):
        """优化提示生成速度"""
        topic, aspect = random.choice(self.topic_pool)
        return random.choice(self.templates).format(topic=topic, aspect=aspect)

    def validate_question(self, question):
        """增强问题验证"""
        # 基本长度检查
        if not (20 <= len(question) <= 360):
            return False
            
        # 完整性检查
        if question.count("，") > 3 or question.count("。") > 1:  # 避免过长或分段
            return False
            
        # 问号检查
        contains_question_mark = QUESTION_END_REGEX.search(question)
        contains_question_words = re.search(r'怎么|如何|什么|为什么|哪些|多少|是否|可以|能否|需要|该不该|建议|方法|步骤|标准|操作', question)
        
        if not (contains_question_mark or contains_question_words):
            return False
            
        # 其他验证规则
        if any(rgx.search(question) for rgx in INVALID_REGEX):
            return False
            
        # 真实性检查
        if re.search(r'我要求你|请生成|请提供一个|我命令你', question):
            return False
            
        # 完整性检查
        if len(question) < 10 or question.endswith("...") or question.endswith("等"):
            self.stats["incomplete"] += 1
            return False
            
        return True

    def api_request(self):
        """优化API请求"""
        for retry in range(API_CONFIG["API_RETRY"]):
            try:
                response = requests.post(
                    API_CONFIG["ENDPOINT"],
                    headers={"Authorization": f"Bearer {API_CONFIG['KEY']}"},
                    json={
                        "model": "glm-4-flash",
                        "messages": [
                            {"role": "system", "content": "你是一名化工安全领域的工程师。生成一个真实的化工安全问题，使其听起来像是一线工作人员在实际工作中提出的。问题应该简洁、直接，体现出工作中的紧迫性或实际问题，不要使用过于正式或教科书式的语言。可以包含工厂常见的简写术语。请确保问题完整，不要出现省略号或未完成的情况。"},
                            {"role": "user", "content": self.generate_prompt()}
                        ],
                        "temperature": API_CONFIG["TEMP"],
                        "max_tokens": API_CONFIG["MAX_TOKENS"]
                    },
                    timeout=API_CONFIG["API_TIMEOUT"]
                )

                # 动态速率控制
                remain = int(response.headers.get('X-RateLimit-Remaining', 100))
                if remain < API_CONFIG["RATE_LIMIT_THRESHOLD"]:
                    time.sleep(max(0.5, 15 / remain))  # 增加等待时间

                content = response.json()["choices"][0]["message"]["content"].strip()
                # 清理不完整的问题
                content = re.sub(r'\.{3,}$|等$', '', content)
                return content.replace('"', '')

            except Exception as e:
                if retry < API_CONFIG["API_RETRY"] - 1:
                    time.sleep(0.5 * (retry + 1))  # 增加重试等待时间
                    self.stats["retries"] += 1
                else:
                    return None
        return None

    def process_batch(self):
        """优化批量处理"""
        with ThreadPoolExecutor(max_workers=API_CONFIG["CONCURRENCY"]) as executor:
            futures = [executor.submit(self.api_request)
                       for _ in range(API_CONFIG["BATCH_SIZE"])]

            batch = []
            for future in as_completed(futures):
                if (resp := future.result()) and self.validate_question(resp):
                    q_hash = hashlib.sha1(resp.encode()).hexdigest()
                    if q_hash not in self.seen:
                        batch.append(resp)
                        self.seen.add(q_hash)
                        self.stats["valid"] += 1
                    self.stats["total"] += 1
            return batch

    def save_results(self):
        """安全编码的保存方法"""
        with self.lock:
            with open(API_CONFIG["OUTPUT_FILE"], "w", encoding="utf-8") as f:
                f.write("questions = [\n")
                # 使用json.dumps确保正确转义
                formatted = [f'    {json.dumps(question, ensure_ascii=False)}'
                             for question in self.questions]
                f.write(",\n".join(formatted))
                f.write("\n]")
            print(f"已保存{len(self.questions)}个问题")

    def check_and_save(self):
        """检查是否需要保存"""
        if len(self.questions) - self.last_save_count >= self.save_interval:
            self.save_results()
            self.last_save_count = len(self.questions)


# === 优化后的主程序 ===
def main():
    gen = HighSpeedGenerator()
    target = 22000

    with tqdm(total=target, desc="生成问题", unit="问") as pbar:
        with ThreadPoolExecutor(max_workers=API_CONFIG["CONCURRENCY"]) as executor:
            futures = []

            while len(gen.questions) < target:
                # 检查是否需要保存
                gen.check_and_save()
                
                while len(futures) < API_CONFIG["CONCURRENCY"] * 2:
                    futures.append(executor.submit(gen.process_batch))

                done, _ = as_completed(futures), []
                for f in done:
                    if (batch := f.result()):
                        with gen.lock:
                            add_num = min(target - len(gen.questions), len(batch))
                            gen.questions.extend(batch[:add_num])
                            pbar.update(add_num)
                    futures.remove(f)

    # 最后保存一次
    gen.save_results()
    print(f"生成统计:")
    print(f"- 总请求: {gen.stats['total']}")
    print(f"- 有效问题: {gen.stats['valid']}")
    print(f"- 不完整问题: {gen.stats['incomplete']}")
    print(f"- 重试次数: {gen.stats['retries']}")
    print(f"- 有效率: {gen.stats['valid'] / gen.stats['total']:.1%}")


if __name__ == "__main__":
    main()
