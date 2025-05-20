import re

# 1. 结构完整性分
def structure_score(text):
    required_fields = [
        "作战任务（Task）", "时间框架（TimeFrame）", "作战地域与环境（Environment）",
        "敌方态势（EnemySituation）", "己方态势（OwnSituation）",
        "初始部署（InitialDeployment）", "关键假设（Assumptions）",
        "限制与约束（Constraints）", "关键事件或作战阶段（KeyEvents）"
    ]
    has_think = bool(re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL))
    has_answer = bool(re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL))
    answer_block = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    fields_covered = 0
    if answer_block:
        for field in required_fields:
            if f"{field}：" in answer_block.group(1):
                fields_covered += 1
    return 0.5 * has_think + 0.5 * has_answer + fields_covered / len(required_fields) / 2  # [0, 1]

# 2. 内容质量分
def quality_score(text):
    answer_block = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    if not answer_block:
        return 0.0
    answer = answer_block.group(1)
    # 判定字段非空且有效
    non_empty_fields = len(re.findall(r"：([^\n]*)", answer))
    valid_fields = len([m for m in re.findall(r"：([^\n]*)", answer) if m.strip() and m.strip() not in ["暂无情报", "见上", "略"] and len(m.strip()) >= 5])
    return valid_fields / (non_empty_fields or 1)

# 3. 输出要素分（推理链、专有词、无重复）
def factor_score(text):
    think_block = re.search(r"<think>(.*?)</think>", text, flags=re.DOTALL)
    keywords = ["战略", "威胁", "部署", "态势", "兵力", "假设", "约束", "关键事件", "机动", "火力"]
    think_text = think_block.group(1) if think_block else ""
    factor_hit = sum([1 for k in keywords if k in think_text])
    total_hit = min(factor_hit, 5) / 5  # 最多给满分
    
    answer_block = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    content_lines = re.findall(r"：([^\n]*)", answer_block.group(1)) if answer_block else []
    unique_lines = len(set([l.strip() for l in content_lines if l.strip()])) / (len(content_lines) or 1)
    return 0.5 * total_hit + 0.5 * unique_lines  # 两部分平均

# 4. 综合奖励
def composite_reward(text, w1=0.35, w2=0.35, w3=0.3):
    s = structure_score(text)
    q = quality_score(text)
    f = factor_score(text)
    return w1 * s + w2 * q + w3 * f

if __name__ == "__main__":
    sample = """<think>
    本问题需综合考量战略威胁与资源调配，初步判定需对敌方沿海部署展开机动……
    </think>
    <answer>
    - 作战任务（Task）：控制关键海域
    - 时间框架（TimeFrame）：2026年夏季
    - 作战地域与环境（Environment）：东部沿海，复杂地形
    - 敌方态势（EnemySituation）：敌军集结，威胁高
    - 己方态势（OwnSituation）：快速反应部队待命
    - 初始部署（InitialDeployment）：海空兵力前置
    - 关键假设（Assumptions）：敌军不会主动进攻
    - 限制与约束（Constraints）：需遵守国际法
    - 关键事件或作战阶段（KeyEvents）：火力准备、协同突击
    </answer>"""
    print("综合奖励分：", composite_reward(sample))
