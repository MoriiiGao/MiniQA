from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Episode:
    """
    该类用于保存一次策略学习/生成任务中的完整轨迹
    Args:
        prefix: 用户输入的问题
        text: 完整的文本序列(text + response)
        prefix_token_ids: prefix对应的ID序列
        prefix_tokens: prefix对应的token字符串
        generated_token_ids: 模型实际生成的 token ID（不包含 prefix，只包含 response 部分）
        is_finished: 	表示本次生成是否正常完成（例如是否遇到 <eos> 停止符）
        reward: 当前 text 得到的奖励值，供 RL 或 GRPO 学习使用
        reward_info: 	用于记录奖励的细节信息（例如多个 reward 组成部分），方便调试和可视化
    """

    prefix: str
    text: str
    prefix_token_ids: List[int]
    prefix_tokens: List[str]
    generated_token_ids: List[int]
    is_finished: bool
    reward: float
    reward_info: Dict[str, float]


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[List[int]]
    target: List[int]