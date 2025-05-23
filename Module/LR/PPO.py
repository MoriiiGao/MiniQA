import numpy as np
import torch

def input_data():
    
    batch_size = 2
    
    log_probs = np.array([
        [-1.0, -2.0, -3.0], 
        [-0.5, -1.5, 0.0]    
    ], dtype=np.float32)

    ref_log_probs = np.array([
        [-0.8, -1.5, -2.5],
        [-0.6, -1.2, 0.0]
    ], dtype=np.float32)

    # 来自奖励模型的原始分数 (未经裁剪)
    raw_reward_scores = np.array([7.0, -6.0]) 

    # Action掩码标记有效token位置 (1有效，0padding)
    action_mask = np.array([
        [1, 1, 1],  # 第一条数据实际生成长度3
        [1, 1, 0]    # 第二条数据实际生成长度2
    ], dtype=np.int32)

    return log_probs, ref_log_probs, raw_reward_scores, action_mask

def compute_rewards(
        log_probs, 
        ref_log_probs, 
        reward_score, 
        action_mask, 
        kl_ctl=0.1, 
        clip_reward_value=5):
    """
    PPO奖励信号函数
    Args:
        log_probs: LLM输出对response各token的log概率（batch, seq_len）
        ref_log_probs: 参考模型对response各token的log概率（batch， seq_len）
        reward_score: 外部奖励得分(batch,) 每条数据一个标量
        action_mask: 有效token的掩码（batch, seq_len） 1表示有效 0表示padding
        kl_ctl: KL惩罚项系数
        clip_reward_values: 奖励裁剪阈值

    对于batch中某个prompt来说，它最终的reward分数为：
    （1）先计算actor和ref_model的prompt的logits相似度：-kl_ctl * (log_probs - ref_log_probs)
    这个值越大，说明ref_model对actor生成的结果的认可度越高（表明rlhf没有训歪）
    没有训练歪的情况下，我们应该给予模型一些奖励，这个奖励就是 kl_ctl * (ref_log_probs - log_probs)

    （2）由于我们只取最后一个token对应位置的分数为reward_score，因此我们只需要：
    kl_ctl * (ref_log_probs - log_porbs)的最后一位 + reward_score

    （3）同时我们对reward_score也做了大小限制，最大不超过clip_reward_value（超过统一给成clip_reward_value）
    最小不低于-clip_reward_value（低于统一给成clip_reward_value）

    （4）最后返回rewards大小为：（batch_size, 各条数据的长度），对batch中的每条数据来说：
        - response的最后一位： kl_ctl * (ref_log_probs - log_probs)的最后一位 + reward_score
        - response的其余位置： kl_ctl * (ref_log_probs - log_probs)

    ** 奖励信号的分配问题：
    （1）如果将整体奖励平均分配到每个token，会导致信用分配问题，模型无法区分哪些roken对最终结果的贡献更大。
    （2）解决方案：将整体奖励直接附加到最后一个token的位置：
        - 最后一个token标志着response的结束，是生成过程的最重点
        - 在序列生成任务中，最终token的概率分布通常与整个序列的完整性相关
    
    ** 最终token的特殊性：
    （1）最终token的奖励 = KL惩罚 + 外部奖励：
        - kl_ctl * (ref_log_probs - log_probs): 保持与参考模型的一致性
        - reward_score: 反应人类偏好或任务目标的整体奖励
    （2）为什么只在最后一个token加：
        - 如果所有的token都加reward_score，会导致重复计算。因为reword_model得出的reward_score是对整体response的分数。
        - 最终token的位置是唯一能够明确关联到完整序列的位置
    """

    # 计算KL奖励 ref_log_probs/log_probs shape: [batch_size, seq_len]
    kl_reward = kl_ctl * (ref_log_probs - log_probs)

    # 2.找到每条数据最后一个有效token的位置
    seq_lengths = np.sum(action_mask, axis=1)
    batch_size, seq_len = log_probs.shape

    # 3.初始化最终奖励
    rewards = kl_reward.clone()

    # 4.处理最后的一个token
    for i in range(batch_size):
        last_pos = seq_lengths[i].item()
        # 裁剪外部reward
        r = torch.clamp(reward_score[i], -clip_reward_value, clip_reward_value)
        # 对第i条 kl散度的last_pos位置加上 r（人类偏好） 
        rewards[i, last_pos] += r
    
    return rewards

def get_advantages_and_returns(values, rewards, start):
    """
    1.优势：
    - 优势是在装S_{t}做出动作 A_{t}得到的回报，公式如下：
        A_{t} = Q(S_{t}, A_{t}) - V_{S_{t}}
     - Q(S_{t}, A_{t}):从状态S_{t} 采取A_{t}，后续所有回报的期望
     - V(S_{t}):状态 S_{t}下所有可能动作的平均回报

    2.基础时序差分（TD）版本
        A_{t} = r_{t} + gamma * V_{t+1} - V_{t}
     
     - r_{t}:t时刻的即时收益
     - gamma：折扣因子（0~1），衡量未来奖励的重要性
     - V_{t+1}:表示未来时刻的预期收益
     - V_{t}:可理解成t时刻的预估收益（是模型，如critic model自己估算出来的）
     - r_{t} + gamma * V_{t+1}：可理解成t时刻的实际预期收益

    3.为什么要引入GAE（广义优势估计）
        直接用前两步的Advantage信号，虽然简单，但噪声大、学习不问。
        GAE通过“加权平均多步TD”，即降低了噪声（方差），又能保持较小的偏差

        A^{GAE}_{t} = delta_{t} + gamma * lambda * delta_{t+1} + (gamma * lambda) ^ {2} * delta_{t+2} + ...
        delta_{t} = r_{t} + gamma * V_{t+1} - V_{t} 

        递归式写法：
            A^{GAE}_{t} = delta_{t} + gamma * lambda * A^{GAE}_{t+1}
            即 “当前优势 = 当前TD误差 + 折扣后的未来优势”

    """

if __name__ == '__main__':

    log_probs, ref_log_probs, raw_reward_score, action_mask = input_data()
    compute_rewards(log_probs, ref_log_probs, raw_reward_score, action_mask)