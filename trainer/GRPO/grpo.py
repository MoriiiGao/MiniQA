"""
该脚本实现了一个基于GRPO算法的强化学习框架，用于微调一个Transformer语言模型
以生成高奖励文本，主要流程：
[Prompt 数据] --> rollout (采样生成) --> [Episode] --> normalize_rewards --> update_policy (梯度更新)

"""
import dataclasses
import gc
import math
from collections import defaultdict
from typing import Callable, List

import numpy as np
import torch

from dataset.data_type import Episode, MiniBatch
from model.qwen2_model import Transformer
from dataset.tokenizer import Tokenizer
import torch.nn.functional as F

@torch.no_grad()
def rollout(
    model: Transformer,
    batch: MiniBatch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype
) -> List[Episode]:
    """
    轨迹采样（采样生成回答）
    对模型进行采样生成，形成一批回答，以及评估他们的奖励
    Args：
        model: 当前 Transformer 模型。
        batch: 输入提示语（prompt）批次。
        tokenizer: 用于token和text之间的转换。
        max_gen_len: 最长生成长度。
        num_answer_per_question: 每个问题生成多少个回答。
        reward_function: 用于对生成内容打分的函数。
        device, dtype: 用于模型运行的设备和精度。
    关键步骤：
        构造填充后的 token 张量。
        逐步生成 token，直到达到最大长度或生成结束符。
        利用 softmax + 采样（multinomial）获取每一步的生成 token。
        判断生成是否结束（是否遇到 eos_token）。
        生成完成后，将文本 detokenize，调用 reward 函数评估生成结果。
        最终返回一个包含每条轨迹信息的 Episode 对象列表。
    """
    end_token = tokenizer.eos_token           # 模型输出结束标记 用于判断输出是否结束
    end_token_id = tokenizer.eos_token_id     
    pad_token_id = tokenizer.pad_token_id
    prefix_token_ids = batch.prefix_token_ids # prompt token ids
    batch_size = len(batch.prefix) * num_answer_per_question # 问题条数 * 每个问题生成答案的数量
    min_prompt_len = min(len(t) for t in prefix_token_ids)
    max_prompt_len = max(len(t) for t in prefix_token_ids)
    total_len = max_gen_len + max_prompt_len # 最长的prompt + 最大的生成response
    model.init_kv_cache(
        max_batch_size=batch_size,
        max_seq_len=total_len,
        device=device,
        dtype=dtype
    )

    # 初始化一批样本数据，先用pad_token_id pre-fill
    tokens = torch.full((batch_size, total_len), 
                        pad_token_id, dtype=torch.long, device=device)
    for idx, cur_token in enumerate(prefix_token_ids):
        offset = idx * num_answer_per_question
        for i in range(num_answer_per_question):
            # 从开始位置 将每条数据的prompt进行一个填入
            tokens[offset + i, :len(cur_token)] = torch.tensor(
                cur_token, dtype=torch.long, device=device
            )

    ### 构建模型推理的初始状态
    # 当前模型生成到的位置（初始为0）
    prev_pos = 0 
    # 构建一个bool mask张量 形状为[batch_size, total_len],有效token的位置为True，padding的地方为Fasle
    input_text_mask = tokens != pad_token_id
    # 检查最小的prompt长度不能超过总长度 
    assert min_prompt_len < total_len
    # 初始化bool向量，表示每个样本是否完成，开始所有都设为False，后续每次生成，如果遇到eos_token，则is_finished为True
    is_finished = torch.zeros((batch_size, ), dtype=torch.bool, device=device)

    ### 下面要在给定前缀tokens的基础上，逐步生成新的token，知道完成或达到最大长度
    for cur_pos in range(min_prompt_len, total_len):
        # 用于显示当前的进度状态
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        # 模型进行混合推理，使用autocast是为了混合精度加速
        with torch.autocast(device_type=device.type, dtype=dtype):
            # tokens.shape:[batch_size, cur_pos - prev_pos] 
            # logits.shape:[batch_size, cur_pos - prev_pos, vocab_size]
            logits = model.inference(tokens[:, prev_pos: cur_pos])
        # 从logits获取token分布, 只关心当前生成的最后一个位置, shape:[batch_size, vocab_size]
        probs = torch.softmax(logits[:, -1], dim=-1)
        # 从softmax概率中 采样下一个token
        next_token = torch.multinomial(probs, num_samples=1) # [batch_size, 1]
        next_token = next_token.reshape(-1) # shape:[batch_size]
        # 保留已有的文本，避免被覆盖
        # 基于input_text_mask在cur_pos的bool值进行判断，如果为true，next_token填入tokens[:, cur_pos]
        # 如果为False，填入模型采样结果next_token
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        next_token = torch.where(is_finished, pad_token_id, next_token)
        # 写入当前token到token序列中
        tokens[:, cur_pos] = next_token
        # 判断哪些序列已经生成结束了（即生成了eos_token）,并更新is_finished 状态
        if end_token_id is not None:
            is_end_token = next_token == end_token_id # [batch_size] 判断next的token True or False
            is_generated_token = ~input_text_mask[:, cur_pos] # ~表示结果取反 shape:[batch_size]
            is_finished = is_finished | (is_end_token & is_generated_token) 
            
        # 更新prev_pos 并判断是否所有sample已经结束
        prev_pos = cur_pos
        if is_finished.all(): # 都为True，跳出循环
            break

    model.del_kv_cache()
    gc.collect() # 调用Python的垃圾回收机制，释放无用内存对象
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist() # tensor[batch_size] -> list
    tokens_list = tokens.tolist() # [batch_size, total_len] -> list

    ### 收尾阶段
    # 1.将模型生成的token转换成文本
    # 2.调用reward函数对文本评分
    # 3.打包为Episode对象，供后续的策略更新使用

    episodes = []
    # i表示第几个原始问题 (batch_size // num_answer_per_question)表示prompt数量
    for i in range(batch_size // num_answer_per_question):
        # 每个prompt都有num_answer_per_question个问题
        for j in range(num_answer_per_question):
            # idx表示第i个问题的第j个回答
            idx = i * num_answer_per_question + j
            # 获取生成内容的token序列 不包括prompt
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i])]
            # 移除padding token
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[
                    : generated_token_ids.index(pad_token_id)
                ]
            # detokenize成文本 将token is -> text str
            generated_text = tokenizer.detokenize(generated_token_ids)
            # 计算奖励
            rewards = reward_function(
                response=generated_text,
                numbers=batch.numbers[i],
                target=batch.target[i],
                end_token=end_token,
            )
            # 封装为一条完整的轨迹
            episode = Episode(
                prefix=batch.prefix[i],    # 原始输入的prompt
                text=batch.prefix[i] + generated_text, # 完整的文本 = prompt + response
                prefix_token_ids=batch.prefix_token_ids[i], # prompt的token id
                prefix_tokens=batch.prefix_tokens[i], # prompt的token字符串形式
                generated_token_ids=generated_token_ids, # response的token id
                is_finished=is_finished_list[idx], # 生成是否结束
                reward=rewards["reward"], # 打分结果
                reward_info=rewards["reward_info"], # 详细的奖励成分
            )
            episodes.append(episode)
    print("\r", end=" " * 100, flush=True)
    return episodes

def normalize_rewards_per_group(episodes: List[Episode]) -> List[Episode]:
    """
    为了稳定训练，将每组相同(prefix)中的奖励标准化
    将具有相同prompt的一组回答的reward做标准化，生成新的Episode列表
    📌 实现细节
        对每个 prefix（问题）分组。
        每组内做 (reward - mean) / (std + 1e-4)，避免标准差为0。
    🧠 为什么需要 reward 标准化？
    1.多个回答可能具有不同尺度的reward（0.1-100），会导致策略更新的不稳定
    2.归一化reward可以确保优化器处理的advantage在一个相对一致的范围
    3.按prompt分组是因为：每个prompt的reward分布可能不同，必须局部归一化
    """
    # 初始化按prompt分组的字典
    groups = defaultdict(list)
    for epsiode in episodes:
        # 把所有的epsiode按prefix分组
        # tuple(epsiode.prefix)是为了让prefix可以作为字典的key
        '''
        结果：
        groups = {
            ("敌人在哪？",): [episode1, episode2],
            ("天气如何？",): [episode3, episode4],
        }
        '''
        groups[tuple(epsiode.prefix)].append(epsiode)

    # 对每组group内部做归一化
    output = list()
    for group in groups.values():
        group_rewards = [item.reward for item in group]
        mean_reward = np.mean(group_rewards)
        std_reward = np.std(group_rewards)
        for episode in group:
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-4)
            epsiode = dataclasses.replace(epsiode, reward=normalized_reward)
            output.append(epsiode)
    return output


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    计算模型输出输出概率分布的信息上
    token: [batch_size, seq_len, vocab_size]

    """
    probs = F.softmax(logits, dim=-1)
    entropy = torch.logsumexp(probs, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy

def update_policy(model, 
                  optimizer,
                  episodes: List[Episode],
                  micro_batch_size: int,
                  pad_token_id: int,
                  max_grad_norm: float,
                  device: torch.device,
                  dtype: torch.dtype):
    """
    更新策略：对模型进行一次策略梯度更新 使用GRPO思想
    🔧 输入参数
        episodes: 所有生成的样本，包含文本、奖励、token。
        optimizer: 优化器（如 Adam）。
        micro_batch_size: 小批量处理大小。
        pad_token_id: 用于 padding。
        max_grad_norm: 梯度裁剪阈值。
        device, dtype: 设备与精度。
    🧠 关键计算步骤
        1.调用normalize_rewards_per_group()归一化奖励。
        2.按序列长度升序排序（减少 padding，提高训练效率）。
        3.对每个 mini-batch：
        4.构造输入 token 和 mask。
        5.模型前向传播得到 logits。
        6.计算 log_probs：负的交叉熵损失。
        7.计算 token_entropy，用于奖励惩罚。
        8.目标函数 = log_probs * advantage（即归一化后的 reward）。
        9.加权平均后反向传播。
        10.整个 batch 更新一次模型参数（梯度裁剪+step）。
    📈 返回值
        loss: 当前训练 loss。
        grad_norm: 当前梯度范数。
        entropy: 当前 batch 的平均熵（反映模型确定性/探索性）。
    """
    episodes = normalize_rewards_per_group(episodes)
    # 按照每个样本的token总长度升序排序
    episodes.sort(key=lambda x: len(x.prefix_token_ids) + len(x.generated_token_ids))
    # 计算总样本下需要分多少micro-batch来训练
    num_micro_batches = math.ceil(len(episodes) / micro_batch_size)
    # 总共要训练的目标token数（目标生成部分），用于后续的平均loss、平均entropy
    num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes)
    entropy = 0.0

    for i in range(0, len(episodes), micro_batch_size): # 按照micro-batch-size划分
        print(
            f"\r* Computing policy gradient: {i:>2d}/{len(episodes):>2d}",
            flush=True,
            end="",
        )
        # 获取一个microbatch的episode的数据
        j = min(i + micro_batch_size, len(episodes))
        batch_episodes = episodes[i: j]
        batch_lengths = [ # 计算每个episode的总长度
            len(episode.prefix_token_ids) + len(episode.generated_token_ids)
            for episode in batch_episodes
        ]
        # 对一个micro batch的数据进行padding
        batch_max_length = max(batch_lengths)
        batch_token_ids = [
            episode.prefix_token_ids
            + episode.generated_token_ids 
            + [pad_token_id] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(batch_episodes)
        ]
        # 对每个micro batch做mask， 只保留response部分
        batch_masks = [
            [0] * len(episode.prefix_token_ids)
            + [1] * len(episode.generated_token_ids)
            + [0] * (batch_max_length - batch_lengths[i])
            for i, episode in enumerate(episodes)
        ]
        # 计算每个batch的优势
        batch_advantages = [episode.reward for episode in batch_episodes]
        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(
            batch_advantages, device=device, dtype=torch.float32
        )

        # 混合精度推理
        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids = batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[: , 1:]
            target_masks = batch_masks[:, 1:] 
            logits = model.forward(input_token_ids).float()
        
        # 用模型输出的logits和目标target_token_ids做逐token的交叉熵损失，并加负号得到log_probs
        # logits: 模型输出的预测值 [batch_size, seq_len, vocab_size]
        # target_token_ids: 目标token [batch_size, seq_len]
        # log_probs: 每个token的对数概率 [batch_size, seq_len]
        log_probs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), # [batch_size, seq_len, vocab_size] -> [batch_size * seq_len, vocab_size]
            target_token_ids.reshape(-1), # [batch_size ,seq_len] -> [batch_size * seq_len]
            ignore_index=pad_token_id,
            reduction="none" # 保留所有位置的loss 不求和不求平均
        ).reshape(input_token_ids.shape[0], -1)

        # 计算当前batch的平均token级别信息熵
        with torch.no_grad():
            # token_entropy： 每个token的softmax熵值 [batch_size, seq_len]
            token_entropy = compute_entropy(logits)
            # (token_entropy * target_masks).sum() / num_target_tokens ： 计算token的平均entropy
            # entropy = entropy + (...) ：多个micro-batch可能会累计的entropy
            entropy = entropy + (token_entropy * target_masks).sum() / num_target_tokens
        
        obj = log_probs * batch_advantages[:, None]
        obj = (obj * target_masks).sum() / num_target_tokens
        loss = -obj
        loss.backward()
       
    # update the policy
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=max_grad_norm
    )
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {
        "loss": loss.item(),
        "grad_norm": grad_norm.item(),
        "entropy": entropy.item(),
    }