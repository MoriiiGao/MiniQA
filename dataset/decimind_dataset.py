"""
author:
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re
import numpy as np
from pathlib import Path
import logging

import json
import random
from typing import Any, Tuple, List, Dict, Optional

import pandas as pd
from dataset.tokenizer import Tokenizer
from dataset.data_type import MiniBatch

import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)

def merge_jsonl_files(source_path: str, target_path: str):

    with open(source_path, 'r', encoding='utf8') as src, \
        open(target_path, 'a', encoding='utf8') as tgt: 
        for line in src:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                tgt.write(line + "\n")
            except json.JSONDecodeError:
                logger.info(f"⚠️ 跳过非法 JSON 行: {line[:100]}...")

    logger.info(f"✅ 合并完成：{source_path} -> {target_path}")

def analyze_dataset(file_path: str, text_key: str = "text"):

    def try_open_file(path, encodings=["utf-8", "gb18030", "gbk"]):
        for enc in encodings:
            try:
                f = open(path, 'r', encoding=enc)
                # 试读一行，确保不会乱码
                f.readline()
                f.seek(0)
                logger.info(f"✅ 使用编码 {enc} 成功读取文件")
                return f
            except UnicodeDecodeError:
                continue
        raise UnicodeDecodeError("❌ 无法读取文件，请检查文件编码是否为 utf-8 / gbk / gb18030")

    total_samples = 0
    total_length = 0
    max_length = 0
    min_length = float('inf')

    with try_open_file(file_path) as f:
        for line in f:
            data = json.loads(line.strip())
            text = data.get(text_key, "")
            text_length = len(text)

            total_samples += 1
            total_length += text_length
            max_length = max(max_length, text_length)
            min_length = min(min_length, text_length)

    if total_samples == 0:
        logger.info("❗ 文件为空，没有找到任何数据。")
        return

    avg_length = total_length / total_samples

    logger.info(f"✅ 样本总数: {total_samples}")
    logger.info(f"✅ 平均文本长度: {avg_length:.2f} 字符")
    logger.info(f"✅ 最长文本长度: {max_length} 字符")
    logger.info(f"✅ 最短文本长度: {min_length} 字符")

class PretrainDataset(Dataset):
    """
    预训练数据集类，用于将文本样本加载并编码为模型输入。
    支持 BOS/EOS token 添加、截断、填充，以及loss mask计算。
    """
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(data_path)

    def _load_data(self, path: str) -> List[str]:
        samples = []
        try:
            with open(path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        if 'text' in data:
                            samples.append(data['text'])
                        else:
                            logger.warning(f"Line {line_num} missing 'text' field")
                    except json.JSONDecodeError as e:
                        logger.warning(f"Line {line_num} JSON decode error: {e}")
        except FileNotFoundError:
            logger.error(f"Data file not found: {path}")
            raise
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        text = self.samples[index]
        #
        full_text = f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"

        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding.input_ids.squeeze(0)  # [seq_len]
        attention_mask = encoding.attention_mask.squeeze(0)  # [seq_len]

        X = input_ids[:-1]  # 输入
        Y = input_ids[1:]   # 标签
        loss_mask = attention_mask[1:]  # 有效部分参与loss计算 (loss mask序列中只有1和0 1表示参与计算的位置，0表示不参与计算的位置)

        return X, Y, loss_mask

class SFTDataset(Dataset):
    """
    Supervised Fine-Tuning Dataset
    用于构造DeciMind的监督微调任务 支持ChatML格式对话构建 损失掩码生成等
    """
    def __init__(self, jsonl_path: str, tokenizer: PreTrainedTokenizer, max_length: int=1024):
        """
        Args:
            jsonl_path(str): 数据文件路,要求每一行为一条JSON格式对话数据
            tokenizer: Huggingface或兼容tokeinzer实例
            max_length(int): 最大序列长度
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self._load_data(jsonl_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)
    
    def _load_data(self, jsonl_path):
        samples = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f, 1):  # 注意这里从1开始编号
                try:
                    data = json.loads(line.strip())
                    samples.append(data)
                except json.JSONDecodeError as e:
                    print(f"❌ JSON 解码错误！出错文件：{jsonl_path}")
                    print(f"🧨 出错行号：{idx}")
                    print(f"🔍 出错内容：{line.strip()}")
                    print(f"📍 错误信息：{e}")
                    raise e
        return samples
        
    def _create_chat_prompt(self, conversations):
        """
        构建符合ChatML模板的提示内容
        Args:
            conversations(List[Dict]):对话轮列表 依次排列user和assistant发言
        Returns:
            str: 构造好的Prompt文本
        """
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": turn["content"]})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    
    def _generate_loss_mask(self, input_ids):
        """
        根据 <|im_start|>assistant 和 <|im_end|> 标记，对 assistant 回复内容生成 loss mask。
        """
        loss_mask = [0] * len(input_ids)
        i = 0
        max_len = min(len(input_ids), self.max_length)

        while i < max_len:
            # 尝试匹配 "<|im_start|>assistant"
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                content_start = i + len(self.bos_id)
                content_end = content_start

                # 在后续 token 中寻找 <|im_end|>
                while content_end < max_len:
                    if input_ids[content_end:content_end + len(self.eos_id)] == self.eos_id:
                        break
                    content_end += 1

                # 只对 assistant 回复内容部分启用 loss（跳过起始 token）
                for j in range(content_start + 1, min(content_end, max_len)):
                    loss_mask[j] = 1

                # 更新 i 为结尾之后（跳过这一段）
                i = content_end + len(self.eos_id) if content_end < max_len else max_len
            else:
                i += 1

        return loss_mask


    def __getitem__(self, index):
        """
        获取单挑训练样本 返回模型输入 标签 掩码巡视

        Returns:
            Tuple[Tensor, Tensor, Tensor]: (input_ids, labels, loss_mask)
        """
        sample = self.samples[index] # 获取第i条数据
        prompt = self._create_chat_prompt(sample["conversations"]) # 获取对话 构造prompt
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        loss_mask = self._generate_loss_mask(input_ids)

        input_tensor = torch.tensor(input_ids[:-1], dtype=torch.long)
        label_tensor = torch.tensor(input_ids[1:], dtype=torch.long)
        mask_tensor = torch.tensor(loss_mask[1:], dtype=torch.long)

        return input_tensor, label_tensor, mask_tensor

class OLDSFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = self.load_data(jsonl_path)
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def load_data(self, path):
        samples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self.samples[index]
        # 构建对话提示
        print(sample["conversations"])
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)  # 对齐预测位置

        return X, Y, loss_mask

class MixDataset_OLD(Dataset):
    """
    LoRA微调 同时读取通用数据(public)与领域数据(domain)
    训练时按 `p_domain` 概率从域内数据抽样，其余概率抽通用数据。

    每行数据格式：
    {
        "conversations": [
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."},
            ...
        ]
    }
    """
    def __init__(self, 
                 domain_path: str, 
                 public_path: str,
                 tokenizer: PreTrainedTokenizer,
                 max_length: int=1024,
                 p_domain: float=0.7):
        super().__init__()
        assert 0.0 <= p_domain <= 1.0, "`p_domain` must be in [0,1]"
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.p_domain = p_domain

        # 载入两份数据
        self.domain_samples = self._load_jsonl(domain_path)
        self.public_samples = self._load_jsonl(public_path)

        # 编码special-tokens id
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

        # 取最大长度
        self._max_len = max(len(self.domain_samples), len(self.public_samples))
        
    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        samples = []
        with open(path, 'r', encoding="utf-8") as fp:
            for idx, line in enumerate(fp, 1):  # 从第1行开始计数
                line = line.strip()
                if not line:
                    continue
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"❌ JSON解析错误：第 {idx} 行，原因：{e}")
                    print(f"出错内容：{line}")
                    raise e  # 如果你希望程序停止运行，否则可删去这一行
        return samples

    def _build_prompt(self, conversations: List[Dict]) -> str:
        msgs = [
            {"role": ("user" if i % 2 == 0 else "assistant"), "content": turn["content"]}
            for i, turn in enumerate(conversations)
        ]
        return self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

    # 生成 loss-mask：仅对 <|im_start|>assistant ... <|im_end|> 之间 token 计算损失
    def _make_loss_mask(self, ids: List[int]) -> List[int]:
        mask = [0] * len(ids)
        i, n = 0, min(len(ids), self.max_length)
        while i < n:
            if ids[i : i + len(self.bos_id)] == self.bos_id:
                s = i + len(self.bos_id)
                e = s
                while e < n and ids[e : e + len(self.eos_id)] != self.eos_id:
                    e += 1
                for j in range(s + 1, min(e, n)):
                    mask[j] = 1
                i = e + len(self.eos_id)
            else:
                i += 1
        return mask
    # ───────────────────── dataset api ─────────────────────
    def __len__(self) -> int:
        return self._max_len

    def __getitem__(self, idx: int):
        # 动态决定使用哪类数据
        if random.random() < self.p_domain and self.domain_samples:
            sample = self.domain_samples[idx % len(self.domain_samples)]
        else:
            sample = self.public_samples[idx % len(self.public_samples)]

        prompt      = self._build_prompt(sample["conversations"])
        input_ids   = self.tokenizer(prompt).input_ids[: self.max_length]
        pad_len     = self.max_length - len(input_ids)
        input_ids  += [self.tokenizer.pad_token_id] * pad_len

        loss_mask   = self._make_loss_mask(input_ids)

        # shift-one-token
        x = torch.tensor(input_ids[:-1],              dtype=torch.long)
        y = torch.tensor(input_ids[1:],               dtype=torch.long)
        m = torch.tensor(loss_mask[1:],               dtype=torch.long)
        return x, y, m

class MixDataset(Dataset):
    def __init__(self, domain_dataset, public_dataset, p_domain=0.7):
        self.domain_dataset = domain_dataset
        self.public_dataset = public_dataset
        self.p_domain = p_domain

    def __len__(self):
        return max(len(self.domain_dataset), len(self.public_dataset))

    def __getitem__(self, idx):
        if random.random() < self.p_domain:
            return self.domain_dataset[idx % len(self.domain_dataset)]
        else:
            return self.public_dataset[idx % len(self.public_dataset)]



def convert_json_list_to_jsonl(input_path: str, output_path: str):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("❌ 输入 JSON 文件结构错误，最外层应为 list。")

    with open(output_path, 'w', encoding='utf-8') as fout:
        for idx, item in enumerate(data):
            if not isinstance(item, dict) or "conversations" not in item:
                raise ValueError(f"❌ 第 {idx + 1} 项不是合法的对话对象，缺少 'conversations' 键。")
            json_line = json.dumps(item, ensure_ascii=False)
            fout.write(json_line + '\n')

    print(f"✅ 已成功将 {len(data)} 条对话写入 JSONL 文件: {output_path}")


SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)
USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. "
    "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
    "Show your work in <think> </think> tags. "
    "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
)
RESPONSE_PROMPT = "Let me solve this step by step.\n<think>"

class CountdownTasksDataset(Dataset):
    """Prepare Countdown Tasks for training"""

    def __init__(
        self,
        tokenizer: Tokenizer,
        data_path: str,
        split: str = "train",
        test_size: int = 100,
    ):
        data = pd.read_parquet(Path(data_path))
        print(data.iloc[0])
        # use the last `test_size` examples for testing
        self.data = (
            data.iloc[:-test_size] if split == "train" else data.iloc[-test_size:]
        )
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["nums"], item["target"]))
        return item

    def encode_prefix(self, numbers: List[int], target: int):
        """Prefix is the *actual* input to the model."""
        user_message = USER_TEMPLATE.format(numbers=numbers, target=target)
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user_message},
            ],
            RESPONSE_PROMPT,
        )
        tokens = self.tokenizer.tokenize(prefix)
        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch."""
        numbers = [item["nums"] for item in batch]
        target = [item["target"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        return MiniBatch(
            numbers=numbers,
            target=target,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )

class NewCountdownTasksDataset(Dataset):
    
    def __init__(self, tokenizer: Tokenizer, data_path: str, split: str = "train", test_size: int = 100):
        data = pd.read_parquet(Path(data_path))
        print("Example record:", data.iloc[0].to_dict())
        self.data = (
            data.iloc[:test_size] if split == "train" else data.iloc[-test_size:]
        ).reset_index(drop=True)
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()

        # 使用 messages 构造对话 prompt
        messages = item["messages"]
        if isinstance(messages, str):
            import json
            messages = json.loads(messages)

        # 如果是 numpy 数组，转成 list
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()

        # 如果是单条 dict，包裹成列表
        if isinstance(messages, dict):
            messages = [messages]

        # 最终确保是 list，否则报错
        if not isinstance(messages, list):
            raise TypeError(f"`messages` should be a list, got: {type(messages)}")

        # 添加 system message
        messages = [{"role": "system", "content": SYSTEM_MESSAGE}] + messages


        # 编码 prefix prompt
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            messages,
            RESPONSE_PROMPT  # 适配 Qwen tokenizer 风格
        )
        tokens = self.tokenizer.tokenize(prefix)

        return {
            "prefix": prefix,
            "prefix_tokens": tokens.tokens,
            "prefix_token_ids": tokens.ids,
            "target": item.get("answer", ""),  # answer 可用于 reward function
            "numbers": [],  # 保持 GRPO 接口兼容性（无具体作用）
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MiniBatch:
        """Collate examples into a batch for GRPO."""
        return MiniBatch(
            numbers=[item["numbers"] for item in batch],
            target=[item["target"] for item in batch],
            prefix=[item["prefix"] for item in batch],
            prefix_tokens=[item["prefix_tokens"] for item in batch],
            prefix_token_ids=[item["prefix_token_ids"] for item in batch],
        )

if __name__ == "__main__":
    # merge_jsonl_files("/root/LLMDataset/MilitaryIssues.jsonl", "/root/LLMDataset/pretrain_data.jsonl")
    # dataset_path = "/root/LLMDataset/pretrain_data.jsonl"
    # analyze_dataset(dataset_path)
    # tokenizerPath = "/root/MiniQA/model/PretrainTokenizer"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizerPath)
    # train_dataset = PretrainDataset(
    #     "/root/LLMDataset/MilitaryIssues.jsonl",
    #     # "/root/LLMDataset/pretrain_hq.jsonl",
    #     tokenizer,
    #     max_length=512)

    # for idx in range(len(train_dataset)):
    #     X, Y, loss_mask = train_dataset[idx]
    #     print(f"\n=== Sample {idx} ===")
    #     print("X (input):", X.tolist())
    #     print("Y (label):", Y.tolist())
    #     print("loss_mask:", loss_mask.tolist())

    #     if idx >= 4:
    #         break

    # ==============================SFTDataset===========================
    # tokenizerPath = "/root/MiniQA/model/PretrainTokenizer"
    # tokenizer = AutoTokenizer.from_pretrained(tokenizerPath)
    # sft_dataset = "/root/LLMDataset/decimin_dataset/sft_512.jsonl"
    # sftdataset = OLDSFTDataset(sft_dataset, tokenizer)

    # for i in range(100):
    #     X, Y, mask = sftdataset[i]
        # print(f"Sample {i}")
        # print("Input IDs:", X.shape)
        # print("Labels:", Y.shape)
        # print(mask)
        # print("Loss Mask:", mask.sum().item(), "positions used for loss")

    # ==============================Mix SFTDataset===========================
    tokenizerPath = "/root/MiniQA/model/PretrainTokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizerPath)
    Domain_path = "/root/LLMDataset/decimin_dataset/scenario.jsonl"
    Public_path = "/root/LLMDataset/decimin_dataset/sft_mini_512.jsonl"
    
    # mixdata = MixDataset(
    #     domain_path = Domain_path,
    #     public_path = Public_path,
    #     tokenizer   = tokenizer,
    #     max_length  = 1024,
    #     p_domain    = 0.7,           # 70% 领域 / 30% 通用
    # )

    # for i in range(100):
    #     X, Y, mask = mixdata[i]

    # convert_json_list_to_jsonl(Domain_path, "/root/LLMDataset/decimin_dataset/scenario.jsonl")