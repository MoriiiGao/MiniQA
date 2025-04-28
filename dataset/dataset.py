"""
author:
"""
import logging
import os
from typing import Tuple, List

import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizer

from dataset.MyselfTokenizer import MyselfTokenizer

logger = logging.getLogger(__name__)

class PretrainDataset(Dataset):
    """
    预训练数据集类，用于将文本样本加载并编码为模型输入。
    支持 BOS/EOS token 添加、截断、填充，以及 loss mask 计算。
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
        loss_mask = attention_mask[1:]  # 有效部分参与loss计算

        return X, Y, loss_mask

if __name__ == "__main__":

    path = 'D:\\iscas\\LLM\\MiniQA\\model\\PretrainTokenizer'
    tokenizer = AutoTokenizer.from_pretrained(path)
    train_dataset = PretrainDataset(
        "D:/iscas/LLM/minimind-master/dataset/pretrain_hq.jsonl",
        tokenizer,
        max_length=512)

    for idx in range(len(train_dataset)):
        X, Y, loss_mask = train_dataset[idx]
        print(f"\n=== Sample {idx} ===")
        print("X (input):", X.tolist())
        print("Y (label):", Y.tolist())
        print("loss_mask:", loss_mask.tolist())

        if idx >= 4:
            break