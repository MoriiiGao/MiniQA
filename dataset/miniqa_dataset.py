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



if __name__ == "__main__":
    merge_jsonl_files("/root/LLMDataset/MilitaryIssues.jsonl", "/root/LLMDataset/pretrain_data.jsonl")
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