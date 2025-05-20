import torch
from typing import Dict, List
from transformers import AutoTokenizer

class MyselfTokenizer:
    def __init__(self, pretrained_tokenizer):
        self.tokenizer = pretrained_tokenizer

        self.pad_token_id = pretrained_tokenizer.pad_token_id or 0
        self.bos_token_id = pretrained_tokenizer.bos_token_id
        self.eos_token_id = pretrained_tokenizer.eos_token_id

    def tokenize(self, text: str) -> List[str]:
        """
        convert text to tokens
        """
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """
        Mapping tokens to ids
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def add_special_tokens(self, token_ids: List[int]) -> List[int]:

        if self.bos_token_id is not None:
            token_ids = [self.bos_token_id] + token_ids
        if self.eos_token_id is not None:
            token_ids = token_ids  + [self.eos_token_id]
        return token_ids

    def __call__(
            self,
            text: str,
            max_length: int,
            padding: str = 'max_length',
            truncation: bool = True,
            return_tensors: str = None
    ) -> Dict[str, torch.Tensor]:

        tokens = self.tokenizer(text)

        token_ids = self.convert_tokens_to_ids(tokens)

        token_ids = self.add_special_tokens(token_ids)

        if truncation and len(tokens) > max_length:
            tokens_ids = token_ids[:max_length]

        attention_mask = [1] * len(token_ids)
        if padding == "max_length" and len(token_ids) < max_length:
            pad_len = max_length - len(token_ids)
            token_ids += [self.pad_token_id] * pad_len
            attention_mask += [0] * pad_len

        if return_tensors == 'pt':
            return {
                "input_ids": torch.tensor([token_ids], dtype=torch.long),
                "attention_mask": torch.tensor([attention_mask], dtype=torch.long)
            }
        else:
            return {
                "input_ids": [token_ids],
                "attention_mask":[attention_mask]
            }


if __name__ == "__main__":
    path = 'D:\\iscas\\LLM\\MiniQA\\model\\PretrainTokenizer'
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer = MyselfTokenizer(tokenizer)

    text = "“军事问题”是指影响国家安全、军事战略、部队建设与作战能力生成的核心矛盾与关键难题，通常聚焦于“打得赢”与“防得住”的现实需求。它不仅包括战争准备和战争指导中的重大决策问题（如打什么仗、如何打仗、以何兵打仗），还涵盖武器装备发展、军种协同作战、力量结构调整、指挥控制体系构建、保障能力建设等方面。军事问题往往具有战略性、复杂性和多变量交互性，解决军事问题需要结合国家战略、军事理论、作战需求和未来战争形态进行系统研究和科学拆解。简而言之，军事问题是推动军队现代化建设、提升战斗力水平、赢得未来战争的源头性、方向性问题。"
    encoding = tokenizer(
        text,
        max_length=1024,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    print("input_ids:\n", encoding["input_ids"])
    print("attention_mask:\n", encoding["attention_mask"])
