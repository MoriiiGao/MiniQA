"""
DeciMind LoRA SFT 使用通用lora注入/保存lora权重
"""
import os
import sys
__package__ = "model"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import json
from typing import List, Dict, Tuple
from model.model_decimind import DeciMindConfig, DeciMindForCausalLM
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """
    配置类
    """
    rank: int = 8
    alpha: int = 32
    dropout: float = 0.
    target_modules: Tuple[str, ...] = (
        # self-attention
        "q_proj", "k_proj", "v_proj", "out_proj"
        # FFN / MoE
        "gate_proj", "up_proj",  "down_proj"
    )

class LoRALinear(nn.Module):
    """
    lora基础层
    """
    def __init__(self,
                 origin: nn.Linear,
                 rank: int=8,
                 alpha: int = 32,
                 dropout: float = 0.):
        super().__init__()
        assert origin.weight.size(1) > 0, "Origin Linear must be initialized"

        # 保留原始权重(被冻结，不参与训练)
        self.origin = origin
        for p in self.origin.parameters():
            p.requires_grad = False
        
        # LoRA params
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        self.A = nn.Linear(origin.in_features, rank, bias=False)
        self.B = nn.Linear(rank, origin.out_features, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()
    
    def forward(self, x):
        return self.origin(x) + self.B(self.dropout(self.A(x))) * self.scaling

def _replace_module(parent: nn.Module,
                    name:   str,
                    lora:   LoRALinear):
    """把 parent.name 替换为 LoRA 包装层"""
    original = dict(parent.named_children())[name]
    setattr(parent, name, lora)
    # 保存以便merge/unmerge
    lora._origin_ref = original     # type: ignore[attr-defined]

def inject_lora(
    model: nn.Module,
    target_modules: Tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "out_proj"),
    rank: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    dump_txt: str | None = None,
) -> None:
    """
    遍历模型，将名字包含在 target_modules 的 nn.Linear 替换为 LoRALinear。

    Args:
        model (nn.Module): 原始模型
        target_modules (Tuple[str]): 需要注入 LoRA 的层名关键词
        rank (int): LoRA rank
        alpha (int): LoRA alpha
        dropout (float): LoRA dropout
        dump_txt (str | None): 若提供路径，则把 model.named_modules()
                               以 JSON Lines 形式写入该文件
    """
    if dump_txt:
        with open(dump_txt, "w", encoding="utf-8") as fp:
            for name, mod in model.named_modules():
                fp.write(
                    json.dumps(
                        {
                            "name": name,
                            "type": mod.__class__.__name__,
                            "repr": repr(mod),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    for module_name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in module_name for t in target_modules):
            parent_name, child_name = module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
            parent = model.get_submodule(parent_name) if parent_name else model
            lora_layer = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
            _replace_module(parent, child_name, lora_layer)

def merge_lora(model: nn.Module) -> None:
    """把 LoRA Delta 合并进原始权重（推理时可调用）"""
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            delta_w = (mod.B.weight @ mod.A.weight) * mod.scaling   # [out, in]
            mod.origin.weight += delta_w.to(mod.origin.weight.dtype)
            mod.merged = True

def unmerge_lora(model: nn.Module) -> None:
    """反向操作：从已合并的权重中减去 LoRA delta（继续训练）"""
    for mod in model.modules():
        if isinstance(mod, LoRALinear) and getattr(mod, "merged", False):
            delta_w = (mod.B.weight @ mod.A.weight) * mod.scaling
            mod.origin.weight -= delta_w.to(mod.origin.weight.dtype)
            mod.merged = False

def _lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            for k, v in module.state_dict().items():
                sd[f"{name}.{k}"] = v
    return sd


def save_lora(model: nn.Module, file: str) -> None:
    torch.save(_lora_state_dict(model), file)

def load_lora(model: nn.Module, file: str, map_location="cpu") -> None:
    sd = torch.load(file, map_location=map_location)
    missing = []
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = f"{name}."
            lora_weights = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if lora_weights:
                module.load_state_dict(lora_weights, strict=True)
            else:
                missing.append(name)
    if missing:
        print(f"[LoRA] Warning: layers not found in checkpoint: {missing}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="DeciMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="/root/models/DeciMind")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DeciMind-Full-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=640, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=True, type=bool)
    parser.add_argument('--tokenizer_path', default="/root/MiniQA/model/PretrainTokenizer", type=str)
    parser.add_argument("--data_path", type=str, default="/root/LLMDataset/decimin_dataset/sft_mini_512.jsonl")

    args = parser.parse_args()
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    lm_config = DeciMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe
    )

    model = DeciMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ""
    ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(args.device)
    print(model)
    cfg   = LoRAConfig(rank=4, alpha=16)
    inject_lora(model, target_modules=cfg.target_modules,
                rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout, dump_txt="decimind_module.txt")

    print("LoRA injected ✓")
    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    lora_params_count = sum(
        p.numel() for n, p in model.named_parameters()
        if (".A." in n or ".B." in n) and p.requires_grad
    )

    print(f"LLM 总参数量: {total_params}")
    print(f"LoRA 参数量: {lora_params_count}")
    print(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")
    # 训练 demo
    # optimiser = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    # dummy_inp = torch.randint(0, 100, (2, 16))
    # out = model(dummy_inp).logits.sum()
    # out.backward()
    # optimiser.step()
    # print("LoRA train step ✓")

    # # 保存 / 加载
    # save_lora(model, "lora_gpt2.pt")
    # load_lora(model, "lora_gpt2.pt")
    # print("LoRA save/load ✓")

