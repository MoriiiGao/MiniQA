import os, sys, argparse, torch, json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.DeciMinid_LoRA import inject_lora, LoRAConfig

def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen-3 Full-Param → LoRA SFT")
    # === 通用训练超参 ===
    parser.add_argument("--out_dir",          type=str, default="/root/models/Qwen3_finetune")
    parser.add_argument("--epochs",           type=int, default=2)
    parser.add_argument("--batch_size",       type=int, default=64)
    parser.add_argument("--learning_rate",    type=float, default=5e-7)
    parser.add_argument("--device",           type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype",            type=str, default="bfloat16")         # fp16/bf16/float32
    # === 数据 & 模型路径 ===
    parser.add_argument("--model_name",       type=str, default="/root/models/Qwen_1.7",
                        help="🤗hub repo 或本地 ckpt 目录（已包含 config.json / pytorch_model.bin）")
    parser.add_argument("--tokenizer_path",   type=str, default="/root/models/Qwen_1.7")
    parser.add_argument("--data_path",        type=str, default="/root/LLMDataset/sft_qa.jsonl")
    # === 记录 / 分布式 ===（其余参数保留）
    parser.add_argument("--log_interval",     type=int, default=100)
    # … 其余 argparse 同上，略 …

    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # ──────────────────────────────────
    # 1. 载入 Qwen-3 权重 & tokenizer
    # ──────────────────────────────────
    print(f"[INFO] Loading base model  ⤵  {args.model_name}")
    torch_dtype = {"fp16": torch.float16, "bfloat16": torch.bfloat16}.get(args.dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch_dtype,
        device_map="auto",             # 多卡自动分配；单卡可用 .to(args.device)
        trust_remote_code=True
    )
    print(model)

    # 若显式指定 device_map = "auto"，模型会直接放到可用 GPU；
    # 如果想保持和原逻辑一致，也可手动: model.to(args.device)

    # ──────────────────────────────────
    # 2. 注入 LoRA
    # ──────────────────────────────────
    lora_cfg = LoRAConfig(rank=4, alpha=16,
                          target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))
    inject_lora(model,
                target_modules=lora_cfg.target_modules,
                rank=lora_cfg.rank,
                alpha=lora_cfg.alpha,
                dropout=lora_cfg.dropout,
                dump_txt=os.path.join(args.out_dir, "qwen3_modules.txt"))

    print("✅ LoRA injected.  Trainable params:",
          sum(p.numel() for n, p in model.named_parameters() if p.requires_grad) / 1e6, "M")

if __name__ == "__main__":
    main()