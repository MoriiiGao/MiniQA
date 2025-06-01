import os
from pathlib import Path
import random
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
from torch import optim, nn
import torch.distributed as dist
from contextlib import nullcontext
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_decimind import DeciMindConfig, DeciMindForCausalLM
from dataset.decimind_dataset import SFTDataset
from model.model_lora import load_lora, save_lora, apply_lora
from torch.utils.data import Subset

warnings.filterwarnings('ignore')


# Logger function
def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, dataloader, wandb, iter_per_epoch, stage_name, stage_epochs):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    model.train()
    for step, (X, Y, loss_mask) in enumerate(dataloader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        assert not torch.isnan(X).any(), "输入 X 有 NaN"
        assert not torch.isnan(Y).any(), "输入 Y 有 NaN"
        assert not torch.isnan(loss_mask).any(), "loss_mask 有 NaN"
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            if torch.isnan(res.logits).any():
                print("⚠️模型输出logits中存在 NaN")
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
            if torch.isnan(res.loss):
                print("❗ res.aux_loss 是 NaN")
            loss += res.loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                f'{stage_name} Epoch:[{epoch+1}/{stage_epochs}]({step}/{iter_per_epoch}) '
                f'loss:{loss.item()*args.accumulation_steps:.3f} '
                f'lr:{optimizer.param_groups[-1]["lr"]:.12f} '
                f'epoch_Time:{spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60}min'
            )
            if (wandb is not None) and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr'],
                           "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            lora_save_path = (
                Path(args.save_dir)
                / "lora"
                / f"{args.lora_name}_{lm_config.hidden_size}.pth"
            )
            print(lora_save_path)
            lora_save_path.parent.mkdir(parents=True, exist_ok=True)
            save_lora(model, lora_save_path)
            model.train()

def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    model = DeciMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    return model.to(args.device), tokenizer

def init_qwen_model():

    torch_dtype = {"fp16": torch.float16, "bfloat16": torch.bfloat16}.get(args.dtype, torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(args.qwen_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.qwen_path,
        torch_dtype=torch_dtype,
        device_map="auto",             # 多卡自动分配；单卡可用 .to(args.device)
        trust_remote_code=True
    )

    return model.to(args.device), tokenizer

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def stage_train(dataloader: DataLoader, stage_name: str, epochs: int):
    """
    多阶段微调
    """
    iter_per_epoch = len(dataloader)
    print(f"\n{'='*30}\n>>> 开始阶段：{stage_name}，共 {epochs} 轮，每轮 {iter_per_epoch} 步\n{'='*30}")
    for epoch in range(epochs):
        train_epoch(epoch, dataloader, wandb, iter_per_epoch, stage_name=stage_name, stage_epochs=epochs)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DeciMind SFT with LoRA")
    parser.add_argument("--out_dir", type=str, default="/root/models/DeciMind")
    parser.add_argument("--orgi_epochs", type=int, default=1)
    parser.add_argument("--identity_epochs", type=int, default=2)
    parser.add_argument("--scenario_epochs", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DeciMind-LoRA-SFT")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--hidden_size', default=640, type=int)
    parser.add_argument('--num_hidden_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=1024, type=int)
    parser.add_argument('--use_moe', default=True, type=bool)
    parser.add_argument('--qwen_path', type=str, default='/root/models/Qwen_1.7')
    parser.add_argument('--tokenizer_path', default="/root/MiniQA/model/PretrainTokenizer", type=str)
    parser.add_argument("--identity_data", type=str, default="/root/LLMDataset/decimin_dataset/lora_identity.jsonl")
    parser.add_argument("--orgi_data", type=str, default="/root/LLMDataset/decimin_dataset/sft_mini_512.jsonl")
    parser.add_argument("--data_path", type=str, default="/root/LLMDataset/decimin_dataset/scenario.jsonl") # scenario.jsonl sft_mini_512.jsonl
    parser.add_argument("--lora_name", type=str, default="lora_deci", help="根据任务保存成lora_XXX")
    args = parser.parse_args()

    lm_config = DeciMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_local_rank, DEVICE = 0, "cuda:0"
    base_seed = 1337
    torch.manual_seed(base_seed)
    torch.cuda.manual_seed(base_seed)

    if ddp:
        init_distributed_mode()
        args.device = torch.device(DEVICE)
        rank = dist.get_rank()
        torch.manual_seed(base_seed + rank)
        # 同时设置 CUDA 的随机种子
        torch.cuda.manual_seed(base_seed + rank)

    args.wandb_run_name = f"DeciMind-Lora-SFT-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"
    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # model, tokenizer = init_model(lm_config)
    model, tokenizer = init_qwen_model()
    apply_lora(model)

    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    lora_params_count = sum(p.numel() for name, p in model.named_parameters() if 'lora' in name)  # LoRA 参数数量
    if not ddp or dist.get_rank() == 0:
        print(f"LLM 总参数量: {total_params}")
        print(f"LoRA 参数量: {lora_params_count}")
        print(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    for name, param in model.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False
    lora_params = []
    for name, param in model.named_parameters():
        if 'lora' in name:
            lora_params.append(param)

    # 只对 LoRA 参数进行优化
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    
    train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_sampler = DistributedSampler(train_ds) if ddp else None
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    iter_per_epoch = len(train_loader)

    identity_ds = SFTDataset(args.identity_data, tokenizer, max_length=args.max_seq_len)
    orgi_ds     = SFTDataset(args.orgi_data,     tokenizer, max_length=args.max_seq_len)
    scenario_ds = SFTDataset(args.data_path,     tokenizer, max_length=args.max_seq_len)

    # 分布式采样器（如果用DDP）
    identity_sampler = DistributedSampler(identity_ds) if ddp else None
    orgi_sampler     = DistributedSampler(orgi_ds)     if ddp else None
    scenario_sampler = DistributedSampler(scenario_ds) if ddp else None

    # 预热
    total_len = len(orgi_ds)
    sample_size = 20000
    indices = random.sample(range(total_len), sample_size)
    orgi_sampled_ds = Subset(orgi_ds, indices)

    orgi_loader = DataLoader(
        orgi_sampled_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
        sampler=orgi_sampler
    )

    # 通识记忆
    identity_loader = DataLoader(
        identity_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(identity_sampler is None),
        num_workers=args.num_workers,
        sampler=identity_sampler
    )

    # 领域知识
    scenario_loader = DataLoader(
        scenario_ds,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=(scenario_sampler is None),
        num_workers=args.num_workers,
        sampler=scenario_sampler
    )

    stage_train(orgi_loader, "orgi", args.orgi_epochs)

    # --- 阶段2：通识记忆（身份数据） ---
    stage_train(identity_loader, "identity", args.identity_epochs)

    # --- 阶段3：领域知识（场景数据） ---
    stage_train(scenario_loader, "scenario", args.scenario_epochs)

    