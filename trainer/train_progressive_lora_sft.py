"""
渐进式LoRA训练方法
"""
import os
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
from dataset.decimind_dataset import SFTDataset, OLDSFTDataset, MixDataset
from model.DeciMinid_LoRA import LoRAConfig, LoRALinear, inject_lora, merge_lora, save_lora, load_lora
from log.LoggerHelper import LoggerHelper

logger = LoggerHelper(name="DeciMindSFT", log_dir="train_logs")
warnings.filterwarnings('ignore')

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def train_epoch(epoch, wandb, mode="public", public_dataset=None, domain_dataset=None):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()

    # 根据mode构造train_loader
    if mode == "public":
        dataset = public_dataset
    elif mode == "domain":
        dataset = domain_dataset
    elif mode == "mix":
        dataset = MixDataset(domain_dataset, public_dataset, p_domain=args.p_domain)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    sampler = DistributedSampler(dataset) if ddp else None
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,
        num_workers=args.num_workers,
        sampler=sampler
    )

    iter_per_epoch = len(train_loader)
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)
        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
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
            logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min'.format(
                    global_epoch + 1 if global_epoch is not None else epoch + 1,
                    total_epochs if total_epochs is not None else args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            if wandb is not None and (not ddp or dist.get_rank() == 0):
                wandb.log({"loss": loss.item() * args.accumulation_steps,
                           "lr": optimizer.param_groups[-1]['lr']})

        if (step + 1) % args.save_interval == 0 and (not ddp or dist.get_rank() == 0):
            model.eval()
            lora_save_path = f'{args.save_dir}/lora/{args.lora_name}_{lm_config.hidden_size}.pth'
            os.makedirs(os.path.dirname(lora_save_path), exist_ok=True)
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

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeciMind Full SFT")
    parser.add_argument("--out_dir", type=str, default="/root/models/DeciMind")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-7)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", type=bool, default=True, help="是否启用 WandB")
    parser.add_argument("--wandb_project", type=str, default="DeciMind-LoRA-SFT")
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
    parser.add_argument("--p_domain", default=0.7, type=float)
    parser.add_argument('--epochs_public', type=int, default=3, help="公共语料训练轮数")
    parser.add_argument('--epochs_mix', type=int, default=7, help="混合语料训练轮数")
    parser.add_argument('--epochs_domain', type=int, default=3, help="领域语料训练轮数")
    parser.add_argument("--lora_name", default="lora", type=str)
    parser.add_argument('--tokenizer_path', default="/root/MiniQA/model/PretrainTokenizer", type=str)
    parser.add_argument("--data_path", type=str, default="/root/LLMDataset/decimin_dataset/sft_mini_512.jsonl")
    parser.add_argument("--domain_data_path", type=str, default="/root/LLMDataset/decimin_dataset/scenario.jsonl")

    args = parser.parse_args()
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    lm_config = DeciMindConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        use_moe=args.use_moe
    )

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
    
    model, tokenizer = init_model(lm_config) # 初始化模型
    # apply_lora(model)
    cfg  = LoRAConfig(rank=4, alpha=16)
    inject_lora(model, target_modules=cfg.target_modules,
                rank=cfg.rank, alpha=cfg.alpha, dropout=cfg.dropout, dump_txt="decimind_module.txt")
    
    
    total_params = sum(p.numel() for p in model.parameters())  # 总参数数量
    lora_params_count = sum(
        p.numel() for n, p in model.named_parameters()
        if (".A." in n or ".B." in n) and p.requires_grad
    )
    if not ddp or dist.get_rank() == 0:
        print(f"LLM 总参数量: {total_params}")
        print(f"LoRA 参数量: {lora_params_count}")
        print(f"LoRA 参数占比: {lora_params_count / total_params * 100:.2f}%")

    # 先全部冻结
    for p in model.parameters():
        p.requires_grad = False

    # 只解冻 LoRA 里的 A/B
    lora_params = []
    for m in model.modules():
        if isinstance(m, LoRALinear):
            for p in m.parameters():
                p.requires_grad = True
                lora_params.append(p)
    
    # 只对 LoRA 参数进行优化
    optimizer = optim.AdamW(lora_params, lr=args.learning_rate)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))


    # 构建数据集
    public_dataset = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    domain_dataset = SFTDataset(args.domain_data_path, tokenizer, max_length=args.max_seq_len)

    total_epochs = args.epochs_public + args.epochs_mix + args.epochs_domain
    global_epoch = 0

    # 阶段一：public数据
    logger.info("📘 阶段一：public语料预热训练")
    for i in range(args.epochs_public):
        train_epoch(
            epoch=i,
            wandb=wandb,
            mode="public",
            public_dataset=public_dataset,
            domain_dataset=domain_dataset,
            global_epoch=global_epoch,
            total_epochs=total_epochs
        )
        global_epoch += 1

    # 阶段二：混合对齐
    logger.info("📗 阶段二：混合 public + domain 语料训练")
    for i in range(args.epochs_mix):
        train_epoch(
            epoch=i,
            wandb=wandb,
            mode="mix",
            public_dataset=public_dataset,
            domain_dataset=domain_dataset,
            global_epoch=global_epoch,
            total_epochs=total_epochs
        )
        global_epoch += 1

    # 阶段三：领域精调
    logger.info("📕 阶段三：仅使用 domain 语料微调")
    for i in range(args.epochs_domain):
        train_epoch(
            epoch=i,
            wandb=wandb,
            mode="domain",
            public_dataset=public_dataset,
            domain_dataset=domain_dataset,
            global_epoch=global_epoch,
            total_epochs=total_epochs
        )
        global_epoch += 1