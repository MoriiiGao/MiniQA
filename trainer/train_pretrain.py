"""
MiniQALM - Industrial Grade Training Bootstrapper
Author: MoriiiGao
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import torch
import math
import argparse
import torch.distributed as dist
from contextlib import nullcontext
from torch import nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from transformers import AutoTokenizer

from dataset.miniqa_dataset import PretrainDataset
from model.model_miniqa import MiniQAConfig
from log.LoggerHelper import LoggerHelper
from model.model_miniqa import MiniQALMLite, MiniQALM

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
   
class MiniQATrainer:
    def __init__(self):
        self.args = self._parse_arguments()
        self.logger = LoggerHelper(name="MiniQATrainer", log_dir="train_logs")
        self.logger.log_dict("\u2728 当前训练配置", vars(self.args))

        self.ddp = int(os.environ.get("RANK", -1)) != -1
        self.ddp_local_rank, self.device = 0, self.args.device
        if self.ddp:
            self._init_distributed_mode()

        # 初始化参数配置
        self.lm_config = self._build_config()
        self.tokenizer, self.model = self._init_model()
        self._prepare_environment()
        self._maybe_init_wandb()

    def _parse_arguments(self):
        parser = argparse.ArgumentParser(description="MiniQA Pretraining Script")
        parser.add_argument("--out_dir", type=str, default="out", help="模型输出目录")

        parser.add_argument("--epochs", 
                            type=int, default=1, help="训练总轮数")
        parser.add_argument("--batch_size", 
                            type=int, default=32, help="每个batch的样本数")
        parser.add_argument("--learning_rate", 
                            type=float, default=5e-4, help="学习率")
        parser.add_argument("--device", 
                            type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="设备")
        parser.add_argument("--dtype", 
                            type=str, default="bfloat16", help="数据类型，如 float16 bfloat16 float32")
        parser.add_argument("--use_wandb", 
                            action="store_true", help="是否启用 wandb 记录")
        parser.add_argument("--wandb_project", 
                            type=str, default="MiniQA-Pretrain", help="wandb 项目名称")
        parser.add_argument("--num_workers", 
                            type=int, default=1, help="DataLoader 工作线程数")
        parser.add_argument("--ddp", 
                            action="store_true", help="是否启用 DDP 分布式")
        parser.add_argument("--accumulation_steps", 
                            type=int, default=8, help="梯度累积步数")
        parser.add_argument("--grad_clip", 
                            type=float, default=1.0, help="梯度裁剪阈值")
        parser.add_argument("--warmup_iters", 
                            type=int, default=0, help="学习率预热步数")
        parser.add_argument("--log_interval", 
                            type=int, default=100, help="日志打印间隔")
        parser.add_argument("--save_interval", 
                            type=int, default=100, help="模型保存间隔")
        parser.add_argument("--local_rank", 
                            type=int, default=-1, help="DDP 本地编号")
        parser.add_argument("--hidden_size", 
                            type=int, default=512)
        parser.add_argument("--num_hidden_layer", 
                            type=int, default=8)
        parser.add_argument("--max_seq_len", 
                            type=int, default=512)
        parser.add_argument("--use_moe", 
                            type=bool, default=False)
        parser.add_argument("--data_path", 
                            type=str, default="/root/LLMDataset/pretrain_data.jsonl",)
        parser.add_argument("--tokenizer_path", 
                            type=str, default="/root/MiniQA/model/PretrainTokenizer")
        return parser.parse_args()

    def _init_distributed_mode(self):
        dist.init_process_group(backend="nccl")
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.device = f"cuda:{self.ddp_local_rank}"
        torch.cuda.set_device(self.device)
        self.args.device = torch.device(self.device)

    def _build_config(self):
        return MiniQAConfig(
            dim=self.args.hidden_size,
            n_layers=self.args.num_hidden_layer,
            max_seq_len=self.args.max_seq_len,
            use_moe=self.args.use_moe
        )

    def _init_model(self):
        """
        初始化分词器 模型
        """
        if not os.path.exists(self.args.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer 路径不存在: {self.args.tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(self.args.tokenizer_path)
        model = MiniQALM(self.lm_config).to(self.args.device)
        self.logger.info(
            f'MiniQA可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万'
        )
        return tokenizer, model

    def _prepare_environment(self):
        os.makedirs(self.args.out_dir, exist_ok=True)
        self.args.save_dir = os.path.join(self.args.out_dir, "checkpoints")
        os.makedirs(self.args.save_dir, exist_ok=True)
        torch.manual_seed(1337)
        self.ctx = nullcontext() if "cpu" in self.args.device else torch.cuda.amp.autocast()
        self.args.tokens_per_iter = self.args.batch_size * self.lm_config.max_seq_len

    def _maybe_init_wandb(self):
        self.args.wandb_run_name = f"MiniQALM-E{self.args.epochs}-B{self.args.batch_size}-LR{self.args.learning_rate}"
        if self.args.use_wandb and (not self.ddp or self.ddp_local_rank == 0):
            import wandb
            self.wandb.init(project=self.args.wandb_project, name=self.args.wandb_run_name)
            self.logger.info("\u2705 WandB 初始化完成")
        else:
            self.logger.info("\u26A0\ufe0f WandB 未启用或非主进程")
            self.wandb = None

    @staticmethod
    def get_cosine_annealing_lr(
            current_step: int,
            total_steps: int,
            lr: float) -> float:
        
        if total_steps == 0:
            raise ValueError("total_steps must be greater than 0")

        # 计算余弦部分( 范围是从 -1， 1) 表示当前进度对应的余弦值 初始节奏是1，最后接近-1
        cosine_decy = math.cos(math.pi * current_step / total_steps)

        # 计算调整后的lr
        adjusted_lr = (lr / 10) + 0.5 * lr * (1 + cosine_decy)

        return adjusted_lr

    def adjust_learning_rate(self, current_step):
        lr = self.get_cosine_annealing_lr(
            current_step,
            self.args.epochs * self.iter_per_epoch,
            self.args.learning_rate
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def compute_loss(self, res, Y, loss_mask):
        loss_fn = nn.CrossEntropyLoss(reduction="none")
        loss = loss_fn(
            res.logits.view(-1, res.logits.size(-1)),
            Y.view(-1)
        ).view(Y.size())  # 恢复到(batch_size, seq_len)形状
        loss = (loss * loss_mask).sum() / loss_mask.sum()  # 只计算有效位置loss
        loss += res.loss  # 加上模型内部其他损失项（如正则化loss）
        loss = loss / self.args.accumulation_steps
        return loss

    def update_optimizer(self):

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

    def log_metrics(self, epoch, step, loss, start_time):
        if step % self.args.log_interval != 0:
            return

        elapsed_time = time.time() - start_time
        eta_minutes = (elapsed_time / (step + 1)) * (self.iter_per_epoch - step) / 60

        msg = (
            f"Epoch:[{epoch+1}/{self.args.epochs}]({step}/{self.iter_per_epoch}) "
            f"loss:{loss.item() * self.args.accumulation_steps:.3f} "
            f"lr:{self.optimizer.param_groups[-1]['lr']:.12f} "
            f"ETA:{eta_minutes:.1f}min"
        )
        self.logger.info(msg)

        if self.wandb and (not self.ddp or dist.get_rank() == 0):
            self.wandb.log({
                "loss": loss.item() * self.args.accumulation_steps,
                "lr": self.optimizer.param_groups[-1]['lr'],
                "ETA_minutes": eta_minutes
            })

    def save_checkpoint(self, epoch, step):
        if (step + 1) % self.args.save_interval != 0 or (self.ddp and dist.get_rank() != 0):
            return
        self.model.eval()

        moe_suffix = '_moe' if self.lm_config.use_moe else ''
        save_path = f'{self.args.save_dir}/pretrain_{self.lm_config.hidden_dim}{moe_suffix}.pth'

        state_dict = self.model.module.state_dict() if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model.state_dict()
        state_dict = {k: v.half() for k, v in state_dict.items()}  # 保存为半精度

        torch.save(state_dict, save_path)
        self.model.train()

    def train_epoch(self, epoch):
        """
        Pretrain the large language model for one epoch
        """
        self.model.train()
        start_time = time.time()

        for step, (X, Y, loss_mask) in enumerate(self.train_loader):
            X, Y, loss_mask = X.to(self.args.device), Y.to(self.args.device), loss_mask.to(self.args.device)

            # 更新学习率
            self.adjust_learning_rate(epoch * self.iter_per_epoch + step)

            with self.ctx:
                res = self.model(X)
                loss = self.compute_loss(res, Y, loss_mask)

            self.scaler.scale(loss).backward()

            if (step + 1) % self.args.accumulation_steps == 0:
                self.update_optimizer()

            self.log_metrics(epoch, step, loss, start_time)
            self.save_checkpoint(epoch, step)

    def run(self):
        self.logger.info("\u26A1\ufe0f 初始化完成，准备开始训练...")
        # 准备训练数据
        train_ds = PretrainDataset(
            self.args.data_path,
            self.tokenizer,
            max_length=self.lm_config.max_seq_len)
        # 分布式采样器
        train_sampler = DistributedSampler(train_ds) if self.ddp else None
        # 构建dataloader
        self.train_loader = DataLoader(
            train_ds,                           # 数据集实例
            batch_size=self.args.batch_size,    # 每个批次加载多少数据
            pin_memory=True,                    # 是否直接将数据拷贝到GPU
            drop_last=False,                    # 是丢弃最后不足一个batch的数据
            shuffle=not self.ddp,               # 是否数据打乱
            num_workers=self.args.num_workers,  # 加载数据的线程数
            sampler=train_sampler
        )
        # 构建梯度缩放器
        # 混合精度训练使用GradScaler自动处理梯度缩放，避免数值不稳定
        # 只有在float16 或者bfloat16训练时启动
        self.scaler = torch.cuda.amp.GradScaler(enabled=(self.args.dtype in ['float16', 'bfloat16']))
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)

        if self.ddp:
            self.model._ddp_params_and_buffers_to_ignore = {"pos_cis"}
            self.model = DistributedDataParallel(self.model, device_ids=[self.ddp_local_rank])

        self.iter_per_epoch = len(self.train_loader)
        self.logger.info(f"\u2708\ufe0f 每轮迭代步数：{self.iter_per_epoch}")

        # TODO: 编写训练循环（训练 + 验证 + 保存）
        # 可继续添加 self._train_epoch() 等模块
        for epoch in range(self.iter_per_epoch):
            self.train_epoch(epoch)

if __name__ == "__main__":
    trainer = MiniQATrainer()
    trainer.run()
