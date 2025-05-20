import json
import os
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import re
import sys
import argparse
import time
import math
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext
from pathlib import Path
from torch import optim
from typing import Any, Callable, Iterator, Optional
from transformers import GenerationConfig
from GRPO.grpo_loss import GRPOLoss
from GRPO.replay_buffer import ReplayBuffer
from GRPO.system_prompt import system_prompt, chinese_system_prompt, decimind_system_prompt
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model_decimind import DeciMindConfig, DeciMindForCausalLM

warnings.filterwarnings('ignore')

def Logger(content):
    if not ddp or dist.get_rank() == 0:
        print(content)

def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))

def load_model(
    model_name_or_path: str,
    trust_remote_code: bool = False,
    bf16: bool = True,
    device_map=None,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if bf16 else "auto",
        device_map=device_map,
    )
    return model, tokenizer



def init_model(lm_config):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    # 初始化训练模型
    model = DeciMindForCausalLM(lm_config)
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    state_dict = torch.load(ckp, map_location=args.device)
    model.load_state_dict(state_dict, strict=False)
    # 初始化参考模型
    ref_model = DeciMindForCausalLM(lm_config)
    ref_model.load_state_dict(state_dict, strict=False)
    ref_model.eval()
    ref_model.requires_grad_(False)

    Logger(f'LLM总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    model = model.to(args.device)
    ref_model = ref_model.to(args.device)

    return model, ref_model, tokenizer



@torch.no_grad()
def rollout(
    model: DeciMindForCausalLM,
    tokenizer: AutoTokenizer,
    task: str,
    oracle_answer: str,
    num_rollouts: int,
    max_length: int = 1024,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    
    """
    给定一个任务（如指标生成），用模型生成多组回答，并用给定的奖励规则对这些回答逐条打分
    最终返回序列ID 奖励 mask和回答文本
    """
    ### 1.模型设置于输入构建
    model.eval()

    # 构建prompt 包含system prompt和用户提问
    chat_messages = [
        {"role": "system", "content": chinese_system_prompt},
        {"role": "user", "content": task},
    ]
    chat_prompt = tokenizer.apply_chat_template(
        chat_messages, tokenize=False, add_generation_prompt=True
    )

    ### 2.Tokenizer编码+多次复制

    model_inputs = tokenizer(
        [chat_prompt],
        return_tensors="pt",
        padding=True,
        padding_side="left",
        return_attention_mask=True,
    ).to(args.device)

    # duplicate prompt num_rollouts times
    model_inputs["attention_mask"] = model_inputs["attention_mask"].repeat(num_rollouts, 1)
    input_ids = model_inputs["input_ids"].repeat(num_rollouts, 1)
    model_inputs["input_ids"] = input_ids

    ### 3.模型采样生成
    # model = DeciMindForCausalLM(lm_config)
    # moe_path = '_moe' if lm_config.use_moe else ''
    # ckp = f'{args.save_dir}/full_sft_{lm_config.hidden_size}{moe_path}.pth'
    # state_dict = torch.load(ckp, map_location=args.device)
    # model.load_state_dict(state_dict, strict=False)
    generation_config = GenerationConfig(
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
        max_length=max_length,
        pad_token_id=pad_token_id,
    )
    sequence_ids = model.generate(**model_inputs, generation_config=generation_config)
    completions = tokenizer.batch_decode(
        sequence_ids[:, input_ids.shape[1]:], skip_special_tokens=True
    )

    ### 4.构造action mask
    action_mask = torch.zeros_like(sequence_ids, dtype=torch.bool)
    action_mask[:, input_ids.shape[1]:] = True
    action_mask[sequence_ids == pad_token_id] = False
    action_mask = action_mask[:, 1:]

    ### 5.奖励打分
    returns = torch.zeros(num_rollouts, 1, dtype=torch.float)
    for i, completion in enumerate(completions):
        # 用正则抽取<answer>标签中的内容
        answer_match = re.search(
            r"<answer>(.*?)</answer>",
            completion,
            flags=re.DOTALL,
        )
        answer = answer_match.group(1) if answer_match else None
        reward = 0
        if answer is not None:
            if answer == oracle_answer:
                reward = 1.0
            elif oracle_answer in answer:
                reward = 0.5
            else:
                reward = 0.01
        returns[i] = reward

    return sequence_ids, returns.to(sequence_ids.device), action_mask, completions

def init_distributed_mode():
    if not ddp: return
    global ddp_local_rank, DEVICE

    dist.init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    DEVICE = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(DEVICE)

def read_jsonl(file_name: str | Path) -> Iterator:
    file_path = Path(file_name)
    with file_path.open(mode="r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def read_prompts(
    file_name: str,
    predicate: Optional[Callable[[Any], bool]] = None,
    max_rows: Optional[int] = None,
) -> list:
    rows = []
    for x in read_jsonl(file_name):
        if predicate is None or predicate(x):
            rows.append(x)
        if max_rows is not None and len(rows) >= max_rows:
            break
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DeciMind GRPO RLHF")
    parser.add_argument("--out_dir", type=str, default="/root/models/DeciMind")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-8)
    # sft阶段学习率为 「5e-6」->「5e-7」长度512，建议离线正负样本「概率」偏好对齐阶段lr <=「1e-8」长度3000，否则很容易遗忘训坏
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DeciMind-GRPO")
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
    parser.add_argument("--model_name", default="/root/models/Qwen_1.7", type=str)
    parser.add_argument('--tokenizer_path', default="/root/MiniQA/model/PretrainTokenizer", type=str)
    parser.add_argument("--data_path", type=str, default="/root/LLMDataset/decimin_dataset/math_tasks.jsonl")
    parser.add_argument('--group_size', type=int, default=12)
    parser.add_argument('--rollouts_per_step', type=int, default=32)
    parser.add_argument('--epochs_per_step', type=int, default=1)
    parser.add_argument('--max_norm', type=float, default=1.0)
    parser.add_argument('--kl_weight', type=float, default=0.01)
    parser.add_argument('--clip_eps', type=float, default=0.2)
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--checkpoint_interval', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_rows', type=int, default=64 * 1024)
    args = parser.parse_args()

    lm_config = DeciMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=args.use_moe)
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len
    device_type = "cuda" if "cuda" in args.device else "cpu"

    args.wandb_run_name = f"DeciMind-Full-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

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

    if args.use_wandb and (not ddp or ddp_local_rank == 0):
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    # model, reference_model, tokenizer = init_model(lm_config)
    # device = torch.device("cuda", device_index)
    # cpu_device = torch.device("cpu")
    reference_model, _ = load_model(args.model_name, device_map=args.device)
    model, tokenizer = load_model(args.model_name, device_map=args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    reference_model.eval()
    # model.gradient_checkpointing_enable(
    #     gradient_checkpointing_kwargs={"use_reentrant": False}
    # )

    pad_token_id = tokenizer.eos_token_id

    prompts = read_prompts(
        args.data_path,
        predicate=lambda x: len(x["question"]) < 128
        and x["num_terms"] <= 3
        and x["num_digits"] <= 3,
        max_rows=64 * 1024,
    )
    print(f"found {len(prompts)} matching prompts")
    prompt_loader = DataLoader(
        prompts,
        batch_size=args.rollouts_per_step,
        shuffle=True,
        drop_last=True,
        pin_memory=False,
    )

    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=args.clip_eps, kl_weight=args.kl_weight)

    for k, prompt_batch in enumerate(prompt_loader):
        
        rollout_returns = []
        replay_buffer.clear()

        questions = prompt_batch["question"]
        answers = prompt_batch["answer"]

        with torch.no_grad():
            for q, a in zip(questions, answers):
                sequence_ids, returns, action_mask, completions = rollout(
                    model,
                    tokenizer,
                    q,
                    a,
                    num_rollouts=args.group_size,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

