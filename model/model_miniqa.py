from transformers import PretrainedConfig
from typing import Optional, Union, Dict


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniQA Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

class MiniQAConfig(PretrainedConfig):
    """
    A configuration class for the littleLM language model.

    This class inherits from HuggingFace's `PretrainedConfig` and is used to store
    model architecture parameters, including transformer structure and Mixture-of-Experts (MoE)
    configurations. It supports loading from and saving to dictionaries and JSON files,
    and is compatible with the HuggingFace ecosystem.

    Attributes:
        model_type (str): Identifier for the model type. Defaults to "minimind".

    Basic Transformer Parameters:
        dim (int): Hidden size of the model. Default is 512.
        n_layers (int): Number of transformer layers. Default is 8.
        n_heads (int): Number of attention heads. Default is 8.
        n_kv_heads (int): Number of key/value heads (used for grouped attention). Default is 2.
        vocab_size (int): Vocabulary size. Default is 6400.
        hidden_dim (Optional[int]): Dimension of the feedforward layer (if None, it's computed automatically).
        multiple_of (int): Rounds hidden_dim up to a multiple of this value. Default is 64.
        norm_eps (float): Epsilon value for LayerNorm. Default is 1e-5.
        max_seq_len (int): Maximum sequence length the model supports. Default is 8192.
        rope_theta (float): Base frequency for rotary positional encoding (RoPE). Default is 1e6.
        dropout (float): Dropout probability. Default is 0.0.
        flash_attn (bool): Whether to enable Flash Attention for speed/memory efficiency. Default is True.

    MoE (Mixture-of-Experts) Parameters:
        use_moe (bool): Whether to enable MoE modules. Default is False.
        num_experts_per_tok (int): Number of experts selected per token. Default is 2.
        n_routed_experts (int): Total number of experts to route across. Default is 4.
        n_shared_experts (bool): Whether to share experts across layers. Default is True.
        scoring_func (str): Scoring function used to rank experts. Usually "softmax" or "topk". Default is "softmax".
        aux_loss_alpha (float): Weighting factor for the auxiliary loss (used in load balancing). Default is 0.1.
        seq_aux (bool): Whether to apply the auxiliary loss at sequence-level. Default is True.
        norm_topk_prob (bool): Whether to normalize the top-k probabilities during routing. Default is True.

    Methods:
        from_dict(config_dict): Class method to instantiate from a Python dict.
        to_dict(): Returns the configuration as a dict, including base class params.

    Example:
        >>> config = MiniQAConfig(dim=256, n_layers=4, use_moe=True)
        >>> print(config.to_dict())
    """
    model_type = "littleLM"

    def __init__(
        self,
        dim: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        vocab_size: int = 6400,
        hidden_dim: Optional[int] = None,
        multiple_of: int = 64,
        norm_eps: float = 1e-5,
        max_seq_len: int = 8192,
        rope_theta: float = 1e6,
        dropout: float = 0.0,
        flash_attn: bool = True,
        ##############################
        # MoE 参数
        #########################
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: bool = True,
        scoring_func: str = 'softmax',
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs
    ):
        self.dim = dim                              # 隐藏层维度
        self.n_layers = n_layers                    # Transformer 层数
        self.n_heads = n_heads                      # 多头注意力的头数
        self.n_kv_heads = n_kv_heads                # KV头数
        self.vocab_size = vocab_size                # 词表大小
        self.hidden_dim = hidden_dim                # FFN 隐藏层维度
        self.multiple_of = multiple_of              # FFN 隐藏层对齐单位
        self.norm_eps = norm_eps                    # LayerNorm 的 eps
        self.max_seq_len = max_seq_len              # 最大序列长度
        self.rope_theta = rope_theta                # RoPE 的频率基数
        self.dropout = dropout                      # dropout 概率
        self.flash_attn = flash_attn                # 是否启用 Flash Attention

        # MoE（混合专家）配置
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok  # 每个token选择的专家数量
        self.n_routed_experts = n_routed_experts  # 总的专家数量
        self.n_shared_experts = n_shared_experts  # 共享专家
        self.scoring_func = scoring_func  # 评分函数，默认为'softmax'
        self.aux_loss_alpha = aux_loss_alpha  # 辅助损失的alpha参数
        self.seq_aux = seq_aux  # 是否在序列级别上计算辅助损失
        self.norm_topk_prob = norm_topk_prob  # 是否标准化top-k概率

        super().__init__(**kwargs)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "MiniQAConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        config = {
            "dim": self.dim,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "n_kv_heads": self.n_kv_heads,
            "vocab_size": self.vocab_size,
            "hidden_dim": self.hidden_dim,
            "multiple_of": self.multiple_of,
            "norm_eps": self.norm_eps,
            "max_seq_len": self.max_seq_len,
            "rope_theta": self.rope_theta,
            "dropout": self.dropout,
            "flash_attn": self.flash_attn,
            "use_moe": self.use_moe,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_routed_experts": self.n_routed_experts,
            "n_shared_experts": self.n_shared_experts,
            "scoring_func": self.scoring_func,
            "aux_loss_alpha": self.aux_loss_alpha,
            "seq_aux": self.seq_aux,
            "norm_topk_prob": self.norm_topk_prob,
        }
        # config.update(self.to_diff_dict())  # 合并 base class 的参数
        return config

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             MiniQA Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from model.model_miniqa import MiniQAConfig
from log.LoggerHelper import LoggerHelper

logger = LoggerHelper(name="Model", log_dir="train_logs")

class CausalMaskModule(nn.Module):
    def __init__(self, max_seq_len: int):
        super().__init__()

        # 步骤1：创建一个形状为[1, 1, max_seq_len, max_seq_len]的四维张量，张量中值全部是-inf
        fourTensorMatrix = torch.full((1, 1, max_seq_len, max_seq_len), float('-inf'))

        # 步骤2：将矩阵的下三角部分（包括主对角线）置为0（保留原值），只保留上三角；
        causalMask = torch.triu(fourTensorMatrix)

        # 创建一个shape为[1, 1, max_seq_len, max_seq_len]的上三角mask
        causal_mask = torch.triu(
            torch.full((1, 1, max_seq_len, max_seq_len), float('-inf'))
        )

        # 使用register_buffer 注册为模块持久属性（不会参与训练，但会保存和转移到GPU）
        self.register_buffer("mask", causal_mask)

    def forward(self):
        # 返回mask用于attention
        # 常见用途：加入attention
        # scores = scores + causal_mask[:, :, :seq_len, :seq_len]
        return self.mask

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)

    RMSNorm 是一种替代 LayerNorm 的归一化方式，它只基于输入向量的 L2 范数（均方根）进行归一化，
    去除了均值计算，具有更低的计算开销和更稳定的训练表现，广泛应用于 LLM（大语言模型）中。

    对于输入向量 x，RMSNorm 计算：
    y = x / RMS(x) * weight
    其中 RMS(x) = sqrt(mean(x^2) + eps)
    weight 是一个可训练的缩放因子

    args:
        dim(int):输入特征维度
        eps(float):为避免除零误差添加的稳定常数，通常为 1e-5 或 1e-6
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        对张量执行RMS归一化

        args:
            x(Tensor): 输入张量，形状为[batch_size, seq_len, dim]
        returns:
            Tensor: 归一化后的张量，形状同输出
        """

        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * (x * norm)

def precompute_pos_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    pos_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return pos_cis

def apply_rotary_emb(xq, xk, pos_cis):
    """
    将旋转位置编码应用于 Query 和 Key。

    Args:
        xq (Tensor): 查询张量，形状为 [batch_size, seq_len, n_heads, head_dim]
        xk (Tensor): 键张量，形状同上
        pos_cis (Tensor): 位置旋转系数，形状为 [seq_len, head_dim]

    Returns:
        Tuple[Tensor, Tensor]: 应用了旋转编码的 xq 和 xk
    """
    def unite_shape(pos_cis, x):
        shape = [d if i in {1, x.ndim - 1} else 1 for i, d in enumerate(x.shape)]
        return pos_cis.view(*shape)

    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    pos_cis = unite_shape(pos_cis, xq_)
    xq_out = torch.view_as_real(xq_ * pos_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * pos_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 KV 重复以匹配多头 Attention。

    Args:
        x (Tensor): KV 输入，形状为 [batch, seq_len, kv_heads, head_dim]
        n_rep (int): 每个 KV 复制的次数

    Returns:
        Tensor: 重复后的张量，形状为 [batch, seq_len, heads, head_dim]
    """
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)

class Attention(nn.Module):
    """
    实现多头注意力（Multi-head Attention）模块，支FlashAttention、KV缓存，实现旋转位置编码
    可用于自回归语言模型中的Transformer Block

    输入：[batch_size, seq_len, dim]
    输出：[batch_size, seq_len,dim]
    """
    def __init__(self, config: MiniQAConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        assert self.n_heads % self.n_kv_heads == 0, "head数需能整除kv_head数"

        self.head_dim = config.dim // self.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        # wq/wk/wv/wo:分别为query、key、value以及输出的线性层映射
        # head_dim：每个注意力头的维度 等于dim // m_heads
        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        if not self.use_flash:
            logger.warning("⚠ Flash Attention 不可用，使用普通 Attention 实现。")

        # 预计算的上三角mask（仅用于 causal attention）
        # mask是上三角矩阵，用于实现因果Mask（防止模型看未来的token）
        self.register_buffer("mask",
                             torch.triu(torch.full((1, 1, config.max_seq_len, config.max_seq_len), float("-inf")), diagonal=1))

    def forward(
            self,
            x: torch.Tensor,
            pos_cis: torch.Tensor,
            past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            use_cache:bool=False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Attntion 前向传播

        args:
            x(Tensor):输入张量 [batch_size, seq_lem. dim]
            pos_cis(Tensor):旋转位置编码系数
            past_key_value(Tuple):历史kv缓存
            use_cache(bool):是否启用缓存（用于推理）

        returns:
            Tuple:
                -output:输出张量[batch_size, seq_len, dim]
                - past_kv: 缓存的KV对（若use_cache=True）
        """
        bsz, seq_len, _ = x.shape

        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        # 实现RoPE（旋转位置编码）机制，模型对长距离序列的处理能力
        xq, xk = apply_rotary_emb(xq, xk, pos_cis)

        # 推理时使用KV缓存 加速生成
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)  # [B, H, T, D]
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        # 自动判断是否使用pytorch的FlashAttention
        if self.use_flash and seq_len != 1:
            dropout_p = self.attn_dropout.p if self.training else 0.0
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + self.mask[:, :, :seq_len, :xk.shape[2]]
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output)), past_kv

class FeedForward(nn.Module):
    """
    FeedForward Module used in Transformer Blocks

    本模块是 Transformer 中的前馈全连接层，采用 SwiGLU（或近似）结构进行非线性变换，
    包括一个上升投影（hidden_dim）、非线性激活、下降投影回 dim 维度。
    引入 dropout 提高训练时的泛化能力。

    输出 y = Dropout(W2(SiLU(W1(x)) * W3(x)))

    - W1: 线性变换 (dim -> hidden_dim)
    - W3: Gate控制 (dim -> hidden_dim)
    - W2: 回投影 (hidden_dim -> dim)
    - SiLU: 激活函数，相比 ReLU 更平滑



    """
    def __init__(self, config: MiniQAConfig):
        super().__init__()

        # 自动计算 hidden_dim（推荐结构）并向上取整到 multiple_of 的倍数
        if config.hidden_dim is None:
            intermediate_dim = int((4 * config.dim * 2) / 3)
            config.hidden_dim = config.multiple_of * ((intermediate_dim + config.multiple_of - 1) // config.multiple_of)

        self.dim = config.dim
        self.hidden_dim = config.hidden_dim

        # 三个线性层：主通道 w1，门控通道 w3，回投影 w2
        self.w1 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.w2 = nn.Linear(self.hidden_dim, self.dim, bias=False)

        # Dropout 在输出投影之后
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：执行siLU激活后的门控乘法，再通过输出投影降维

        args:
            x (Tensor):输入张量[batch_size, seq_len, dim]

        return:
            Tensor: 输出张量 [batch_size, seq_len, dim]
        """
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class MoEGate(nn.Module):
    """
    Mixture of Experts (MoE) Gate Module

    MoE门控模块根据输入的隐藏状态 `hidden_states` 对专家路由权重进行评分，
    并选择前 top_k 个专家进行路由。支持 softmax 评分机制和辅助损失（aux_loss），
    以提升路由的均匀性。



    """
    def __init__(self, config: MiniQAConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha  # auxiliary loss scale
        self.seq_aux = config.seq_aux       # 是否基于序列计算 aux loss
        self.norm_topk_prob = config.norm_topk_prob

        self.gating_dim = config.dim
        self.weight = nn.Parameter(torch.empty(self.n_experts, self.gating_dim))  # [n_experts, dim]
        self.reset_parameters()

    def reset_parameters(self):
        """
        初始化专家权重矩阵
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播函数

        Args:
            hidden_states: Tensor, [batch_size, seq_len, dim]

        Returns:
            topk_idx: Tensor, [batch_size * seq_len, top_k]
            topk_weight: Tensor, [batch_size * seq_len, top_k]
            aux_loss: Tensor or float
        """
        bsz, seq_len, dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, dim)  # [bsz * seq_len, dim]

        # 计算 gate scores
        logits = F.linear(hidden_states, self.weight)  # [tokens, n_experts]

        if self.scoring_func == 'softmax':
            scores = F.softmax(logits, dim=-1)
        else:
            raise NotImplementedError(f"Unsupported scoring function: {self.scoring_func}")

        # top-k 选择
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)  # [tokens, top_k]

        if self.top_k > 1 and self.norm_topk_prob:
            topk_weight = topk_weight / (topk_weight.sum(dim=-1, keepdim=True) + 1e-20)

        # Auxiliary Loss (训练阶段使用)
        aux_loss = self.compute_aux_loss(scores, topk_idx, bsz, seq_len) if self.training and self.alpha > 0.0 else 0.0

        return topk_idx, topk_weight, aux_loss

    def compute_aux_loss(self, scores: torch.Tensor, topk_idx: torch.Tensor, bsz: int, seq_len: int) -> torch.Tensor:
        """
        计算辅助损失 aux loss，以促进 MoE 路由的均衡性

        Returns:
            aux_loss: Tensor
        """
        aux_topk = self.top_k
        scores_flat = scores  # [tokens, n_experts]
        topk_idx_flat = topk_idx.view(bsz, -1)  # [bsz, seq_len * top_k]

        if self.seq_aux:
            # 基于序列计算aux loss
            scores_seq = scores_flat.view(bsz, seq_len, -1)  # [bsz, seq_len, n_experts]
            ce = torch.zeros(bsz, self.n_experts, device=scores.device)
            ce.scatter_add_(1, topk_idx_flat,
                            torch.ones_like(topk_idx_flat, dtype=torch.float32, device=scores.device))
            ce = ce / (seq_len * aux_topk / self.n_experts)  # normalize
            aux_loss = (ce * scores_seq.mean(dim=1)).sum(dim=1).mean() * self.alpha
        else:
            # 基于全局 token 分布
            one_hot_mask = F.one_hot(topk_idx_flat.view(-1), num_classes=self.n_experts).float()
            ce = one_hot_mask.mean(0)  # shape: [n_experts]
            Pi = scores_flat.mean(0)
            fi = ce * self.n_experts
            aux_loss = (Pi * fi).sum() * self.alpha

        return aux_loss


class MOEFeedForward(nn.Module):
    """
    Mixture-of-Experts FeedForward 模块

    - 基于 MoEGate 动态选择多个专家执行前馈操作（FeedForward）
    - 支持训练阶段的 top-k 路由与辅助损失计算
    - 推理阶段仅使用 top-1 专家，提高效率
    - 支持共享专家路径（Shared Experts）
    """
    def __init__(self, config: MiniQAConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        self.use_shared_expert = config.n_shared_experts is not None
        self.shared_expert = FeedForward(config) if self.use_shared_expert else None
        self.aux_loss = None  # type: Optional[torch.Tensor]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播：根据 gate 选择专家并路由计算
        """
        identity = x
        bsz, seq_len, dim = x.shape
        orig_shape = x.shape

        # 路由 Gate
        topk_idx, topk_weight, aux_loss = self.gate(x)  # shape: [tokens, top_k]
        self.aux_loss = aux_loss

        x = x.view(-1, dim)  # shape: [tokens, dim]
        flat_topk_idx = topk_idx.view(-1)  # [tokens * top_k]

        if self.training:
            # 训练阶段（全路径专家参与）
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)  # [tokens * top_k, dim]
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                mask = flat_topk_idx == i
                if mask.any():
                    y[mask] = expert(x[mask]).to(y.dtype)
            y = (y.view(-1, self.config.num_experts_per_tok, dim) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            # 推理阶段（仅用 top-1 路由）
            y = self._moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)

        # 共享专家通路（残差融合）
        if self.use_shared_expert:
            y = y + self.shared_expert(identity)

        return y

    @torch.no_grad()
    def _moe_infer(
        self,
        x: torch.Tensor,
        flat_expert_indices: torch.Tensor,
        flat_expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        推理阶段使用：只执行 top-1 路由专家
        使用 scatter_add 实现专家路由和结果合并

        Args:
            x: 输入 token [tokens, dim]
            flat_expert_indices: 每个 token 的专家 ID [tokens]
            flat_expert_weights: 每个 token 的权重 [tokens, 1]

        Returns:
            expert_cache: 所有专家加权输出后的结果 [tokens, dim]
        """
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        sorted_indices = flat_expert_indices[idxs]
        tokens_per_expert = torch.bincount(sorted_indices).cpu().cumsum(0)

        token_idxs = idxs // self.config.num_experts_per_tok  # 每个专家处理哪些 token

        for i, end in enumerate(tokens_per_expert):
            start = 0 if i == 0 else tokens_per_expert[i - 1]
            if start == end:
                continue
            expert = self.experts[i]
            token_ids = token_idxs[start:end]
            tokens = x[token_ids]
            outputs = expert(tokens).to(expert_cache.dtype)
            outputs.mul_(flat_expert_weights[idxs[start:end]])
            expert_cache.scatter_add_(
                0,
                token_ids.view(-1, 1).expand(-1, x.shape[1]),
                outputs
            )

        return expert_cache

class MiniQABlock(nn.Module):
    """
    MiniQABlock 是 LLM 模型中的基本构建单元之一，通常被多次堆叠以构成完整的 Transformer 结构。

    模块结构：
    - 规范化 + 多头注意力（Attention）
    - 残差连接（Residual）
    - 规范化 + 前馈网络（FeedForward 或 MoE）
    - 残差连接（Residual）

    参数:
        layer_id (int): 当前层编号，仅用于追踪或调试。
        config (MiniQAConfig): 模型配置参数，包含维度、头数、是否启用 MoE 等。
    """
    def __init__(self, layer_id: int, config: MiniQAConfig):
        super().__init__()
        self.layer_id = layer_id
        self.n_heads = config.n_heads
        self.dim = config.dim
        self.head_dim = self.dim // self.n_heads

        # 多头注意力模块
        self.attention = Attention(config)

        # RMS规范化（注意力前）
        self.attention_norm = RMSNorm(self.dim, eps=config.norm_eps)

        # RMS规范化（FFN前）
        self.ffn_norm = RMSNorm(self.dim, eps=config.norm_eps)

        # 前馈网络（普通或 MoE）
        self.feed_forward = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(
        self,
        x: torch.Tensor,
        pos_cis: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, seq_len, dim]
            pos_cis (torch.Tensor): 旋转位置编码向量
            past_key_value (Optional[Tuple]): 用于 KV Cache 加速推理
            use_cache (bool): 是否启用 KV Cache

        返回:
            Tuple[torch.Tensor, Optional[Tuple]]: 输出张量 和 可选的 KV 缓存
        """
        # 注意力层
        h_attn, past_kv = self.attention(
            self.attention_norm(x),
            pos_cis,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        h = x + h_attn  # 残差连接

        # 前馈网络层
        out = h + self.feed_forward(self.ffn_norm(h))  # 残差连接

        return out, past_kv

class MiniQALM(PreTrainedModel):

    config_class = MiniQAConfig

    def __init__(self, config: MiniQAConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        # Embedding层（输入嵌入 + 位置编码）
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer Block 结构
        self.layers = nn.ModuleList([MiniQABlock(i, config) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # 输出层
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight  # 权重共享

        # RoPE旋转位置编码缓存
        self.register_buffer(
            "pos_cis",
            precompute_pos_cis(dim=config.dim // config.n_heads, theta=config.rope_theta),
            persistent=False
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> CausalLMOutputWithPast:
        """
        标准前向传播，用于训练或推理。

        Args:
            input_ids: 输入token序列 (batch, seq_len)
            past_key_values: KV缓存
            use_cache: 是否缓存KV用于推理
            kwargs: 可包含 start_pos

        Returns:
            CausalLMOutputWithPast，包括 logits、past_key_values、aux_loss
        """
        ### 1.起始位置的处理
        # 在生成时，如果是单token增量生成，需要设置start_pos 用来告诉模型 位置编码从哪个位置开始算
        # 如果是训练 一般从0开始
        start_pos = kwargs.get("start_pos", 0)

        ### 2. KV 缓存初始化
        # 如果没有传入历史KV缓存，就是初始化为[None] * 层数
        past_key_values = past_key_values or [None] * self.n_layers

        ### 3. Token embedding + Dropout
        # input_id:[batch_size, seq_len]
        # embedding将Input_id每个token编码成向量[dim]
        hidden_states = self.dropout(self.tok_embeddings(input_ids))

        ### 4. 提取位置编码
        pos_cis = self.pos_cis[start_pos: start_pos + input_ids.size(1)]

        ### 5. 多层Transformer前向传播
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            hidden_states, pkv = layer(
                hidden_states,
                pos_cis,
                past_key_value=past_key_values[i],
                use_cache=use_cache
            )
            new_past_key_values.append(pkv)

        ### 6.输出层（logits）
        logits = self.output(self.norm(hidden_states))
        ### 7. 计算MoE的辅助损失
        aux_loss = sum(
            layer.feed_forward.aux_loss
            for layer in self.layers
            if isinstance(layer.feed_forward, MOEFeedForward)
        )

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=new_past_key_values,
            loss=aux_loss,
            hidden_states=None,
            attentions=None,
            # cross_attentions=None,
            # loss=aux_loss,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        eos_token_id: int = 2,
        max_new_tokens: int = 1024,
        temperature: float = 0.75,
        top_p: float = 0.9,
        stream: bool = False,
        repetition_penalty: float = 1.0,
        use_cache: bool = True,
        pad_token_id: int = 0,
        **kwargs
    ):
        """
        推理阶段生成文本接口（支持流式与静态生成）。
        """
        if stream:
            return self._stream(
                input_ids, eos_token_id, max_new_tokens,
                temperature, top_p, repetition_penalty,
                use_cache, **kwargs
            )

        generated = []
        for i in range(input_ids.size(0)):
            non_pad = input_ids[i][input_ids[i] != pad_token_id].unsqueeze(0)
            out = self._stream(non_pad, eos_token_id, max_new_tokens, temperature, top_p, repetition_penalty, use_cache, **kwargs)
            tokens_list = [tokens[:, -1:] for tokens in out]
            gen = torch.cat(tokens_list, dim=-1) if tokens_list else non_pad
            full_sequence = torch.cat([non_pad, gen], dim=-1)
            generated.append(full_sequence)

        max_length = max(seq.size(1) for seq in generated)
        padded = [
            torch.cat([seq, torch.full((1, max_length - seq.size(1)), pad_token_id, dtype=seq.dtype, device=seq.device)], dim=-1)
            for seq in generated
        ]
        return torch.cat(padded, dim=0)

    def _stream(
        self, input_ids, eos_token_id, max_new_tokens,
        temperature, top_p, repetition_penalty, use_cache, **kwargs
    ):
        """
        流式生成器（逐token返回新生成的tokens）。
        """
        past_key_values = None
        first_token = True
        start = input_ids.shape[1]

        while input_ids.shape[1] < max_new_tokens:
            if first_token or not use_cache:
                out = self(
                    input_ids, past_key_values=past_key_values,
                    use_cache=use_cache, **kwargs
                )
                first_token = False
            else:
                out = self(
                    input_ids[:, -1:], past_key_values=past_key_values,
                    use_cache=use_cache, start_pos=input_ids.shape[1] - 1,
                    **kwargs
                )

            logits, past_key_values = out.logits[:, -1, :], out.past_key_values
            logits[:, list(set(input_ids[0].tolist()))] /= repetition_penalty
            logits = logits / (temperature + 1e-8)

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = sorted_probs.cumsum(dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1]
                sorted_mask[:, 0] = False
                mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[mask] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            yield input_ids[:, start:]

            if next_token.item() == eos_token_id:
                break

class MiniQALMLite(PreTrainedModel):
    """
    MiniQALM（简化版）用于轻量级本地调试，仅保留 Embedding、Dropout、Linear 层，
    无 Transformer Block、无 RoPE。
    """
    config_class = MiniQAConfig

    def __init__(self, config: MiniQAConfig):
        super().__init__(config)

        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers  # 保留字段，forward 中用作 past_kv 长度占位

        # 嵌入层（token embedding）
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)

        # Dropout（增强训练鲁棒性）
        self.dropout = nn.Dropout(config.dropout)

        # RMSNorm 层（代替 LayerNorm）
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # 输出层（映射到 vocab logits）
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 权重共享（Embedding 与输出层）
        self.output.weight = self.tok_embeddings.weight

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
            use_cache: bool = False,
            **kwargs
    ) -> CausalLMOutputWithPast:
        """
        简化版前向传播：用于本地轻量调试，去除Transformer结构
        """
        # 1. 处理起始位置
        start_pos = kwargs.get("start_pos", 0)

        # 2. 模拟 KV 缓存结构（初始化为空）
        past_key_values = past_key_values or [None] * self.n_layers

        # 3. Token embedding + Dropout
        hidden_states = self.dropout(self.tok_embeddings(input_ids))

        # 4. 模拟位置编码（不再使用pos_cis）
        # 你可以忽略或者直接使用 embedding 后的 hidden_states

        # 5. 跳过Transformer，只用一个线性层做映射
        # （实际等价于随机初始化的一层 MLP）
        hidden_states = self.norm(hidden_states)
        logits = self.output(hidden_states)

        # 6. 伪造MoE aux_loss（用于保持结构一致）
        aux_loss = torch.tensor(0.0, device=input_ids.device)

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,  # 伪造空缓存
            loss=aux_loss,
            hidden_states=None,
            attentions=None,
            # aux_loss=aux_loss,
        )

if __name__ == "__main__":
    # 测试mask
    mask = CausalMaskModule(max_seq_len=8)
    print(mask())


