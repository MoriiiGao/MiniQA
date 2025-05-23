import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import PretrainedConfig
from typing import Optional, Union, Dict

# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             DeciMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from transformers import PretrainedConfig
from typing import Dict

class DeciMindConfig(PretrainedConfig):
    """
    A configuration class for the DeciMind language model.

    Inherits from HuggingFace's PretrainedConfig and includes parameters
    for transformer architecture and Mixture-of-Experts (MoE) configurations.
    """

    model_type = "decimind"

    def __init__(
            self,
            dropout: float = 0.0,
            bos_token_id: int = 1,
            eos_token_id: int = 2,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_hidden_layers: int = 8,
            num_key_value_heads: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-05,
            rope_theta: float = 1e6,
            flash_attn: bool = False,
            use_moe: bool = False,
            num_experts_per_tok: int = 2,
            n_routed_experts: int = 4,
            n_shared_experts: int = 1,
            scoring_func: str = 'softmax',
            aux_loss_alpha: float = 0.1,
            seq_aux: bool = True,
            norm_topk_prob: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.flash_attn = flash_attn

        # MoE config
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.scoring_func = scoring_func
        self.aux_loss_alpha = aux_loss_alpha
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "DeciMindConfig":
        return cls(**config_dict)

    def to_dict(self) -> Dict:
        config = super().to_dict()
        config.update({
            "dropout": self.dropout,
            "bos_token_id": self.bos_token_id,
            "eos_token_id": self.eos_token_id,
            "hidden_act": self.hidden_act,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "num_attention_heads": self.num_attention_heads,
            "num_hidden_layers": self.num_hidden_layers,
            "num_key_value_heads": self.num_key_value_heads,
            "vocab_size": self.vocab_size,
            "rms_norm_eps": self.rms_norm_eps,
            "rope_theta": self.rope_theta,
            "flash_attn": self.flash_attn,
            "use_moe": self.use_moe,
            "num_experts_per_tok": self.num_experts_per_tok,
            "n_routed_experts": self.n_routed_experts,
            "n_shared_experts": self.n_shared_experts,
            "scoring_func": self.scoring_func,
            "aux_loss_alpha": self.aux_loss_alpha,
            "seq_aux": self.seq_aux,
            "norm_topk_prob": self.norm_topk_prob
        })
        return config


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             DeciMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import CausalLMOutputWithPast

# from model.model_decimind import DeciMindConfig
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

class RMSNorm(torch.nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

def precompute_freqs_cis(dim: int, end: int = int(32 * 1024), theta: float = 1e6):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed, k_embed

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, num_key_value_heads, n_rep, head_dim)
        .reshape(bs, slen, num_key_value_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: DeciMindConfig):
        super().__init__()
        self.num_key_value_heads = args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads
        assert args.num_attention_heads % self.num_key_value_heads == 0
        self.n_local_heads = args.num_attention_heads
        self.n_local_kv_heads = self.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.hidden_size // args.num_attention_heads
        self.q_proj = nn.Linear(args.hidden_size, args.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(args.num_attention_heads * self.head_dim, args.hidden_size, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and args.flash_attn
        # print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

    def forward(self,
                x: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],  # 修改为接收cos和sin
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache=False,
                attention_mask: Optional[torch.Tensor] = None):
        bsz, seq_len, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos[:seq_len], sin[:seq_len])

        # kv_cache实现
        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2)
        )

        if self.flash and seq_len != 1:
            print(self.flash)
            dropout_p = self.dropout if self.training else 0.0
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1)
                attn_mask = attn_mask.bool() if attention_mask is not None else None

            output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=True)
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)
            scores = scores + torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=scores.device),
                diagonal=1
            ).unsqueeze(0).unsqueeze(0)  # scores+mask

            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = scores @ xv

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv

class FeedForward(nn.Module):
    def __init__(self, config: DeciMindConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = int(config.hidden_size * 8 / 3)
            config.intermediate_size = 64 * ((intermediate_size + 64 - 1) // 64)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))

class MoEGate(nn.Module):
    def __init__(self, config: DeciMindConfig):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts

        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.seq_aux = config.seq_aux

        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
            if self.seq_aux:
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                ce.scatter_add_(1, topk_idx_for_aux_loss,
                                torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device)).div_(
                    seq_len * aux_topk / self.n_routed_experts)
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                ce = mask_ce.float().mean(0)
                Pi = scores_for_aux.mean(0)
                fi = ce * self.n_routed_experts
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0
        return topk_idx, topk_weight, aux_loss


class MOEFeedForward(nn.Module):
    def __init__(self, config: DeciMindConfig):
        super().__init__()
        self.config = config
        self.experts = nn.ModuleList([
            FeedForward(config)
            for _ in range(config.n_routed_experts)
        ])
        self.gate = MoEGate(config)
        if config.n_shared_experts > 0:
            self.shared_experts = nn.ModuleList([
                FeedForward(config)
                for _ in range(config.n_shared_experts)
            ])

    def forward(self, x):
        identity = x
        orig_shape = x.shape
        bsz, seq_len, _ = x.shape
        # 使用门控机制选择专家
        topk_idx, topk_weight, aux_loss = self.gate(x)
        x = x.view(-1, x.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            x = x.repeat_interleave(self.config.num_experts_per_tok, dim=0)
            y = torch.empty_like(x, dtype=torch.float16)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(x[flat_topk_idx == i]).to(y.dtype)  # 确保类型一致
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.view(*orig_shape)
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.config.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y

    @torch.no_grad()
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        expert_cache = torch.zeros_like(x)
        idxs = flat_expert_indices.argsort()
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        token_idxs = idxs // self.config.num_experts_per_tok
        # 当tokens_per_expert = [6, 15, 20, 26]，tokens_per_expert.shape[0]即为专家数量（此时为4）
        # 且token_idxs = [3, 7, 19, 21, 24, 25,  4,  5,  6, 10, 11, 12...] 时
        # 意味token_idxs[:6] -> [3, 7, 19, 21, 24, 25]这6个位置属于专家0处理的token（每个token有可能被多个专家处理，这取决于num_experts_per_tok）
        # 接下来9个位置token_idxs[6:15] -> [4,  5,  6, 10, 11, 12...]属于专家1处理的token...依此类推
        for i, end_idx in enumerate(tokens_per_expert):
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            expert = self.experts[i]
            exp_token_idx = token_idxs[start_idx:end_idx]
            expert_tokens = x[exp_token_idx]
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache

class DeciMindBlock(nn.Module):
    def __init__(self, layer_id: int, config: DeciMindConfig):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.self_attn = Attention(config)

        self.layer_id = layer_id
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            past_key_value, use_cache, attention_mask
        )
        hidden_states += residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value

class DeciMindLM(nn.Module):
    def __init__(self, config: DeciMindConfig):
        super().__init__()
        self.config = config
        self.vocab_size, self.num_hidden_layers = config.vocab_size, config.num_hidden_layers
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([DeciMindBlock(l, config) for l in range(self.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs_cos, freqs_sin = precompute_freqs_cis(dim=config.hidden_size // config.num_attention_heads,
                                                    end=config.max_position_embeddings, theta=config.rope_theta)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs):
        batch_size, seq_length = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0

        hidden_states = self.dropout(self.embed_tokens(input_ids))

        position_embeddings = (
            self.freqs_cos[start_pos:start_pos + seq_length],
            self.freqs_sin[start_pos:start_pos + seq_length]
        )

        presents = []
        for layer_idx, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)

        hidden_states = self.norm(hidden_states)

        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MOEFeedForward)
        )

        return hidden_states, presents, aux_loss

# class DeciMindForCausalLM(PreTrainedModel, GenerationMixin):
#     config_class = DeciMindConfig

#     def __init__(self, config: DeciMindConfig = None):
#         self.config = config or DeciMindConfig()
#         super().__init__(self.config)
#         self.model = DeciMindLM(self.config)
#         self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
#         self.model.embed_tokens.weight = self.lm_head.weight
#         self.OUT = CausalLMOutputWithPast()

#     def forward(self,
#                 input_ids: Optional[torch.Tensor] = None,
#                 attention_mask: Optional[torch.Tensor] = None,
#                 past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
#                 use_cache: bool = False,
#                 logits_to_keep: Union[int, torch.Tensor] = 0,
#                 **args):
#         h, past_kvs, aux_loss = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             **args
#         )
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(h[:, slice_indices, :])
#         self.OUT.__setitem__('last_hidden_state', h)
#         self.OUT.__setitem__('logits', logits)
#         self.OUT.__setitem__('aux_loss', aux_loss)
#         self.OUT.__setitem__('past_key_values', past_kvs)
#         return self.OUT

class DeciMindForCausalLM(PreTrainedModel, GenerationMixin):

    # 是否支持梯度检查点
    # supports_gradient_checkpointing = True  

    def __init__(self, config: DeciMindConfig = None):
        # 接收模型配置对象
        self.config = config or DeciMindConfig
        super().__init__(self.config)
        
        # 初始化 transformer      
        self.model = DeciMindLM(self.config)
        # 投影层 从隐藏状态 -> vocab logits
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
        # 权重共享 嵌入层与输出层共享权重 节省参数 提升性能
        self.model.embed_tokens.weight = self.lm_head.weight
        # 构造输出容器
        self.OUT = CausalLMOutputWithPast()
    
    def forward(
            self, 
            input_ids: Optional[torch.Tensor] = None, 
            attention_mask: Optional[torch.Tensor] = None, 
            past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
            use_cache: bool = False, 
            logits_to_keep: Union[int, torch.Tensor] = 0, 
            return_dict: Optional[bool] = True,
            **kwargs):
        
        # 调用transofmrer模型主体 输出最后一层的hidden states/缓存的kv/专家路由损失
        hidden_states, past_kvs, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs
        )
        
        # 计算logits(词预测概率) 将hidden states 传入lm_head投影为[batch, seq, vocab_size]
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        
        # 构造标准输出
        if not return_dict:
            return ()
        
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_kvs,
            hidden_states=hidden_states,
            # attentions=None,
            loss=aux_loss  
        )


# class DeciMindForCausalLM(PreTrainedModel, GenerationMixin):

#     # supports_gradient_checkpointing = True

#     def __init__(self, config: DeciMindConfig = None):
#         self.config = config or DeciMindConfig
#         super().__init__(self.config)
#         self.model = DeciMindLM(self.config)
#         self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
#         self.model.embed_tokens.weight = self.lm_head.weight
#         self.OUT = CausalLMOutputWithPast()

#     def forward(
#             self, 
#             input_ids: Optional[torch.Tensor] = None, 
#             attention_mask: Optional[torch.Tensor] = None, 
#             token_type_ids: Optional[torch.Tensor] = None,
#             past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None, 
#             use_cache: bool = False, 
#             logits_to_keep: Union[int, torch.Tensor] = 0, 
#             return_dict: Optional[bool] = True,
#             **kwargs):
#         hidden_states, past_kvs, aux_loss = self.model(
#             input_ids=input_ids,
#             attention_mask=attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             **kwargs
#         )
#         slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
#         logits = self.lm_head(hidden_states[:, slice_indices, :])
#         if not return_dict:
#             return ()
#         return CausalLMOutputWithPast(
#             logits=logits,
#             past_key_values=past_kvs,
#             hidden_states=hidden_states,
#             attentions=None,
#             loss=aux_loss  
#         )

#     def prepare_inputs_for_generation(
#         self,
#         input_ids,
#         past_key_values=None,
#         attention_mask=None,
#         token_type_ids= None,
#         **kwargs
#     ):
        
#         return {
#             "input_ids": input_ids,
#             "past_key_values": past_key_values,
#             "attention_mask": attention_mask,
#             "token_type_ids": token_type_ids,
#             "use_cache": True,
#         }

#     @property
#     def device(self):
#         return next(self.parameters()).device

#     def _update_model_kwargs_for_generation(
#         self, outputs, model_kwargs, is_encoder_decoder=False
#     ):
#         # 兼容 HuggingFace generate 底层的 state 传递
#         model_kwargs["past_key_values"] = outputs.past_key_values
#         if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
#             model_kwargs["attention_mask"] = torch.cat(
#                 [model_kwargs["attention_mask"], torch.ones((model_kwargs["attention_mask"].shape[0], 1), dtype=model_kwargs["attention_mask"].dtype, device=model_kwargs["attention_mask"].device)], dim=-1)
#         return model_kwargs



if __name__ == "__main__":
    # 测试mask
    # mask = CausalMaskModule(max_seq_len=8)
    # print(mask())

    # 构造模型配置
    config = DeciMindConfig(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
        dropout=0.1,
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=2
    )

    # 初始化模型
    model = DeciMindForCausalLM(config)
    model.eval()

    # 3. 构造输入
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # 4. 执行前向传播（不计算梯度）
    with torch.no_grad():
        outputs = model(input_ids=input_ids, logits_to_keep=seq_len)

    # 5. 输出结果结构
    print("Input shape:", input_ids.shape)
    print("Logits shape:", outputs.logits.shape)     # [batch_size, seq_len, vocab_size]
    print("Aux loss:", outputs.loss)                 # 如果 use_moe=True，会输出 aux_loss
