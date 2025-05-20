import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import PretrainedConfig
from typing import Optional, Union, Dict


# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘
#                                             DeciMind Config
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

class DeciMindConfig(PretrainedConfig):

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
    def from_dict(cls, config_dict: Dict) -> "DeciMindConfig":
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
#                                             DeciMind Model
# 📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘📘

from typing import Optional, Tuple, List
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
 
from log.LoggerHelper import LoggerHelper

logger = LoggerHelper(name="testModel", log_dir="train_logs")

def apply_rotary_emb(x_query, x_key, pos_cis):
    """
    用复数运算的方式将位置信息编码于Query和Key

    目标：在不引入额外参数的前提下，将 相对位置关系 自然的从融入到Query和Key中，从而让注意力具备位置感知能力

    复数视角下的位置编码：
        假设head_dim = 2d,将每两个维度看成一个负数：Z=x_{2i} + i \dot {x_{2i+1}}
        RoPE定义一个旋转矩阵，其本质是将这个复数向量 固定频率旋转:RoPE(z,theta)=z\dot(e^{i theta})

    输入：查询张量(x_query) 键张量(key_value) 旋转位置编码(pos_cis)
    返回: 应用了旋转位置编码的x_query x_key
    """
    x_query_ = torch.view_as_complex(x_query.float().reshape(*x_query[:-1], -1, 2))
    x_key_ = torch.view_as_complex(x_key.float().reshape(*x_key[:-1], -1, -2))


class Attention(nn.Module):
    def __init__(self, config: DeciMindConfig):
        super().__init__()

        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        # 保证Q head对KV head进行统一的分组和映射
        # 可能存在一个Q head 对 一个 kv head
        # 也可能多个Qhead 对一个kv head
        assert self.n_heads % self.n_kv_heads == 0, "head数需要整除kv head数"

        self.head_dim = config.dim // self.n_heads
        # kv head共享次数 用于MQA/GQA
        self.n_rep = self.n_head // self.n_kv_heads

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)

        self.attn_dropput = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.use_flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn
        if not self.use_flash:
            logger.warning("⚠ Flash Attention 不可用，使用普通 Attention 实现。")
        # 注册生成上三角矩阵 
        # mask是上三角矩阵 用于实现因果mask （防止模型看到未来信息）
        self.register_buffer("mask", 
            torch.triu(torch.full(1, 1, config.max_seq_len, max_seq_len), float("-inf"), diagonal=1))
    
    def forward(
        self,
        x: torch.Tensor,
        pos_cis = torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache:bool=True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Attention前向传播
        
        args:
            x(Tensor): input tensor [batch_size, seq_len, dim]
            pos_cis(Tensor): 旋转位置编码系数
            past_key_value(tuple): 历史kv缓存
            use_cache(bool): 是否启用缓存（用于推理）
        returns:
            Tuple:
             - output : output tensor of shape [batch_size, seq_len, dim]
             - past_kv: 
        """
        batch_size, seq_len, dim = x.shape
        x_query = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        x_key = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        x_value = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)


class DeciMindBlock(nn.Module):
    """
    
    """
    def __init__(self, layer_id: int, config: DeciMindConfig):
        super().__init__()

        self.layer_id = layer_id
        self.n_hedas = config.n_heads
        self.dim = config.dim
        self.head_dim = self.dim // self.n_hedas # 注意力每个头纬度

        # 多头注意力模块

        
class DeciMindLM(PreTrainedModel):
    def __init__(self, config: DeciMindConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers

        # Embedding
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)

        # Transformer Block


if __name__ == "__main__":

    # 创建一个形状为[1,1,max_seq_len,max_seq_len]的思维张量 张量中值全部是-inf
    max_seq_len=128
    TensorMatrix = torch.full((1, 1, max_seq_len, max_seq_len), float('-inf'))
    # 将矩阵的下三角部分（包括主对角线）置为0（保留原值），只保留上三角
    causal_mask = torch.triu(TensorMatrix, diagonal=1)
    
    config = DeciMindConfig(
        vocab_size=1000,
        dim=256,
        n_layers=2,
        n_heads=4,
        max_seq_len=128,
        dropout=0.1,
        use_moe=True,
        n_routed_experts=4,
        num_experts_per_tok=2
    )
    # 模型初始化
    # model = DeciMindLM(config)
    # model.eval()

    # 构造输入
    batch_size = 2
    seq_len = 128
    vocab_size = config.vocab_size
    # input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    # print("input_ids.shape:", input_ids.shape)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(input_ids.shape)

    # 均匀分布输入
    x_query = torch.rand(batch_size, seq_len, config.n_heads, config.dim // config.n_heads) 
    x_key = torch.rand(batch_size, seq_len, config.n_heads, config.dim // config.n_heads)
    print(x_query.shape)
    pos_cis = config.rope_theta
    # [batch_size, seq_len, head_num, head_dim] -> 
    # [batch_size, seq_len, head_num, head_dim // 2, 2]
    x_query_ = x_query.float().reshape(*x_query.shape[:-1], -1, 2)
    x_key_ = x_key.float().reshape(*x_key.shape[:-1], -1, 2)
    # 将 x_query_看成一个复数
    # real = x_query_[...,0] imag = x_query_[...,1]
    # x_query_complex = real + i.imag
    x_query_complex = torch.view_as_complex(x_query_)
    k_key_complex = torch.view_as_complex(x_key_)
    print(x_query_.shape)




    


    