from transformers import PretrainedConfig
from typing import Optional, Union, Dict


class LLMConfig(PretrainedConfig):
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
        >>> config = LLMConfig(dim=256, n_layers=4, use_moe=True)
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
    def from_dict(cls, config_dict: Dict) -> "LMConfig":
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
        config.update(self.to_diff_dict())  # 合并 base class 的参数
        return config
