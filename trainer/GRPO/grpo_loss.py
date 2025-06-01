from typing import Optional
import torch
import torch.nn as nn

from replay_buffer import Experience


def approx_kl_divergence(
    log_probs: torch.Tensor,
    log_probs_ref: torch.Tensor,
    action_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    使用Monte Carlo方法近似计算策略的KL散度，衡量当前策略与近似策略的差异
    该计算结果用于在训练中作为正则项，防止策略发生过大的变化

    log_probs: 当前策略对动作的对数概率
    log_probs_ref: 参考策略对动作的对数概率
    action_mask: 用于指示哪些动作被考虑在内的掩码
    
    """

    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        # 该处掩码只计算有效位置处的掩码，掩码用1/0位表示
        log_ratio = log_ratio * action_mask

    return log_ratio.exp() - log_ratio - 1


def masked_mean(
    tensor: torch.Tensor,
    mask: Optional[torch.Tensor],
    dim: int = None,
) -> torch.Tensor:
    """
    计算张量在指定维度上的加权平均值，使用掩码来指示哪些元素被考虑
    """
    if mask is None:
        return tensor.mean(axis=dim)
    return (tensor * mask).sum(axis=dim) / mask.sum(axis=dim)


class GRPOLoss(nn.Module):
    """
    GRPO actor loss
    这段代码实现了GRPO。替代传统PPO算法，旨在通过组内比较来优化策略，减少对价值函数的依赖，从而提高训练效率和稳定性
    """

    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        # 用于限制梯度策略更新幅度的裁剪阈值，防止策略发生剧烈变化
        self.clip_eps = clip_eps
        # 控制KL散度项在总损失中的权重，用于惩罚当前策略与参考策略之间的差异，防止策略偏离参考策略过远
        self.kl_weight = kl_weight

    def forward(
        self,
        log_probs: torch.Tensor,
        experience: Experience,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        log_probs:当前策略下每个动作的对数概率
        experience:包括旧策略信息和优势值的experience实现
        """

        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages
        # 1. 计算kl散度
        kl = approx_kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )
        # 2.计算新老策略概率比
        ratio = (log_probs - old_log_probs).exp()
        # 3.计算裁剪后 及带有优势的损失
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl
        # 4.应用掩码并激素那平均损失
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss, kl.mean()