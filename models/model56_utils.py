"""
图结构处理工具函数
"""
import torch
from typing import Optional


def preprocess_adj(adj: torch.Tensor, L: int) -> torch.Tensor:
    """
    预处理邻接矩阵：归一化到 0/1 并屏蔽 [1,9] 范围内的边（如果有）。
    """
    adj = adj.masked_fill(torch.logical_and(adj > 1, adj < 9), 0)
    adj = adj.masked_fill(adj != 0, 1).float()
    return adj


def mask_cls_in_adj(adj: torch.Tensor) -> torch.Tensor:
    """
    将第 0 行和第 0 列（cls token 所在）置零，防止信息泄露。
    返回修改后的副本。
    """
    adj = adj.clone()
    adj[:, 0, :] = 0
    adj[:, :, 0] = 0
    return adj


def build_structural_bias(
    adj: torch.Tensor,
    reachability: Optional[torch.Tensor],
    n_head: int,
) -> torch.Tensor:
    """
    构建 6 通道结构感知偏置矩阵，对应 6 个 attention head。

    通道顺序: [adj, adj^T, reach_fwd, reach_bwd, common_succ, common_pred]
    """
    adj_local = mask_cls_in_adj(adj)

    adj_fwd = adj_local.float()
    adj_bwd = adj_local.mT.float()

    if reachability is not None:
        reach_fwd = (reachability > 0).float()
        reach_bwd = (reachability.mT > 0).float()
    else:
        reach_fwd = (adj_local @ adj_local > 0).float()
        reach_bwd = (adj_local.mT @ adj_local.mT > 0).float()

    common_succ = (torch.bmm(adj_local, adj_local.mT) > 0).float()
    common_pred = (torch.bmm(adj_local.mT, adj_local) > 0).float()

    pe = torch.stack(
        [adj_fwd, adj_bwd, reach_fwd, reach_bwd, common_succ, common_pred], dim=1
    )
    return pe


def apply_structural_bias(
    score: torch.Tensor,
    pe: torch.Tensor,
    L: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将结构偏置应用到 attention score，并加入自环。
    score: (B, H, L, L)
    pe:    (B, 6, L, L)  -> 广播到 H
    """
    pe_expanded = pe.unsqueeze(2).expand(-1, -1, score.size(1) // pe.size(1), -1, -1)
    pe_expanded = pe_expanded.reshape(score.size(0), score.size(1), L, L)

    eye = torch.eye(L, dtype=pe.dtype, device=device)
    pe_expanded = (pe_expanded + eye).int()

    score = score.masked_fill(pe_expanded == 0, -torch.inf)
    return score
