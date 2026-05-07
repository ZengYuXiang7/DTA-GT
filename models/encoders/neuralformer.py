from typing import List

import torch
import torch.nn.functional as F


def tokenizer(
    ops: List[int],
    adj,
    depth: int,
    op_depth: List[int],
    dim_x: int = 192,
    embed_type: str = "onehot_op",
):
    """Build the architecture token features consumed by NASBench datasets."""
    if embed_type != "onehot_op":
        raise ValueError(f"Unsupported embed_type: {embed_type}")

    adj = torch.tensor(adj)
    code_ops = F.one_hot(torch.tensor(ops), num_classes=dim_x)
    code_depth = F.one_hot(torch.tensor([depth]), num_classes=dim_x)
    code_op_depth = F.one_hot(torch.tensor(op_depth), num_classes=dim_x)
    return (
        code_ops.to(torch.int8),
        adj.to(torch.int8),
        code_depth.to(torch.int8),
        code_op_depth.to(torch.int8),
    )
