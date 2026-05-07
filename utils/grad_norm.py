"""梯度范数计算工具 - 在 loss.backward() 后、optimizer.step() 前调用"""


def compute_grad_norm(model) -> float:
    """计算模型所有参数梯度的 L2 范数，用于监控梯度爆炸/消失"""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
