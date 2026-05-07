"""
model56_cls_readout: 消融 - 使用 CLS Token 进行图级表示汇聚

与 model56_full 的区别：
  - graph_readout 强制为 "cls"
  - 使用 CLS Token 而非 Attention Pooling
  - 邻接矩阵需要扩展以包含 CLS Token

消融目的：验证 CLS Token vs Attention Pooling 的效果

注意：这是与 model56_full 的对比实验
- model56_full: 使用 attention pooling (graph_readout="att")
- model56_cls_readout: 使用 CLS token (graph_readout="cls")
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import register_model
from typing import Optional
from torch import Tensor
from timm.models.layers import to_2tuple

from models.model56_utils import (
    preprocess_adj,
    build_structural_bias,
    apply_structural_bias,
)


# ============================================================
# 公共工具
# ============================================================


def make_activation(act_layer: str) -> nn.Module:
    act = act_layer.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {act_layer}")


# ============================================================
# 注意力层（与 model56_full 相同）
# ============================================================


class MultiHeadAttention(nn.Module):
    """带结构感知偏置的多头注意力"""

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_head == 0
        self.n_head = n_head
        self.head_size = dim // n_head
        self.scale = math.sqrt(self.head_size)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _compute_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        return q, k, v

    def _build_bias(self, adj: Tensor, reachability: Optional[Tensor]) -> Tensor:
        return build_structural_bias(adj, reachability, self.n_head)

    def forward(
        self, x: Tensor, adj: Tensor, reachability: Optional[Tensor] = None
    ) -> Tensor:
        B, L, C = x.shape
        q, k, v = self._compute_qkv(x)
        score = torch.matmul(q, k.mT) / self.scale

        adj = preprocess_adj(adj, L)
        pe = self._build_bias(adj, reachability)
        score = apply_structural_bias(score, pe, L, adj.device)

        attn = F.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, C)
        return self.resid_dropout(self.proj(out))


# ============================================================
# FFN 层（与 model56_full 相同）
# ============================================================


class BatchedMoEGraphFFN(nn.Module):
    """MoE-Graph FFN：3个专家"""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
        gate_hidden_ratio: float = 0.5,
        temperature: float = 1.0,
        has_cls: bool = True,
    ):
        super().__init__()
        self.has_cls = has_cls
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        p1, p2 = to_2tuple(drop)

        self.self_expert = nn.Linear(in_features, hidden_features, bias=False)
        self.in_expert = nn.Linear(in_features, hidden_features, bias=False)
        self.out_expert = nn.Linear(in_features, hidden_features, bias=False)

        gate_hidden = int(in_features * gate_hidden_ratio)
        if gate_hidden_ratio > 0:
            self.gate = nn.Sequential(
                nn.Linear(in_features, gate_hidden, bias=False),
                nn.GELU(),
                nn.Linear(gate_hidden, 3, bias=False),
            )
        else:
            self.gate = nn.Linear(in_features, 3, bias=False)

        self.temperature = float(temperature)
        self.act = make_activation(act_layer)
        self.drop1 = nn.Dropout(p1)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop2 = nn.Dropout(p2)

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        L = x.size(1)
        adj = adj.float()
        if self.has_cls:
            adj[:, 0, :] = 0
            adj[:, :, 0] = 0
        adj = adj.masked_fill(torch.eye(L, device=adj.device, dtype=torch.bool), 0)

        e0 = self.self_expert(x)
        e1 = torch.bmm(adj, self.in_expert(x))
        e2 = torch.bmm(adj.transpose(1, 2), self.out_expert(x))

        w = F.softmax(self.gate(x) / self.temperature, dim=-1)
        out = w[..., 0:1] * e0 + w[..., 1:2] * e1 + w[..., 2:3] * e2

        out = self.drop1(self.act(self.fc2(out)))
        out = self.drop2(out)

        routing_penalty = (w * (w + 1e-8).log()).sum(dim=-1).mean()
        return out, routing_penalty


class GraphTransformerLayer(nn.Module):
    """Pre-norm Transformer 层"""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float,
        dropout: float,
        activation: str = "relu",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = BatchedMoEGraphFFN(
            d_model, mlp_ratio=mlp_ratio, act_layer=activation, drop=dropout
        )

    def forward(
        self, x: Tensor, adj: Tensor, reachability: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        x = x + self.self_attn(self.norm1(x), adj, reachability)
        ffn_out, penalty = self.ffn(self.norm2(x), adj)
        x = x + ffn_out
        return x, penalty


class TransformerEncoder(nn.Module):
    """多层 GraphTransformerEncoder"""

    def __init__(
        self,
        d_model: int,
        d_ff_ratio: float,
        gcn_layers: int,
        n_head: int = 4,
        activation: str = "relu",
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                GraphTransformerLayer(
                    d_model=d_model,
                    n_head=n_head,
                    mlp_ratio=d_ff_ratio,
                    dropout=0.1,
                    activation=activation,
                )
                for _ in range(gcn_layers)
            ]
        )

    def forward(
        self, x: Tensor, adj: Tensor, reachability: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        adj = preprocess_adj(adj, x.size(1))
        penalties = []
        for layer in self.layers:
            x, penalty = layer(x, adj, reachability)
            penalties.append(penalty)
        aux_loss = torch.stack(penalties).mean()
        return x, aux_loss


# ============================================================
# 主模型（使用 CLS Readout）
# ============================================================


@register_model("model56_cls_readout")
class Net(nn.Module):
    """
    Graph Transformer（消融版：CLS Token Readout）

    与 model56_full 的区别：
      - graph_readout 强制为 "cls"
      - 使用 CLS Token 进行图级表示汇聚
      - 需要扩展邻接矩阵以包含 CLS Token

    对比：
      - model56_full: Attention Pooling
      - model56_cls_readout: CLS Token
    """

    NUM_OPS = 32
    NUM_DEGREE = 32
    NUM_DEPTH_NASBENCH = 32

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config.dataset

        self.d_model = int(getattr(config, "d_model", 192))
        self.dropout = float(getattr(config, "dropout", 0.1))
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))
        # 【消融】强制使用 CLS readout
        self.graph_readout = "cls"
        self.d_ff_ratio = float(getattr(config, "d_ff_ratio", 4))
        self.graph_n_head = int(getattr(config, "graph_n_head", 6))
        self.act_layer = str(getattr(config, "act_layer", "relu"))
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))
        self.pe_k = int(getattr(config, "pe_k", 5))

        self._build_embedding_layers()

        # PE 编码器（保留）
        from models.ablations.model56_full import MagLapEncoder
        pe_hidden = min(self.d_model, 128)
        self.ml_encoder = MagLapEncoder(
            k=self.pe_k,
            hidden_dim=pe_hidden,
            output_dim=self.d_model,
        )

        self.encoder = TransformerEncoder(
            self.d_model,
            self.d_ff_ratio,
            self.gcn_layers,
            self.graph_n_head,
            self.act_layer,
        )

        self.post_norm = nn.LayerNorm(self.d_model)
        # 【消融】不使用 attention pooling
        self.predictor = nn.Linear(self.d_model, 1)

        self._init_weights()

    def _build_embedding_layers(self):
        # 【消融】必须使用 CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))

        depth_dim = self.NUM_DEPTH_NASBENCH

        self.op_embed = nn.Linear(self.NUM_OPS, self.d_model)
        self.in_degree_embed = nn.Linear(self.NUM_DEGREE, self.d_model)
        self.out_degree_embed = nn.Linear(self.NUM_DEGREE, self.d_model)
        self.depth_embed = nn.Linear(depth_dim, self.d_model)
        self.embed_proj = nn.Linear(4 * self.d_model, self.d_model)

    def _init_weights(self):
        self.apply(self._init_module_weights)
        if self.cls_token is not None:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    @staticmethod
    def _init_module_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.constant_(m.weight, 0.02)

    def get_data(self, sample, static_feature):
        return (
            sample["ops"],
            sample["code_adj"],
            sample["in_degree"],
            sample["out_degree"],
            sample["op_depth"],
            sample["reachability"],
        )

    def _embed_features(
        self, ops: Tensor, in_degree: Tensor, out_degree: Tensor, depth: Tensor
    ) -> Tensor:

        ops_onehot = F.one_hot(ops.long(), num_classes=self.NUM_OPS).float()
        in_deg_onehot = F.one_hot(in_degree.long(), num_classes=self.NUM_DEGREE).float()
        out_deg_onehot = F.one_hot(
            out_degree.long(), num_classes=self.NUM_DEGREE
        ).float()
        NUM_DEPTH = self.NUM_DEPTH_NASBENCH
        depth_onehot = F.one_hot(depth.long(), num_classes=NUM_DEPTH).float()

        ops_enc = self.op_embed(ops_onehot)
        in_enc = self.in_degree_embed(in_deg_onehot)
        out_enc = self.out_degree_embed(out_deg_onehot)
        dep_enc = self.depth_embed(depth_onehot)

        return self.embed_proj(torch.cat([ops_enc, in_enc, out_enc, dep_enc], dim=-1))

    def _add_pe(self, x: Tensor, sample) -> Tensor:
        pe = self.ml_encoder(sample["dir_pe_ml"])
        return x + pe

    def _add_cls_token(
        self, x: Tensor, adj: Tensor, reachability: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """添加 CLS Token"""
        B, L = x.size(0), adj.size(1)
        new_adj = torch.ones(B, L + 1, L + 1, device=adj.device)
        new_adj[:, 1:, 1:] = adj

        new_reach = torch.ones(B, L + 1, L + 1, device=reachability.device)
        new_reach[:, 1:, 1:] = reachability

        cls_token = self.cls_token.expand(B, 1, self.d_model)
        return torch.cat([cls_token, x], dim=1), new_adj, new_reach

    def _readout(self, x: Tensor) -> Tensor:
        """【消融】使用 CLS Token"""
        return x[:, 0, :]

    def forward(self, sample, static_feature) -> tuple[Tensor, Tensor]:
        ops, adj, in_degree, out_degree, depth, reachability = self.get_data(
            sample, static_feature
        )

        x = self._embed_features(ops, in_degree, out_degree, depth)
        x = self._add_pe(x, sample)
        x, adj, reachability = self._add_cls_token(x, adj, reachability)
        x, aux_loss = self.encoder(x, adj, reachability)
        x = self.post_norm(x)

        # 【消融】使用 CLS token 读取
        graph_feat = self._readout(x)
        predict = self.predictor(graph_feat)
        if getattr(self.config, "predict_target", None) != "latency":
            predict = predict + 0.5
        return predict, aux_loss if self.use_aux_loss else 0