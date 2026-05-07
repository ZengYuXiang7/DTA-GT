"""
model56_wo_topo: 消融 - 去掉拓扑特征（入度、出度、深度）

与 model56_full 的区别：
  - 去掉 in_degree_embed、out_degree_embed、depth_embed
  - 节点嵌入仅使用 op_embed（操作类型），不拼接拓扑信息
  - embed_proj 随之去掉（单路径直接输出 d_model）

消融目的：验证入度/出度/深度等拓扑信息对预测的贡献
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
# 公共工具（与 model56_full 相同）
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
# 位置编码（与 model56_full 相同）
# ============================================================


class MagLapEncoder(nn.Module):
    def __init__(
        self,
        k: int = 25,
        hidden_dim: int = 64,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim

        self.f_elem = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=2,
            dropout=dropout,
            batch_first=True,
            bias=False,
        )
        self.drop = nn.Dropout(dropout)
        self.f_mix = nn.Sequential(
            nn.Linear(k * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, N, K, _ = x.shape
        h = self.norm(self.f_elem(x))
        h = h.reshape(B * N, K, self.hidden_dim)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.drop(h)
        h = h.reshape(B, N, K * self.hidden_dim)
        return self.f_mix(h)


# ============================================================
# 注意力层（与 model56_full 相同）
# ============================================================


class MultiHeadAttention(nn.Module):
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


# ============================================================
# 编码器层（与 model56_full 相同）
# ============================================================


class GraphTransformerLayer(nn.Module):
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
# 主模型（去掉拓扑特征）
# ============================================================


@register_model("model56_wo_topo")
class Net(nn.Module):
    """
    Graph Transformer（消融版：去掉拓扑特征）

    与 model56_full 的唯一区别：
      - 节点嵌入只用 op_embed，不使用 in_degree_embed / out_degree_embed / depth_embed
      - embed_proj 去掉，op_embed 直接输出 d_model 维表示
      - 其余（PE、结构偏置、MoE-FFN 等）完全保留
    """

    NUM_OPS = 32

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config.dataset

        self.d_model = int(getattr(config, "d_model", 192))
        self.dropout = float(getattr(config, "dropout", 0.1))
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))
        self.graph_readout = str(getattr(config, "graph_readout", "cls"))
        self.d_ff_ratio = float(getattr(config, "d_ff_ratio", 4))
        self.graph_n_head = int(getattr(config, "graph_n_head", 6))
        self.act_layer = str(getattr(config, "act_layer", "relu"))
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))
        self.pe_k = int(getattr(config, "pe_k", 5))

        self._build_embedding_layers()

        # PE 编码器保留
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
        if self.graph_readout == "att":
            self.att_pool = nn.Linear(self.d_model, 1)
        self.predictor = nn.Linear(self.d_model, 1)

        self._init_weights()

    def _build_embedding_layers(self):
        """【消融】只保留 op_embed，去掉入度/出度/深度嵌入及 embed_proj。"""
        if self.graph_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        self.op_embed = nn.Linear(self.NUM_OPS, self.d_model)

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
            sample["reachability"],
        )

    def _embed_features(self, ops: Tensor) -> Tensor:
        """【消融】仅使用操作类型嵌入，不包含任何拓扑信息。"""

        ops_onehot = F.one_hot(ops.long(), num_classes=self.NUM_OPS).float()
        return self.op_embed(ops_onehot)

    def _add_pe(self, x: Tensor, sample) -> Tensor:
        pe = self.ml_encoder(sample["dir_pe_ml"])
        return x + pe

    def _add_cls_token(
        self, x: Tensor, adj: Tensor, reachability: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        if self.graph_readout != "cls":
            return x, adj, reachability

        B, L = x.size(0), adj.size(1)
        new_adj = torch.ones(B, L + 1, L + 1, device=adj.device)
        new_adj[:, 1:, 1:] = adj

        new_reach = torch.ones(B, L + 1, L + 1, device=reachability.device)
        new_reach[:, 1:, 1:] = reachability

        cls_token = self.cls_token.expand(B, 1, self.d_model)
        return torch.cat([cls_token, x], dim=1), new_adj, new_reach

    def _readout(self, x: Tensor) -> Tensor:
        if self.graph_readout == "cls":
            return x[:, 0, :]
        elif self.graph_readout == "att":
            w = torch.softmax(self.att_pool(x).squeeze(-1), dim=1).unsqueeze(-1)
            return (x * w).sum(dim=1)
        else:
            raise ValueError(f"Unsupported graph_readout: {self.graph_readout}")

    def forward(self, sample, static_feature) -> tuple[Tensor, Tensor]:
        # 1. 提取原始数据（不需要 in_degree / out_degree / op_depth）
        ops, adj, reachability = self.get_data(sample, static_feature)

        # 2. 仅用 op 嵌入（【消融】去掉拓扑特征）
        x = self._embed_features(ops)

        # 3. 叠加位置编码
        x = self._add_pe(x, sample)

        # 4. 添加 [CLS] token
        x, adj, reachability = self._add_cls_token(x, adj, reachability)

        # 5. 主编码器
        x, aux_loss = self.encoder(x, adj, reachability)
        x = self.post_norm(x)

        # 6. 图汇聚 + 预测
        graph_feat = self._readout(x)
        predict = self.predictor(graph_feat)
        if getattr(self.config, "predict_target", None) != "latency":
            predict = predict + 0.5
        return predict, aux_loss if self.use_aux_loss else 0
