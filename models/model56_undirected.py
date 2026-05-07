"""
model56_undirected: undirected graph ablation for model56.

Main changes from model56_full:
  1. Node input is only op + undirected degree.
  2. Attention is fixed to two heads:
     - head 0 uses undirected adjacency A.
     - head 1 uses symmetrized reachability.
  3. FFN is changed to self + A@H with the undirected A.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.registry import register_model
from timm.models.layers import to_2tuple
from torch import Tensor

from models.model56_utils import preprocess_adj


def make_activation(act_layer: str) -> nn.Module:
    act = act_layer.lower()
    if act == "relu":
        return nn.ReLU()
    if act == "leaky_relu":
        return nn.LeakyReLU()
    raise ValueError(f"Unsupported activation: {act_layer}")


def _zero_diagonal(adj: Tensor) -> Tensor:
    L = adj.size(-1)
    eye = torch.eye(L, dtype=torch.bool, device=adj.device).unsqueeze(0)
    return adj.masked_fill(eye, 0)


def build_undirected_adj(adj: Tensor) -> Tensor:
    adj = preprocess_adj(adj, adj.size(-1))
    adj = ((adj + adj.transpose(1, 2)) > 0).to(adj.dtype)
    return _zero_diagonal(adj)


def build_undirected_reachability(
    adj: Tensor,
    reachability: Optional[Tensor] = None,
) -> Tensor:
    if reachability is not None:
        reach = (reachability > 0) | (reachability.transpose(1, 2) > 0)
        return _zero_diagonal(reach.to(adj.dtype))

    reach = build_undirected_adj(adj).bool()
    L = reach.size(-1)
    for k in range(L):
        reach = reach | (reach[:, :, k : k + 1] & reach[:, k : k + 1, :])
    return _zero_diagonal(reach.to(adj.dtype))


def build_two_head_masks(adj: Tensor, reachability: Optional[Tensor]) -> Tensor:
    adj_u = build_undirected_adj(adj)
    reach = build_undirected_reachability(adj_u, reachability)
    masks = torch.stack([adj_u, reach], dim=1).bool()

    L = adj.size(-1)
    eye = torch.eye(L, dtype=torch.bool, device=adj.device).view(1, 1, L, L)
    return masks | eye


class TwoHeadUndirectedAttention(nn.Module):
    """Two-head attention: one undirected-adj head and one reachability head."""

    def __init__(self, dim: int, n_head: int = 2, dropout: float = 0.0):
        super().__init__()
        if n_head != 2:
            raise ValueError("model56_undirected uses exactly 2 heads")
        if dim % n_head != 0:
            raise ValueError(f"dim={dim} must be divisible by n_head={n_head}")

        self.n_head = n_head
        self.head_size = dim // n_head
        self.scale = math.sqrt(self.head_size)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _compute_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        B, L, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        return q, k, v

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        reachability: Optional[Tensor] = None,
    ) -> Tensor:
        B, L, C = x.shape
        q, k, v = self._compute_qkv(x)

        score = torch.matmul(q, k.mT) / self.scale
        masks = build_two_head_masks(adj, reachability)
        score = score.masked_fill(~masks, -torch.inf)

        attn = F.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.resid_dropout(self.proj(out))


class SelfPlusHAFFN(nn.Module):
    """FFN block using self features plus A@H on the undirected graph."""

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
        has_cls: bool = True,
    ):
        super().__init__()
        self.has_cls = has_cls
        out_features = out_features or dim
        hidden_features = int(mlp_ratio * dim)
        p1, p2 = to_2tuple(drop)

        self.self_expert = nn.Linear(dim, hidden_features, bias=False)
        self.ha_expert = nn.Linear(dim, hidden_features, bias=False)
        self.act = make_activation(act_layer)
        self.drop1 = nn.Dropout(p1)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop2 = nn.Dropout(p2)

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        adj_u = build_undirected_adj(adj)
        if self.has_cls:
            adj_u = adj_u.clone()
            adj_u[:, 0, :] = 0
            adj_u[:, :, 0] = 0

        ha = torch.bmm(adj_u, x)
        out = self.self_expert(x) + self.ha_expert(ha)
        out = self.drop1(self.act(self.fc2(out)))
        out = self.drop2(out)
        return out, torch.zeros((), dtype=x.dtype, device=x.device)


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
        self.self_attn = TwoHeadUndirectedAttention(d_model, n_head, dropout)
        self.ffn = SelfPlusHAFFN(
            d_model,
            mlp_ratio=mlp_ratio,
            act_layer=activation,
            drop=dropout,
        )

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        reachability: Optional[Tensor] = None,
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
        n_head: int = 2,
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
        self,
        x: Tensor,
        adj: Tensor,
        reachability: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        adj = preprocess_adj(adj, x.size(1))
        penalties = []
        for layer in self.layers:
            x, penalty = layer(x, adj, reachability)
            penalties.append(penalty)

        aux_loss = torch.stack(penalties).mean()
        return x, aux_loss


@register_model("model56_undirected")
class Net(nn.Module):
    NUM_OPS = 32
    NUM_DEGREE = 32

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.predict_target = getattr(config, "predict_target", None)
        self.dataset = config.dataset

        self.d_model = int(getattr(config, "d_model", 192))
        self.dropout = float(getattr(config, "dropout", 0.1))
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))
        self.graph_readout = str(getattr(config, "graph_readout", "cls"))
        self.d_ff_ratio = float(getattr(config, "d_ff_ratio", 4))
        self.graph_n_head = 2
        self.act_layer = str(getattr(config, "act_layer", "relu"))
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))

        self._build_embedding_layers()
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
        if self.graph_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

        self.op_embed = nn.Linear(self.NUM_OPS, self.d_model)
        self.degree_embed = nn.Linear(self.NUM_DEGREE, self.d_model)
        self.embed_proj = nn.Linear(2 * self.d_model, self.d_model)

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
        del static_feature
        return sample["ops"], sample["code_adj"], sample.get("reachability")

    def _degree_from_adj(self, adj: Tensor) -> Tensor:
        adj_u = build_undirected_adj(adj)
        degree = adj_u.sum(dim=-1).long()
        return degree.clamp(min=0, max=self.NUM_DEGREE - 1)

    def _embed_features(self, ops: Tensor, degree: Tensor) -> Tensor:

        degree = degree.to(device=ops.device)
        ops_onehot = F.one_hot(ops.long(), num_classes=self.NUM_OPS).float()
        degree_onehot = F.one_hot(
            degree.long(),
            num_classes=self.NUM_DEGREE,
        ).float()

        ops_enc = self.op_embed(ops_onehot)
        degree_enc = self.degree_embed(degree_onehot)
        return self.embed_proj(torch.cat([ops_enc, degree_enc], dim=-1))

    def _add_cls_token(
        self,
        x: Tensor,
        adj: Tensor,
        reachability: Optional[Tensor],
    ) -> tuple[Tensor, Tensor, Optional[Tensor]]:
        if self.graph_readout != "cls":
            return x, adj, reachability

        B, L = x.size(0), adj.size(1)
        new_adj = torch.zeros(B, L + 1, L + 1, dtype=adj.dtype, device=adj.device)
        new_adj[:, 1:, 1:] = adj
        new_adj[:, 0, 1:] = 1
        new_adj[:, 1:, 0] = 1

        new_reach = None
        if reachability is not None:
            new_reach = torch.zeros(
                B,
                L + 1,
                L + 1,
                dtype=reachability.dtype,
                device=reachability.device,
            )
            new_reach[:, 1:, 1:] = reachability
            new_reach[:, 0, 1:] = 1
            new_reach[:, 1:, 0] = 1

        cls_token = self.cls_token.expand(B, 1, self.d_model)
        return torch.cat([cls_token, x], dim=1), new_adj, new_reach

    def _readout(self, x: Tensor) -> Tensor:
        if self.graph_readout == "cls":
            return x[:, 0, :]
        if self.graph_readout == "att":
            w = torch.softmax(self.att_pool(x).squeeze(-1), dim=1).unsqueeze(-1)
            return (x * w).sum(dim=1)
        raise ValueError(f"Unsupported graph_readout: {self.graph_readout}")

    def forward(self, sample, static_feature) -> tuple[Tensor, Tensor]:
        ops, adj, reachability = self.get_data(sample, static_feature)

        degree = self._degree_from_adj(adj)
        x = self._embed_features(ops, degree)
        x, adj, reachability = self._add_cls_token(x, adj, reachability)

        x, aux_loss = self.encoder(x, adj, reachability)
        x = self.post_norm(x)

        graph_feat = self._readout(x)
        predict = self.predictor(graph_feat)
        if self.predict_target != "latency":
            predict = predict + 0.5
        return predict, aux_loss if self.use_aux_loss else 0


__all__ = ["Net"]
