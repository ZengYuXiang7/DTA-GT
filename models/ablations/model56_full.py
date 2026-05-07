"""
model56_full: 完整版Graph Transformer（消融实验基准模型）

架构组成：
  1. MagLapEncoder        - Magnetic Laplacian 位置编码
  2. MultiHeadAttention   - 结构感知多头注意力（6通道偏置）
  3. BatchedMoEGraphFFN   - MoE图前馈网络（3专家门控）
  4. CLS Token            - 用于图级表示汇聚

这是消融实验的基准（full）模型，所有消融变体都基于此减去某个组件。
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
    """根据字符串名称创建激活函数模块。"""
    act = act_layer.lower()
    if act == "relu":
        return nn.ReLU()
    elif act == "leaky_relu":
        return nn.LeakyReLU()
    else:
        raise ValueError(f"Unsupported activation: {act_layer}")


# ============================================================
# 位置编码
# ============================================================


class MagLapEncoder(nn.Module):
    """
    Magnetic Laplacian Position Encoder.

    对每个节点的 k 条特征边 (B, N, K, 2) 进行：
      1. 逐边线性投影 + LayerNorm
      2. Self-Attention 聚合
      3. 拼接后投影到 d_model 维
    """

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
        """
        x: (B, N, K, 2)
        returns: (B, N, output_dim)
        """
        B, N, K, _ = x.shape
        h = self.norm(self.f_elem(x))  # (B, N, K, hidden_dim)
        h = h.reshape(B * N, K, self.hidden_dim)  # (BN, K, hidden_dim)
        h, _ = self.attn(h, h, h, need_weights=False)
        h = self.drop(h)
        h = h.reshape(B, N, K * self.hidden_dim)  # (B, N, K*hidden_dim)
        return self.f_mix(h)


# ============================================================
# 注意力层
# ============================================================


class MultiHeadAttention(nn.Module):
    """
    手动实现的多头注意力，带结构感知偏置（6 个 head 对应 6 种图结构信号）。

    结构偏置通过 masked_fill 直接作用于 attention score，而非通过 attn_mask 参数。
    """

    def __init__(self, dim: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        assert dim % n_head == 0, f"dim={dim} must be divisible by n_head={n_head}"
        self.n_head = n_head
        self.head_size = dim // n_head
        self.scale = math.sqrt(self.head_size)

        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def _compute_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """QKV 投影并 reshape 为 (B, H, L, d)。"""
        B, L, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, L, self.n_head, self.head_size).transpose(1, 2)
        return q, k, v

    def _build_bias(self, adj: Tensor, reachability: Optional[Tensor]) -> Tensor:
        """构建 6 通道结构偏置矩阵。"""
        return build_structural_bias(adj, reachability, self.n_head)

    def forward(
        self, x: Tensor, adj: Tensor, reachability: Optional[Tensor] = None
    ) -> Tensor:
        """
        x:       (B, L, C)
        adj:     (B, L, L)
        reachability: (B, L, L) 或 None
        """
        B, L, C = x.shape

        q, k, v = self._compute_qkv(x)
        score = torch.matmul(q, k.mT) / self.scale  # (B, H, L, L)

        # --- 结构感知掩码 ---
        adj = preprocess_adj(adj, L)
        pe = self._build_bias(adj, reachability)
        score = apply_structural_bias(score, pe, L, adj.device)

        attn = F.softmax(score, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B, H, L, d)

        out = out.transpose(1, 2).reshape(B, L, C)
        return self.resid_dropout(self.proj(out))


# ============================================================
# FFN 层
# ============================================================


class GINMlp(nn.Module):
    """
    GIN-style MLP：结合节点自身特征与邻居聚合特征。
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        out_features: Optional[int] = None,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        p1, p2 = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.gcn = nn.Linear(in_features, hidden_features, bias=False)
        self.act = make_activation(act_layer)
        self.drop1 = nn.Dropout(p1)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop2 = nn.Dropout(p2)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        h1, h2 = self.gcn(x).chunk(2, dim=-1)
        out = self.fc1(x) + torch.cat([adj @ h1, adj.mT @ h2], dim=-1)
        out = self.drop1(self.act(self.fc2(out)))
        return self.drop2(out)


class BatchedMoEGraphFFN(nn.Module):
    """
    MoE-Graph FFN：3 个专家分别处理 self / in-neighbor / out-neighbor 信息。
    """

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

        # --- 门控网络 ---
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
        """
        返回 (输出, 路由熵惩罚)
        """
        L = x.size(1)
        adj = adj.float()
        if self.has_cls:
            adj[:, 0, :] = 0
            adj[:, :, 0] = 0
        adj = adj.masked_fill(torch.eye(L, device=adj.device, dtype=torch.bool), 0)

        # 三个专家的输出
        e0 = self.self_expert(x)
        e1 = torch.bmm(adj, self.in_expert(x))
        e2 = torch.bmm(adj.transpose(1, 2), self.out_expert(x))

        # 软门控加权
        w = F.softmax(self.gate(x) / self.temperature, dim=-1)
        out = w[..., 0:1] * e0 + w[..., 1:2] * e1 + w[..., 2:3] * e2

        out = self.drop1(self.act(self.fc2(out)))
        out = self.drop2(out)

        routing_penalty = (w * (w + 1e-8).log()).sum(dim=-1).mean()
        return out, routing_penalty


class StandardFFN(nn.Module):
    """
    标准FFN（消融用）：替代MoE-Graph FFN的简单版本。
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: str = "relu",
        drop: float = 0.0,
    ):
        super().__init__()
        hidden_features = int(mlp_ratio * dim)
        self.fc1 = nn.Linear(dim, hidden_features, bias=False)
        self.act = make_activation(act_layer)
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, dim, bias=False)

    def forward(self, x: Tensor, adj: Tensor) -> tuple[Tensor, Tensor]:
        """标准FFN不需要adj，返回(输出, 0)保持接口一致"""
        del adj  # 未使用
        out = self.drop(self.act(self.fc1(x)))
        out = self.drop(self.fc2(out))
        return out, torch.tensor(0.0, device=x.device)


# ============================================================
# 编码器层
# ============================================================


class GraphTransformerLayer(nn.Module):
    """
    Pre-norm Transformer 层：
      1. Multi-Head Self-Attention（带结构感知偏置）
      2. MoE-Graph FFN
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float,
        dropout: float,
        activation: str = "relu",
        use_struct_bias: bool = True,
        use_moe_ffn: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.use_moe_ffn = use_moe_ffn

        if use_moe_ffn:
            self.ffn = BatchedMoEGraphFFN(
                d_model, mlp_ratio=mlp_ratio, act_layer=activation, drop=dropout
            )
        else:
            self.ffn = StandardFFN(
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
    """
    多层 GraphTransformerEncoder。
    邻接矩阵预处理在入口处执行一次，各层复用。
    """

    def __init__(
        self,
        d_model: int,
        d_ff_ratio: float,
        gcn_layers: int,
        n_head: int = 4,
        activation: str = "relu",
        use_struct_bias: bool = True,
        use_moe_ffn: bool = True,
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
                    use_struct_bias=use_struct_bias,
                    use_moe_ffn=use_moe_ffn,
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
# 主模型
# ============================================================


@register_model("model56_full")
class Net(nn.Module):
    """
    Graph Transformer 主模型（完整版基准模型）。

    数据流程：
      1. get_data()        -> 从 sample 中提取原始特征
      2. _embed_features() -> one-hot 编码 + 线性投影
      3. _add_pe()         -> 叠加 Magnetic Laplacian PE
      4. _add_cls_token()  -> 添加 [CLS] token 并扩展 adj/reachability
      5. encoder()          -> 多层 GraphTransformer
      6. _readout()        -> 汇聚图级表示
      7. predictor()        -> 预测回归值
    """

    # 各特征 one-hot 编码维度
    NUM_OPS = 32
    NUM_DEGREE = 32
    NUM_DEPTH_NASBENCH = 32

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dataset = config.dataset

        # --- 超参数解析 ---
        self.d_model = int(getattr(config, "d_model", 192))
        self.dropout = float(getattr(config, "dropout", 0.1))
        self.gcn_layers = int(getattr(config, "gcn_layers", 2))
        self.graph_readout = str(getattr(config, "graph_readout", "cls"))
        self.d_ff_ratio = float(getattr(config, "d_ff_ratio", 4))
        self.graph_n_head = int(getattr(config, "graph_n_head", 6))
        self.act_layer = str(getattr(config, "act_layer", "relu"))
        self.use_aux_loss = bool(getattr(config, "use_aux_loss", False))
        self.pe_k = int(getattr(config, "pe_k", 5))

        # --- 特征嵌入层 ---
        self._build_embedding_layers()

        # --- PE 编码器 ---
        pe_hidden = min(self.d_model, 128)
        self.ml_encoder = MagLapEncoder(
            k=self.pe_k,
            hidden_dim=pe_hidden,
            output_dim=self.d_model,
        )

        # --- 主编码器 ---
        self.encoder = TransformerEncoder(
            self.d_model,
            self.d_ff_ratio,
            self.gcn_layers,
            self.graph_n_head,
            self.act_layer,
            use_struct_bias=True,
            use_moe_ffn=True,
        )

        # --- 预测头 ---
        self.post_norm = nn.LayerNorm(self.d_model)
        if self.graph_readout == "att":
            self.att_pool = nn.Linear(self.d_model, 1)
        self.predictor = nn.Linear(self.d_model, 1)

        self._init_weights()

    # --------------------------------------------------
    # 构建方法
    # --------------------------------------------------

    def _build_embedding_layers(self):
        """构建操作嵌入、入度/出度嵌入、深度嵌入。"""
        if self.graph_readout == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        else:
            self.cls_token = None

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

    # --------------------------------------------------
    # 数据提取与预处理
    # --------------------------------------------------

    def get_data(self, sample, static_feature):
        """从 sample 中提取模型所需的所有原始特征。"""
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
        """将原始特征 one-hot 编码后投影到 d_model 维。"""

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
        """叠加 Magnetic Laplacian Position Encoding。"""
        pe = self.ml_encoder(sample["dir_pe_ml"])
        return x + pe

    def _add_cls_token(
        self, x: Tensor, adj: Tensor, reachability: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """在图前添加 [CLS] token 并相应扩展邻接矩阵和可达性矩阵。"""
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
        """图级表示汇聚。"""
        if self.graph_readout == "cls":
            return x[:, 0, :]
        elif self.graph_readout == "att":
            w = torch.softmax(self.att_pool(x).squeeze(-1), dim=1).unsqueeze(-1)
            return (x * w).sum(dim=1)
        else:
            raise ValueError(f"Unsupported graph_readout: {self.graph_readout}")

    # --------------------------------------------------
    # 前向传播
    # --------------------------------------------------

    def forward(self, sample, static_feature) -> tuple[Tensor, Tensor]:
        # 1. 提取原始数据
        ops, adj, in_degree, out_degree, depth, reachability = self.get_data(
            sample, static_feature
        )

        # 2. 特征嵌入
        x = self._embed_features(ops, in_degree, out_degree, depth)

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
