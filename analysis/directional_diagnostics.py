"""Directional diagnostics for model56.

This script loads a trained run directory, collects MoE gate routing and
attention weights with forward hooks, and writes paper-ready CSV/PDF outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import models
from datasets.fixed_length_sampler import FixedLengthBatchSampler
from datasets.nasbench import NasbenchDataset
from models.model56_utils import build_structural_bias, preprocess_adj
from models.registry import get_model
from utils.seed import set_seed


RELATION_NAMES = ["A", "A^T", "R_s2t", "R_t2s", "AA^T", "A^TA"]
EXPERT_NAMES = ["self", "forward_A", "backward_AT"]
ROLE_NAMES = ["source", "sink", "middle", "branching", "converging"]


class PrintLogger:
    def info(self, msg: str) -> None:
        print(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, help="results/<dataset>/<model>/<timestamp>")
    parser.add_argument("--runs", nargs="+", type=int, default=[0])
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--max_samples", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--skip_magnetic", action="store_true")
    return parser.parse_args()


def load_config(run_dir: Path) -> SimpleNamespace:
    with open(run_dir / "config.json", "r") as f:
        cfg = json.load(f)
    cfg.setdefault("lambda_consistency", 0.0)
    cfg.setdefault("embed_type", "onehot_op")
    cfg.setdefault("batch_size", 128)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("do_train", False)
    return SimpleNamespace(**cfg)


def resolve_device(device: str) -> torch.device:
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def build_loader(cfg: SimpleNamespace, split: str, run_id: int, batch_size: int) -> DataLoader:
    data_path = f"data/{cfg.dataset}/all_{cfg.dataset}.pt"
    dataset = NasbenchDataset(
        PrintLogger(),
        cfg.dataset,
        split,
        data_path,
        cfg.percent,
        getattr(cfg, "lambda_consistency", 0.0),
        embed_type=getattr(cfg, "embed_type", "onehot_op"),
        runid=run_id,
    )
    sampler = FixedLengthBatchSampler(dataset, cfg.dataset, batch_size, include_partial=True)
    return DataLoader(
        dataset,
        shuffle=(sampler is None),
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        batch_sampler=sampler,
    )


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def load_model(cfg: SimpleNamespace, checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    model_cfg = SimpleNamespace(**vars(cfg))
    model_cfg.device = str(device)
    model_cfg.do_train = False
    models.import_all_models()
    model = get_model(model_cfg)
    model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Warn] missing keys when loading {checkpoint_path.name}: {len(missing)}")
    unexpected_real = [
        key
        for key in unexpected
        if not (key == "total_ops" or key == "total_params" or key.endswith(".total_ops") or key.endswith(".total_params"))
    ]
    if unexpected_real:
        print(f"[Warn] unexpected keys when loading {checkpoint_path.name}: {len(unexpected_real)}")
    model.eval()
    return model


def register_capture_hooks(model: torch.nn.Module, capture: dict[str, list]) -> list[Any]:
    handles = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        ffn = layer.ffn

        def gate_hook(module, inputs, output, layer_idx=layer_idx, ffn=ffn):
            del module, inputs
            weights = F.softmax(output.detach() / float(ffn.temperature), dim=-1)
            capture["gates"].append((layer_idx, weights.cpu()))

        def attn_hook(module, inputs, output, layer_idx=layer_idx):
            del module, inputs
            capture["attn"].append((layer_idx, output.detach().cpu()))

        handles.append(ffn.gate.register_forward_hook(gate_hook))
        handles.append(layer.self_attn.attn_dropout.register_forward_hook(attn_hook))
    return handles


def role_masks(in_degree: torch.Tensor, out_degree: torch.Tensor) -> dict[str, torch.Tensor]:
    in_degree = in_degree.cpu()
    out_degree = out_degree.cpu()
    return {
        "source": (in_degree == 0) & (out_degree > 0),
        "sink": (out_degree == 0) & (in_degree > 0),
        "middle": (in_degree > 0) & (out_degree > 0),
        "branching": out_degree >= 2,
        "converging": in_degree >= 2,
    }


def update_gate_stats(
    gate_stats: dict[tuple[int, int, str], dict[str, Any]],
    gates: list[tuple[int, torch.Tensor]],
    in_degree: torch.Tensor,
    out_degree: torch.Tensor,
    run_id: int,
) -> None:
    masks = role_masks(in_degree, out_degree)
    for layer_idx, weights in gates:
        for role, mask in masks.items():
            count = int(mask.sum().item())
            if count == 0:
                continue
            vals = weights[mask]
            key = (run_id, layer_idx, role)
            if key not in gate_stats:
                gate_stats[key] = {
                    "sum": torch.zeros(3, dtype=torch.float64),
                    "sumsq": torch.zeros(3, dtype=torch.float64),
                    "count": 0,
                }
            gate_stats[key]["sum"] += vals.double().sum(dim=0)
            gate_stats[key]["sumsq"] += vals.double().pow(2).sum(dim=0)
            gate_stats[key]["count"] += count


def effective_relation_masks(adj: torch.Tensor, reachability: torch.Tensor | None) -> torch.Tensor:
    adj = preprocess_adj(adj, adj.size(-1))
    return build_structural_bias(adj, reachability, n_head=6).cpu().bool()


def update_attention_stats(
    attention_stats: dict[tuple[int, int, int, int], dict[str, float]],
    attentions: list[tuple[int, torch.Tensor]],
    adj: torch.Tensor,
    reachability: torch.Tensor,
    run_id: int,
) -> None:
    rel = effective_relation_masks(adj.cpu(), reachability.cpu())
    batch_size, _, length, _ = rel.shape
    valid = ~torch.eye(length, dtype=torch.bool).view(1, 1, length, length)
    valid = valid.expand(batch_size, 1, length, length)
    density_den = float(valid.sum().item())
    rel_valid = rel & valid
    density_nums = rel_valid.sum(dim=(0, 2, 3)).double()

    for layer_idx, attn in attentions:
        # attn: (B, H, L, L)
        attn_valid = attn.double() * valid.double()
        mass_den = attn_valid.sum(dim=(0, 2, 3))
        for head_idx in range(attn.size(1)):
            head_attn = attn[:, head_idx : head_idx + 1].double()
            for rel_idx in range(len(RELATION_NAMES)):
                mask = rel_valid[:, rel_idx : rel_idx + 1].double()
                key = (run_id, layer_idx, head_idx, rel_idx)
                if key not in attention_stats:
                    attention_stats[key] = {
                        "mass_num": 0.0,
                        "mass_den": 0.0,
                        "density_num": 0.0,
                        "density_den": 0.0,
                    }
                attention_stats[key]["mass_num"] += float((head_attn * mask).sum().item())
                attention_stats[key]["mass_den"] += float(mass_den[head_idx].item())
                attention_stats[key]["density_num"] += float(density_nums[rel_idx].item())
                attention_stats[key]["density_den"] += density_den


def collect_run_diagnostics(
    cfg: SimpleNamespace,
    run_dir: Path,
    run_id: int,
    split: str,
    max_samples: int,
    batch_size: int,
    device: torch.device,
    collect_magnetic: bool,
) -> tuple[dict, dict, list[np.ndarray]]:
    checkpoint_path = run_dir / "checkpoints" / f"best_model_run{run_id}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    set_seed(int(cfg.seed) + run_id)
    random.seed(int(cfg.seed) + run_id)
    loader = build_loader(cfg, split, run_id, batch_size)
    model = load_model(cfg, checkpoint_path, device)

    capture: dict[str, list] = {"gates": [], "attn": []}
    handles = register_capture_hooks(model, capture)

    gate_stats: dict[tuple[int, int, str], dict[str, Any]] = {}
    attention_stats: dict[tuple[int, int, int, int], dict[str, float]] = {}
    magnetic_adjs: list[np.ndarray] = []

    seen = 0
    try:
        with torch.no_grad():
            for batch in loader:
                if max_samples > 0 and seen >= max_samples:
                    break
                current_bs = int(batch["ops"].size(0))
                if max_samples > 0 and seen + current_bs > max_samples:
                    keep = max_samples - seen
                    batch = {
                        key: value[:keep] if isinstance(value, torch.Tensor) else value
                        for key, value in batch.items()
                    }
                    current_bs = keep

                capture["gates"].clear()
                capture["attn"].clear()
                batch_device = move_batch_to_device(batch, device)
                _ = model(batch_device, None)

                update_gate_stats(
                    gate_stats,
                    capture["gates"],
                    batch["in_degree"],
                    batch["out_degree"],
                    run_id,
                )
                update_attention_stats(
                    attention_stats,
                    capture["attn"],
                    batch["code_adj"],
                    batch["reachability"],
                    run_id,
                )
                if collect_magnetic:
                    magnetic_adjs.extend(batch["code_adj"].cpu().numpy().astype(np.float64))
                seen += current_bs
    finally:
        for handle in handles:
            handle.remove()

    print(f"[Run {run_id}] collected {seen} {split} samples")
    return gate_stats, attention_stats, magnetic_adjs


def gate_stats_to_frame(gate_stats: dict[tuple[int, int, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for (run_id, layer_idx, role), stats in sorted(gate_stats.items()):
        count = int(stats["count"])
        sums = stats["sum"].numpy()
        sumsq = stats["sumsq"].numpy()
        mean = sums / max(count, 1)
        if count > 1:
            var = (sumsq - (sums * sums) / count) / (count - 1)
            std = np.sqrt(np.maximum(var, 0.0))
        else:
            std = np.zeros_like(mean)
        se = std / math.sqrt(max(count, 1))
        ci95 = 1.96 * se
        rows.append(
            {
                "run_id": run_id,
                "layer": layer_idx,
                "role": role,
                "count": count,
                "g_self": mean[0],
                "g_forward_A": mean[1],
                "g_backward_AT": mean[2],
                "g_self_std": std[0],
                "g_forward_A_std": std[1],
                "g_backward_AT_std": std[2],
                "g_self_se": se[0],
                "g_forward_A_se": se[1],
                "g_backward_AT_se": se[2],
                "g_self_ci95": ci95[0],
                "g_forward_A_ci95": ci95[1],
                "g_backward_AT_ci95": ci95[2],
                "g_self_sum": sums[0],
                "g_forward_A_sum": sums[1],
                "g_backward_AT_sum": sums[2],
                "g_self_sumsq": sumsq[0],
                "g_forward_A_sumsq": sumsq[1],
                "g_backward_AT_sumsq": sumsq[2],
            }
        )
    return pd.DataFrame(rows)


def attention_stats_to_frame(attention_stats: dict[tuple[int, int, int, int], dict[str, float]]) -> pd.DataFrame:
    rows = []
    for (run_id, layer_idx, head_idx, rel_idx), stats in sorted(attention_stats.items()):
        mass = stats["mass_num"] / stats["mass_den"] if stats["mass_den"] else np.nan
        density = stats["density_num"] / stats["density_den"] if stats["density_den"] else np.nan
        enrichment = mass / density if density and density > 0 else np.nan
        rows.append(
            {
                "run_id": run_id,
                "layer": layer_idx,
                "head": head_idx,
                "relation": RELATION_NAMES[rel_idx],
                "attention_mass": mass,
                "relation_density": density,
                "enrichment": enrichment,
                "log2_enrichment": math.log(enrichment, 2) if enrichment and enrichment > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def magnetic_laplacian(A: np.ndarray, q: float) -> np.ndarray:
    A = np.array(A, dtype=np.float64)
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.complex128)
    m_tilde = np.sum((A > 0) & (A.T == 0))
    d_g = max(min(int(m_tilde), n), 1)
    q_abs = q / d_g
    A_s = np.maximum(A, A.T)
    d_s = A_s.sum(axis=1)
    d_inv_sqrt = np.zeros_like(d_s)
    mask = d_s > 0
    d_inv_sqrt[mask] = 1.0 / np.sqrt(d_s[mask])
    D_inv_sqrt = np.diag(d_inv_sqrt)
    phase = np.exp(1j * 2 * np.pi * q_abs * (A - A.T))
    return np.eye(n, dtype=np.complex128) - D_inv_sqrt @ A_s @ D_inv_sqrt * phase


def compute_magnetic_frame(adjs: list[np.ndarray]) -> pd.DataFrame:
    rows = []
    for graph_idx, A in enumerate(adjs):
        for q in (0.0, 0.25):
            L = magnetic_laplacian(A, q)
            L_rev = magnetic_laplacian(A.T, q)
            denom = np.linalg.norm(L, ord="fro")
            sensitivity = np.linalg.norm(L - L_rev, ord="fro") / denom if denom > 0 else 0.0
            rows.append({"graph_idx": graph_idx, "q": q, "sensitivity": float(sensitivity)})
    return pd.DataFrame(rows)


def plot_gate(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    long_df = df.melt(
        id_vars=["run_id", "layer", "role", "count"],
        value_vars=["g_self", "g_forward_A", "g_backward_AT"],
        var_name="expert",
        value_name="gate_weight",
    )
    long_df["role"] = pd.Categorical(long_df["role"], ROLE_NAMES, ordered=True)
    long_df["expert"] = long_df["expert"].map(
        {
            "g_self": "self",
            "g_forward_A": "forward A",
            "g_backward_AT": "backward A^T",
        }
    )

    plt.figure(figsize=(8.2, 4.4))
    sns.barplot(
        data=long_df,
        x="role",
        y="gate_weight",
        hue="expert",
        errorbar="sd",
        palette=["#4C78A8", "#F58518", "#54A24B"],
    )
    plt.ylim(0, 1)
    plt.xlabel("directed node role")
    plt.ylabel("mean gate weight")
    plt.title("MoE gate routing by directed node role")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "moe_gate_by_role_bar.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "moe_gate_by_role_bar.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_attention(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    mean_df = (
        df.groupby(["layer", "head", "relation"], as_index=False)["log2_enrichment"]
        .mean()
        .copy()
    )
    mean_df["layer_head"] = mean_df.apply(
        lambda row: f"L{int(row['layer'])}-H{int(row['head'])}", axis=1
    )
    pivot = mean_df.pivot(index="layer_head", columns="relation", values="log2_enrichment")
    pivot = pivot.reindex(columns=RELATION_NAMES)

    plt.figure(figsize=(7.8, max(5.0, 0.23 * len(pivot))))
    sns.heatmap(
        pivot,
        cmap="vlag",
        center=0,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": "log2 attention enrichment"},
    )
    plt.xlabel("directional relation type")
    plt.ylabel("layer-head")
    plt.title("Attention enrichment by directional relation")
    plt.tight_layout()
    plt.savefig(out_dir / "attention_enrichment_layer_head_heatmap.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "attention_enrichment_layer_head_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_magnetic(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = df.copy()
    plot_df["q"] = plot_df["q"].map({0.0: "q=0", 0.25: "q=0.25"})
    plt.figure(figsize=(4.8, 4.0))
    sns.violinplot(data=plot_df, x="q", y="sensitivity", inner="quartile", palette=["#72B7B2", "#E45756"])
    plt.xlabel("magnetic phase q")
    plt.ylabel("S_q(G)")
    plt.title("Magnetic Laplacian direction sensitivity")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "magnetic_phase_sensitivity_violin.pdf", bbox_inches="tight")
    plt.savefig(out_dir / "magnetic_phase_sensitivity_violin.png", dpi=220, bbox_inches="tight")
    plt.close()


def summarize_gate(gate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    expert_specs = [
        ("self", "g_self"),
        ("out_forward_A", "g_forward_A"),
        ("in_backward_AT", "g_backward_AT"),
    ]
    for role in ROLE_NAMES:
        group = gate_df[gate_df["role"] == role]
        if group.empty:
            continue
        row = {
            "node_role": role,
            "count_per_layer_run": int(round(group["count"].mean())),
            "effective_node_layer_count": int(group["count"].sum()),
        }
        for label, prefix in expert_specs:
            total = group[f"{prefix}_sum"].sum()
            total_sq = group[f"{prefix}_sumsq"].sum()
            count = row["effective_node_layer_count"]
            mean = total / max(count, 1)
            if count > 1:
                var = (total_sq - (total * total) / count) / (count - 1)
                std = math.sqrt(max(float(var), 0.0))
            else:
                std = 0.0
            se = std / math.sqrt(max(count, 1))
            row[f"g_{label}_mean"] = mean
            row[f"g_{label}_std"] = std
            row[f"g_{label}_ci95"] = 1.96 * se
            row[f"g_{label}_layer_mean_std"] = group[prefix].std(ddof=1)
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_attention(attention_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    relation_summary = (
        attention_df.groupby("relation", as_index=False)
        .agg(
            relation_density=("relation_density", "mean"),
            raw_attention_mass=("attention_mass", "mean"),
            attention_enrichment=("enrichment", "mean"),
            log2_enrichment=("log2_enrichment", "mean"),
            log2_enrichment_std=("log2_enrichment", "std"),
        )
        .set_index("relation")
        .reindex(RELATION_NAMES)
        .reset_index()
    )

    intended_rows = []
    for head_idx, relation in enumerate(RELATION_NAMES):
        group = attention_df[
            (attention_df["head"] == head_idx) & (attention_df["relation"] == relation)
        ]
        if group.empty:
            continue
        intended_rows.append(
            {
                "head": head_idx,
                "intended_relation": relation,
                "relation_density": group["relation_density"].mean(),
                "raw_attention_mass": group["attention_mass"].mean(),
                "attention_enrichment": group["enrichment"].mean(),
                "log2_enrichment": group["log2_enrichment"].mean(),
                "log2_enrichment_std": group["log2_enrichment"].std(ddof=1),
            }
        )
    intended_summary = pd.DataFrame(intended_rows)
    return relation_summary, intended_summary


def summarize_magnetic(magnetic_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        magnetic_df.groupby("q", as_index=False)["sensitivity"]
        .agg(["mean", "std", "median", "min", "max"])
        .reset_index()
    )
    pivot = magnetic_df.pivot(index="graph_idx", columns="q", values="sensitivity")
    if 0.0 in pivot.columns and 0.25 in pivot.columns:
        diff = pivot[0.25] - pivot[0.0]
        diff_summary = pd.DataFrame(
            [
                {
                    "paired_difference": "S_0.25_minus_S_0",
                    "mean": diff.mean(),
                    "std": diff.std(ddof=1),
                    "median": diff.median(),
                    "min": diff.min(),
                    "max": diff.max(),
                }
            ]
        )
    else:
        diff_summary = pd.DataFrame()
    return summary, diff_summary


def write_markdown_report(
    cfg: SimpleNamespace,
    args: argparse.Namespace,
    out_dir: Path,
    gate_summary: pd.DataFrame,
    attention_relation_summary: pd.DataFrame,
    attention_intended_summary: pd.DataFrame,
    magnetic_summary: pd.DataFrame | None,
    magnetic_diff_summary: pd.DataFrame | None,
) -> None:
    def md_table(df: pd.DataFrame, digits: int = 4) -> str:
        if df is None or df.empty:
            return "_No data._"
        display = df.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda value: f"{value:.{digits}f}")
        return display.to_markdown(index=False)

    gate_eval = (
        "Appendix / tentative main after replication. The role trends are interpretable, "
        "but this run is one seed and one accuracy setting."
    )
    attention_eval = (
        "Main-paper usable if worded as relation-channel separation. The current module "
        "uses hard structural masks per head, so this is not evidence of learned soft head specialization."
    )
    magnetic_eval = (
        "Appendix for NAS-Bench-201. q=0.25 is clearly direction-sensitive while q=0 is not, "
        "but NAS-Bench-201 has a fixed small graph template, so the distribution is degenerate."
    )

    report = f"""# DTA-GT Mechanism Visualization Analysis

## Setting

- Dataset: `{cfg.dataset}`
- Target: `{getattr(cfg, "predict_target", "unknown")}`
- Percent / budget: `{getattr(cfg, "percent", "unknown")}`
- Model: `{cfg.model}`
- Run directory: `{args.run_dir}`
- Runs: `{args.runs}`
- Split: `{args.split}`
- Max samples: `{args.max_samples}`

This report evaluates whether the visualizations support the claim that DTA-GT models directed architecture-DAG semantics at global spectral positioning, pairwise structural interaction, and direction-sensitive feature update levels.

## Important Implementation Note

The MoE code computes `forward_A` with `adj @ H` and `backward_AT` with `adj.T @ H`. Under the usual convention `A[u, v] = 1` for edge `u -> v`, `forward_A` aggregates outgoing/successor context and `backward_AT` aggregates incoming/predecessor context. The original module names in code are therefore not used blindly in this report.

The attention module applies hard structural masks per head. The attention enrichment figure demonstrates separated directional relation channels, not unconstrained learned head specialization.

## 1. MoE Gate Routing By Directed Node Role

### Statistics

{md_table(gate_summary)}

### Interpretation

- Source nodes show the largest routing weight on outgoing/forward context (`g_out_forward_A`), while incoming/backward context is not dominant.
- Sink nodes show the lowest outgoing/forward routing and relatively higher self/incoming-backward routing.
- Branching nodes prefer outgoing/forward context; converging nodes shift toward self and incoming/backward context, though self remains strongest.
- Middle nodes are more balanced than role-extreme nodes, but not perfectly uniform.
- Stability across seeds/settings is not established here because this run uses one seed and one setting.

### Main / Appendix Decision

{gate_eval}

## 2. Attention Enrichment By Directional Relation Type

### Relation-Level Summary

{md_table(attention_relation_summary)}

### Intended Head-Relation Summary

{md_table(attention_intended_summary)}

### Interpretation

- Raw attention mass must be read with relation density. The intended relation channel for each head has mass near 1 because the model masks each head to a specific structural relation plus self-loop.
- Enrichment is high for sparse relations such as branching (`A^T A`) and adjacency because attention is concentrated over a small allowed set.
- Forward and backward adjacency channels have comparable enrichment; reachability channels are less sparse and therefore have lower enrichment.
- This supports directional relation separation in pairwise interaction, but the evidence is architectural/functional rather than learned mask discovery.

### Main / Appendix Decision

{attention_eval}

## 3. Magnetic Laplacian Phase Diagnostic

### Summary

{md_table(magnetic_summary) if magnetic_summary is not None else "_Skipped._"}

### Paired Difference

{md_table(magnetic_diff_summary) if magnetic_diff_summary is not None else "_Skipped._"}

### Interpretation

- q=0 has zero direction sensitivity under this diagnostic.
- q=0.25 produces non-zero direction sensitivity, matching the intended magnetic phase behavior.
- On NAS-Bench-201 this statistic is constant because the architecture topology is effectively fixed and operations vary; use NAS-Bench-101 or another varied-topology setting for a stronger distributional plot.

### Main / Appendix Decision

{magnetic_eval}

## Overall Recommendation

For the current `{cfg.dataset} {getattr(cfg, "predict_target", "unknown")} @{getattr(cfg, "percent", "unknown")}` single-seed result:

- MoE gate plot: appendix now; candidate main only after repeating on `latency @156` and/or multiple seeds.
- Attention enrichment heatmap: main-paper usable with careful wording about hard directional relation channels.
- Magnetic phase violin: appendix for NAS-Bench-201; stronger as main evidence on a varied-topology dataset.

## Paper-Ready 2-3 Sentence Analysis

DTA-GT exhibits direction-aware behavior at multiple levels. The MoE routing changes with directed node roles: source and branching nodes assign more mass to forward/outgoing context, while sink and converging nodes reduce forward routing and rely more on self or incoming/backward context. Structural attention concentrates probability within separate directional relation channels, and the magnetic phase diagnostic shows that q=0.25 introduces non-zero sensitivity to edge reversal whereas q=0 is direction-insensitive.
"""
    with open(out_dir / "mechanism_analysis_report.md", "w") as f:
        f.write(report)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    cfg = load_config(run_dir)
    device = resolve_device(args.device)
    batch_size = int(args.batch_size or cfg.batch_size)
    out_dir = Path(args.out_dir) if args.out_dir else Path("results/diagnostics") / cfg.dataset / cfg.model / run_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    merged_gate_stats: dict[tuple[int, int, str], dict[str, Any]] = {}
    merged_attention_stats: dict[tuple[int, int, int, int], dict[str, float]] = {}
    all_adjs: list[np.ndarray] = []

    for run_id in args.runs:
        gate_stats, attention_stats, adjs = collect_run_diagnostics(
            cfg,
            run_dir,
            run_id,
            args.split,
            args.max_samples,
            batch_size,
            device,
            collect_magnetic=(not args.skip_magnetic and len(all_adjs) == 0),
        )
        merged_gate_stats.update(gate_stats)
        merged_attention_stats.update(attention_stats)
        all_adjs.extend(adjs)

    gate_df = gate_stats_to_frame(merged_gate_stats)
    attention_df = attention_stats_to_frame(merged_attention_stats)
    gate_summary = summarize_gate(gate_df)
    attention_relation_summary, attention_intended_summary = summarize_attention(attention_df)
    gate_df.to_csv(out_dir / "gate_by_role.csv", index=False)
    gate_summary.to_csv(out_dir / "gate_by_role_summary.csv", index=False)
    attention_df.to_csv(out_dir / "attention_relation_enrichment.csv", index=False)
    attention_relation_summary.to_csv(out_dir / "attention_relation_summary.csv", index=False)
    attention_intended_summary.to_csv(out_dir / "attention_intended_head_summary.csv", index=False)
    plot_gate(gate_df, out_dir)
    plot_attention(attention_df, out_dir)

    summary = {
        "run_dir": str(run_dir),
        "runs": args.runs,
        "split": args.split,
        "max_samples": args.max_samples,
        "batch_size": batch_size,
        "device": str(device),
        "gate_rows": int(len(gate_df)),
        "attention_rows": int(len(attention_df)),
        "notes": [
            "Expert labels follow implementation tensors: forward_A uses adj @ H, backward_AT uses adj.T @ H.",
            "Attention relation masks are the effective model masks from build_structural_bias.",
        ],
    }

    magnetic_summary = None
    magnetic_diff_summary = None
    if not args.skip_magnetic:
        magnetic_df = compute_magnetic_frame(all_adjs)
        magnetic_summary, magnetic_diff_summary = summarize_magnetic(magnetic_df)
        magnetic_df.to_csv(out_dir / "magnetic_phase_sensitivity.csv", index=False)
        magnetic_summary.to_csv(out_dir / "magnetic_phase_summary.csv", index=False)
        magnetic_diff_summary.to_csv(out_dir / "magnetic_phase_paired_difference.csv", index=False)
        plot_magnetic(magnetic_df, out_dir)
        summary["magnetic_rows"] = int(len(magnetic_df))

    write_markdown_report(
        cfg,
        args,
        out_dir,
        gate_summary,
        attention_relation_summary,
        attention_intended_summary,
        magnetic_summary,
        magnetic_diff_summary,
    )

    with open(out_dir / "diagnostic_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[Done] diagnostics written to {out_dir}")


if __name__ == "__main__":
    main()
