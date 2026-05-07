#!/usr/bin/env python3
"""Prepare paper-ready mechanism figures from second-budget diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


ROLE_ORDER = ["source", "sink", "middle", "branching", "converging"]
ROLE_LABELS = {
    "source": "Source",
    "sink": "Sink",
    "middle": "Middle",
    "branching": "Branching",
    "converging": "Converging",
}
RELATION_ORDER = ["A", "A^T", "R_s2t", "R_t2s", "AA^T", "A^TA"]
RELATION_LABELS = {
    "A": "Fwd Adj\n$A$",
    "A^T": "Bwd Adj\n$A^\\top$",
    "R_s2t": "Fwd Reach\n$R_{s2t}$",
    "R_t2s": "Bwd Reach\n$R_{t2s}$",
    "AA^T": "Converge\n$AA^\\top$",
    "A^TA": "Branch\n$A^\\top A$",
}
SETTING_LABELS = {
    "NB101 Acc @172": "NB101 Acc",
    "NB101 Lat @172": "NB101 Lat",
    "NB201 Acc @469": "NB201 Acc",
    "NB201 Lat @469": "NB201 Lat",
}
EXPERTS = [
    ("g_self_mean", "Self", "#6E6E6E"),
    ("g_in_backward_AT_mean", "Incoming", "#2A9D8F"),
    ("g_out_forward_A_mean", "Outgoing", "#E76F51"),
]


def setup_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.6,
            "axes.titlesize": 8.4,
            "axes.labelsize": 7.8,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.8,
            "grid.linewidth": 0.4,
            "lines.linewidth": 1.2,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def read_gate(root: Path, key: str) -> pd.DataFrame:
    df = pd.read_csv(root / key / "gate_by_role_summary.csv")
    df["node_role"] = pd.Categorical(df["node_role"], categories=ROLE_ORDER, ordered=True)
    return df.sort_values("node_role")


def read_magnetic(root: Path, key: str, sample_per_q: int | None = 60000) -> pd.DataFrame:
    df = pd.read_csv(root / key / "magnetic_phase_sensitivity.csv")
    df["q_label"] = df["q"].map({0.0: "$q=0$", 0.25: "$q=0.25$"})
    if sample_per_q is None:
        return df
    sampled = []
    rng = np.random.default_rng(20260504)
    for _, group in df.groupby("q", sort=True):
        if len(group) > sample_per_q:
            idx = rng.choice(group.index.to_numpy(), size=sample_per_q, replace=False)
            sampled.append(group.loc[idx])
        else:
            sampled.append(group)
    return pd.concat(sampled, ignore_index=True)


def magnetic_stats(root: Path, key: str) -> pd.DataFrame:
    return pd.read_csv(root / key / "magnetic_phase_summary.csv")


def plot_magnetic_nb101(
    ax: plt.Axes,
    root: Path,
    *,
    full_range: bool = False,
    title: str = "Magnetic phase sensitivity",
) -> None:
    plot_df = read_magnetic(root, "NB101_Acc_172", sample_per_q=60000)
    sns.violinplot(
        data=plot_df,
        x="q_label",
        y="sensitivity",
        order=["$q=0$", "$q=0.25$"],
        ax=ax,
        inner=None,
        cut=0,
        linewidth=0.8,
        palette=["#B8B8B8", "#4C9A8A"],
    )
    sns.boxplot(
        data=plot_df,
        x="q_label",
        y="sensitivity",
        order=["$q=0$", "$q=0.25$"],
        ax=ax,
        width=0.22,
        showcaps=True,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.8},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        medianprops={"color": "#111111", "linewidth": 1.0},
    )
    stats = magnetic_stats(root, "NB101_Acc_172")
    q025 = stats.loc[np.isclose(stats["q"], 0.25)].iloc[0]
    ax.axhline(0.0, color="#555555", linewidth=0.7)
    ax.set_xlabel("")
    ax.set_ylabel("Direction sensitivity $S_q(G)$")
    ax.set_title(title)
    ax.text(
        0.04,
        0.95,
        "$q=0$: 0.000\n"
        f"$q=0.25$: {q025['mean']:.3f}$\\pm${q025['std']:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )
    if full_range:
        ax.set_ylim(-0.02, 1.48)
    else:
        ax.set_ylim(-0.015, 0.37)
        ax.text(
            0.04,
            0.80,
            "Bulk view; max=1.414",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=6.8,
            color="#555555",
        )
    ax.grid(axis="x", visible=False)


def plot_magnetic_dataset_panel(ax: plt.Axes, root: Path, key: str, title: str, *, ylim: tuple[float, float]) -> None:
    plot_df = read_magnetic(root, key, sample_per_q=60000)
    sns.violinplot(
        data=plot_df,
        x="q_label",
        y="sensitivity",
        order=["$q=0$", "$q=0.25$"],
        ax=ax,
        inner=None,
        cut=0,
        linewidth=0.8,
        palette=["#B8B8B8", "#4C9A8A"],
    )
    sns.boxplot(
        data=plot_df,
        x="q_label",
        y="sensitivity",
        order=["$q=0$", "$q=0.25$"],
        ax=ax,
        width=0.22,
        showfliers=False,
        boxprops={"facecolor": "white", "edgecolor": "#333333", "linewidth": 0.8},
        whiskerprops={"color": "#333333", "linewidth": 0.8},
        medianprops={"color": "#111111", "linewidth": 1.0},
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("Sensitivity $S_q(G)$")
    ax.set_ylim(*ylim)
    ax.grid(axis="x", visible=False)


def plot_moe(ax: plt.Axes, gate: pd.DataFrame, title: str, *, legend: bool = True) -> None:
    x = np.arange(len(ROLE_ORDER), dtype=float)
    width = 0.24
    offsets = [-width, 0.0, width]
    for (col, label, color), offset in zip(EXPERTS, offsets):
        values = gate.set_index("node_role").loc[ROLE_ORDER, col].to_numpy()
        ax.bar(x + offset, values, width=width, label=label, color=color, edgecolor="white", linewidth=0.5)
    ax.axhline(1.0 / 3.0, color="#444444", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([ROLE_LABELS[r] for r in ROLE_ORDER], rotation=22, ha="right")
    ax.set_ylim(0.25, 0.47)
    ax.set_ylabel("Average gate weight")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    if legend:
        ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.995))


def figure_main_magnetic(root: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    plot_magnetic_nb101(ax, root, full_range=False, title="Magnetic phase sensitivity (NB101)")
    save_figure(fig, out_dir, "main_magnetic_nb101_bulk")

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    plot_magnetic_nb101(ax, root, full_range=True, title="Magnetic phase sensitivity (NB101)")
    save_figure(fig, out_dir, "main_magnetic_nb101_fullrange")


def figure_main_magnetic_moe(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.55), gridspec_kw={"width_ratios": [0.90, 1.32]})
    plot_magnetic_nb101(axes[0], root, full_range=False, title="Magnetic phase sensitivity")
    axes[0].text(-0.15, 1.06, "(a)", transform=axes[0].transAxes, fontsize=9, fontweight="bold")
    gate = read_gate(root, "NB201_Acc_469")
    plot_moe(axes[1], gate, "MoE routing on NB201 Acc @469", legend=True)
    axes[1].text(-0.09, 1.06, "(b)", transform=axes[1].transAxes, fontsize=9, fontweight="bold")
    fig.subplots_adjust(wspace=0.38)
    save_figure(fig, out_dir, "main_mechanistic_magnetic_moe")


def figure_appendix_moe(root: Path, out_dir: Path) -> None:
    settings = [
        ("NB101_Acc_172", "NB101 Acc @172"),
        ("NB101_Lat_172", "NB101 Lat @172"),
        ("NB201_Acc_469", "NB201 Acc @469"),
        ("NB201_Lat_469", "NB201 Lat @469"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.1), sharey=True)
    for ax, (key, title) in zip(axes.ravel(), settings):
        plot_moe(ax, read_gate(root, key), title, legend=False)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.01))
    fig.subplots_adjust(hspace=0.42, wspace=0.18, top=0.90)
    save_figure(fig, out_dir, "appendix_moe_routing_all_settings")


def figure_appendix_attention(root: Path, out_dir: Path) -> None:
    df = pd.read_csv(root / "combined_attention_relation_summary.csv")
    df["setting_short"] = df["label"].map(SETTING_LABELS)
    heat = (
        df.pivot(index="setting_short", columns="relation", values="attention_enrichment")
        .loc[["NB101 Acc", "NB101 Lat", "NB201 Acc", "NB201 Lat"], RELATION_ORDER]
    )
    fig, ax = plt.subplots(figsize=(7.1, 2.45))
    sns.heatmap(
        heat,
        ax=ax,
        cmap="YlGnBu",
        vmin=0.0,
        vmax=5.0,
        annot=True,
        fmt=".2f",
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"label": "Attention enrichment", "shrink": 0.86},
    )
    ax.set_xlabel("Directional relation channel")
    ax.set_ylabel("")
    ax.set_xticklabels([RELATION_LABELS[r] for r in RELATION_ORDER], rotation=0)
    ax.set_title("Attention enrichment over explicit directional relation channels")
    ax.text(
        0.0,
        -0.36,
        "Structural masks define the channels; this figure verifies explicit relation-channel separation, not learned head specialization.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.2,
        color="#444444",
    )
    save_figure(fig, out_dir, "appendix_attention_relation_channels")


def figure_appendix_magnetic(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.55), sharey=False)
    plot_magnetic_dataset_panel(axes[0], root, "NB101_Acc_172", "NAS-Bench-101", ylim=(-0.015, 0.37))
    axes[0].text(0.50, 0.84, "Bulk view; max=1.414", transform=axes[0].transAxes, fontsize=7.2, color="#555555")
    plot_magnetic_dataset_panel(axes[1], root, "NB201_Acc_469", "NAS-Bench-201", ylim=(-0.015, 0.25))
    axes[1].text(0.46, 0.84, "Fixed topology", transform=axes[1].transAxes, fontsize=7.2, color="#555555")
    fig.subplots_adjust(wspace=0.32)
    save_figure(fig, out_dir, "appendix_magnetic_nb101_nb201")


def write_guide(out_dir: Path) -> None:
    guide = """# Second-Budget Mechanism Figure Guide

## Main Paper Recommendation

Use `main_magnetic_nb101_bulk.pdf` as the cleanest main-paper figure.
It shows NAS-Bench-101 direction sensitivity for q=0 versus q=0.25.
The y-axis focuses on the bulk distribution; `main_magnetic_nb101_fullrange.pdf` is also provided for a full-range check.

Optional two-panel main figure:
Use `main_mechanistic_magnetic_moe.pdf`.
Panel (a) is magnetic phase sensitivity on NAS-Bench-101.
Panel (b) is MoE routing by directed node role on NB201 Acc @469, the clearest representative MoE setting.

Suggested caption for the two-panel figure:
Mechanistic diagnostics of directional modeling. (a) Magnetic phase sensitivity compares the spectral operator under A and A^T. The q=0 case removes the magnetic phase and yields zero sensitivity, while q=0.25 produces positive sensitivity on NAS-Bench-101. (b) A representative MoE routing diagnostic on NAS-Bench-201 accuracy shows role-dependent shifts: source and branching nodes assign more weight to outgoing contexts, whereas sink and converging nodes place more weight on incoming contexts. Full cross-setting diagnostics are reported in the appendix.

## Appendix Figures

Use `appendix_moe_routing_all_settings.pdf` for full MoE routing across all four second-budget settings.
Use cautious wording: role-dependent shifts are visible but not uniformly strong across all settings.

Use `appendix_attention_relation_channels.pdf` for attention relation-channel analysis.
Do not describe it as learned head specialization. The correct wording is explicit directional relation-channel separation under hard structural masks.

Use `appendix_magnetic_nb101_nb201.pdf` as a full magnetic sanity check.
NB101 gives the useful topology-diverse distribution; NB201 is a fixed-topology sanity check with constant q=0.25 sensitivity.

## File Formats

Every figure is saved as both PDF and PNG in this directory.
"""
    (out_dir / "FIGURE_GUIDE.md").write_text(guide, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", default="results/diagnostics_second_budget")
    parser.add_argument("--out-dir", default="results/paper_figures_second_budget")
    args = parser.parse_args()

    setup_style()
    root = Path(args.diagnostics_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    figure_main_magnetic(root, out_dir)
    figure_main_magnetic_moe(root, out_dir)
    figure_appendix_moe(root, out_dir)
    figure_appendix_attention(root, out_dir)
    figure_appendix_magnetic(root, out_dir)
    write_guide(out_dir)

    print(f"Paper mechanism figures written to {out_dir}")


if __name__ == "__main__":
    main()
