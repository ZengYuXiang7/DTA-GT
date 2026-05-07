#!/usr/bin/env python3
"""NeurIPS-style mechanism figures from second-budget diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROLE_ORDER = ["source", "branching", "middle", "converging", "sink"]
ROLE_LABEL = {
    "source": "Source",
    "branching": "Branching",
    "middle": "Middle",
    "converging": "Converging",
    "sink": "Sink",
}
SETTING_ORDER = [
    ("NB101_Acc_172", "NB101 Acc"),
    ("NB101_Lat_172", "NB101 Lat"),
    ("NB201_Acc_469", "NB201 Acc"),
    ("NB201_Lat_469", "NB201 Lat"),
]
SETTING_FULL_LABEL = {
    "NB101_Acc_172": "NB101 Acc @172",
    "NB101_Lat_172": "NB101 Lat @172",
    "NB201_Acc_469": "NB201 Acc @469",
    "NB201_Lat_469": "NB201 Lat @469",
}
REL_ORDER = ["A", "A^T", "R_s2t", "R_t2s", "AA^T", "A^TA"]
REL_LABEL = {
    "A": "Fwd\nAdj.",
    "A^T": "Bwd\nAdj.",
    "R_s2t": "Fwd\nReach",
    "R_t2s": "Bwd\nReach",
    "AA^T": "Conv.",
    "A^TA": "Branch",
}
EXPERTS = [
    ("g_self_mean", "Self", "#666666"),
    ("g_in_backward_AT_mean", "Incoming", "#4477AA"),
    ("g_out_forward_A_mean", "Outgoing", "#CC6677"),
]
MARKERS = ["s", "o", "^"]


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Nimbus Roman", "Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 6.4,
            "axes.labelsize": 6.6,
            "axes.titlesize": 6.8,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "legend.fontsize": 5.8,
            "axes.linewidth": 0.45,
            "xtick.major.width": 0.45,
            "ytick.major.width": 0.45,
            "xtick.major.size": 2.2,
            "ytick.major.size": 2.2,
            "legend.handlelength": 1.0,
            "legend.handletextpad": 0.35,
            "legend.columnspacing": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.pad_inches": 0.012,
        }
    )


def finish_axis(ax: plt.Axes, *, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444444")
    ax.spines["bottom"].set_color("#444444")
    ax.tick_params(colors="#222222")
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E5E5E5", linewidth=0.35)
        ax.set_axisbelow(True)


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def magnetic_values(root: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(root / key / "magnetic_phase_sensitivity.csv")
    return (
        df.loc[np.isclose(df["q"], 0.0), "sensitivity"].to_numpy(),
        df.loc[np.isclose(df["q"], 0.25), "sensitivity"].to_numpy(),
    )


def magnetic_summary(root: Path, key: str) -> pd.DataFrame:
    return pd.read_csv(root / key / "magnetic_phase_summary.csv")


def attention_summary(root: Path, key: str) -> pd.DataFrame:
    return pd.read_csv(root / key / "attention_relation_summary.csv").set_index("relation").loc[REL_ORDER].reset_index()


def role_contrast_table(root: Path) -> pd.DataFrame:
    df = pd.read_csv(root / "combined_gate_role_contrasts.csv")
    return pd.DataFrame(
        {
            "Setting": df["label"].str.replace(" @", "\n@", regex=False),
            "Source\nout-in": df["source_out_minus_in"],
            "Branching\nout-in": df["branching_out_minus_in"],
            "Converging\nin-out": df["converging_in_minus_out"],
            "Sink\nin-out": df["sink_in_minus_out"],
        }
    )


def draw_magnetic_box(
    ax: plt.Axes,
    root: Path,
    key: str = "NB101_Acc_172",
    *,
    bulk_ylim: tuple[float, float] = (-0.012, 0.36),
    annotate: bool = True,
    show_ylabel: bool = True,
) -> None:
    q0, q25 = magnetic_values(root, key)
    q25_bulk = q25[(q25 >= bulk_ylim[0]) & (q25 <= bulk_ylim[1])]
    rng = np.random.default_rng(20260504)
    idx = rng.choice(np.arange(q25_bulk.size), size=min(1400, q25_bulk.size), replace=False)
    jitter = rng.normal(1.0, 0.035, size=idx.size)

    vp = ax.violinplot(
        [q25_bulk],
        positions=[1],
        widths=0.52,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor("#66A89A")
        body.set_edgecolor("none")
        body.set_alpha(0.16)

    bp = ax.boxplot(
        [q0, q25_bulk],
        positions=[0, 1],
        widths=0.34,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#111111", "linewidth": 0.72},
        whiskerprops={"color": "#333333", "linewidth": 0.52},
        capprops={"color": "#333333", "linewidth": 0.52},
        boxprops={"edgecolor": "#333333", "linewidth": 0.52},
    )
    for patch, color in zip(bp["boxes"], ["#D7D7D7", "#66A89A"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.95)

    ax.scatter(jitter, q25_bulk[idx], s=1.15, color="#2B7A73", alpha=0.06, linewidth=0)
    ax.axhline(0, color="#555555", linewidth=0.42)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([r"$q=0$", r"$q=0.25$"])
    ax.set_ylim(*bulk_ylim)
    ax.set_xlim(-0.55, 1.55)
    if show_ylabel:
        ax.set_ylabel(r"Sensitivity $S_q(G)$")
    else:
        ax.set_ylabel("")
    if annotate:
        summary = magnetic_summary(root, key)
        q25_row = summary.loc[np.isclose(summary["q"], 0.25)].iloc[0]
        ax.text(
            0.50,
            0.94,
            rf"$\mu={q25_row['mean']:.3f},\ \sigma={q25_row['std']:.3f}$",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=5.9,
        )
    finish_axis(ax, grid_axis="y")


def gate_summary(root: Path, key: str) -> pd.DataFrame:
    df = pd.read_csv(root / key / "gate_by_role_summary.csv")
    return df.set_index("node_role").loc[ROLE_ORDER].reset_index()


def draw_moe_delta_weights(
    ax: plt.Axes,
    gate: pd.DataFrame,
    *,
    xlim: tuple[float, float],
    show_ylabel: bool = True,
    legend: bool = False,
) -> None:
    y = np.arange(len(ROLE_ORDER), dtype=float)
    offset_step = 0.16
    offsets = [-offset_step, 0, offset_step]
    baseline = 1.0 / 3.0
    for (col, name, color), marker, offset in zip(EXPERTS, MARKERS, offsets):
        vals = gate[col].to_numpy() - baseline
        for yi, val in zip(y + offset, vals):
            ax.plot([0, val], [yi, yi], color=color, alpha=0.24, linewidth=0.85, solid_capstyle="round")
        ax.scatter(
            vals,
            y + offset,
            s=12,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.25,
            label=name,
            zorder=3,
        )
    ax.axvline(0, color="#777777", linestyle=(0, (2.2, 2.2)), linewidth=0.58)
    ax.set_yticks(y)
    ax.set_yticklabels([ROLE_LABEL[r] for r in ROLE_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(r"Gate weight $-\,1/3$")
    if not show_ylabel:
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    if legend:
        ax.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.01))
    finish_axis(ax, grid_axis="x")


def draw_moe_role_contrast(ax: plt.Axes, gate: pd.DataFrame) -> None:
    """Compact main-paper view of the clearest NB201 Acc role-aligned contrasts."""
    gate = gate.set_index("node_role")
    rows = [
        ("source", gate.loc["source", "g_out_forward_A_mean"] - gate.loc["source", "g_in_backward_AT_mean"], r"$g_{\rm out}-g_{\rm in}$", "#CC6677"),
        ("branching", gate.loc["branching", "g_out_forward_A_mean"] - gate.loc["branching", "g_in_backward_AT_mean"], r"$g_{\rm out}-g_{\rm in}$", "#CC6677"),
        ("converging", gate.loc["converging", "g_in_backward_AT_mean"] - gate.loc["converging", "g_out_forward_A_mean"], r"$g_{\rm in}-g_{\rm out}$", "#4477AA"),
        ("sink", gate.loc["sink", "g_in_backward_AT_mean"] - gate.loc["sink", "g_out_forward_A_mean"], r"$g_{\rm in}-g_{\rm out}$", "#4477AA"),
    ]
    y = np.arange(len(rows), dtype=float)
    vals = np.array([r[1] for r in rows])
    colors = [r[3] for r in rows]
    ax.barh(y, vals, height=0.42, color=colors, alpha=0.72, edgecolor="#333333", linewidth=0.35)
    ax.axvline(0, color="#555555", linewidth=0.55)
    ax.set_yticks(y)
    ax.set_yticklabels([ROLE_LABEL[r[0]] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(-0.006, 0.050)
    ax.set_xticks([0.00, 0.02, 0.04])
    ax.set_xlabel("Role-aligned gate contrast")
    for yi, (_, val, _label, _) in enumerate(rows):
        ax.text(val + 0.0014, yi, f"{val:.3f}", ha="left", va="center", fontsize=5.5, color="#333333")
    finish_axis(ax, grid_axis="x")


def draw_moe_role_contrast_heatmap(ax: plt.Axes, table: pd.DataFrame, *, annotate: bool = True) -> None:
    value_cols = [c for c in table.columns if c != "Setting"]
    vals = table[value_cols].to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=-0.012, vcenter=0.0, vmax=0.045)
    im = ax.imshow(vals, aspect="auto", cmap="RdBu_r", norm=norm)
    ax.set_yticks(np.arange(table.shape[0]))
    ax.set_yticklabels(table["Setting"].tolist())
    ax.set_xticks(np.arange(len(value_cols)))
    ax.set_xticklabels(value_cols)
    if annotate:
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]):
                ax.text(j, i, f"{vals[i, j]:+.3f}", ha="center", va="center", fontsize=5.0, color="#111111")
    ax.set_xticks(np.arange(-0.5, len(value_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, table.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.70)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def draw_attention_enrichment_bars(ax: plt.Axes, rel: pd.DataFrame, *, show_ylabel: bool = True) -> None:
    vals = rel["attention_enrichment"].to_numpy(dtype=float)
    colors = ["#88CCEE", "#88CCEE", "#DDCC77", "#DDCC77", "#999933", "#44AA99"]
    x = np.arange(len(REL_ORDER))
    ax.bar(x, vals, color=colors, alpha=0.78, edgecolor="#333333", linewidth=0.35, width=0.68)
    ax.axhline(1.0, color="#666666", linestyle=(0, (2.2, 2.2)), linewidth=0.58)
    ax.set_xticks(x)
    ax.set_xticklabels([REL_LABEL[r] for r in REL_ORDER])
    ax.set_ylim(0, max(5.1, vals.max() * 1.12))
    if show_ylabel:
        ax.set_ylabel("Enrichment")
    for xi, val in zip(x, vals):
        ax.text(xi, val + 0.08, f"{val:.2f}", ha="center", va="bottom", fontsize=4.9, color="#333333")
    finish_axis(ax, grid_axis="y")


def draw_attention_mass_density(ax: plt.Axes, rel: pd.DataFrame, *, show_ylabel: bool = True) -> None:
    x = np.arange(len(REL_ORDER))
    width = 0.34
    ax.bar(
        x - width / 2,
        rel["relation_density"].to_numpy(dtype=float),
        width=width,
        color="#BBBBBB",
        alpha=0.82,
        edgecolor="#333333",
        linewidth=0.28,
        label="Density",
    )
    ax.bar(
        x + width / 2,
        rel["raw_attention_mass"].to_numpy(dtype=float),
        width=width,
        color="#77AADD",
        alpha=0.78,
        edgecolor="#333333",
        linewidth=0.28,
        label="Mass",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([REL_LABEL[r] for r in REL_ORDER])
    ax.set_ylim(0, 0.55)
    if show_ylabel:
        ax.set_ylabel("Fraction")
    ax.legend(frameon=False, ncol=2, loc="upper right")
    finish_axis(ax, grid_axis="y")


def fig_main_magnetic(root: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.35, 1.72))
    draw_magnetic_box(ax, root, annotate=True)
    ax.set_title("")
    save(fig, out_dir, "neurips_main_magnetic_nb101")


def fig_main_magnetic_full(root: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(2.35, 1.72))
    draw_magnetic_box(ax, root, bulk_ylim=(-0.04, 1.48), annotate=False)
    ax.set_title("")
    save(fig, out_dir, "neurips_main_magnetic_nb101_fullrange")


def fig_main_two_panel(root: Path, out_dir: Path) -> None:
    fig = plt.figure(figsize=(5.35, 1.78))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.92, 1.12], wspace=0.44)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    draw_magnetic_box(ax0, root, annotate=True)
    ax0.set_title("Magnetic sensitivity", pad=2)
    gate = gate_summary(root, "NB201_Acc_469")
    draw_moe_role_contrast(ax1, gate)
    ax1.set_title("MoE role contrast", pad=2)
    ax0.text(-0.19, 1.04, "(a)", transform=ax0.transAxes, fontweight="bold", fontsize=6.8)
    ax1.text(-0.18, 1.04, "(b)", transform=ax1.transAxes, fontweight="bold", fontsize=6.8)
    save(fig, out_dir, "neurips_main_magnetic_moe")


def fig_appendix_moe(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(6.45, 3.80), sharex=True)
    for ax, (key, label) in zip(axes.ravel(), SETTING_ORDER):
        draw_moe_delta_weights(ax, gate_summary(root, key), xlim=(-0.065, 0.105), legend=False)
        ax.set_title(label, pad=2)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.52, 1.00))
    fig.subplots_adjust(top=0.88, hspace=0.34, wspace=0.24)
    save(fig, out_dir, "neurips_appendix_moe_all_settings")


def fig_appendix_attention(root: Path, out_dir: Path) -> None:
    df = pd.read_csv(root / "combined_attention_relation_summary.csv")
    setting_map = {
        "NB101 Acc @172": "NB101 Acc",
        "NB101 Lat @172": "NB101 Lat",
        "NB201 Acc @469": "NB201 Acc",
        "NB201 Lat @469": "NB201 Lat",
    }
    df["setting"] = df["label"].map(setting_map)
    heat = (
        df.pivot(index="setting", columns="relation", values="attention_enrichment")
        .loc[["NB101 Acc", "NB101 Lat", "NB201 Acc", "NB201 Lat"], REL_ORDER]
    )
    fig, ax = plt.subplots(figsize=(6.25, 1.58))
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.8, vmax=4.8)
    ax.set_xticks(np.arange(len(REL_ORDER)))
    ax.set_xticklabels([REL_LABEL[r] for r in REL_ORDER])
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index.tolist())
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            color = "white" if val >= 3.35 else "#111111"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.2, color=color)
    ax.set_xticks(np.arange(-0.5, len(REL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.70)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xlabel("Directional relation channel")
    cbar = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.016)
    cbar.set_label("Attention enrichment")
    cbar.ax.tick_params(labelsize=5.5)
    save(fig, out_dir, "neurips_appendix_attention_channels")


def fig_appendix_magnetic(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4.90, 1.78), sharey=False)
    draw_magnetic_box(axes[0], root, "NB101_Acc_172", annotate=True)
    axes[0].set_title("NAS-Bench-101", pad=2)
    draw_magnetic_box(axes[1], root, "NB201_Acc_469", bulk_ylim=(-0.012, 0.24), annotate=False)
    axes[1].set_title("NAS-Bench-201", pad=2)
    axes[1].text(
        0.06,
        0.94,
        "$q=0$: 0.000\n$q=0.25$: 0.206",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.8,
    )
    axes[1].text(
        0.06,
        0.73,
        "fixed topology",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=5.3,
        color="#555555",
    )
    fig.subplots_adjust(wspace=0.34)
    save(fig, out_dir, "neurips_appendix_magnetic_nb101_nb201")


def fig_all_settings_moe_contrast(root: Path, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.65, 1.82))
    im = draw_moe_role_contrast_heatmap(ax, role_contrast_table(root))
    ax.set_xlabel("Role-aligned directional contrast")
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.018)
    cbar.set_label("Gate contrast")
    cbar.ax.tick_params(labelsize=5.5)
    save(fig, out_dir, "neurips_all_settings_moe_role_contrast_heatmap")


def _combined_attention_matrix(root: Path, value_col: str) -> pd.DataFrame:
    df = pd.read_csv(root / "combined_attention_relation_summary.csv")
    setting_map = {
        "NB101 Acc @172": "NB101 Acc\n@172",
        "NB101 Lat @172": "NB101 Lat\n@172",
        "NB201 Acc @469": "NB201 Acc\n@469",
        "NB201 Lat @469": "NB201 Lat\n@469",
    }
    df["setting"] = df["label"].map(setting_map)
    return (
        df.pivot(index="setting", columns="relation", values=value_col)
        .loc[["NB101 Acc\n@172", "NB101 Lat\n@172", "NB201 Acc\n@469", "NB201 Lat\n@469"], REL_ORDER]
    )


def _draw_metric_heatmap(
    ax: plt.Axes,
    heat: pd.DataFrame,
    *,
    title: str,
    cmap: str,
    vmin: float,
    vmax: float,
    fmt: str,
) -> None:
    im = ax.imshow(heat.to_numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, pad=2)
    ax.set_xticks(np.arange(len(REL_ORDER)))
    ax.set_xticklabels([REL_LABEL[r] for r in REL_ORDER])
    ax.set_yticks(np.arange(heat.shape[0]))
    ax.set_yticklabels(heat.index.tolist())
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            val = heat.iloc[i, j]
            color = "white" if val > (vmin + 0.70 * (vmax - vmin)) else "#111111"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=4.8, color=color)
    ax.set_xticks(np.arange(-0.5, len(REL_ORDER), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, heat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.65)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return im


def fig_all_settings_attention_stats(root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.85, 1.95), sharey=True)
    specs = [
        ("relation_density", "Relation density", "Greys", 0.0, 0.50, ".2f"),
        ("raw_attention_mass", "Attention mass", "Blues", 0.0, 0.50, ".2f"),
        ("attention_enrichment", "Enrichment", "YlGnBu", 0.8, 4.8, ".2f"),
    ]
    ims = []
    for ax, (col, title, cmap, vmin, vmax, fmt) in zip(axes, specs):
        ims.append(_draw_metric_heatmap(ax, _combined_attention_matrix(root, col), title=title, cmap=cmap, vmin=vmin, vmax=vmax, fmt=fmt))
    for ax in axes[1:]:
        ax.set_yticklabels([])
    for ax, im, (_, title, _, _, _, _) in zip(axes, ims, specs):
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.020)
        cbar.ax.tick_params(labelsize=5.0)
    fig.subplots_adjust(wspace=0.16)
    save(fig, out_dir, "neurips_all_settings_attention_density_mass_enrichment")


def fig_all_settings_magnetic_summary(root: Path, out_dir: Path) -> None:
    df = pd.read_csv(root / "combined_magnetic_phase_summary.csv")
    q25 = df[np.isclose(df["q"], 0.25)].copy()
    q25["setting"] = q25["label"].str.replace(" @", "\n@", regex=False)
    fig, ax = plt.subplots(figsize=(3.85, 1.75))
    x = np.arange(q25.shape[0])
    ax.bar(x, q25["mean"], yerr=q25["std"], color="#66A89A", alpha=0.78, edgecolor="#333333", linewidth=0.35, capsize=2)
    ax.axhline(0, color="#555555", linewidth=0.50)
    ax.set_xticks(x)
    ax.set_xticklabels(q25["setting"].tolist())
    ax.set_ylabel(r"Sensitivity $S_{0.25}(G)$")
    ax.set_ylim(0, 0.28)
    ax.text(0.02, 0.94, r"$q=0$ gives zero sensitivity in all four settings", transform=ax.transAxes, ha="left", va="top", fontsize=5.6)
    finish_axis(ax, grid_axis="y")
    save(fig, out_dir, "neurips_all_settings_magnetic_summary")


def fig_all_settings_overview(root: Path, out_dir: Path) -> None:
    fig = plt.figure(figsize=(6.85, 2.08))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.78, 1.08, 1.34], wspace=0.42)

    ax0 = fig.add_subplot(gs[0, 0])
    df = pd.read_csv(root / "combined_magnetic_phase_summary.csv")
    q25 = df[np.isclose(df["q"], 0.25)].copy()
    q25["setting"] = q25["label"].str.replace(" @", "\n@", regex=False)
    x = np.arange(q25.shape[0])
    ax0.bar(x, q25["mean"], yerr=q25["std"], color="#66A89A", alpha=0.78, edgecolor="#333333", linewidth=0.35, capsize=1.8)
    ax0.set_xticks(x)
    ax0.set_xticklabels(q25["setting"].tolist(), rotation=45, ha="right")
    ax0.set_ylabel(r"$S_{0.25}(G)$")
    ax0.set_ylim(0, 0.28)
    ax0.set_title("Magnetic", pad=2)
    finish_axis(ax0, grid_axis="y")

    ax1 = fig.add_subplot(gs[0, 1])
    im1 = draw_moe_role_contrast_heatmap(ax1, role_contrast_table(root), annotate=True)
    ax1.set_title("MoE contrast", pad=2)
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.040, pad=0.018)
    cbar1.ax.tick_params(labelsize=4.8)

    ax2 = fig.add_subplot(gs[0, 2])
    im2 = _draw_metric_heatmap(
        ax2,
        _combined_attention_matrix(root, "attention_enrichment"),
        title="Attention enrichment",
        cmap="YlGnBu",
        vmin=0.8,
        vmax=4.8,
        fmt=".1f",
    )
    ax2.set_yticklabels([])
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.033, pad=0.018)
    cbar2.ax.tick_params(labelsize=4.8)

    ax0.text(-0.28, 1.05, "(a)", transform=ax0.transAxes, fontweight="bold", fontsize=6.8)
    ax1.text(-0.27, 1.05, "(b)", transform=ax1.transAxes, fontweight="bold", fontsize=6.8)
    ax2.text(-0.17, 1.05, "(c)", transform=ax2.transAxes, fontweight="bold", fontsize=6.8)
    save(fig, out_dir, "neurips_all_settings_mechanism_overview")


def fig_setting_moe(root: Path, out_dir: Path, key: str) -> None:
    fig, ax = plt.subplots(figsize=(3.20, 2.20))
    draw_moe_delta_weights(ax, gate_summary(root, key), xlim=(-0.065, 0.105), legend=True)
    ax.set_title(SETTING_FULL_LABEL[key], pad=2)
    save(fig, out_dir, f"neurips_setting_{key}_moe_gate_delta")


def fig_setting_attention(root: Path, out_dir: Path, key: str) -> None:
    rel = attention_summary(root, key)
    fig, axes = plt.subplots(1, 2, figsize=(5.75, 1.95))
    draw_attention_mass_density(axes[0], rel)
    axes[0].set_title("Density and mass", pad=2)
    draw_attention_enrichment_bars(axes[1], rel)
    axes[1].set_title("Enrichment", pad=2)
    fig.suptitle(SETTING_FULL_LABEL[key], y=1.02, fontsize=7.0)
    fig.subplots_adjust(wspace=0.26)
    save(fig, out_dir, f"neurips_setting_{key}_attention_stats")


def fig_setting_magnetic(root: Path, out_dir: Path, key: str) -> None:
    dataset_ylim = (-0.012, 0.36) if key.startswith("NB101") else (-0.012, 0.24)
    fig, ax = plt.subplots(figsize=(2.35, 1.72))
    draw_magnetic_box(ax, root, key, bulk_ylim=dataset_ylim, annotate=True)
    ax.set_title(SETTING_FULL_LABEL[key], pad=2)
    save(fig, out_dir, f"neurips_setting_{key}_magnetic")


def fig_setting_diagnostics(root: Path, out_dir: Path, key: str) -> None:
    dataset_ylim = (-0.012, 0.36) if key.startswith("NB101") else (-0.012, 0.24)
    fig = plt.figure(figsize=(6.85, 2.10))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1.12, 0.86], wspace=0.34)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])
    draw_moe_delta_weights(ax0, gate_summary(root, key), xlim=(-0.065, 0.105), legend=True)
    ax0.set_title("MoE gate", pad=2)
    draw_attention_enrichment_bars(ax1, attention_summary(root, key))
    ax1.set_title("Attention relations", pad=2)
    draw_magnetic_box(ax2, root, key, bulk_ylim=dataset_ylim, annotate=True)
    ax2.set_title("Magnetic phase", pad=2)
    ax0.text(-0.22, 1.04, "(a)", transform=ax0.transAxes, fontweight="bold", fontsize=6.8)
    ax1.text(-0.18, 1.04, "(b)", transform=ax1.transAxes, fontweight="bold", fontsize=6.8)
    ax2.text(-0.26, 1.04, "(c)", transform=ax2.transAxes, fontweight="bold", fontsize=6.8)
    fig.suptitle(SETTING_FULL_LABEL[key], y=1.03, fontsize=7.2)
    save(fig, out_dir, f"neurips_setting_{key}_full_diagnostics")


def fig_all_per_setting(root: Path, out_dir: Path) -> None:
    for key, _label in SETTING_ORDER:
        fig_setting_diagnostics(root, out_dir, key)
        fig_setting_moe(root, out_dir, key)
        fig_setting_attention(root, out_dir, key)
        fig_setting_magnetic(root, out_dir, key)


def write_analysis_tables(root: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    role_contrast_table(root).to_csv(out_dir / "table_moe_role_aligned_contrasts.csv", index=False)
    pd.read_csv(root / "combined_gate_by_role_summary.csv").to_csv(out_dir / "table_moe_gate_by_role_all_settings.csv", index=False)
    pd.read_csv(root / "combined_attention_relation_summary.csv").to_csv(out_dir / "table_attention_relation_stats_all_settings.csv", index=False)
    pd.read_csv(root / "combined_magnetic_phase_summary.csv").to_csv(out_dir / "table_magnetic_phase_summary_all_settings.csv", index=False)


def write_guide(out_dir: Path) -> None:
    text = """# NeurIPS-Style Mechanism Figure Guide

Use these figures rather than the raw diagnostic plots.

## Main Paper

Preferred single-panel figure:
- `neurips_main_magnetic_nb101.pdf`

Optional two-panel mechanism figure:
- `neurips_main_magnetic_moe.pdf`

Recommended caption for the two-panel figure:
Mechanistic diagnostics of directional modeling. (a) Magnetic phase sensitivity on NAS-Bench-101. Removing the magnetic phase (`q=0`) yields zero sensitivity to transposing the adjacency matrix, whereas `q=0.25` produces positive direction sensitivity. (b) A representative MoE routing diagnostic on NAS-Bench-201 accuracy shows positive role-aligned directional gate contrasts: `g_out - g_in` for source/branching nodes and `g_in - g_out` for converging/sink nodes. Full cross-setting gate weights are shown in the appendix.

## Appendix

- `neurips_appendix_moe_all_settings.pdf`: full MoE routing across all four second-budget settings, plotted as gate weight minus the uniform value 1/3.
- `neurips_appendix_attention_channels.pdf`: explicit directional relation-channel separation. Do not call this learned head specialization because the channels are hard-masked.
- `neurips_appendix_magnetic_nb101_nb201.pdf`: magnetic phase sanity check on NB101 and NB201.

## Full Four-Setting Analysis Pack

Use these when you want to discuss all four settings rather than selecting a representative case.

- `neurips_all_settings_mechanism_overview.pdf`: compact three-panel overview for Magnetic, MoE role contrast, and Attention enrichment across all four settings.
- `neurips_all_settings_magnetic_summary.pdf`: q=0.25 Magnetic sensitivity for all four settings, with q=0 noted as zero.
- `neurips_all_settings_moe_role_contrast_heatmap.pdf`: role-aligned MoE contrasts across all four settings.
- `neurips_all_settings_attention_density_mass_enrichment.pdf`: relation density, raw attention mass, and enrichment across all four settings.

Per-setting complete diagnostic figures:
- `neurips_setting_NB101_Acc_172_full_diagnostics.pdf`
- `neurips_setting_NB101_Lat_172_full_diagnostics.pdf`
- `neurips_setting_NB201_Acc_469_full_diagnostics.pdf`
- `neurips_setting_NB201_Lat_469_full_diagnostics.pdf`

Per-setting single-module figures are also saved with suffixes:
- `_moe_gate_delta.pdf`
- `_attention_stats.pdf`
- `_magnetic.pdf`

Analysis tables copied into this directory:
- `table_moe_role_aligned_contrasts.csv`
- `table_moe_gate_by_role_all_settings.csv`
- `table_attention_relation_stats_all_settings.csv`
- `table_magnetic_phase_summary_all_settings.csv`

All figures are saved as PDF and 600 dpi PNG.
"""
    (out_dir / "NEURIPS_FIGURE_GUIDE.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-root", default="results/diagnostics_second_budget")
    parser.add_argument("--out-dir", default="results/neurips_figures_second_budget")
    args = parser.parse_args()
    root = Path(args.diagnostics_root)
    out_dir = Path(args.out_dir)
    set_style()
    fig_main_magnetic(root, out_dir)
    fig_main_magnetic_full(root, out_dir)
    fig_main_two_panel(root, out_dir)
    fig_appendix_moe(root, out_dir)
    fig_appendix_attention(root, out_dir)
    fig_appendix_magnetic(root, out_dir)
    fig_all_settings_overview(root, out_dir)
    fig_all_settings_magnetic_summary(root, out_dir)
    fig_all_settings_moe_contrast(root, out_dir)
    fig_all_settings_attention_stats(root, out_dir)
    fig_all_per_setting(root, out_dir)
    write_analysis_tables(root, out_dir)
    write_guide(out_dir)
    print(f"NeurIPS-style figures written to {out_dir}")


if __name__ == "__main__":
    main()
