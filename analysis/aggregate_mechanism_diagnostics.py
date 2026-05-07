#!/usr/bin/env python3
"""Aggregate per-setting DTA-GT mechanism diagnostics into handoff files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


GATE_COLUMNS = {
    "g_self_mean": "g_self",
    "g_out_forward_A_mean": "g_out",
    "g_in_backward_AT_mean": "g_in",
}


def read_index(root: Path) -> pd.DataFrame:
    index_path = root / "run_index.tsv"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing {index_path}")
    return pd.read_csv(index_path, sep="\t")


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def add_setting(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    df = df.copy()
    for col in ["setting_key", "label", "dataset", "target", "percent", "run_dir", "diag_dir"]:
        df.insert(0, col, row[col])
    return df


def summarize_training(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = Path(row["run_dir"]) / "summary_all_runs.json"
    rows = json.load(open(summary_path))
    runs = pd.DataFrame(rows)
    runs = add_setting(runs, row)

    metrics = {}
    for key in ["best_epoch", "tau", "mape", "err", "test_tau", "test_mape", "test_err", "train_time_sec"]:
        vals = pd.to_numeric(runs[key], errors="coerce").dropna() if key in runs else pd.Series(dtype=float)
        metrics[f"{key}_mean"] = vals.mean() if len(vals) else np.nan
        metrics[f"{key}_std"] = vals.std(ddof=1) if len(vals) > 1 else 0.0 if len(vals) == 1 else np.nan
    summary = pd.DataFrame([{**{c: row[c] for c in ["setting_key", "label", "dataset", "target", "percent", "run_dir", "diag_dir"]}, **metrics}])
    return runs, summary


def markdown_table(df: pd.DataFrame, columns: Iterable[str] | None = None, max_rows: int | None = None) -> str:
    if columns is not None:
        df = df[list(columns)]
    if max_rows is not None:
        df = df.head(max_rows)
    if df.empty:
        return "_No rows._"

    def fmt(value):
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value)

    rows = [[fmt(v) for v in record] for record in df.to_numpy()]
    headers = [str(c) for c in df.columns]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(v)) for w, v in zip(widths, row)]

    header = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |" for row in rows]
    return "\n".join([header, sep, *body])


def compact_gate(gate: pd.DataFrame) -> pd.DataFrame:
    out = gate.copy()
    rename = {v: k for k, v in GATE_COLUMNS.items()}
    out = out.rename(columns=GATE_COLUMNS)
    cols = ["node_role", "count_per_layer_run", "g_self", "g_in", "g_out"]
    return out[[c for c in cols if c in out.columns]]


def role_contrasts(gate: pd.DataFrame, label: str) -> dict[str, float | str]:
    by_role = {r["node_role"]: r for _, r in gate.iterrows()}

    def val(role: str, col: str) -> float:
        if role not in by_role or col not in by_role[role]:
            return float("nan")
        return float(by_role[role][col])

    return {
        "label": label,
        "source_out_minus_in": val("source", "g_out_forward_A_mean") - val("source", "g_in_backward_AT_mean"),
        "sink_in_minus_out": val("sink", "g_in_backward_AT_mean") - val("sink", "g_out_forward_A_mean"),
        "branching_out_minus_in": val("branching", "g_out_forward_A_mean") - val("branching", "g_in_backward_AT_mean"),
        "converging_in_minus_out": val("converging", "g_in_backward_AT_mean") - val("converging", "g_out_forward_A_mean"),
    }


def recommendation_from_gate(contrast_df: pd.DataFrame) -> str:
    checks = {
        "source_out_minus_in": (contrast_df["source_out_minus_in"] > 0).mean(),
        "sink_in_minus_out": (contrast_df["sink_in_minus_out"] > 0).mean(),
        "branching_out_minus_in": (contrast_df["branching_out_minus_in"] > 0).mean(),
        "converging_in_minus_out": (contrast_df["converging_in_minus_out"] > 0).mean(),
    }
    passed = sum(v >= 0.75 for v in checks.values())
    mean_abs = contrast_df[[c for c in checks]].abs().mean().mean()
    if passed >= 3 and mean_abs >= 0.005:
        return "Main-paper candidate: role-aligned MoE contrasts are mostly consistent across settings."
    if passed >= 2:
        return "Appendix / tentative main: role trends exist but are not uniformly strong."
    return "Do not use as primary evidence: role-aligned gate contrasts are weak or inconsistent."


def write_outputs(root: Path, title: str, report_name: str) -> None:
    index = read_index(root)
    setting_rows = []
    training_runs = []
    training_summary = []
    gate_rows = []
    attention_rows = []
    attention_intended_rows = []
    magnetic_rows = []
    magnetic_diff_rows = []
    contrasts = []

    for _, row in index.iterrows():
        diag_dir = Path(row["diag_dir"])
        run_dir = Path(row["run_dir"])

        setting_rows.append({
            **row.to_dict(),
            "moe_fig": str(diag_dir / "moe_gate_by_role_bar.pdf"),
            "attention_fig": str(diag_dir / "attention_enrichment_layer_head_heatmap.pdf"),
            "magnetic_fig": str(diag_dir / "magnetic_phase_sensitivity_violin.pdf"),
            "report": str(diag_dir / "mechanism_analysis_report.md"),
        })

        runs, train_summary = summarize_training(row)
        training_runs.append(runs)
        training_summary.append(train_summary)

        gate = read_csv_if_exists(diag_dir / "gate_by_role_summary.csv")
        if gate is not None:
            gate_rows.append(add_setting(gate, row))
            contrasts.append(role_contrasts(gate, row["label"]))

        attention = read_csv_if_exists(diag_dir / "attention_relation_summary.csv")
        if attention is not None:
            attention_rows.append(add_setting(attention, row))

        intended = read_csv_if_exists(diag_dir / "attention_intended_head_summary.csv")
        if intended is not None:
            attention_intended_rows.append(add_setting(intended, row))

        mag = read_csv_if_exists(diag_dir / "magnetic_phase_summary.csv")
        if mag is not None:
            magnetic_rows.append(add_setting(mag, row))

        diff = read_csv_if_exists(diag_dir / "magnetic_phase_paired_difference.csv")
        if diff is not None:
            magnetic_diff_rows.append(add_setting(diff, row))

        if not run_dir.exists():
            raise FileNotFoundError(run_dir)

    setting_index = pd.DataFrame(setting_rows)
    setting_index.to_csv(root / "setting_index.csv", index=False)

    combined_training_runs = pd.concat(training_runs, ignore_index=True)
    combined_training_runs.to_csv(root / "combined_training_metrics_runs.csv", index=False)

    combined_training_summary = pd.concat(training_summary, ignore_index=True)
    combined_training_summary.to_csv(root / "combined_training_metrics_summary.csv", index=False)

    combined_gate = pd.concat(gate_rows, ignore_index=True) if gate_rows else pd.DataFrame()
    combined_gate.to_csv(root / "combined_gate_by_role_summary.csv", index=False)

    combined_attention = pd.concat(attention_rows, ignore_index=True) if attention_rows else pd.DataFrame()
    combined_attention.to_csv(root / "combined_attention_relation_summary.csv", index=False)

    combined_intended = pd.concat(attention_intended_rows, ignore_index=True) if attention_intended_rows else pd.DataFrame()
    combined_intended.to_csv(root / "combined_attention_intended_head_summary.csv", index=False)

    combined_mag = pd.concat(magnetic_rows, ignore_index=True) if magnetic_rows else pd.DataFrame()
    combined_mag.to_csv(root / "combined_magnetic_phase_summary.csv", index=False)

    combined_diff = pd.concat(magnetic_diff_rows, ignore_index=True) if magnetic_diff_rows else pd.DataFrame()
    combined_diff.to_csv(root / "combined_magnetic_phase_paired_difference.csv", index=False)

    contrast_df = pd.DataFrame(contrasts)
    contrast_df.to_csv(root / "combined_gate_role_contrasts.csv", index=False)

    gate_rec = recommendation_from_gate(contrast_df) if not contrast_df.empty else "No MoE gate data found."

    lines = [
        f"# {title}",
        "",
        "This file is intended for downstream ChatGPT analysis and paper-figure decision making. It contains paths, training metrics, mechanism statistics, stability checks, and wording caveats for all four second-budget settings.",
        "",
        "## Global Caveats",
        "",
        "- All four settings use `model56`, `graph_readout=att`, `graph_n_head=6`, `gcn_layers=10`, and `rounds=3`.",
        "- MoE naming follows implementation-aware semantics: `g_out` is `adj @ H` / forward-successor context, and `g_in` is `adj.T @ H` / backward-predecessor context under `A[u,v]=1` for `u -> v`.",
        "- The structural attention module uses hard structural masks per head. Attention results therefore support explicit directional relation-channel separation, not learned free-form head specialization.",
        "- Magnetic phase is model-independent and computed on the requested split. NB201 has fixed small topology, so magnetic sensitivity is expected to be constant; NB101 provides distributional evidence.",
        "",
        "## Setting Index And File Paths",
        "",
        markdown_table(setting_index[["label", "run_dir", "diag_dir", "moe_fig", "attention_fig", "magnetic_fig", "report"]]),
        "",
        "## Training Metrics Summary",
        "",
        markdown_table(combined_training_summary[[
            "label",
            "best_epoch_mean",
            "best_epoch_std",
            "test_tau_mean",
            "test_tau_std",
            "test_mape_mean",
            "test_mape_std",
            "test_err_mean",
            "test_err_std",
            "train_time_sec_mean",
        ]]),
        "",
        "## Cross-Setting Stability Checks",
        "",
        "### MoE Role Contrasts",
        "",
        "Positive values match the expected role-aligned direction: source/branching prefer `g_out`, sink/converging prefer `g_in` over `g_out`.",
        "",
        markdown_table(contrast_df) if not contrast_df.empty else "_No contrast data._",
        "",
        f"Automated recommendation: {gate_rec}",
        "",
        "### Attention Relation Channels",
        "",
        "Use these results as evidence for directional relation-channel separation. If intended-head raw mass is exactly 1.0, the correct wording is explicit structural separation rather than learned head specialization.",
        "",
        markdown_table(combined_attention[["label", "relation", "relation_density", "raw_attention_mass", "attention_enrichment", "log2_enrichment"]] if not combined_attention.empty else pd.DataFrame()),
        "",
        "### Magnetic Phase",
        "",
        "The key diagnostic is whether `q=0.25` has non-zero sensitivity while `q=0` remains zero or much smaller.",
        "",
        markdown_table(combined_mag[["label", "q", "mean", "std", "median", "min", "max"]] if not combined_mag.empty else pd.DataFrame()),
        "",
        "## Per-Setting Details",
        "",
    ]

    for _, row in index.iterrows():
        label = row["label"]
        diag_dir = Path(row["diag_dir"])
        train_runs = combined_training_runs[combined_training_runs["label"] == label]
        gate = combined_gate[combined_gate["label"] == label] if not combined_gate.empty else pd.DataFrame()
        att = combined_attention[combined_attention["label"] == label] if not combined_attention.empty else pd.DataFrame()
        intended = combined_intended[combined_intended["label"] == label] if not combined_intended.empty else pd.DataFrame()
        mag = combined_mag[combined_mag["label"] == label] if not combined_mag.empty else pd.DataFrame()
        diff = combined_diff[combined_diff["label"] == label] if not combined_diff.empty else pd.DataFrame()

        lines.extend([
            f"### Setting: {label}",
            "",
            f"- Run dir: `{row['run_dir']}`",
            f"- Diagnostics dir: `{row['diag_dir']}`",
            f"- Full per-setting report: `{diag_dir / 'mechanism_analysis_report.md'}`",
            "",
            "#### Training Runs",
            "",
            markdown_table(train_runs[["run_id", "best_epoch", "tau", "mape", "err", "test_tau", "test_mape", "test_err", "train_time_sec"]]),
            "",
            "#### MoE Gate Routing By Node Role",
            "",
            markdown_table(compact_gate(gate)) if not gate.empty else "_No MoE gate data._",
            "",
            "#### Attention Relation Analysis",
            "",
            markdown_table(att[["relation", "relation_density", "raw_attention_mass", "attention_enrichment", "log2_enrichment"]] if not att.empty else pd.DataFrame()),
            "",
            "Intended head-channel view:",
            "",
            markdown_table(intended[["head", "intended_relation", "relation_density", "raw_attention_mass", "attention_enrichment", "log2_enrichment"]] if not intended.empty else pd.DataFrame()),
            "",
            "#### Magnetic Phase Diagnostic",
            "",
            markdown_table(mag[["q", "mean", "std", "median", "min", "max"]] if not mag.empty else pd.DataFrame()),
            "",
            "Paired difference:",
            "",
            markdown_table(diff[["paired_difference", "mean", "std", "median", "min", "max"]] if not diff.empty else pd.DataFrame()),
            "",
            "#### Figure Paths",
            "",
            f"- MoE: `{diag_dir / 'moe_gate_by_role_bar.pdf'}`",
            f"- Attention: `{diag_dir / 'attention_enrichment_layer_head_heatmap.pdf'}`",
            f"- Magnetic: `{diag_dir / 'magnetic_phase_sensitivity_violin.pdf'}`",
            "",
        ])

    lines.extend([
        "## Initial Paper-Use Guidance",
        "",
        "- MoE: use the cross-setting role contrasts above to decide main text versus appendix. Main text needs clear, stable role-direction alignment.",
        "- Attention: usable if worded as explicit directional relation-channel separation; do not claim unconstrained learned head specialization when intended-head mass is fixed by masks.",
        "- Magnetic: use NB101 as the stronger distributional diagnostic and treat NB201 as a fixed-topology sanity check unless the figure layout explicitly notes the degeneracy.",
        "",
        "## Paper-Ready Draft Analysis",
        "",
        "Across the second-budget settings, DTA-GT can be evaluated at three directed-semantics levels: MoE routing over node roles, structural attention over directional relation channels, and magnetic spectral sensitivity to edge reversal. The strongest claim should be assigned to the diagnostic whose cross-setting statistics show stable role- or relation-specific structure; attention results should be described as explicit directional channel separation because the effective masks are structurally imposed. The magnetic diagnostic is useful as a phase sanity check: non-zero `q=0.25` sensitivity with zero `q=0` sensitivity supports the direction-aware spectral encoding, with NB101 giving the more informative distribution.",
    ])

    (root / report_name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--title", default="DTA-GT Mechanism Diagnostics Handoff")
    parser.add_argument("--report-name", default="MECHANISM_ANALYSIS.md")
    args = parser.parse_args()

    write_outputs(Path(args.root), args.title, args.report_name)


if __name__ == "__main__":
    main()
