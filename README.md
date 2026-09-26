# DTA-GT

> 🎉 **Accepted at NeurIPS 2026!** 🏆

This is the official PyTorch implementation of the paper **"DTA-GT: Direction- and Topology-Aware Graph Transformer for Neural Network Representation Learning"** (NeurIPS 2026).

## Abstract

Architecture attribute predictors reduce the cost of neural architecture search by estimating accuracy or latency from a small set of evaluated candidates. The key difficulty is that neural architectures are operation-labeled DAGs: edge direction defines computational flow, but existing predictors often treat direction as a local structural cue. We propose DTA-GT, a Direction- and Topology-Aware Graph Transformer based on **encoder-wide directional consistency**, which preserves directed computational semantics across node initialization, global spectral encoding, pairwise interaction, and feature update. DTA-GT realizes this principle with direction- and topology-aware node initialization, Magnetic Laplacian spectral encoding, direction-dual structural attention, and a Direction-Sensitive MoE-FFN. Across accuracy and latency prediction on NAS-Bench-101 and NAS-Bench-201, DTA-GT consistently outperforms representative sequence-, GNN-, Transformer-, and hybrid predictors under limited-budget settings. Ablations and Magnetic spectral controls show that the improvements come from coordinated direction-aware stages and edge-orientation-sensitive spectral information, while mechanistic diagnostics confirm that the modules exhibit the intended directional behavior. These results support encoder-wide directional consistency as an effective design principle for architecture DAG representation.
