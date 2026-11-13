#!/usr/bin/env python3
"""ILLUSTRATIVE k = 4 DENDROGRAM (supplementary alternative-clustering figure).

Companion to script 04, which draws the *chosen* k = 2 grouping
(`cross_family/dendrogram.pdf`). This script draws the SAME linkage but cut at
k = 4 — the grouping the scipy default colour threshold (0.7 × max merge
height) renders — so the dissenting k-selection diagnostic can be seen
directly. It does NOT change the analysis; k = 2 remains the reported cut.

Outputs (output/main/cross_family/):
  dendrogram_k4.{pdf,png}   — Supplementary Figure S2
  cluster_signatures_k4.csv — Supplementary Table S5 (shared / distinctive
                              concepts, within-cluster ρ, mean ρ with Universal)

Usage:
    python src/05_dendrogram_k4.py       (requires script 04 to have run)
"""
import csv
import importlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform

from config import MAIN_DIR

CF = MAIN_DIR / "cross_family"
SIM_CSV = CF / "similarity_matrix.csv"
K = 4
# matplotlib tab10, one colour per k=4 cluster A1..A4: A1/A2 (the k=2 injury side,
# orange in Fig. 2) get orange/red, A3/A4 (the disease+grief side, green) get green/blue
PALETTE = ["#ff7f0e", "#d62728", "#2ca02c", "#1f77b4"]
FONT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
}


def load_families_matrix():
    rows = list(csv.reader(open(SIM_CSV)))
    header = rows[0][1:]
    sim = {}
    for r in rows[1:]:
        a = r[0]
        for j, v in enumerate(r[1:]):
            sim[(a, header[j])] = float(v)
    fams = [f for f in header if f != "Universal"]
    n = len(fams)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = 1.0 - sim[(fams[i], fams[j])]
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0  # enforce exact symmetry for squareform
    return fams, D


def leaf_ids(node, Z, n):
    if node < n:
        return [node]
    a, b = int(Z[node - n, 0]), int(Z[node - n, 1])
    return leaf_ids(a, Z, n) + leaf_ids(b, Z, n)


def main():
    fams, D = load_families_matrix()
    n = len(fams)
    Z = linkage(squareform(D, checks=False), method="average")

    heights = np.sort(Z[:, 2])
    scipy_default = 0.7 * heights[-1]                       # 0.494 -> the k=4 diagnostic
    k2_cut = 0.5 * (heights[-1] + heights[-2])              # 0.642 -> the reported k=2 cut

    clab = fcluster(Z, K, criterion="maxclust")            # cluster id per family (1..K)

    # link colour: a merge gets its cluster's colour iff all members share a k=4
    # cluster, else grey (above the k=4 cut)
    cluster_colour = {c: PALETTE[(c - 1) % len(PALETTE)] for c in sorted(set(clab))}
    link_colour = {}
    for i in range(Z.shape[0]):
        node = n + i
        cls = {clab[m] for m in leaf_ids(node, Z, n)}
        link_colour[node] = cluster_colour[next(iter(cls))] if len(cls) == 1 else "#9a9a9a"

    with plt.rc_context(FONT_RC):
        fig, ax = plt.subplots(figsize=(12, 7))
        dendrogram(
            Z, labels=fams, ax=ax,
            link_color_func=lambda x: link_colour[x],
            above_threshold_color="#9a9a9a",
        )

        # colour each leaf tick label by its k=4 cluster (so the singleton is visible too)
        fam_to_cluster = dict(zip(fams, clab))
        for lbl in ax.get_xticklabels():
            lbl.set_color(cluster_colour[fam_to_cluster[lbl.get_text()]])
            lbl.set_rotation(45)
            lbl.set_ha("right")
            lbl.set_fontsize(15)
        ax.tick_params(axis="y", labelsize=13)

        ax.axhline(scipy_default, color="black", linestyle="--", linewidth=1.1)
        ax.text(0.995, scipy_default, f" 0.7×max = {scipy_default:.3f}  → k = 4",
                transform=ax.get_yaxis_transform(), va="bottom", ha="right",
                color="black", fontsize=13)
        ax.axhline(k2_cut, color="grey", linestyle=":", linewidth=0.9)
        ax.text(0.995, k2_cut, f" k = 2 cut = {k2_cut:.3f} (reported)",
                transform=ax.get_yaxis_transform(), va="bottom", ha="right",
                color="grey", fontsize=12)

        ax.set_ylabel("Distance (1 − Spearman ρ)", fontsize=14)
        # No title: the manuscript caption carries it (journal style).

        # group legend, A1..A4 in fcluster order
        groups = {}
        for f, c in fam_to_cluster.items():
            groups.setdefault(c, []).append(f)
        lines = []
        for c in sorted(groups):
            mem = ", ".join(sorted(groups[c]))
            lines.append(f"A{c} (n={len(groups[c])}): {mem}")
        handles = [Line2D([], [], color=cluster_colour[c], linewidth=3) for c in sorted(groups)]
        ax.legend(handles, lines, loc="upper left", fontsize=15, frameon=True,
                  edgecolor="#cccccc", handlelength=1.5)

        ax.set_ylim(0, heights[-1] * 1.42)  # headroom so the enlarged legend clears the tree
        fig.tight_layout()
        for ext in ("pdf", "png"):
            out = CF / f"dendrogram_k4.{ext}"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved {out}")

    print(f"\nheights (asc): {[round(float(h),3) for h in heights]}")
    print(f"scipy default 0.7×max = {scipy_default:.3f}  |  k=2 cut = {k2_cut:.3f}")
    print("k=4 groups:")
    for line in lines:
        print("  " + line)

    # Supplementary Table S5: signatures and ρ summaries per k = 4 cluster
    fc = importlib.import_module("04_family_clustering")
    profile = pd.read_csv(CF / "profile_matrix.csv", index_col=0)
    sim = pd.read_csv(SIM_CSV, index_col=0)
    clusters = {f"A{c}": (f"A{c}", sorted(groups[c])) for c in sorted(groups)}
    table = fc.presence_signature_table(profile, clusters, sim)
    out = CF / "cluster_signatures_k4.csv"
    table.to_csv(out, index=False)
    print(f"\nSaved {out}  (Supplementary Table S5)")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
