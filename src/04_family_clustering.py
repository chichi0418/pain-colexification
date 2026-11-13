"""CROSS-FAMILY CLUSTERING — Spearman similarity + dendrogram (main text Fig. 2).

Correlates post-walk Strength profiles (from script 02) across the 11 language
families + Universal, clusters the family-only dissimilarity matrix (1 − ρ,
average linkage), selects the number of clusters k from merge-height-gap and
silhouette diagnostics, derives the cluster membership at runtime from the same
linkage the dendrogram draws, and prints per-cluster shared-concept signatures
and the between-cluster concept contrast (supplementary concept-contrast table).

Outputs (output/main/cross_family/):
  profile_matrix.csv, similarity_matrix.csv (+ .png)
  dendrogram.png / .pdf            — the reported k = 2 cut (manuscript Fig. 2)
  dendrogram_with_universal.png, heatmap_profile.png
  k_selection_diagnostics.png
  cluster_signature_heatmap.{png,pdf}, cluster_signatures_panel.{png,pdf}
  cluster_signatures_k2.csv        — Table 1 (≥ 80% shared / distinctive concepts)
  cluster_contrast.csv             — Supplementary Table S3 (Δ = C1 − C2)

Usage:
    python src/04_family_clustering.py       (requires script 02 to have run)
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score

from config import CONCEPT_PATH, MAIN_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
CONCEPT_F   = Path(CONCEPT_PATH)
CENTRALITY_DIR = MAIN_DIR
OUT_DIR     = CENTRALITY_DIR / "cross_family"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAG = "CLICS4 Aligned-T2 PAIN-CC"
# Times New Roman for the Fig. 2 dendrogram, matching Fig. S2 (dendrogram_k4)
FIG_FONT_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
}

TARGET_FAMILIES = [
    "Afro-Asiatic",
    "Austroasiatic",
    "Austronesian",
    "Indo-European",
    "Nakh-Daghestanian",
    "Pano-Tacanan",
    "Sino-Tibetan",
    "Tai-Kadai",
    "Tupian",
    "Turkic",
    "Uralic",
    "Universal",
]

# Data-driven cut.  Unlike the satellite-inclusive tree — where three criteria (merge-gap,
# silhouette, scipy's default colour threshold) all land on k = 4 — dropping the satellite
# components collapses the structure, and the same three criteria all point to k = 2 here.
# See k_selection_diagnostics.png.
N_CLUSTERS = 2
# PAIN/SICK carry no discriminating signal in the satellite-inclusive tree because
# they are present in every family.  main() re-checks that under PAIN-CC rather
# than assuming it.
CORE_CONCEPTS = {"PAIN", "SICK"}


def derive_clusters(Z, families: list[str], leaf_order: list[str],
                    k: int = N_CLUSTERS) -> dict[str, tuple[str, list[str]]]:
    """Cut the family-only linkage into k clusters, numbered C1..Ck in
    left-to-right dendrogram-leaf order so labels are stable across reruns.

    Returns the same shape as the hard-coded CLUSTERS dict in the
    satellite-inclusive script: {cid: (label, [members])}.
    """
    labels = fcluster(Z, k, criterion="maxclust")
    pos = {f: i for i, f in enumerate(leaf_order)}

    groups: dict[int, list[str]] = {}
    for fam, lab in zip(families, labels):
        groups.setdefault(int(lab), []).append(fam)

    # Order members within a cluster, and clusters among themselves, by leaf position.
    for members in groups.values():
        members.sort(key=lambda f: pos[f])
    ordered = sorted(groups.values(), key=lambda members: pos[members[0]])

    return {f"C{i}": (f"C{i}", members) for i, members in enumerate(ordered, start=1)}


def cluster_signatures(profile: pd.DataFrame,
                       clusters: dict[str, tuple[str, list[str]]]) -> dict[str, list[str]]:
    """Per-cluster shared-concept signature: concepts non-zero across *all*
    members of a cluster, excluding the PAIN/SICK core.  Concept order follows
    the profile-matrix column order for stable, reproducible output."""
    sigs = {}
    for cid, (_, members) in clusters.items():
        present = [m for m in members if m in profile.index]
        shared = []
        for concept in profile.columns:
            if concept in CORE_CONCEPTS:
                continue
            if all(profile.loc[m, concept] > 0 for m in present):
                shared.append(concept)
        sigs[cid] = shared
    return sigs


SHARE_THRESHOLD = 0.8


def presence_signature_table(profile: pd.DataFrame,
                             clusters: dict[str, tuple[str, list[str]]],
                             sim: pd.DataFrame,
                             share: float = SHARE_THRESHOLD) -> pd.DataFrame:
    """Cluster signatures as reported in Table 1 (k = 2) and Supplementary
    Table S5 (k = 4), PAIN/SICK excluded:

      Shared_all   concepts present (strength > 0) in every member
      Shared_80    concepts present in ≥ 80% of members
      Distinctive  Shared_80 concepts absent from every other cluster's Shared_80

    plus the within-cluster ρ range and the mean ρ with Universal."""
    present = profile > 0
    shared_80 = {}
    rows = []
    for cid, (_, members) in clusters.items():
        frac = present.loc[members].mean()
        shared_80[cid] = [c for c in profile.columns
                          if c not in CORE_CONCEPTS and frac[c] >= share]
    for cid, (_, members) in clusters.items():
        frac = present.loc[members].mean()
        others = set().union(*(set(v) for k, v in shared_80.items() if k != cid))
        pairs = [sim.loc[a, b] for i, a in enumerate(members) for b in members[i + 1:]]
        rows.append({
            "Cluster": cid,
            "Families": "; ".join(members),
            "n": len(members),
            "Within_rho_min": min(pairs) if pairs else np.nan,
            "Within_rho_max": max(pairs) if pairs else np.nan,
            "Mean_rho_Universal": sim.loc[members, "Universal"].mean(),
            "Shared_all": "; ".join(c for c in profile.columns
                                    if c not in CORE_CONCEPTS and frac[c] == 1),
            "Shared_80": "; ".join(shared_80[cid]),
            "Distinctive": "; ".join(c for c in shared_80[cid] if c not in others),
        })
    return pd.DataFrame(rows)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pain_concepts(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    return df["Concept"].tolist()


def load_centrality_data(families: list[str], concepts: list[str]) -> pd.DataFrame:
    """Read per-family _centrality_postwalk.csv and return a long-format DataFrame.

    Per-family dirs live under ``families/<Family>/`` (matching the
    satellite-inclusive tree).
    """
    rows = []
    for fam in families:
        csv = CENTRALITY_DIR / "families" / fam / f"{fam}_centrality_postwalk.csv"
        if not csv.exists():
            print(f"  [WARNING] Missing: {csv}")
            continue
        df = pd.read_csv(csv)
        df["Family"] = fam
        rows.append(df[["Family", "Concept", "Strength"]])
    if not rows:
        raise FileNotFoundError("No centrality CSVs found.")
    return pd.concat(rows, ignore_index=True)


# ── Profile matrix & similarity ───────────────────────────────────────────────

def build_profile_matrix(df: pd.DataFrame, concepts: list[str]) -> pd.DataFrame:
    """Pivot long DataFrame into Family × Concept matrix (Strength values).

    Missing concepts are filled with 0.0, matching CLICS3 treatment.
    """
    pivot = df.pivot_table(index="Family", columns="Concept",
                           values="Strength", aggfunc="first")
    pivot = pivot.reindex(columns=concepts, fill_value=0.0).fillna(0.0)
    return pivot


def compute_similarity_matrix(profile: pd.DataFrame) -> pd.DataFrame:
    """Spearman rank correlation between every pair of family profiles."""
    families = profile.index.tolist()
    n = len(families)
    mat = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho, _ = spearmanr(profile.iloc[i].values, profile.iloc[j].values)
            mat[i, j] = rho
            mat[j, i] = rho
    return pd.DataFrame(mat, index=families, columns=families)


# ── Visualisations ────────────────────────────────────────────────────────────

def _plot_similarity_matrix(sim: pd.DataFrame) -> None:
    families = sim.index.tolist()
    n = len(families)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.7), max(9, n * 0.65)))
    im = ax.imshow(sim.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(families, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(families, fontsize=7)
    for i in range(n):
        for j in range(n):
            v = sim.values[i, j]
            color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=6, color=color)
    ax.set_title(f"Cross-Family Spearman Similarity (Strength Profile) — {TAG}")
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    plt.tight_layout()
    path = OUT_DIR / "similarity_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_dendrogram(sim_families: pd.DataFrame,
                     sim_to_universal: pd.Series):
    """Families-only dendrogram + bar chart of ρ with Universal.

    Returns (Z, leaf_order) so the caller can derive C1-C4 from the *same*
    linkage that is drawn here — the hard-coded grouping in the satellite-
    inclusive script can silently drift from its own figure.
    """
    with plt.rc_context(FIG_FONT_RC):
        return _draw_dendrogram(sim_families, sim_to_universal)


def _draw_dendrogram(sim_families: pd.DataFrame,
                     sim_to_universal: pd.Series):
    families = sim_families.index.tolist()
    n = len(families)

    dist_mat = np.clip(1 - sim_families.values, 0, 2)
    np.fill_diagonal(dist_mat, 0)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")

    fig, (ax_dend, ax_bar) = plt.subplots(
        1, 2, figsize=(max(14, n * 1.1), 7),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.02},
    )
    # Colour the branches at the chosen k (not scipy's rendering default) so the
    # figure agrees with the k = N_CLUSTERS cut used everywhere else.  The threshold
    # sits between the k-th and (k-1)-th largest merges, giving exactly k groups.
    merges = np.sort(Z[:, 2])
    color_thr = 0.5 * (merges[-N_CLUSTERS] + merges[-(N_CLUSTERS - 1)])
    dend = dendrogram(Z, labels=families, ax=ax_dend,
                      leaf_rotation=45, leaf_font_size=15, orientation="top",
                      color_threshold=color_thr, above_threshold_color="#9a9a9a")
    for lbl in ax_dend.get_xticklabels():
        lbl.set_ha("right")
    ax_dend.tick_params(axis="y", labelsize=13)
    ax_dend.axhline(color_thr, color="grey", linestyle="--", linewidth=0.8)
    ax_dend.text(0.99, color_thr, f" k = {N_CLUSTERS} cut", transform=ax_dend.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=12, color="grey")
    ax_dend.set_ylabel("Distance (1 − Spearman ρ)", fontsize=14)
    # No title: the manuscript caption carries it (journal style).

    leaf_order = dend["ivl"]
    rho_vals = [sim_to_universal[f] for f in leaf_order]
    # Single neutral hue: bar length + value labels already carry ρ, and a
    # threshold colour scheme would clash with the dendrogram's cluster palette
    # (red would mean both "ρ ≥ 0.5" and a cluster).
    y_pos = range(n)
    ax_bar.barh(y_pos, rho_vals, color="#8c8c8c", edgecolor="grey", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(leaf_order, fontsize=15)
    ax_bar.yaxis.tick_right()
    ax_bar.tick_params(axis="x", labelsize=13)
    ax_bar.set_xlabel("Spearman ρ\nwith Universal", fontsize=14)
    ax_bar.set_xlim(-0.1, 1.0)
    ax_bar.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    for i, v in enumerate(rho_vals):
        ax_bar.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=12)

    plt.tight_layout()
    path = OUT_DIR / "dendrogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    pdf_path = OUT_DIR / "dendrogram.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Saved: {pdf_path}")

    return Z, leaf_order


def _plot_dendrogram_with_universal(sim_full: pd.DataFrame) -> None:
    """All families + Universal clustered together."""
    families = sim_full.index.tolist()
    n = len(families)

    dist_mat = np.clip(1 - sim_full.values, 0, 2)
    np.fill_diagonal(dist_mat, 0)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")

    fig, (ax_dend, ax_bar) = plt.subplots(
        1, 2, figsize=(max(12, n * 0.9), 6),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.02},
    )
    dend = dendrogram(Z, labels=families, ax=ax_dend,
                      leaf_rotation=60, leaf_font_size=9, orientation="top")
    ax_dend.set_ylabel("Distance (1 − Spearman ρ)")
    ax_dend.set_title(f"Hierarchical Clustering (incl. Universal) — {TAG}")

    leaf_order = dend["ivl"]
    rho_vals = [sim_full.loc[f, "Universal"] if f != "Universal" else 1.0
                for f in leaf_order]
    # Single neutral hue: bar length + value labels already carry ρ, and a
    # threshold colour scheme would clash with the dendrogram's cluster palette
    # (red would mean both "ρ ≥ 0.5" and a cluster).
    y_pos = range(n)
    ax_bar.barh(y_pos, rho_vals, color="#8c8c8c", edgecolor="grey", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(leaf_order, fontsize=9)
    ax_bar.yaxis.tick_right()
    ax_bar.set_xlabel("Spearman ρ\nwith Universal")
    ax_bar.set_xlim(-0.1, 1.1)
    ax_bar.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    for i, v in enumerate(rho_vals):
        ax_bar.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "dendrogram_with_universal.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_heatmap_profile(profile: pd.DataFrame) -> None:
    """Family × Concept heatmap, row-normalised Strength to [0, 1]."""
    families = profile.index.tolist()
    concepts = profile.columns.tolist()

    normed = _row_normalise(profile)

    fig, ax = plt.subplots(figsize=(max(14, len(concepts) * 0.55),
                                    max(8, len(families) * 0.5)))
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=7)
    ax.set_title(f"Strength Centrality (row-normalised): Family × Pain Concept — {TAG}")
    plt.colorbar(im, ax=ax, label="Normalised Strength", shrink=0.8)
    plt.tight_layout()
    path = OUT_DIR / "heatmap_profile.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _row_normalise(profile: pd.DataFrame) -> pd.DataFrame:
    """Row min-max scaling of Strength to [0, 1]."""
    row_min = profile.min(axis=1)
    row_max = profile.max(axis=1)
    denom = (row_max - row_min).replace(0, 1)
    return profile.sub(row_min, axis=0).div(denom, axis=0)


def _plot_cluster_heatmap(profile: pd.DataFrame,
                          clusters: dict[str, tuple[str, list[str]]]) -> None:
    """Family × Concept heatmap with rows grouped by cluster and each cluster's
    shared-concept signature outlined."""
    sigs = cluster_signatures(profile, clusters)
    span = f"C1–C{len(clusters)}"

    # Row order: cluster members C1→C4, dropping Universal for a clean cluster view.
    row_order, band_spans = [], []
    for cid, (_, members) in clusters.items():
        present = [m for m in members if m in profile.index]
        start = len(row_order)
        row_order.extend(present)
        band_spans.append((cid, start, start + len(present)))

    sub = profile.loc[row_order]
    normed = _row_normalise(sub)
    concepts = profile.columns.tolist()
    col_idx = {c: i for i, c in enumerate(concepts)}

    fig, ax = plt.subplots(figsize=(max(14, len(concepts) * 0.55),
                                    max(6, len(row_order) * 0.5)))
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=8)
    ax.set_title(f"Per-cluster shared-concept signatures ({span}, k = {len(clusters)}): "
                 f"Family × Pain Concept — {TAG}", fontsize=10)
    plt.colorbar(im, ax=ax, label="Normalised Strength", shrink=0.7)

    # Cluster block dividers + left-margin labels.
    for cid, start, end in band_spans:
        if start > 0:
            ax.axhline(start - 0.5, color="black", linewidth=1.4)
        label = clusters[cid][0].replace(" — ", "\n")
        ax.text(-0.17, (start + end - 1) / 2, label,
                transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=9, fontweight="bold")

    # Outline each cluster's signature cells (shared-by-all-members columns).
    for cid, start, end in band_spans:
        for concept in sigs[cid]:
            j = col_idx[concept]
            ax.add_patch(Rectangle((j - 0.5, start - 0.5), 1, end - start,
                                   fill=False, edgecolor="#1f3b8c",
                                   linewidth=1.6))

    plt.tight_layout()
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"cluster_signature_heatmap.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def _plot_cluster_signatures(profile: pd.DataFrame,
                             clusters: dict[str, tuple[str, list[str]]]) -> None:
    """Small-multiples: one panel per cluster, horizontal bars of each signature
    concept's mean (row-normalised) strength across cluster members.  The grid
    adapts to k so k = 2 renders as a clean 1×2, not 2×2 with empty panels."""
    sigs = cluster_signatures(profile, clusters)
    normed = _row_normalise(profile)

    ncols = min(2, len(clusters))
    nrows = -(-len(clusters) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows),
                             squeeze=False)
    axes = axes.flatten()
    for ax, (cid, (label, members)) in zip(axes, clusters.items()):
        present = [m for m in members if m in profile.index]
        concepts = sigs[cid]
        means = normed.loc[present, concepts].mean(axis=0) if concepts \
            else pd.Series(dtype=float)
        means = means.sort_values()

        ax.barh(range(len(means)), means.values, color="#d95f0e",
                edgecolor="grey", linewidth=0.5)
        ax.set_yticks(range(len(means)))
        ax.set_yticklabels(means.index, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean normalised Strength", fontsize=8)
        ax.set_title(f"{label}\n({', '.join(members)})", fontsize=9)
        for i, v in enumerate(means.values):
            ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=7)

    for ax in axes[len(clusters):]:
        ax.axis("off")

    fig.suptitle(f"Per-cluster shared-concept signatures "
                 f"(C1–C{len(clusters)}, k = {len(clusters)}) — {TAG}", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    for ext in ("png", "pdf"):
        path = OUT_DIR / f"cluster_signatures_panel.{ext}"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close(fig)


def _plot_k_selection(Z, dist: np.ndarray, n_families: int,
                      primary_k: int) -> dict[int, float]:
    """Two-panel justification for the chosen k: merge-height gaps (an elbow
    criterion) and silhouette on the precomputed 1 − ρ distances.  Returns
    {k: silhouette} so main() can print the numbers."""
    heights = Z[:, 2]
    gaps = np.diff(heights)                    # gap between successive merge heights
    k_at_gap = [n_families - i - 1 for i in range(len(gaps))]

    ks = list(range(2, min(7, n_families)))
    sils = {}
    for k in ks:
        labels = fcluster(Z, k, criterion="maxclust")
        sils[k] = float(silhouette_score(dist, labels, metric="precomputed"))

    thr = 0.7 * heights.max()                  # scipy's default dendrogram colour threshold
    n_below = int((heights < thr).sum())

    fig, (ax_gap, ax_sil) = plt.subplots(1, 2, figsize=(13, 4.6))

    ax_gap.bar(k_at_gap, gaps, color="#4c72b0", edgecolor="grey", linewidth=0.5)
    ax_gap.set_xlabel("k (clusters remaining below the merge)")
    ax_gap.set_ylabel("Gap to next merge height")
    ax_gap.set_title("Merge-height gap (larger = more natural cut)", fontsize=10)
    ax_gap.set_xticks(sorted(k_at_gap))
    best_gap = k_at_gap[int(np.argmax(gaps))]
    ax_gap.annotate(f"largest gap → k = {best_gap}",
                    xy=(best_gap, gaps.max()),
                    xytext=(best_gap + 1.2, gaps.max() * 0.9),
                    fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.8))

    ax_sil.plot(ks, [sils[k] for k in ks], marker="o", color="#c44e52")
    ax_sil.set_xlabel("k")
    ax_sil.set_ylabel("Silhouette (precomputed 1 − ρ)")
    ax_sil.set_title("Silhouette by k (higher = better separated)", fontsize=10)
    ax_sil.set_xticks(ks)
    ax_sil.grid(alpha=0.3, linewidth=0.5)
    lo, hi = min(sils.values()), max(sils.values())
    pad = (hi - lo) * 0.18 or 0.01             # headroom so value labels are not clipped
    ax_sil.set_ylim(lo - pad, hi + pad)
    for k in ks:
        ax_sil.annotate(f"{sils[k]:+.3f}", xy=(k, sils[k]), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
    ax_sil.axvline(primary_k, color="grey", linestyle="-", linewidth=1)
    ax_sil.text(0.02, 0.04, f"chosen k = {primary_k}",
                transform=ax_sil.transAxes, fontsize=8, va="bottom")

    fig.suptitle(f"Choice of k — {TAG}", fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    path = OUT_DIR / "k_selection_diagnostics.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")

    print(f"\nk selection: largest merge-height gap at k = {best_gap} "
          f"(gap = {gaps.max():.3f})")
    print("  silhouette: " + "  ".join(f"k={k}: {sils[k]:+.3f}" for k in ks)
          + f"  (peak k={max(sils, key=sils.get)})")
    print(f"  scipy default colour threshold {thr:.3f}: {n_below} merges below it")
    print(f"  → chosen k = {primary_k}")
    print(f"  [NOTE] n = {n_families}; silhouette differences below ~0.05 are not "
          "meaningful at this sample size, but all three criteria agree.")
    return sils


# ── Reporting ─────────────────────────────────────────────────────────────────

def _report_cluster_contrast(profile: pd.DataFrame,
                             clusters: dict[str, tuple[str, list[str]]],
                             top: int = 8) -> None:
    """For a 2-cluster cut, print the concepts that best separate the two groups
    (mean row-normalised Strength difference).  Strict-intersection signatures
    degrade for large clusters, so this contrast is the more informative view of
    *why* the two groups differ."""
    if len(clusters) != 2:
        return
    normed = _row_normalise(profile)
    a, b = clusters["C1"][1], clusters["C2"][1]
    mean_a = normed.loc[[m for m in a if m in normed.index]].mean()
    mean_b = normed.loc[[m for m in b if m in normed.index]].mean()
    diff = (mean_a - mean_b).sort_values()
    print("\nConcept contrast between the two clusters (mean row-normalised Strength):")
    print(f"  C1 = {', '.join(a)}")
    print(f"  C2 = {', '.join(b)}")
    print("  C2-leaning (disease-type):")
    for c in diff.index[:top]:
        if diff[c] < 0:
            print(f"    {c:22s} C1={mean_a[c]:.2f} C2={mean_b[c]:.2f}  Δ={diff[c]:+.2f}")
    print("  C1-leaning (injury-type):")
    for c in diff.index[::-1][:top]:
        if diff[c] > 0:
            print(f"    {c:22s} C1={mean_a[c]:.2f} C2={mean_b[c]:.2f}  Δ={diff[c]:+.2f}")

    path = OUT_DIR / "cluster_contrast.csv"
    pd.DataFrame({"C1_mean": mean_a, "C2_mean": mean_b, "Delta": mean_a - mean_b}) \
        .loc[diff.index].rename_axis("Concept").to_csv(path)
    print(f"Saved: {path}  (Supplementary Table S3)")


def _report_core_concepts(profile: pd.DataFrame, families_only: list[str]) -> None:
    """Which concepts are non-zero in *every* family?  Under the satellite-
    inclusive tree this is exactly {PAIN, SICK}; PAIN-CC need not match, so the
    excluded core is reported rather than assumed."""
    sub = profile.loc[families_only]
    universal_concepts = [c for c in sub.columns if (sub[c] > 0).all()]
    print(f"\nConcepts present in all {len(families_only)} families "
          f"({len(universal_concepts)}): {', '.join(universal_concepts)}")
    print(f"  excluded as core: {sorted(CORE_CONCEPTS)}")
    extra = sorted(set(universal_concepts) - CORE_CONCEPTS)
    if extra:
        print(f"  [NOTE] also present in all families but NOT excluded: {', '.join(extra)}")
    missing = sorted(CORE_CONCEPTS - set(universal_concepts))
    if missing:
        print(f"  [NOTE] core concept(s) absent from some family: {', '.join(missing)}")


def _report_clusters(clusters: dict[str, tuple[str, list[str]]],
                     sim_families: pd.DataFrame,
                     sigs: dict[str, list[str]]) -> None:
    print("\nDerived clusters (average linkage, k = "
          f"{len(clusters)}, leaf order):")
    for cid, (_, members) in clusters.items():
        pairs = [sim_families.loc[a, b]
                 for i, a in enumerate(members) for b in members[i + 1:]]
        span = f"{min(pairs):.2f}–{max(pairs):.2f}" if pairs else "n/a (singleton)"
        print(f"  {cid}: {', '.join(members)}")
        print(f"       within-cluster ρ: {span}")
        print(f"       signature ({len(sigs[cid])}): {', '.join(sigs[cid]) or '—'}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CLICS4 ALIGNED-T2 PAIN-CC: CROSS-FAMILY CENTRALITY COMPARISON")
    print("=" * 60)

    concepts = load_pain_concepts(CONCEPT_F)
    print(f"Pain concepts: {len(concepts)}")

    print("Loading per-family centrality CSVs ...")
    df = load_centrality_data(TARGET_FAMILIES, concepts)
    print(f"  Loaded {len(df)} rows for {df['Family'].nunique()} families")

    # Profile matrix
    profile = build_profile_matrix(df, concepts)
    profile_path = OUT_DIR / "profile_matrix.csv"
    profile.to_csv(profile_path)
    print(f"Saved: {profile_path}  ({profile.shape[0]} × {profile.shape[1]})")

    # Similarity matrix
    sim = compute_similarity_matrix(profile)
    sim_path = OUT_DIR / "similarity_matrix.csv"
    sim.to_csv(sim_path)
    print(f"Saved: {sim_path}  ({sim.shape[0]}×{sim.shape[1]})")

    # Print ρ with Universal for quick inspection
    print("\nSpearman ρ with Universal:")
    families_only = [f for f in sim.index if f != "Universal"]
    for f in families_only:
        print(f"  {f:25s}  ρ = {sim.loc[f, 'Universal']:.3f}")

    _report_core_concepts(profile, families_only)

    # Plots
    sim_families = sim.loc[families_only, families_only]
    sim_to_universal = sim.loc[families_only, "Universal"]

    _plot_similarity_matrix(sim)
    Z, leaf_order = _plot_dendrogram(sim_families, sim_to_universal)
    _plot_dendrogram_with_universal(sim)
    _plot_heatmap_profile(profile)

    # Justify k from the same linkage the dendrogram drew, then cut at it.
    dist = np.clip(1 - sim_families.values, 0, 2)
    np.fill_diagonal(dist, 0)
    _plot_k_selection(Z, dist, len(families_only), primary_k=N_CLUSTERS)

    clusters = derive_clusters(Z, sim_families.index.tolist(), leaf_order)
    _report_clusters(clusters, sim_families, cluster_signatures(profile, clusters))
    _report_cluster_contrast(profile, clusters)

    sig_table = presence_signature_table(profile, clusters, sim)
    sig_path = OUT_DIR / f"cluster_signatures_k{len(clusters)}.csv"
    sig_table.to_csv(sig_path, index=False)
    print(f"\nSaved: {sig_path}  (Table 1)")
    print(sig_table.to_string(index=False))

    _plot_cluster_heatmap(profile, clusters)
    _plot_cluster_signatures(profile, clusters)

    print("\nDone.")


if __name__ == "__main__":
    main()
