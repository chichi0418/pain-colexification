"""SATELLITE-INCLUSIVE CROSS-FAMILY CLUSTERING (secondary analysis).

Satellite-inclusive counterpart of script 04: correlates post-walk Strength
profiles from the satellite-inclusive centrality run (script 12) and clusters
the families. Here the four-cluster grouping (C1–C4) is the fixed grouping
obtained from this tree's dendrogram. Its similarity matrix also feeds the
satellite-inclusive geography control (script 14).

Outputs: output/satellite/cross_family/ (same file set as script 04).

Usage:
    python src/13_satellite_clustering.py    (requires script 12 to have run)
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

from config import CONCEPT_PATH, SATELLITE_DIR

# ── Paths ─────────────────────────────────────────────────────────────────────
CONCEPT_F   = Path(CONCEPT_PATH)
CENTRALITY_DIR = SATELLITE_DIR
OUT_DIR     = CENTRALITY_DIR / "cross_family"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# ── §4.5 per-cluster grouping ──────────────────────────────────────────────────
# Four clusters of the 11 families (Universal excluded) from average-linkage
# clustering on the family-only Spearman matrix.  See SUMMARY.md §4.5.
CLUSTERS = {
    "C1": ("C1 — Disease + grief",
           ["Nakh-Daghestanian", "Uralic", "Afro-Asiatic", "Turkic"]),
    "C2": ("C2 — Injury + grief",
           ["Austroasiatic", "Tai-Kadai", "Pano-Tacanan"]),
    "C3": ("C3 — Injury + heat",
           ["Austronesian", "Tupian"]),
    "C4": ("C4 — Disease + weakness",
           ["Indo-European", "Sino-Tibetan"]),
}
# PAIN/SICK are present in all 11 families, so they carry no discriminating signal.
CORE_CONCEPTS = {"PAIN", "SICK"}


def cluster_signatures(profile: pd.DataFrame) -> dict[str, list[str]]:
    """Per-cluster shared-concept signature: concepts non-zero across *all*
    members of a cluster, excluding the PAIN/SICK core.  Concept order follows
    the profile-matrix column order for stable, reproducible output."""
    sigs = {}
    for cid, (_, members) in CLUSTERS.items():
        present = [m for m in members if m in profile.index]
        shared = []
        for concept in profile.columns:
            if concept in CORE_CONCEPTS:
                continue
            if all(profile.loc[m, concept] > 0 for m in present):
                shared.append(concept)
        sigs[cid] = shared
    return sigs


# ── Data loading ──────────────────────────────────────────────────────────────

def load_pain_concepts(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    return df["Concept"].tolist()


def load_centrality_data(families: list[str], concepts: list[str]) -> pd.DataFrame:
    """Read per-family _centrality_postwalk.csv and return a long-format DataFrame."""
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
    ax.set_title("Cross-Family Spearman Similarity (Strength Profile) — CLICS4 Aligned-T2")
    plt.colorbar(im, ax=ax, label="Spearman ρ", shrink=0.8)
    plt.tight_layout()
    path = OUT_DIR / "similarity_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _plot_dendrogram(sim_families: pd.DataFrame,
                     sim_to_universal: pd.Series) -> None:
    """Families-only dendrogram + bar chart of ρ with Universal."""
    families = sim_families.index.tolist()
    n = len(families)

    dist_mat = np.clip(1 - sim_families.values, 0, 2)
    np.fill_diagonal(dist_mat, 0)
    condensed = squareform(dist_mat, checks=False)
    Z = linkage(condensed, method="average")

    fig, (ax_dend, ax_bar) = plt.subplots(
        1, 2, figsize=(max(12, n * 0.9), 6),
        gridspec_kw={"width_ratios": [3, 1], "wspace": 0.02},
    )
    # Colour the branches at the chosen k = len(CLUSTERS) so the figure agrees with
    # the cut used everywhere else.  Threshold sits between the k-th and (k-1)-th
    # largest merges, giving exactly k groups.
    k = len(CLUSTERS)
    merges = np.sort(Z[:, 2])
    color_thr = 0.5 * (merges[-k] + merges[-(k - 1)])
    dend = dendrogram(Z, labels=families, ax=ax_dend,
                      leaf_rotation=60, leaf_font_size=9, orientation="top",
                      color_threshold=color_thr, above_threshold_color="#9a9a9a")
    ax_dend.axhline(color_thr, color="grey", linestyle="--", linewidth=0.8)
    ax_dend.text(0.99, color_thr, f" k = {k} cut", transform=ax_dend.get_yaxis_transform(),
                 ha="right", va="bottom", fontsize=8, color="grey")
    ax_dend.set_ylabel("Distance (1 − Spearman ρ)")

    leaf_order = dend["ivl"]
    rho_vals = [sim_to_universal[f] for f in leaf_order]
    # Single neutral hue: bar length + value labels already carry ρ, and a
    # threshold colour scheme would clash with the dendrogram's cluster palette
    # (red would mean both "ρ ≥ 0.5" and a cluster).
    y_pos = range(n)
    ax_bar.barh(y_pos, rho_vals, color="#8c8c8c", edgecolor="grey", linewidth=0.5)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(leaf_order, fontsize=9)
    ax_bar.yaxis.tick_right()
    ax_bar.set_xlabel("Spearman ρ\nwith Universal")
    ax_bar.set_xlim(-0.1, 1.0)
    ax_bar.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    for i, v in enumerate(rho_vals):
        ax_bar.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    path = OUT_DIR / "dendrogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    pdf_path = OUT_DIR / "dendrogram.pdf"
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    print(f"Saved: {pdf_path}")
    return Z


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
    ax_dend.set_title("Hierarchical Clustering (incl. Universal) — CLICS4 Aligned-T2")

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

    row_min = profile.min(axis=1)
    row_max = profile.max(axis=1)
    denom = (row_max - row_min).replace(0, 1)
    normed = profile.sub(row_min, axis=0).div(denom, axis=0)

    fig, ax = plt.subplots(figsize=(max(14, len(concepts) * 0.55),
                                    max(8, len(families) * 0.5)))
    im = ax.imshow(normed.values, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(len(concepts)))
    ax.set_xticklabels(concepts, rotation=55, ha="right", fontsize=7)
    ax.set_yticks(range(len(families)))
    ax.set_yticklabels(families, fontsize=7)
    ax.set_title("Strength Centrality (row-normalised): Family × Pain Concept — CLICS4 Aligned-T2")
    plt.colorbar(im, ax=ax, label="Normalised Strength", shrink=0.8)
    plt.tight_layout()
    path = OUT_DIR / "heatmap_profile.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def _row_normalise(profile: pd.DataFrame) -> pd.DataFrame:
    """Row min-max scaling of Strength to [0, 1] (matches _plot_heatmap_profile)."""
    row_min = profile.min(axis=1)
    row_max = profile.max(axis=1)
    denom = (row_max - row_min).replace(0, 1)
    return profile.sub(row_min, axis=0).div(denom, axis=0)


def _plot_cluster_heatmap(profile: pd.DataFrame) -> None:
    """Family × Concept heatmap with rows grouped by C1–C4 and each cluster's
    shared-concept signature outlined.  Mirrors §4.5 of SUMMARY.md."""
    sigs = cluster_signatures(profile)

    # Row order: cluster members C1→C4, dropping Universal for a clean cluster view.
    row_order, row_clusters, band_spans = [], [], []
    for cid, (_, members) in CLUSTERS.items():
        present = [m for m in members if m in profile.index]
        start = len(row_order)
        row_order.extend(present)
        row_clusters.extend([cid] * len(present))
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
    ax.set_title("Per-cluster shared-concept signatures (C1–C4): "
                 "Family × Pain Concept — CLICS4 Aligned-T2", fontsize=10)
    plt.colorbar(im, ax=ax, label="Normalised Strength", shrink=0.7)

    # Cluster block dividers + left-margin labels.
    for cid, start, end in band_spans:
        if start > 0:
            ax.axhline(start - 0.5, color="black", linewidth=1.4)
        label = CLUSTERS[cid][0].replace(" — ", "\n")
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


def _plot_cluster_signatures(profile: pd.DataFrame) -> None:
    """2×2 small-multiples: one panel per cluster, horizontal bars of each
    signature concept's mean (row-normalised) strength across cluster members."""
    sigs = cluster_signatures(profile)
    normed = _row_normalise(profile)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (cid, (label, members)) in zip(axes.flat, CLUSTERS.items()):
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

    fig.suptitle("Per-cluster shared-concept signatures (C1–C4) — "
                 "CLICS4 Aligned-T2", fontsize=12)
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
    {k: silhouette} so main() can print the numbers.  Shares its shape with the
    PAIN-CC script; here primary_k = 4."""
    heights = Z[:, 2]
    gaps = np.diff(heights)
    k_at_gap = [n_families - i - 1 for i in range(len(gaps))]

    ks = list(range(2, min(7, n_families)))
    sils = {}
    for k in ks:
        labels = fcluster(Z, k, criterion="maxclust")
        sils[k] = float(silhouette_score(dist, labels, metric="precomputed"))

    thr = 0.7 * heights.max()
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
    pad = (hi - lo) * 0.18 or 0.01
    ax_sil.set_ylim(lo - pad, hi + pad)
    for k in ks:
        ax_sil.annotate(f"{sils[k]:+.3f}", xy=(k, sils[k]), xytext=(0, 7),
                        textcoords="offset points", ha="center", fontsize=8)
    ax_sil.axvline(primary_k, color="grey", linestyle="-", linewidth=1)
    ax_sil.text(0.02, 0.04, f"chosen k = {primary_k}",
                transform=ax_sil.transAxes, fontsize=8, va="bottom")

    fig.suptitle("Choice of k — CLICS4 Aligned-T2 (satellite-inclusive)", fontsize=12)
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
    return sils


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CLICS4 ALIGNED-T2: CROSS-FAMILY CENTRALITY COMPARISON")
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

    # Plots
    sim_families = sim.loc[families_only, families_only]
    sim_to_universal = sim.loc[families_only, "Universal"]

    _plot_similarity_matrix(sim)
    Z = _plot_dendrogram(sim_families, sim_to_universal)
    _plot_dendrogram_with_universal(sim)
    _plot_heatmap_profile(profile)

    # Same k-selection method as the PAIN-CC tree; here all three criteria land on k = 4.
    dist = np.clip(1 - sim_families.values, 0, 2)
    np.fill_diagonal(dist, 0)
    _plot_k_selection(Z, dist, len(families_only), primary_k=len(CLUSTERS))

    _plot_cluster_heatmap(profile)
    _plot_cluster_signatures(profile)

    print("\nDone.")


if __name__ == "__main__":
    main()
