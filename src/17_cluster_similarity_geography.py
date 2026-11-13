"""CLUSTER SIMILARITY vs GEOGRAPHY — within- vs between-cluster ρ ranges.

Reports the range of cross-family Spearman similarity (ρ) within and between the
k = 2 clusters of script 04, and tests whether the two clusters are separated
geographically.

Two deliberately different standards of evidence apply here:

  * The within/between **ρ** comparison is reported as pure description —
    ranges, medians, IQR, and the overlap interval — with *no* significance
    test.  C1/C2 were derived by average-linkage clustering of this very ρ
    matrix, and linkage maximises within-cluster similarity by construction, so
    any test of "within ρ > between ρ" is circular and would measure the
    clustering algorithm rather than the data.

  * The **geographic** comparison is inferential, because the clusters were not
    derived from geography.  Each cluster is tested separately against every
    same-size subset of the 11 families (C(11, 5) = C(11, 6) = 462 subsets), so
    the p-value is exact rather than sampled.  Separately rather than pooled:
    C1 and C2 turn out to have *opposite* geographic character, and a pooled
    within-vs-between median is set by whichever cluster contributes more pairs,
    so it represents neither and hides both signals.

Outputs (output/main/cross_family/cluster_similarity/):
  cluster_pairs.csv, family_silhouette.csv, cluster_similarity_summary.json,
  cluster_similarity_report.md, cluster_similarity_geography.{png,pdf}

Usage:
    python src/17_cluster_similarity_geography.py
      (requires scripts 04, 06, and — for the satellite appendix — 13)
"""

from __future__ import annotations

import importlib
import itertools
import json
import os
import tempfile
from pathlib import Path
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "pain_cluster_similarity_cache"
_mpl_cache = _cache_root / "mpl"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

from config import MAIN_DIR, PUBLIC_ROOT, SATELLITE_DIR

ROOT = PUBLIC_ROOT
CROSS_FAMILY_DIR = MAIN_DIR / "cross_family"
SIMILARITY_CSV = CROSS_FAMILY_DIR / "similarity_matrix.csv"
GEO_PAIRS_CSV = CROSS_FAMILY_DIR / "geography_control" / "geography_control_pairs.csv"
SAT_SIMILARITY_CSV = SATELLITE_DIR / "cross_family" / "similarity_matrix.csv"

OUT_DIR = CROSS_FAMILY_DIR / "cluster_similarity"
OUT_PAIRS_CSV = OUT_DIR / "cluster_pairs.csv"
OUT_SILHOUETTE_CSV = OUT_DIR / "family_silhouette.csv"
OUT_SUMMARY_JSON = OUT_DIR / "cluster_similarity_summary.json"
OUT_REPORT_MD = OUT_DIR / "cluster_similarity_report.md"
OUT_FIG_STEM = OUT_DIR / "cluster_similarity_geography"

N_CLUSTERS = 2  # matches script 04

# The manuscript's reported k = 2 membership.  derive_clusters() rebuilds this
# from the linkage; the check exists so a change in the upstream ρ matrix fails
# loudly here instead of silently renumbering the clusters in the report.
EXPECTED_CLUSTERS = {
    "C1": {"Austroasiatic", "Tai-Kadai", "Pano-Tacanan", "Austronesian", "Tupian"},
    "C2": {"Nakh-Daghestanian", "Turkic", "Uralic", "Indo-European",
           "Afro-Asiatic", "Sino-Tibetan"},
}

COLOR_WITHIN = "#1f77b4"
COLOR_BETWEEN = "#d62728"
COLOR_OVERLAP = "#f0e2a8"


# ── Cluster derivation (mirrors script 04) ────────────────────────────────────

def load_similarity(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing similarity matrix: {path}\n"
            "Run script 04 (and 13 for the satellite appendix) first."
        )
    sim = pd.read_csv(path, index_col=0)
    sim.index = sim.index.astype(str)
    sim.columns = sim.columns.astype(str)
    return sim


def build_linkage(sim_families: pd.DataFrame):
    """Average linkage on 1 - ρ — byte-for-byte the construction in script 04."""
    dist_mat = np.clip(1 - sim_families.values, 0, 2)
    np.fill_diagonal(dist_mat, 0)
    condensed = squareform(dist_mat, checks=False)
    return linkage(condensed, method="average"), dist_mat


def derive_clusters(Z, families: list[str], k: int = N_CLUSTERS) -> dict[str, list[str]]:
    """Cut the linkage into k clusters, numbered C1..Ck in dendrogram-leaf order
    so the labels agree with script 04's figure and with the manuscript."""
    labels = fcluster(Z, k, criterion="maxclust")
    leaf_order = [families[i] for i in dendrogram(Z, labels=families, no_plot=True)["leaves"]]
    pos = {f: i for i, f in enumerate(leaf_order)}

    groups: dict[int, list[str]] = {}
    for fam, lab in zip(families, labels):
        groups.setdefault(int(lab), []).append(fam)
    for members in groups.values():
        members.sort(key=lambda f: pos[f])
    ordered = sorted(groups.values(), key=lambda members: pos[members[0]])
    return {f"C{i}": members for i, members in enumerate(ordered, start=1)}


def verify_clusters(clusters: dict[str, list[str]]) -> None:
    got = {cid: set(members) for cid, members in clusters.items()}
    if got != EXPECTED_CLUSTERS:
        raise SystemExit(
            "Derived k = 2 clusters do not match the manuscript's reported membership.\n"
            f"  derived:  {({c: sorted(m) for c, m in got.items()})}\n"
            f"  expected: {({c: sorted(m) for c, m in EXPECTED_CLUSTERS.items()})}\n"
            "The upstream similarity matrix has changed — update EXPECTED_CLUSTERS "
            "and the manuscript together, or re-check script 04."
        )


# ── Pair table ────────────────────────────────────────────────────────────────

def build_pair_table(sim_families: pd.DataFrame,
                     clusters: dict[str, list[str]]) -> pd.DataFrame:
    """One row per unordered family pair, tagged within/between, joined to the
    geographic distances script 06 already assembled."""
    label = {f: cid for cid, members in clusters.items() for f in members}
    families = sim_families.index.tolist()

    geo_lookup: dict[tuple[str, str], dict[str, float]] = {}
    if GEO_PAIRS_CSV.exists():
        geo = pd.read_csv(GEO_PAIRS_CSV)
        for _, row in geo.iterrows():
            key = tuple(sorted((str(row["family_a"]), str(row["family_b"]))))
            geo_lookup[key] = {
                "geographic_distance_km": float(row["geographic_distance_km"]),
                "n_common_glosses": row.get("n_common_glosses", np.nan),
            }

    rows = []
    for a, b in itertools.combinations(families, 2):
        geo_row = geo_lookup.get(tuple(sorted((a, b))), {})
        rows.append({
            "family_a": a,
            "family_b": b,
            "cluster_a": label[a],
            "cluster_b": label[b],
            "relation": "within" if label[a] == label[b] else "between",
            "within_cluster": label[a] if label[a] == label[b] else "",
            "spearman_rho": float(sim_families.loc[a, b]),
            "geographic_distance_km": geo_row.get("geographic_distance_km", np.nan),
            "n_common_glosses": geo_row.get("n_common_glosses", np.nan),
        })
    return pd.DataFrame(rows)


def describe(values: np.ndarray) -> dict[str, Any]:
    """Descriptive summary only — see the module docstring on why no test."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"n": 0}
    q1, q3 = np.percentile(values, [25, 75])
    return {
        "n": int(len(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "mean": float(values.mean()),
        "q1": float(q1),
        "q3": float(q3),
    }


def overlap_summary(within: np.ndarray, between: np.ndarray) -> dict[str, Any]:
    low = max(within.min(), between.min())
    high = min(within.max(), between.max())
    overlaps = low <= high
    return {
        "overlaps": bool(overlaps),
        "low": float(low) if overlaps else None,
        "high": float(high) if overlaps else None,
        "n_between_above_within_min": int((between > within.min()).sum()),
        "n_between_total": int(len(between)),
        "n_within_below_between_max": int((within < between.max()).sum()),
        "n_within_total": int(len(within)),
    }


# ── Exact cluster-label permutation test (geography only) ─────────────────────

def exact_compactness_test(pair_df: pd.DataFrame,
                           value_col: str,
                           members: list[str]) -> dict[str, Any]:
    """Is *this* cluster geographically tighter than a random group of families
    of the same size?

    Observed statistic: the median pairwise distance among the cluster's own
    members.  Null: the same statistic over **every** subset of the 11 families
    of that size — C(11, 5) = C(11, 6) = 462 — so the p-value is exact and needs
    no seed.  Two-sided (2 x the smaller tail, capped at 1), because the claim
    being supported is the *absence* of a geographic effect, and a one-sided
    test has no power against a cluster that is unusually dispersed.

    A test per cluster rather than one pooled within-vs-between test: C1 and C2
    turn out to have opposite geographic character, so pooling their within-pairs
    lets the larger cluster dominate the pooled median and hides both signals.
    """
    usable = pair_df.dropna(subset=[value_col])
    families = sorted(set(usable["family_a"]).union(usable["family_b"]))
    k = len(members)

    dist: dict[frozenset[str], float] = {
        frozenset((row["family_a"], row["family_b"])): float(row[value_col])
        for _, row in usable.iterrows()
    }

    def median_within(group: tuple[str, ...]) -> float:
        vals = [dist[key] for a, b in itertools.combinations(group, 2)
                if (key := frozenset((a, b))) in dist]
        return float(np.median(vals)) if vals else float("nan")

    observed = median_within(tuple(members))
    null = np.asarray([median_within(g)
                       for g in itertools.combinations(families, k)])
    null = null[~np.isnan(null)]

    n = len(null)
    p_low = float(np.sum(null <= observed) / n)    # tighter than observed
    p_high = float(np.sum(null >= observed) / n)   # more dispersed than observed
    p_two_sided = float(min(1.0, 2.0 * min(p_low, p_high)))
    return {
        "value": value_col,
        "n_members": int(k),
        "n_pairs": int(k * (k - 1) // 2),
        "observed_median_km": observed,
        "null_median_km": float(np.median(null)),
        "null_min_km": float(null.min()),
        "null_max_km": float(null.max()),
        "percentile_in_null": float(100.0 * p_low),
        "p_one_sided_tighter": p_low,
        "p_one_sided_more_dispersed": p_high,
        "p_two_sided_exact": p_two_sided,
        "n_subsets_enumerated": int(n),
        "minimum_attainable_p": float(2.0 / n),
        "exhaustive": True,
    }


# ── Silhouette ────────────────────────────────────────────────────────────────

def family_silhouette(sim_families: pd.DataFrame, dist_mat: np.ndarray,
                      clusters: dict[str, list[str]]) -> pd.DataFrame:
    """Per-family silhouette on the same 1 - ρ distances the k-selection used.

    The overall score (S = +0.331 at k = 2) is already reported in the
    manuscript; the per-family breakdown shows *which* families sit near the
    boundary, and so which ones generate the within/between overlap.
    """
    families = sim_families.index.tolist()
    label = {f: cid for cid, members in clusters.items() for f in members}
    labels = np.array([label[f] for f in families])
    scores = silhouette_samples(dist_mat, labels, metric="precomputed")
    df = pd.DataFrame({
        "family": families,
        "cluster": labels,
        "silhouette": scores,
    }).sort_values("silhouette")
    return df.reset_index(drop=True)


# ── Ranked pair tables ────────────────────────────────────────────────────────

TOP_N = 10


def ranked_pairs(pair_df: pd.DataFrame, column: str, ascending: bool,
                 n: int = TOP_N) -> list[dict[str, Any]]:
    """The n most extreme pairs by `column`, each tagged within/between.

    Script 06 prints the same two rankings (nearest geographically, most similar
    in concept structure) without cluster labels.  Adding the label is the point:
    it shows directly that geographic adjacency and cluster co-membership come
    apart, which is the question under review.
    """
    df = pair_df.dropna(subset=[column])
    ordered = df.nsmallest(n, column) if ascending else df.nlargest(n, column)
    return [
        {
            "pair": f"{row['family_a']}–{row['family_b']}",
            "relation": ("within " + row["within_cluster"]) if row["relation"] == "within"
                        else "between",
            "spearman_rho": float(row["spearman_rho"]),
            "geographic_distance_km": float(row["geographic_distance_km"]),
        }
        for _, row in ordered.iterrows()
    ]


# ── Satellite appendix (descriptive only) ─────────────────────────────────────

def satellite_summary() -> dict[str, Any] | None:
    """k = 4 within/between ρ from the satellite-inclusive tree, for comparison.

    Descriptive only and figure-free: the professor's question is about the
    main k = 2 split, and the satellite grouping was likewise derived from its
    own ρ matrix, so the same circularity applies.
    """
    if not SAT_SIMILARITY_CSV.exists():
        return None
    try:
        sat_clusters = {
            cid: members
            for cid, (_, members) in
            importlib.import_module("13_satellite_clustering").CLUSTERS.items()
        }
    except Exception:
        return None

    sim = load_similarity(SAT_SIMILARITY_CSV)
    label = {f: cid for cid, members in sat_clusters.items() for f in members}
    families = [f for f in label if f in sim.index]

    within, between = [], []
    for a, b in itertools.combinations(families, 2):
        (within if label[a] == label[b] else between).append(float(sim.loc[a, b]))
    within_arr, between_arr = np.asarray(within), np.asarray(between)

    return {
        "clusters": {cid: sorted(m) for cid, m in sat_clusters.items()},
        "within": describe(within_arr),
        "between": describe(between_arr),
        "overlap": overlap_summary(within_arr, between_arr),
    }


# ── Figure ────────────────────────────────────────────────────────────────────

def _annotate(ax, x, y, text, dx, dy):
    """Label a point, placing the text on whichever side dx points to so it
    stays inside the axes."""
    ax.annotate(text, xy=(x, y), xytext=(x + dx, y + dy), fontsize=7.5,
                color="#333333", ha="right" if dx < 0 else "left", va="center",
                arrowprops=dict(arrowstyle="-", color="#999999", linewidth=0.7))


def plot_figure(pair_df: pd.DataFrame, overlap: dict[str, Any]) -> None:
    geo_df = pair_df.dropna(subset=["geographic_distance_km"])
    fig, (ax_sc, ax_box) = plt.subplots(
        1, 2, figsize=(12, 5), layout="constrained",
        gridspec_kw={"width_ratios": [2, 1]},
    )

    # ── Left: geographic distance vs ρ ────────────────────────────────────────
    for relation, color in (("within", COLOR_WITHIN), ("between", COLOR_BETWEEN)):
        sub = geo_df[geo_df["relation"] == relation]
        sizes = np.clip(sub["n_common_glosses"].to_numpy(dtype=float) * 12.0, 20, 160)
        ax_sc.scatter(sub["geographic_distance_km"], sub["spearman_rho"],
                      s=sizes, color=color, alpha=0.72, edgecolor="white",
                      linewidth=0.6, label=f"{relation}-cluster pair (n = {len(sub)})")

    x = geo_df["geographic_distance_km"].to_numpy(dtype=float)
    y = geo_df["spearman_rho"].to_numpy(dtype=float)
    if len(geo_df) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(x.min(), x.max(), 100)
        ax_sc.plot(x_line, slope * x_line + intercept, color="#555555",
                   linewidth=1.8, label="OLS fit (all pairs)")

    ax_sc.set_xlabel("Geographic distance (km)")
    ax_sc.set_ylabel("PAIN concept-structure similarity (Spearman ρ)")
    ax_sc.set_title("Similarity vs geographic distance\n(marker size ∝ shared concepts)",
                    fontsize=10)
    ax_sc.legend(frameon=False, fontsize=8, loc="upper right")
    ax_sc.grid(alpha=0.2)

    # ── Right: the ρ distributions, and where they overlap ────────────────────
    groups = [("within", COLOR_WITHIN), ("between", COLOR_BETWEEN)]
    data = [pair_df.loc[pair_df["relation"] == g, "spearman_rho"].to_numpy()
            for g, _ in groups]

    if overlap["overlaps"]:
        ax_box.axhspan(overlap["low"], overlap["high"], color=COLOR_OVERLAP,
                       alpha=0.55, zorder=0,
                       label=f"overlap {overlap['low']:.3f}–{overlap['high']:.3f}")

    bp = ax_box.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                        medianprops=dict(color="#222222", linewidth=1.6),
                        zorder=2)
    for patch, (_, color) in zip(bp["boxes"], groups):
        patch.set_facecolor("white")
        patch.set_edgecolor(color)
        patch.set_linewidth(1.4)

    rng = np.random.default_rng(0)  # jitter is cosmetic only
    for i, (values, (_, color)) in enumerate(zip(data, groups), start=1):
        jitter = rng.uniform(-0.13, 0.13, size=len(values))
        ax_box.scatter(np.full(len(values), i) + jitter, values, s=22,
                       color=color, alpha=0.7, edgecolor="white",
                       linewidth=0.5, zorder=3)

    # The two pairs the overlap argument turns on.
    for fam_a, fam_b, group_i, dx, dy in (
        ("Sino-Tibetan", "Tupian", 2, -0.22, 0.09),
        ("Indo-European", "Nakh-Daghestanian", 1, 0.22, -0.09),
    ):
        key = pair_df[
            (pair_df[["family_a", "family_b"]].apply(frozenset, axis=1)
             == frozenset((fam_a, fam_b)))
        ]
        if len(key):
            rho = float(key.iloc[0]["spearman_rho"])
            _annotate(ax_box, group_i, rho, f"{fam_a}–{fam_b}\nρ = {rho:.3f}", dx, dy)

    ax_box.set_xticks([1, 2])
    ax_box.set_xticklabels([f"within\n(n = {len(data[0])})",
                            f"between\n(n = {len(data[1])})"])
    ax_box.set_ylabel("Spearman ρ")
    ax_box.set_title("Within- vs between-cluster ρ\n(descriptive; see report)", fontsize=10)
    ax_box.legend(frameon=False, fontsize=8, loc="lower left")
    ax_box.grid(alpha=0.2, axis="y")

    fig.suptitle("CLICS4 PAIN-CC (k = 2): cross-family similarity, cluster membership "
                 "and geography", fontsize=12)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT_FIG_STEM}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── Report ────────────────────────────────────────────────────────────────────

def _fmt_p(p: float) -> str:
    return f"{p:.2e}" if p < 0.001 else f"{p:.3f}"


def _desc_row(name: str, d: dict[str, Any]) -> str:
    return (f"| {name} | {d['n']} | {d['min']:+.3f} – {d['max']:+.3f} | "
            f"{d['median']:+.3f} | {d['q1']:+.3f} – {d['q3']:+.3f} | {d['mean']:+.3f} |")


def write_report(summary: dict[str, Any], sil: pd.DataFrame) -> None:
    rho = summary["rho"]
    ov = rho["overlap"]
    geo_desc = summary["geography_descriptive"]
    geo_tests = summary["geography_tests"]
    first_test = geo_tests[next(iter(geo_tests))]
    lines = [
        "# Within- vs Between-Cluster Similarity and Geography (CLICS4 PAIN-CC, k = 2)",
        "",
        "Generated by: `src/17_cluster_similarity_geography.py`",
        "",
        "## Inputs",
        "",
        f"- Cross-family similarity: `{SIMILARITY_CSV.relative_to(ROOT)}` (script 04)",
        f"- Geographic distances: `{GEO_PAIRS_CSV.relative_to(ROOT)}` (script 06)",
        "",
        "## Cluster membership",
        "",
        "Re-derived here from the same average linkage on 1 − ρ that script 04 draws, "
        "then checked against the membership reported in the manuscript.",
        "",
    ]
    for cid, members in summary["clusters"].items():
        lines.append(f"- **{cid}** ({len(members)}): {', '.join(members)}")

    lines += [
        "",
        "## Within- vs between-cluster ρ",
        "",
        "| Group | n pairs | Range | Median | IQR | Mean |",
        "|---|---:|---|---:|---|---:|",
        _desc_row("**Within (pooled)**", rho["within_pooled"]),
    ]
    for cid, d in rho["within_by_cluster"].items():
        lines.append(_desc_row(f"— {cid}", d))
    lines.append(_desc_row("**Between**", rho["between"]))

    lines += [
        "",
        f"**The two ranges overlap across ρ = {ov['low']:.3f} – {ov['high']:.3f}.** "
        f"{ov['n_between_above_within_min']} of {ov['n_between_total']} between-cluster "
        f"pairs exceed the lowest within-cluster pair, and "
        f"{ov['n_within_below_between_max']} of {ov['n_within_total']} within-cluster "
        "pairs fall below the highest between-cluster pair.",
        "",
        "The clearest single case is **Sino-Tibetan–Tupian**, a between-cluster pair at "
        f"ρ = {summary['highlight']['Sino-Tibetan–Tupian']:.3f}, which is higher than "
        "most within-cluster pairs; the lowest within-cluster pair is "
        "**Indo-European–Nakh-Daghestanian** at "
        f"ρ = {summary['highlight']['Indo-European–Nakh-Daghestanian']:.3f}.",
        "",
        "### Why no significance test is reported here",
        "",
        "C1 and C2 were obtained by average-linkage clustering **of this same ρ "
        "matrix**, and linkage maximises within-cluster similarity by construction. "
        "A test of \"within ρ > between ρ\" would therefore be circular: it would "
        "report a property of the clustering algorithm, not evidence that the split "
        "is real. The numbers above are descriptive, and the overlap is reported as a "
        "genuine feature of the data rather than explained away. The inferential "
        "weight in this report rests entirely on the geographic comparison below, "
        "which is not circular because the clusters were not derived from geography.",
        "",
        "### Per-family silhouette (k = 2, distance = 1 − ρ)",
        "",
        "The overall silhouette at k = 2 is modest (S = +0.331), so substantial "
        "overlap is the expected pairwise signature of this clustering, not a "
        "contradiction of it. The per-family breakdown shows which families sit near "
        "the boundary and therefore generate most of that overlap.",
        "",
        "| Family | Cluster | Silhouette |",
        "|---|---|---:|",
    ]
    for _, row in sil.iterrows():
        lines.append(f"| {row['family']} | {row['cluster']} | {row['silhouette']:+.3f} |")

    lines += [
        "",
        "## Geography: are the clusters geographically compact?",
        "",
        "### Distances, by group",
        "",
        "| Group | n pairs | Median | Range |",
        "|---|---:|---:|---|",
    ]
    for cid in summary["clusters"]:
        g = geo_desc[cid]
        lines.append(f"| {cid} (within) | {g['n']} | {g['median']:,.0f} km | "
                     f"{g['min']:,.0f} – {g['max']:,.0f} km |")
    g = geo_desc["between"]
    lines.append(f"| Between | {g['n']} | {g['median']:,.0f} km | "
                 f"{g['min']:,.0f} – {g['max']:,.0f} km |")

    lines += [
        "",
        "**The two clusters have opposite geographic character** ("
        + ", ".join(f"{cid} median within-pair distance {geo_desc[cid]['median']:,.0f} km"
                    for cid in summary["clusters"])
        + f"). Pooling their within-pairs gives a median of "
          f"{geo_desc['within_pooled']['median']:,.0f} km, set by whichever cluster "
          "contributes more pairs and describing neither — so no pooled "
          "within-vs-between test is reported, and each cluster is tested on its own "
          "below.",
        "",
        "### Exact compactness test, per cluster",
        "",
        "For each cluster, the median distance among its own members is compared "
        "against **every** subset of the 11 families of the same size "
        f"({first_test['n_subsets_enumerated']} subsets), so the p-value is exact and "
        "needs no seed. Two-sided (twice the smaller tail), because the claim under "
        "test is the *absence* of a geographic effect, and a one-sided test has no "
        "power against a cluster that is unusually dispersed.",
        "",
        "| Cluster | n families | Observed median | Null median | Percentile | Exact two-sided p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cid, t in geo_tests.items():
        lines.append(
            f"| {cid} | {t['n_members']} | {t['observed_median_km']:,.0f} km | "
            f"{t['null_median_km']:,.0f} km | {t['percentile_in_null']:.1f} | "
            f"{_fmt_p(t['p_two_sided_exact'])} |")

    lines += [
        "",
        summary["geography_verdict"],
        "",
        "### Ranked pairs, with cluster membership",
        "",
        "Script 06 prints these two rankings without cluster labels. The labels are "
        "what makes them informative here.",
        "",
        f"**The {TOP_N} geographically nearest family pairs**",
        "",
        "| Pair | Distance | ρ | Relation |",
        "|---|---:|---:|---|",
    ]
    for r in summary["nearest_geographic_pairs"]:
        lines.append(f"| {r['pair']} | {r['geographic_distance_km']:,.0f} km | "
                     f"{r['spearman_rho']:+.3f} | {r['relation']} |")

    lines += [
        "",
        f"**The {TOP_N} most similar family pairs in concept structure**",
        "",
        "| Pair | ρ | Distance | Relation |",
        "|---|---:|---:|---|",
    ]
    for r in summary["most_similar_pairs"]:
        lines.append(f"| {r['pair']} | {r['spearman_rho']:+.3f} | "
                     f"{r['geographic_distance_km']:,.0f} km | {r['relation']} |")

    lines += [
        "",
        "This is a different question from the one script 06 answers. Script 06 asks "
        "whether geographic distance predicts concept-structure similarity across all "
        "55 pairs (it does not: ρ = −0.122, p = 0.373). The test here asks whether the "
        "cluster *memberships* align with geography — which a null correlation does "
        "not by itself rule out.",
        "",
    ]

    sat = summary.get("satellite")
    if sat:
        lines += [
            "## Appendix: the satellite-inclusive k = 4 grouping",
            "",
            "Descriptive comparison only (the same circularity applies, and this "
            "grouping is not the one under discussion). Reported because the overlap "
            "is markedly narrower there, which suggests that the wider overlap in the "
            "main analysis follows from the PAIN-CC restriction rather than from the "
            "clustering method.",
            "",
            "| Group | n pairs | Range | Median |",
            "|---|---:|---|---:|",
            (f"| Within | {sat['within']['n']} | {sat['within']['min']:+.3f} – "
             f"{sat['within']['max']:+.3f} | {sat['within']['median']:+.3f} |"),
            (f"| Between | {sat['between']['n']} | {sat['between']['min']:+.3f} – "
             f"{sat['between']['max']:+.3f} | {sat['between']['median']:+.3f} |"),
            "",
            (f"Overlap: ρ = {sat['overlap']['low']:.3f} – {sat['overlap']['high']:.3f}, "
             f"with {sat['overlap']['n_between_above_within_min']} of "
             f"{sat['overlap']['n_between_total']} between-cluster pairs above the "
             "lowest within-cluster pair."),
            "",
        ]

    lines += [
        "## Suggested wording",
        "",
        "> Within-cluster Spearman correlations range from "
        f"ρ = {rho['within_pooled']['min']:.3f} to {rho['within_pooled']['max']:.3f} "
        f"(median {rho['within_pooled']['median']:.3f}; C1 "
        f"{rho['within_by_cluster']['C1']['min']:.3f}–{rho['within_by_cluster']['C1']['max']:.3f}, "
        f"C2 {rho['within_by_cluster']['C2']['min']:.3f}–{rho['within_by_cluster']['C2']['max']:.3f}), "
        f"and between-cluster correlations from {rho['between']['min']:.3f} to "
        f"{rho['between']['max']:.3f} (median {rho['between']['median']:.3f}). The two "
        f"ranges overlap across ρ = {ov['low']:.3f}–{ov['high']:.3f}: "
        f"{ov['n_between_above_within_min']} of {ov['n_between_total']} between-cluster "
        "pairs exceed the weakest within-cluster pair, Sino-Tibetan–Tupian "
        f"(ρ = {summary['highlight']['Sino-Tibetan–Tupian']:.3f}) most conspicuously. "
        "We report this overlap rather than a significance test, because the clusters "
        "were derived from this same correlation matrix and any such test would be "
        "circular. The overlap is consistent with the modest silhouette of the k = 2 "
        "solution (S = +0.331), which indicates a weak rather than a categorical "
        "partition.",
        "",
        "> The two clusters differ sharply in geographic character. "
        + " ".join(
            f"{cid} members lie a median {t['observed_median_km']:,.0f} km apart "
            f"(percentile {t['percentile_in_null']:.1f} of all "
            f"{t['n_subsets_enumerated']} same-size subsets of the 11 families, exact "
            f"two-sided p = {_fmt_p(t['p_two_sided_exact'])})."
            for cid, t in geo_tests.items())
        + f" Between-cluster pairs lie a median {geo_desc['between']['median']:,.0f} km "
          "apart. Each cluster is therefore tested against same-size random groups "
          "rather than pooled, because pooling lets the larger cluster set the "
          "within-group median.",
        "",
        "## Outputs",
        "",
        f"- Pair table: `{OUT_PAIRS_CSV.relative_to(ROOT)}`",
        f"- Per-family silhouette: `{OUT_SILHOUETTE_CSV.relative_to(ROOT)}`",
        f"- Summary JSON: `{OUT_SUMMARY_JSON.relative_to(ROOT)}`",
        f"- Figure: `{OUT_FIG_STEM.relative_to(ROOT)}.png` / `.pdf`",
    ]
    OUT_REPORT_MD.write_text("\n".join(lines) + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print("CLICS4 PAIN-CC: WITHIN- vs BETWEEN-CLUSTER SIMILARITY AND GEOGRAPHY")
    print("=" * 66)

    sim = load_similarity(SIMILARITY_CSV)
    families = [f for f in sim.index if f != "Universal"]
    sim_families = sim.loc[families, families]

    Z, dist_mat = build_linkage(sim_families)
    clusters = derive_clusters(Z, families)
    verify_clusters(clusters)
    print(f"\nDerived k = {N_CLUSTERS} clusters (verified against the manuscript):")
    for cid, members in clusters.items():
        print(f"  {cid} ({len(members)}): {', '.join(members)}")

    pair_df = build_pair_table(sim_families, clusters)
    pair_df.to_csv(OUT_PAIRS_CSV, index=False)

    within = pair_df.loc[pair_df["relation"] == "within", "spearman_rho"].to_numpy()
    between = pair_df.loc[pair_df["relation"] == "between", "spearman_rho"].to_numpy()
    within_by_cluster = {
        cid: describe(pair_df.loc[pair_df["within_cluster"] == cid, "spearman_rho"].to_numpy())
        for cid in clusters
    }
    ov = overlap_summary(within, between)

    print("\nSpearman ρ (descriptive only — clusters were derived from this matrix):")
    print(f"  within  n={len(within):2d}  {within.min():+.3f} – {within.max():+.3f}  "
          f"median {np.median(within):+.3f}")
    for cid, d in within_by_cluster.items():
        print(f"    {cid}   n={d['n']:2d}  {d['min']:+.3f} – {d['max']:+.3f}  "
              f"median {d['median']:+.3f}")
    print(f"  between n={len(between):2d}  {between.min():+.3f} – {between.max():+.3f}  "
          f"median {np.median(between):+.3f}")
    print(f"  OVERLAP {ov['low']:+.3f} – {ov['high']:+.3f}   "
          f"({ov['n_between_above_within_min']}/{ov['n_between_total']} between-pairs "
          "above the lowest within-pair)")

    sil = family_silhouette(sim_families, dist_mat, clusters)
    sil.to_csv(OUT_SILHOUETTE_CSV, index=False)
    print("\nPer-family silhouette (k = 2, ascending):")
    for _, row in sil.iterrows():
        print(f"  {row['family']:22s} {row['cluster']}  {row['silhouette']:+.3f}")

    geo_desc = {
        "between": describe(
            pair_df.loc[pair_df["relation"] == "between", "geographic_distance_km"].to_numpy()),
        "within_pooled": describe(
            pair_df.loc[pair_df["relation"] == "within", "geographic_distance_km"].to_numpy()),
        **{cid: describe(
            pair_df.loc[pair_df["within_cluster"] == cid, "geographic_distance_km"].to_numpy())
           for cid in clusters},
    }
    print("\nGeographic distance (km), three groups:")
    for name in [*clusters, "between"]:
        d = geo_desc[name]
        tag = f"{name} within" if name in clusters else "between    "
        print(f"  {tag:14s} n={d['n']:2d}  median {d['median']:>9,.0f}  "
              f"({d['min']:,.0f} – {d['max']:,.0f})")

    geo_tests = {
        cid: exact_compactness_test(pair_df, "geographic_distance_km", members)
        for cid, members in clusters.items()
    }
    print("\nGeographic compactness, exact test per cluster "
          f"({geo_tests['C1']['n_subsets_enumerated']} same-size subsets each):")
    verdicts = []
    for cid, t in geo_tests.items():
        print(f"  {cid}: median {t['observed_median_km']:>9,.0f} km vs null median "
              f"{t['null_median_km']:,.0f} km — percentile {t['percentile_in_null']:.1f}, "
              f"two-sided exact p = {_fmt_p(t['p_two_sided_exact'])}")
        if t["p_two_sided_exact"] >= 0.05:
            verdicts.append(
                f"**{cid}** is no more geographically compact than a random group of "
                f"{t['n_members']} families (percentile {t['percentile_in_null']:.1f} of "
                f"the null, two-sided exact p = {_fmt_p(t['p_two_sided_exact'])}), so its "
                "membership is not explained by geographic proximity.")
        elif t["p_one_sided_tighter"] < t["p_one_sided_more_dispersed"]:
            verdicts.append(
                f"**{cid}** is significantly *tighter* geographically than a random group "
                f"of {t['n_members']} families (percentile {t['percentile_in_null']:.1f}, "
                f"two-sided exact p = {_fmt_p(t['p_two_sided_exact'])}); geographic "
                "proximity cannot be ruled out as a contributor to this cluster.")
        else:
            verdicts.append(
                f"**{cid}** is significantly *more dispersed* geographically than a random "
                f"group of {t['n_members']} families (percentile "
                f"{t['percentile_in_null']:.1f}, two-sided exact p = {_fmt_p(t['p_two_sided_exact'])}); its "
                "members group together despite being far apart.")
    verdict = " ".join(verdicts)
    print()
    for v in verdicts:
        print(f"  - {v.replace('**', '')}")

    nearest = ranked_pairs(pair_df, "geographic_distance_km", ascending=True)
    most_similar = ranked_pairs(pair_df, "spearman_rho", ascending=False)
    print(f"\nThe {TOP_N} geographically nearest pairs:")
    for r in nearest:
        print(f"  {r['pair']:42s} {r['geographic_distance_km']:>9,.0f} km  "
              f"ρ = {r['spearman_rho']:+.3f}  {r['relation']}")
    print(f"\nThe {TOP_N} most similar pairs in concept structure:")
    for r in most_similar:
        print(f"  {r['pair']:42s} ρ = {r['spearman_rho']:+.3f}  "
              f"{r['geographic_distance_km']:>9,.0f} km  {r['relation']}")

    sat = satellite_summary()
    if sat:
        print(f"\nSatellite k = 4 appendix: within {sat['within']['min']:+.3f}–"
              f"{sat['within']['max']:+.3f}, between {sat['between']['min']:+.3f}–"
              f"{sat['between']['max']:+.3f}, overlap {sat['overlap']['low']:+.3f}–"
              f"{sat['overlap']['high']:+.3f}")
    else:
        print("\nSatellite k = 4 appendix skipped (run script 13 to include it).")

    def _pair_rho(a: str, b: str) -> float:
        return float(sim_families.loc[a, b])

    summary = {
        "clusters": clusters,
        "rho": {
            "within_pooled": describe(within),
            "within_by_cluster": within_by_cluster,
            "between": describe(between),
            "overlap": ov,
            "test_reported": None,
            "test_omitted_because": (
                "Clusters were derived by average-linkage clustering of this same "
                "Spearman matrix; a within-vs-between test would be circular."
            ),
        },
        "highlight": {
            "Sino-Tibetan–Tupian": _pair_rho("Sino-Tibetan", "Tupian"),
            "Indo-European–Nakh-Daghestanian": _pair_rho("Indo-European", "Nakh-Daghestanian"),
        },
        "silhouette_overall_reported_in_manuscript": 0.331,
        "geography_descriptive": geo_desc,
        "geography_tests": geo_tests,
        "geography_verdict": verdict,
        "nearest_geographic_pairs": nearest,
        "most_similar_pairs": most_similar,
        "geography_pooled_test_omitted_because": (
            "C1 and C2 have opposite geographic character ("
            + ", ".join(f"{cid} median within-pair distance {geo_desc[cid]['median']:,.0f} km"
                        for cid in clusters)
            + "), so a pooled within-vs-between median is dominated by the larger "
              "cluster and represents neither."
        ),
        "satellite": sat,
    }
    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")

    plot_figure(pair_df, ov)
    write_report(summary, sil)

    print(f"\nSaved: {OUT_PAIRS_CSV}")
    print(f"Saved: {OUT_SILHOUETTE_CSV}")
    print(f"Saved: {OUT_SUMMARY_JSON}")
    print(f"Saved: {OUT_REPORT_MD}")
    print(f"Saved: {OUT_FIG_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
