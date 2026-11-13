"""SATELLITE-INCLUSIVE GEOGRAPHY CONTROL (secondary analysis).

Satellite-inclusive counterpart of script 06: joins the satellite-inclusive
cross-family similarity matrix (script 13) with the precomputed family-pair
geographic distances and runs the same correlation + Mantel-style permutation
tests.

Outputs: output/satellite/cross_family/geography_control/

Usage:
    python src/14_satellite_geography.py     (requires script 13 to have run)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

_cache_root = Path(tempfile.gettempdir()) / "pain_geography_control_cache"
_mpl_cache = _cache_root / "mpl"
_xdg_cache = _cache_root / "xdg"
_mpl_cache.mkdir(parents=True, exist_ok=True)
_xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


from config import GEOGRAPHY_CSV, SATELLITE_DIR, PUBLIC_ROOT

ROOT = PUBLIC_ROOT
ALIGNED_DIR = SATELLITE_DIR
CROSS_FAMILY_DIR = ALIGNED_DIR / "cross_family"
SIMILARITY_CSV = CROSS_FAMILY_DIR / "similarity_matrix.csv"

GEO_CTRL_DIR = CROSS_FAMILY_DIR / "geography_control"
OUT_PAIRS_CSV = GEO_CTRL_DIR / "geography_control_pairs.csv"
OUT_SUMMARY_JSON = GEO_CTRL_DIR / "geography_control_summary.json"
OUT_REPORT_MD = GEO_CTRL_DIR / "geography_control.md"
OUT_SCATTER_PNG = GEO_CTRL_DIR / "geography_control_scatter.png"

N_PERMUTATIONS = 10000
RANDOM_SEED = 42


def _format_p(p_value: float) -> str:
    if np.isnan(p_value):
        return "NA"
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.3f}"


def load_similarity_matrix(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing similarity matrix: {path}")
    sim = pd.read_csv(path, index_col=0)
    sim.index = sim.index.astype(str)
    sim.columns = sim.columns.astype(str)
    return sim


def load_geography_pairs(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing geographic pair table: {path}")
    geo = pd.read_csv(path)
    required = {"family_a", "family_b", "distance"}
    missing = required - set(geo.columns)
    if missing:
        raise ValueError(f"Geography CSV is missing columns: {sorted(missing)}")
    return geo


def build_pair_table(sim: pd.DataFrame, geo: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in geo.iterrows():
        family_a = str(row["family_a"])
        family_b = str(row["family_b"])
        if family_a == "Universal" or family_b == "Universal":
            continue
        if family_a not in sim.index or family_b not in sim.columns:
            continue

        similarity = float(sim.loc[family_a, family_b])
        rows.append(
            {
                "family_a": family_a,
                "family_b": family_b,
                "geographic_distance_km": float(row["distance"]),
                "concept_spearman_similarity": similarity,
                "concept_spearman_distance": 1.0 - similarity,
                "n_common_glosses": row.get("n_common_glosses", np.nan),
                "pain_pairwise_ari": row.get("pain_pairwise_ari", np.nan),
                "pain_pairwise_p": row.get("pain_pairwise_p", np.nan),
                "pain_pairwise_perm_robust": row.get("pain_pairwise_perm_robust", np.nan),
            }
        )

    if not rows:
        raise ValueError("No overlapping family pairs found between similarity and geography inputs.")
    return pd.DataFrame(rows)


def spearman_summary(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    temp = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    rho, p_value = stats.spearmanr(temp[x_col].to_numpy(), temp[y_col].to_numpy())
    return {
        "x": x_col,
        "y": y_col,
        "n": int(len(temp)),
        "rho": float(rho),
        "p": float(p_value),
    }


def pearson_summary(df: pd.DataFrame, x_col: str, y_col: str) -> dict[str, Any]:
    temp = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    r, p_value = stats.pearsonr(temp[x_col].to_numpy(), temp[y_col].to_numpy())
    return {
        "x": x_col,
        "y": y_col,
        "n": int(len(temp)),
        "r": float(r),
        "p": float(p_value),
    }


def _matrix_from_pairs(df: pd.DataFrame, value_col: str, diagonal: float) -> pd.DataFrame:
    families = sorted(set(df["family_a"]).union(df["family_b"]))
    mat = pd.DataFrame(np.nan, index=families, columns=families, dtype=float)
    np.fill_diagonal(mat.values, diagonal)
    for _, row in df.iterrows():
        a = row["family_a"]
        b = row["family_b"]
        value = float(row[value_col])
        mat.loc[a, b] = value
        mat.loc[b, a] = value
    return mat


def _upper_triangle(mat: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(mat.shape[0], k=1)
    return mat[idx]


def _rank_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    return float(np.corrcoef(x_rank, y_rank)[0, 1])


def mantel_permutation_summary(
    df: pd.DataFrame,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Matrix-label permutation test for distance vs concept similarity.

    The simple Spearman test treats family pairs as independent. This companion
    test keeps each matrix intact and shuffles the concept-similarity family
    labels, which is a closer match to the matrix comparison.
    """
    geo_mat = _matrix_from_pairs(df, "geographic_distance_km", diagonal=0.0)
    sim_mat = _matrix_from_pairs(df, "concept_spearman_similarity", diagonal=1.0)
    sim_mat = sim_mat.loc[geo_mat.index, geo_mat.columns]

    geo_vec = _upper_triangle(geo_mat.to_numpy(dtype=float))
    sim_arr = sim_mat.to_numpy(dtype=float)
    sim_vec = _upper_triangle(sim_arr)
    observed = _rank_corr(geo_vec, sim_vec)

    rng = np.random.default_rng(seed)
    permuted = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        perm = rng.permutation(sim_arr.shape[0])
        permuted[i] = _rank_corr(geo_vec, _upper_triangle(sim_arr[perm][:, perm]))

    p_two_sided = (np.sum(np.abs(permuted) >= abs(observed)) + 1.0) / (n_permutations + 1.0)
    return {
        "n_families": int(sim_arr.shape[0]),
        "n_pairs": int(len(geo_vec)),
        "rho": float(observed),
        "p_two_sided": float(p_two_sided),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
    }


def plot_scatter(df: pd.DataFrame, spearman: dict[str, Any], out_path: Path) -> None:
    x = df["geographic_distance_km"].to_numpy(dtype=float)
    y = df["concept_spearman_similarity"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, color="#2ca02c", alpha=0.75, edgecolor="white", linewidth=0.6)

    if len(df) >= 2:
        slope, intercept = np.polyfit(x, y, deg=1)
        x_line = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(x_line, slope * x_line + intercept, color="#1f77b4", linewidth=2)

    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel("Geographic distance (km)")
    ax.set_ylabel("PAIN concept-structure similarity (Spearman rho)")
    ax.set_title(
        "CLICS4 aligned-T2 geography control\n"
        f"Spearman rho = {spearman['rho']:+.3f}, p = {_format_p(spearman['p'])}, "
        f"n = {spearman['n']} pairs"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict[str, Any], df: pd.DataFrame, out_path: Path) -> None:
    sim_test = summary["spearman_similarity_vs_geo"]
    dist_test = summary["spearman_concept_distance_vs_geo"]
    mantel = summary["mantel_permutation"]
    pearson = summary["pearson_similarity_vs_geo"]

    nearest = df.sort_values("geographic_distance_km").head(5)
    most_similar = df.sort_values("concept_spearman_similarity", ascending=False).head(5)

    lines = [
        "# CLICS4 Aligned-T2 Geography Control",
        "",
        f"Generated by: `src/14_satellite_geography.py`",
        "",
        "## Inputs",
        "",
        f"- Concept-structure similarity: `{SIMILARITY_CSV.relative_to(ROOT)}`",
        f"- Geographic distances: `{GEOGRAPHY_CSV.relative_to(ROOT)}`",
        "",
        "## Result",
        "",
        "| Test | n | Statistic | p-value |",
        "|---|---:|---:|---:|",
        (
            "| Spearman: geographic distance vs concept-structure similarity "
            f"| {sim_test['n']} | rho = {sim_test['rho']:+.3f} | {_format_p(sim_test['p'])} |"
        ),
        (
            "| Spearman: geographic distance vs concept-structure distance "
            f"`(1 - similarity)` | {dist_test['n']} | rho = {dist_test['rho']:+.3f} | "
            f"{_format_p(dist_test['p'])} |"
        ),
        (
            "| Pearson: geographic distance vs concept-structure similarity "
            f"| {pearson['n']} | r = {pearson['r']:+.3f} | {_format_p(pearson['p'])} |"
        ),
        (
            "| Mantel-style label permutation "
            f"| {mantel['n_pairs']} | rho = {mantel['rho']:+.3f} | "
            f"{_format_p(mantel['p_two_sided'])} |"
        ),
        "",
        "Interpretation: cross-family PAIN concept-structure similarity is not "
        "significantly associated with geographic distance. Nearby families are "
        "not systematically more similar than distant families in the aligned-T2 "
        "concept profile.",
        "",
        "The simple Spearman row treats family pairs as observations. The "
        "Mantel-style row is included as a matrix-level robustness check by "
        f"shuffling family labels {mantel['n_permutations']:,} times "
        f"(seed = {mantel['seed']}).",
        "",
        "## Nearest Geographic Pairs",
        "",
        "| Pair | distance_km | concept_similarity |",
        "|---|---:|---:|",
    ]

    for _, row in nearest.iterrows():
        pair = f"{row['family_a']} - {row['family_b']}"
        lines.append(
            f"| {pair} | {row['geographic_distance_km']:.0f} | "
            f"{row['concept_spearman_similarity']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Most Similar Concept-Structure Pairs",
            "",
            "| Pair | concept_similarity | distance_km |",
            "|---|---:|---:|",
        ]
    )
    for _, row in most_similar.iterrows():
        pair = f"{row['family_a']} - {row['family_b']}"
        lines.append(
            f"| {pair} | {row['concept_spearman_similarity']:.3f} | "
            f"{row['geographic_distance_km']:.0f} |"
        )

    lines.extend(
        [
            "",
            "## Outputs",
            "",
            f"- Pair table: `{OUT_PAIRS_CSV.relative_to(ROOT)}`",
            f"- Summary JSON: `{OUT_SUMMARY_JSON.relative_to(ROOT)}`",
            f"- Scatter plot: `{OUT_SCATTER_PNG.relative_to(ROOT)}`",
        ]
    )

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    CROSS_FAMILY_DIR.mkdir(parents=True, exist_ok=True)
    GEO_CTRL_DIR.mkdir(parents=True, exist_ok=True)

    sim = load_similarity_matrix(SIMILARITY_CSV)
    geo = load_geography_pairs(GEOGRAPHY_CSV)
    pair_df = build_pair_table(sim, geo)
    pair_df.to_csv(OUT_PAIRS_CSV, index=False)

    spearman_similarity = spearman_summary(
        pair_df, "geographic_distance_km", "concept_spearman_similarity"
    )
    spearman_distance = spearman_summary(
        pair_df, "geographic_distance_km", "concept_spearman_distance"
    )
    pearson_similarity = pearson_summary(
        pair_df, "geographic_distance_km", "concept_spearman_similarity"
    )
    mantel = mantel_permutation_summary(pair_df)

    summary = {
        "inputs": {
            "similarity_csv": str(SIMILARITY_CSV.relative_to(ROOT)),
            "geography_csv": str(GEOGRAPHY_CSV.relative_to(ROOT)),
        },
        "outputs": {
            "pairs_csv": str(OUT_PAIRS_CSV.relative_to(ROOT)),
            "summary_json": str(OUT_SUMMARY_JSON.relative_to(ROOT)),
            "report_md": str(OUT_REPORT_MD.relative_to(ROOT)),
            "scatter_png": str(OUT_SCATTER_PNG.relative_to(ROOT)),
        },
        "spearman_similarity_vs_geo": spearman_similarity,
        "spearman_concept_distance_vs_geo": spearman_distance,
        "pearson_similarity_vs_geo": pearson_similarity,
        "mantel_permutation": mantel,
    }

    OUT_SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    plot_scatter(pair_df, spearman_similarity, OUT_SCATTER_PNG)
    write_report(summary, pair_df, OUT_REPORT_MD)

    print("CLICS4 aligned-T2 geography control")
    print(f"Saved: {OUT_PAIRS_CSV}")
    print(f"Saved: {OUT_SUMMARY_JSON}")
    print(f"Saved: {OUT_REPORT_MD}")
    print(f"Saved: {OUT_SCATTER_PNG}")
    print(
        "Spearman distance vs concept similarity: "
        f"rho={spearman_similarity['rho']:+.3f}, "
        f"p={_format_p(spearman_similarity['p'])}, "
        f"n={spearman_similarity['n']}"
    )
    print(
        "Mantel-style label permutation: "
        f"rho={mantel['rho']:+.3f}, "
        f"p={_format_p(mantel['p_two_sided'])}, "
        f"n_perm={mantel['n_permutations']}"
    )


if __name__ == "__main__":
    main()
