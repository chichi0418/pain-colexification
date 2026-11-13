#!/usr/bin/env python3
"""SECONDARY ANALYSIS: SATELLITE-INCLUSIVE CENTRALITY.

Satellite-inclusive counterpart of script 02: per-concept centrality on the
pre-walk pain subgraph and on the satellite-inclusive post-walk graph
(build_aligned_pain_subgraph, no PAIN-CC restriction). The per-family postwalk
CSVs feed the satellite-inclusive cross-family clustering (script 13).

Outputs (output/satellite/):
  families/{Family}/{Family}_centrality_prewalk.csv
  families/{Family}/{Family}_centrality_postwalk.csv
  centrality/phy_nonphy/centrality_summary_prewalk.csv
  centrality/phy_nonphy/centrality_summary_postwalk.csv

Usage:
    python src/12_satellite_centrality.py
"""
import os
import random

import numpy as np
import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"

from config import (
    SATELLITE_DIR, WALK_T, GLOSS_ATTR,
    load_graph, load_concept_set, load_families, load_label_map,
)
from network_utils import build_aligned_pain_subgraph, build_prewalk_pain_subgraph

OUT_DIR = SATELLITE_DIR


def centrality_df(g, label_map):
    glosses = g.vs[GLOSS_ATTR]
    weights = g.es["weight"]
    inv_w = [1.0 / w if w and w > 0 else 0.0 for w in weights]

    # igraph's eigenvector_centrality uses an ARPACK solver seeded from a
    # random start vector, so its last few digits jitter run-to-run. Seed
    # igraph's RNG (Python's `random`) for reproducible output.
    random.seed(0)
    eigenvector = g.eigenvector_centrality(weights="weight")

    df = pd.DataFrame({
        "Concept":     glosses,
        "Comm":        [label_map.get(gl, "") for gl in glosses],
        "Degree":      g.degree(),
        "Strength":    g.strength(weights="weight"),
        "Betweenness": g.betweenness(weights=inv_w),
        "Closeness":   g.closeness(weights=inv_w),
        "Eigenvector": eigenvector,
        "PageRank":    g.pagerank(weights="weight"),
    })
    df.sort_values("Strength", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def family_summary(df):
    """Mean strength per phy/nonphy, top concept by strength + its betweenness."""
    phy  = df[df["Comm"] == "phy"]["Strength"].tolist()
    npy  = df[df["Comm"] == "nonphy"]["Strength"].tolist()

    def _ranked(metric):
        # stable descending order; ties keep df's existing (strength-desc) order
        order = df.sort_values(metric, ascending=False, kind="stable")
        names = [str(c).title() for c, v in zip(order["Concept"], order[metric]) if v > 0]
        first  = names[0] if len(names) > 0 else "—"
        second = names[1] if len(names) > 1 else "—"
        return first, second

    if not df.empty and df["Strength"].max() > 0:
        top_by_strength, top2_by_strength = _ranked("Strength")
    else:
        top_by_strength = top2_by_strength = "—"

    if not df.empty and df["Betweenness"].max() > 0:
        top_by_betw, top2_by_betw = _ranked("Betweenness")
        top_betw = round(float(df["Betweenness"].max()), 2)
    else:
        top_by_betw = top2_by_betw = "—"
        top_betw = 0.0

    return {
        "Phys.":            round(float(np.mean(phy)), 2) if phy else 0.0,
        "Nonphys.":         round(float(np.mean(npy)), 2) if npy else 0.0,
        "Phys. conc.":      len(phy),
        "Nonphys. conc.":   len(npy),
        "Betw.":            top_betw,
        "Top_by_strength":  top_by_strength,
        "Top2_by_strength": top2_by_strength,
        "Top_by_betw":      top_by_betw,
        "Top2_by_betw":     top2_by_betw,
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    concept_set = load_concept_set()
    label_map   = load_label_map()
    families    = load_families()

    g_full = load_graph()

    pre_rows, post_rows = [], []

    for i, (family_name, family_code) in enumerate(families, 1):
        print(f"[{i}/{len(families)}] {family_name} ...", flush=True)

        fam_dir = OUT_DIR / "families" / family_name
        fam_dir.mkdir(parents=True, exist_ok=True)

        # prewalk
        g_pre = build_prewalk_pain_subgraph(g_full, family_code, concept_set)
        if g_pre is not None:
            df_pre = centrality_df(g_pre, label_map)
            df_pre.to_csv(fam_dir / f"{family_name}_centrality_prewalk.csv",
                          index=False)
            summ = family_summary(df_pre)
            pre_rows.append({"Family": family_name, **summ})
            print(f"  prewalk : n={g_pre.vcount()} e={g_pre.ecount()} "
                  f"phy={summ['Phys.']} nonphy={summ['Nonphys.']} "
                  f"top_S={summ['Top_by_strength']} "
                  f"top_B={summ['Top_by_betw']} (B={summ['Betw.']})")
        else:
            print("  prewalk : skipped")

        # postwalk (satellite-inclusive, T=2)
        g_post, _ = build_aligned_pain_subgraph(
            g_full, family_code, concept_set, walk_steps=WALK_T
        )
        if g_post is not None:
            df_post = centrality_df(g_post, label_map)
            df_post.to_csv(fam_dir / f"{family_name}_centrality_postwalk.csv",
                           index=False)
            summ = family_summary(df_post)
            post_rows.append({"Family": family_name, **summ})
            print(f"  postwalk: n={g_post.vcount()} e={g_post.ecount()} "
                  f"phy={summ['Phys.']} nonphy={summ['Nonphys.']} "
                  f"top_S={summ['Top_by_strength']} "
                  f"top_B={summ['Top_by_betw']} (B={summ['Betw.']})")
        else:
            print("  postwalk: skipped")

    cols = ["Family", "Phys.", "Nonphys.", "Phys. conc.", "Nonphys. conc.",
            "Betw.", "Top_by_strength", "Top2_by_strength",
            "Top_by_betw", "Top2_by_betw"]
    cent_dir = OUT_DIR / "centrality" / "phy_nonphy"
    cent_dir.mkdir(parents=True, exist_ok=True)
    pre_path  = cent_dir / "centrality_summary_prewalk.csv"
    post_path = cent_dir / "centrality_summary_postwalk.csv"
    pd.DataFrame(pre_rows,  columns=cols).to_csv(pre_path,  index=False)
    pd.DataFrame(post_rows, columns=cols).to_csv(post_path, index=False)
    print(f"\nSaved {pre_path}")
    print(f"Saved {post_path}")


if __name__ == "__main__":
    main()
