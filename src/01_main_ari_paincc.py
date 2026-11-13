#!/usr/bin/env python3
"""MAIN ANALYSIS — Supplementary Table S2 steps 4–11.

Per family (11 language families + the pooled Universal network):
  build the PAIN-CC graph (family filter → random walk T = 2 on the full
  family network → post-walk extraction of the 26 PAIN concepts → restriction
  to the connected component containing PAIN), detect communities with exact
  optimal modularity, evaluate ARI against the physical/non-physical ground
  truth, and assess significance with a one-sided label-permutation test
  (1,000-shuffle Monte Carlo estimate plus the exact enumerated p-value).

Outputs (output/main/):
  summary.csv                                  — ARI, Monte Carlo p (p_Local), exact p (p_Exact; reported), minimum attainable p, k, node counts, density per family
  families/{Family}/{Family}_local_detail.csv  — per-concept detected cluster vs. truth
  families/{Family}/{Family}_local_perm.csv    — the 1,000 permuted ARIs
  families/{Family}/{Family}_ari_comparison.png — diagnostic panel

Usage:
    python src/01_main_ari_paincc.py
    python src/01_main_ari_paincc.py --labels LABELS.csv --outdir DIR   # label ablation (script 16)
"""
import argparse
import os
from pathlib import Path

import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"

from config import (
    LABEL_CSV, MAIN_DIR, WALK_T, N_PERM, SEED, GLOSS_ATTR,
    load_graph, load_concept_set, load_families,
)
from network_utils import build_paincc_subgraph
from ari_utils import (
    fmt_p,
    calculate_ari_for_subgraph, permutation_test, visualize_local_with_permutation,
    min_attainable_p, exact_permutation_p,
)


def main(label_csv=LABEL_CSV, out_dir=MAIN_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_set = load_concept_set()
    families = load_families()

    g_full = load_graph()

    all_results = []
    for i, (family_name, family_code) in enumerate(families, 1):
        print(f"[{i}/{len(families)}] {family_name} ({family_code}) ...", end=" ", flush=True)

        try:
            g_sub, n_pain, n_paincc = build_paincc_subgraph(
                g_full, family_code, concept_set, walk_steps=WALK_T
            )
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if g_sub is None:
            print(f"skipped (n_pain={n_pain}, n_paincc={n_paincc}, no valid g_sub)")
            continue

        loc = calculate_ari_for_subgraph(g_sub, label_csv, gloss_attr=GLOSS_ATTR)
        if not loc:
            print("skipped (ARI returned None)")
            continue

        p_val, perm_aris = permutation_test(
            loc["y_true"], loc["y_pred"], loc["ari"], n_perm=N_PERM, seed=SEED
        )

        family_dir = out_dir / "families" / family_name
        family_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"perm_ari": perm_aris}).to_csv(
            family_dir / f"{family_name}_local_perm.csv", index=False
        )
        loc["comparison_df"].to_csv(
            family_dir / f"{family_name}_local_detail.csv", index=False
        )
        visualize_local_with_permutation(
            g_sub, loc, perm_aris, label_csv,
            family_name=family_name, save_dir=family_dir,
            p_val=p_val, gloss_attr=GLOSS_ATTR,
        )

        all_results.append({
            "Family":         family_name,
            "ARI_Local":      loc["ari"],
            "p_Local":        max(p_val, 1 / N_PERM),
            "p_Exact":        fmt_p(exact_permutation_p(loc['y_true'], loc['y_pred'], loc['ari'])),
            "p_MinAttainable": fmt_p(min_attainable_p(loc['y_true'], loc['y_pred'])),
            "k_Local":        loc["k_detected"],
            "k_GroundTruth":  loc["k_groundtruth"],
            "Nodes":          loc["n_total"],
            "Matched":        loc["n_matched"],
            "PainNodesInFam": n_pain,
            "PainCCNodes":    n_paincc,
            "Density":        round(g_sub.density(), 4),
        })
        print(f"n_pain={n_pain}  paincc={n_paincc}  ARI={loc['ari']:+.4f}  p={p_val:.4f}")

    summary_df = pd.DataFrame(all_results)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {len(summary_df)} rows → {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=LABEL_CSV,
                        help="phy/nonphy label CSV (default: the manuscript labels)")
    parser.add_argument("--outdir", default=str(MAIN_DIR),
                        help="output directory (default: %(default)s)")
    args = parser.parse_args()
    main(label_csv=args.labels, out_dir=args.outdir)
