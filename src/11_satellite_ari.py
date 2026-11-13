#!/usr/bin/env python3
"""SECONDARY ANALYSIS: SATELLITE-INCLUSIVE ARI (supplementary satellite table).

Identical to the main analysis (script 01) except that the post-walk pain graph
is NOT restricted to the PAIN-connected component: disconnected satellite
components are retained and enter community detection. This is the
satellite-inclusive analysis reported as a supplementary comparison — it is
less conservative, because internally homogeneous satellites can raise ARI
without being semantically connected to PAIN.

Outputs (output/satellite/): same file layout as script 01
  (summary.csv + families/{Family}/ detail, permutations, diagnostic panel).

Usage:
    python src/11_satellite_ari.py
    python src/11_satellite_ari.py --labels LABELS.csv --outdir DIR   # label ablation (script 16)
"""
import argparse
import os
from pathlib import Path

import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"

from config import (
    LABEL_CSV, SATELLITE_DIR, WALK_T, N_PERM, SEED, GLOSS_ATTR,
    load_graph, load_concept_set, load_families,
)
from network_utils import build_aligned_pain_subgraph
from ari_utils import (
    calculate_ari_for_subgraph, permutation_test, visualize_local_with_permutation,
)


def main(label_csv=LABEL_CSV, out_dir=SATELLITE_DIR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    concept_set = load_concept_set()
    families = load_families()

    g_full = load_graph()

    all_results = []
    for i, (family_name, family_code) in enumerate(families, 1):
        print(f"[{i}/{len(families)}] {family_name} ({family_code}) ...", end=" ", flush=True)

        try:
            g_sub, n_pain = build_aligned_pain_subgraph(
                g_full, family_code, concept_set, walk_steps=WALK_T
            )
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()
            continue

        if g_sub is None:
            print(f"skipped (n_pain={n_pain}, no valid g_sub)")
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
            "Family":        family_name,
            "ARI_Local":     loc["ari"],
            "p_Local":       max(p_val, 1 / N_PERM),
            "k_Local":       loc["k_detected"],
            "k_GroundTruth": loc["k_groundtruth"],
            "Nodes":         loc["n_total"],
            "Matched":       loc["n_matched"],
            "PainNodesInFam": n_pain,
        })
        print(f"n_pain={n_pain}  ARI={loc['ari']:+.4f}  p={p_val:.4f}")

    summary_df = pd.DataFrame(all_results)
    summary_path = out_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved {len(summary_df)} rows → {summary_path}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labels", default=LABEL_CSV,
                        help="phy/nonphy label CSV (default: the manuscript labels)")
    parser.add_argument("--outdir", default=str(SATELLITE_DIR),
                        help="output directory (default: %(default)s)")
    args = parser.parse_args()
    main(label_csv=args.labels, out_dir=args.outdir)
