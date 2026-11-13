#!/usr/bin/env python3
"""RATER AGREEMENT — S2 step 3 (annotate semantic categories).

Recomputes the inter-rater reliability reported in the Methods from the
individual physical/non-physical labels of the four raters, and checks that
the ground truth read by scripts 01–18 is the majority vote of those ratings.

Inputs
  data/rater_labels.csv            ID, Concept, R1–R4 (individual labels), Final
  data/pain_phy_nonphy_labels.csv  ground-truth labels used by the pipeline

Output
  output/main/rater_agreement.json
    fleiss_kappa, mean_observed_agreement (P-bar), expected_agreement (P-bar_e),
    n_unanimous / pct_unanimous, and the non-unanimous concepts with their
    splits (marked with a dagger in Supplementary Table S1)

Fleiss' kappa (Fleiss 1971), N concepts, n raters, categories j:
  P_i   = (sum_j n_ij^2 - n) / (n (n - 1))   agreement on concept i
  P-bar = mean_i P_i
  p_j   = sum_i n_ij / (N n)                 overall share of category j
  P_e   = sum_j p_j^2
  kappa = (P-bar - P_e) / (1 - P_e)

Values are stored at full precision; the manuscript reports them rounded
(kappa = 0.825, P-bar = 0.923, 84.6% unanimous). Runs in a second and
does not need the network file.
"""
from __future__ import annotations

import json

import pandas as pd

from config import LABEL_CSV, MAIN_DIR, RATER_CSV

CATEGORIES = ["phy", "nonphy"]


def fleiss_kappa(counts: pd.DataFrame) -> dict:
    """Fleiss' kappa from a concepts x categories matrix of rater counts."""
    per_concept = counts.sum(axis=1)
    if per_concept.nunique() != 1:
        raise ValueError("Every concept must be rated by the same number of raters.")
    n = int(per_concept.iloc[0])
    N = len(counts)
    P_i = ((counts ** 2).sum(axis=1) - n) / (n * (n - 1))
    P_bar = float(P_i.mean())
    p_j = counts.sum(axis=0) / (N * n)
    P_e = float((p_j ** 2).sum())
    return {
        "n_concepts": N,
        "n_raters": n,
        "fleiss_kappa": (P_bar - P_e) / (1 - P_e),
        "mean_observed_agreement": P_bar,
        "expected_agreement": P_e,
        "category_proportions": {k: float(v) for k, v in p_j.items()},
    }


def main() -> None:
    df = pd.read_csv(RATER_CSV)
    raters = [c for c in df.columns if c.startswith("R") and c[1:].isdigit()]
    ratings = df.set_index("Concept")[raters]

    unknown = set(pd.unique(ratings.values.ravel())) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unexpected labels in {RATER_CSV.name}: {sorted(unknown)}")

    counts = pd.DataFrame({c: (ratings == c).sum(axis=1) for c in CATEGORIES})
    stats = fleiss_kappa(counts)

    n = len(raters)
    unanimous = counts.max(axis=1) == n
    ties = counts.index[counts["phy"] == counts["nonphy"]].tolist()
    if ties:
        raise ValueError(f"Tied ratings, no majority label: {ties}")
    majority = counts.idxmax(axis=1)

    final = df.set_index("Concept")["Final"]
    not_majority = final.index[final != majority].tolist()
    ground_truth = pd.read_csv(LABEL_CSV).set_index("Concept")["Comm"]
    not_ground_truth = sorted(
        set(final.index) ^ set(ground_truth.index)
        | {c for c in final.index.intersection(ground_truth.index)
           if final[c] != ground_truth[c]}
    )

    split_concepts = [
        {"concept": c, "phy": int(counts.at[c, "phy"]),
         "nonphy": int(counts.at[c, "nonphy"]), "final": final[c]}
        for c in counts.index[~unanimous]
    ]
    stats.update({
        "n_unanimous": int(unanimous.sum()),
        "pct_unanimous": 100 * float(unanimous.mean()),
        "non_unanimous": split_concepts,
        "final_equals_majority_vote": not not_majority,
        "final_equals_ground_truth_file": not not_ground_truth,
    })

    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    out = MAIN_DIR / "rater_agreement.json"
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print(f"Fleiss' kappa = {stats['fleiss_kappa']:.3f} "
          f"({stats['n_concepts']} concepts, {stats['n_raters']} raters)")
    print(f"Mean observed agreement P-bar = {stats['mean_observed_agreement']:.3f}; "
          f"expected P_e = {stats['expected_agreement']:.3f}")
    print(f"Unanimous: {stats['n_unanimous']}/{stats['n_concepts']} "
          f"({stats['pct_unanimous']:.1f}%)")
    for s in split_concepts:
        print(f"  split: {s['concept']:<10s} phy {s['phy']} / nonphy {s['nonphy']}"
              f"  -> {s['final']}")
    if not_majority:
        raise SystemExit(f"Final label is not the majority vote for: {not_majority}")
    if not_ground_truth:
        raise SystemExit(f"Final labels differ from {LABEL_CSV}: {not_ground_truth}")
    print("Final labels = majority vote = data/pain_phy_nonphy_labels.csv: OK")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
