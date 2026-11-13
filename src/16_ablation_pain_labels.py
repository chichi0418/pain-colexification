#!/usr/bin/env python3
"""ABLATION: sensitivity of ARI to the PAIN / PAINFUL phy-nonphy labels.

The phy/nonphy ground truth is a manual annotation, and PAIN and PAINFUL are
the two concepts whose labels are most open to dispute (PAINFUL is defined as
causing distress "either physical or mental"). This script reruns the main
PAIN-CC analysis (script 01) and the satellite-inclusive analysis (script 11)
under all four PAIN × PAINFUL labelings, leaving every other concept's label,
the networks, community detection, and the permutation seed unchanged:

  A0_pain-phy_painful-nonphy      manuscript labels (baseline)
  A1_pain-phy_painful-phy
  A2_pain-nonphy_painful-phy      both flipped
  A3_pain-nonphy_painful-nonphy

Detected communities do not depend on the labels, so any difference between
conditions comes from the ground truth alone. A family whose network lacks
PAINFUL gives identical results under the two conditions that differ only in
PAINFUL (A0/A1 and A2/A3); these are flagged in the comparison.

Inputs:  data/labels/ablation/{condition}.csv
Outputs (output/ablation/):
  {condition}/main/       — script 01 outputs for that labeling
  {condition}/satellite/  — script 11 outputs for that labeling
  comparison_main.csv, comparison_satellite.csv — family × condition table
  SUMMARY.md              — comparison tables + significant-family counts

Usage:
    python src/16_ablation_pain_labels.py              # run all + compare
    python src/16_ablation_pain_labels.py --compare-only
"""
import argparse
import importlib
import os
from pathlib import Path

import pandas as pd

os.environ["PYTHONWARNINGS"] = "ignore"

from config import DATA_DIR, OUTPUT_DIR, N_PERM, SEED

ABLATION_LABEL_DIR = DATA_DIR / "labels" / "ablation"
ABLATION_DIR = OUTPUT_DIR / "ablation"
ALPHA = 0.05

CONDITIONS = [
    ("A0_pain-phy_painful-nonphy",    "phy",    "nonphy"),
    ("A1_pain-phy_painful-phy",       "phy",    "phy"),
    ("A2_pain-nonphy_painful-phy",    "nonphy", "phy"),
    ("A3_pain-nonphy_painful-nonphy", "nonphy", "nonphy"),
]
BASELINE = CONDITIONS[0][0]

# analysis name → script module (run with --labels/--outdir equivalents)
ANALYSES = {
    "main":      "01_main_ari_paincc",
    "satellite": "11_satellite_ari",
}


def run_all():
    for analysis, module_name in ANALYSES.items():
        script = importlib.import_module(module_name)
        for cond, _, _ in CONDITIONS:
            print(f"\n=== {analysis} · {cond} ===")
            script.main(
                label_csv=str(ABLATION_LABEL_DIR / f"{cond}.csv"),
                out_dir=ABLATION_DIR / cond / analysis,
            )


def has_painful(cond_dir: Path, family: str) -> bool:
    """Whether PAINFUL is among the matched concepts of this family's network."""
    detail = cond_dir / "families" / family / f"{family}_local_detail.csv"
    df = pd.read_csv(detail)
    return "PAINFUL" in set(df.iloc[:, 0])


def compare(analysis: str) -> pd.DataFrame:
    rows = []
    for cond, pain, painful in CONDITIONS:
        cond_dir = ABLATION_DIR / cond / analysis
        s = pd.read_csv(cond_dir / "summary.csv")
        for _, r in s.iterrows():
            rows.append({
                "Family": r["Family"],
                "Condition": cond,
                "PAIN": pain,
                "PAINFUL": painful,
                "ARI": r["ARI_Local"],
                "p": r["p_Local"],
                "p_Exact": r.get("p_Exact", float("nan")),
                "p_MinAttainable": r.get("p_MinAttainable", float("nan")),
                "k": r["k_Local"],
                "Matched": r["Matched"],
                "Significant": r["p_Local"] < ALPHA,
                "HasPAINFUL": has_painful(cond_dir, r["Family"]),
            })
    long = pd.DataFrame(rows)
    base = long[long.Condition == BASELINE].set_index("Family")["Significant"]
    long["SigChangedVsBaseline"] = long.apply(
        lambda r: r["Significant"] != base[r["Family"]], axis=1
    )
    long.to_csv(ABLATION_DIR / f"comparison_{analysis}.csv", index=False)
    return long


def fmt_cell(r) -> str:
    star = "*" if r["Significant"] else ""
    flag = " †" if r["SigChangedVsBaseline"] else ""
    return f"{r['ARI']:+.3f} (p={r['p']:.3f}){star}{flag}"


def markdown_section(analysis: str, long: pd.DataFrame) -> str:
    families = list(dict.fromkeys(long["Family"]))
    conds = [c for c, _, _ in CONDITIONS]
    short = {c: c.split("_")[0] for c in conds}

    lines = [
        f"### {analysis}",
        "",
        "| Family | Matched | PAINFUL in network | "
        + " | ".join(f"{short[c]}" for c in conds) + " |",
        "|---|---|---|" + "---|" * len(conds),
    ]
    for fam in families:
        sub = long[long.Family == fam].set_index("Condition")
        cells = [fmt_cell(sub.loc[c]) for c in conds]
        lines.append(
            f"| {fam} | {sub.loc[BASELINE, 'Matched']} | "
            f"{'yes' if sub.loc[BASELINE, 'HasPAINFUL'] else 'no'} | "
            + " | ".join(cells) + " |"
        )
    counts = long.groupby("Condition")["Significant"].sum()
    n_fam = len(families)
    lines.append(
        f"| **Significant (p < {ALPHA})** | | | "
        + " | ".join(f"**{int(counts[c])}/{n_fam}**" for c in conds) + " |"
    )

    lines += [""]
    lacking = [f for f in families
               if not long[(long.Family == f) & (long.Condition == BASELINE)]["HasPAINFUL"].iloc[0]]
    if lacking:
        lines.append(
            "PAINFUL is absent from the network of: " + ", ".join(lacking)
            + ". For these families A0 = A1 and A2 = A3 by construction."
        )
        lines.append("")
    changed = long[long.SigChangedVsBaseline]
    if changed.empty:
        lines.append("No family changes significance relative to the baseline (A0).")
    else:
        lines += ["Significance changes relative to the baseline (A0):", ""]
        for _, r in changed.iterrows():
            direction = "becomes significant" if r["Significant"] else "loses significance"
            lines.append(f"- {r['Family']} · {r['Condition']}: {direction} "
                         f"(ARI {r['ARI']:+.3f}, p = {r['p']:.3f})")
    lines.append("")
    return "\n".join(lines)


def write_summary(results: dict):
    parts = [
        "# PAIN / PAINFUL label ablation",
        "",
        "Sensitivity of community–label alignment (ARI) to the manually assigned "
        "phy/nonphy labels of PAIN and PAINFUL. All other labels, the networks, "
        f"community detection, and the permutation test ({N_PERM} permutations, "
        f"seed {SEED}) are identical across conditions.",
        "",
        "| Condition | PAIN | PAINFUL | Note |",
        "|---|---|---|---|",
    ]
    notes = {BASELINE: "manuscript labels (baseline)"}
    for cond, pain, painful in CONDITIONS:
        parts.append(f"| {cond} | {pain} | {painful} | {notes.get(cond, '')} |")
    parts += [
        "",
        "## Method",
        "",
        "**Rationale.** The phy/nonphy ground truth is a manual annotation. PAIN "
        "and PAINFUL are the concepts whose labels are most open to dispute: PAIN "
        "is the seed concept of the whole domain, and PAINFUL is defined as causing "
        "distress \"either physical or mental\". The ablation asks whether the "
        "manuscript's conclusions depend on how these two concepts are labelled.",
        "",
        "**Design.** A 2 × 2 crossing of PAIN ∈ {phy, nonphy} and PAINFUL ∈ "
        "{phy, nonphy}. Each condition is a copy of the manuscript label file "
        "(`data/pain_phy_nonphy_labels.csv`) with only these two rows changed "
        "(`data/labels/ablation/{condition}.csv`); the other 24 labels are "
        "untouched. The phy/nonphy class sizes are therefore 17/9 (A0, A2), "
        "18/8 (A1) and 16/10 (A3).",
        "",
        "**Pipeline.** Unchanged from the manuscript. For each of the 11 families "
        "and the Universal network, script 01 (primary) builds the family network, "
        "applies random-walk smoothing (T = 2) on the full family network, extracts "
        "the 26 PAIN concepts, restricts to the PAIN-connected component (PAIN-CC), "
        "and detects communities with exact optimal modularity. Script 11 "
        "(secondary) is identical but keeps satellite components disconnected from "
        "PAIN. Because networks and community detection do not use the labels, the "
        "detected communities are the same in all four conditions; only the ground "
        "truth they are compared with changes.",
        "",
        "**Evaluation.** ARI between detected communities and the phy/nonphy "
        "labels over the concepts present in both. Significance by a one-sided "
        f"label-permutation test ({N_PERM} permutations, NumPy seed {SEED}, "
        f"p = P(ARI_perm ≥ ARI_obs), floored at {1 / N_PERM:g}), with the "
        f"same seed in every condition; significant = p < {ALPHA}. No "
        "multiple-comparison correction, matching the manuscript.",
        "",
        "**Reading the results.**",
        "",
        "- Where PAINFUL is absent from a family's network, A0 = A1 and A2 = A3 "
        "by construction; only the PAIN label can matter there.",
        "- Several families have few matched concepts (6–8), so a single label "
        "change moves ARI substantially and their minimum attainable p can be "
        "close to or above 0.05 (see `p_MinAttainable` in "
        "`comparison_main.csv`).",
        "- Significance changes with p close to 0.05 should be read as "
        "borderline, not as a qualitative shift.",
        "- Supplementary Tables S13–S14 report these Monte Carlo p-values, except "
        "where the labelled network is identical to the primary analysis (A0, and "
        "A1 without PAINFUL, in the PAIN-CC analysis; Universal A0 in the "
        "satellite-inclusive analysis), where the exact p from Table 3 "
        "(`p_Exact` in `comparison_main.csv`) is reported instead.",
        "",
        "**Reproduce.** `python src/16_ablation_pain_labels.py` (about 1 min); "
        "`--compare-only` rebuilds the tables from existing outputs. A0 "
        "reproduces `expected_output/main/summary.csv` and "
        "`expected_output/satellite/summary.csv` byte for byte.",
        "",
        "## Results",
        "",
        "Cells: ARI (permutation p). `*` p < 0.05; `†` significance differs from A0.",
        "`main` = PAIN-CC analysis (script 01); `satellite` = satellite-inclusive "
        "analysis (script 11).",
        "",
    ]
    for analysis, long in results.items():
        parts.append(markdown_section(analysis, long))
    path = ABLATION_DIR / "SUMMARY.md"
    path.write_text("\n".join(parts))
    print(f"\nSaved → {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compare-only", action="store_true",
                        help="skip the reruns; rebuild the comparison from existing outputs")
    args = parser.parse_args()

    if not args.compare_only:
        run_all()
    results = {analysis: compare(analysis) for analysis in ANALYSES}
    write_summary(results)


if __name__ == "__main__":
    main()
