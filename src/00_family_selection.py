#!/usr/bin/env python3
"""FAMILY SELECTION — S2 step 2 (select language families).

The family filtering was carried out by hand before the analysis pipeline was
written. This script re-applies the rules stated in the Methods to the
per-family counts used at the time and checks that they give the 11 families
analyzed by scripts 01–18.

Input
  data/family_selection.csv   93 families with at least one PAIN-related
                              colexification: Glottocode, name, PainEdges (edges
                              of data/clics4-network-3-families.gml attested in
                              the family and linking two of the 26 concepts) and
                              PainNodes (concepts on those edges). These counts
                              were checked against the network file.
  data/major_families.csv     the 11 analyzed families plus Universal (read by 01–18)

Filters (Methods; Supplementary Table S2, step 2)
  1. at least one PAIN-related colexification     -> 93 families (input)
  2. more than one edge                           -> 51 families
  3. edges/nodes above the median of step 2       -> 24 families
     (median = 2/3; reported as > 0.6667)
  4. edges above the median of step 3             -> 11 families
     (median = 6; reported as > 6 internal edges)

Output
  output/main/family_selection.csv   input plus EdgesPerNode, one pass/fail
                                     column per filter, and the last filter
                                     passed (StepReached)

Runs in a second and does not need the network file.
"""
from __future__ import annotations

import pandas as pd

from config import FAMILIES_CSV, FAMILY_SELECTION_CSV, MAIN_DIR


def main() -> None:
    df = pd.read_csv(FAMILY_SELECTION_CSV)
    if df["Family"].duplicated().any():
        raise ValueError("Duplicate Glottocodes in family_selection.csv")
    df["EdgesPerNode"] = df["PainEdges"] / df["PainNodes"]

    step1 = df["PainEdges"] >= 1
    step2 = step1 & (df["PainEdges"] > 1)
    ratio_median = df.loc[step2, "EdgesPerNode"].median()
    step3 = step2 & (df["EdgesPerNode"] > ratio_median)
    edge_median = df.loc[step3, "PainEdges"].median()
    step4 = step3 & (df["PainEdges"] > edge_median)

    df["Pass1_AnyPainEdge"] = step1
    df["Pass2_MoreThanOneEdge"] = step2
    df["Pass3_EdgesPerNodeAboveMedian"] = step3
    df["Pass4_EdgesAboveMedian"] = step4
    df["StepReached"] = (step1.astype(int) + step2.astype(int)
                         + step3.astype(int) + step4.astype(int))

    print(f"Step 1  >= 1 PAIN-related colexification : {int(step1.sum())} families")
    print(f"Step 2  > 1 edge                         : {int(step2.sum())} families")
    print(f"Step 3  edges/nodes > median ({ratio_median:.4f})   : {int(step3.sum())} families")
    print(f"Step 4  edges > median ({edge_median:g})              : {int(step4.sum())} families")

    selected = set(df.loc[step4, "Family"])
    analyzed = set(pd.read_csv(FAMILIES_CSV)["Family"].dropna())  # skip Universal
    if selected != analyzed:
        raise SystemExit(
            "Selected families differ from data/major_families.csv: "
            f"only selected {sorted(selected - analyzed)}, "
            f"only analyzed {sorted(analyzed - selected)}")
    print("Selected families = data/major_families.csv: OK")

    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    out = MAIN_DIR / "family_selection.csv"
    df.sort_values(["StepReached", "PainEdges", "EdgesPerNode", "Family"],
                   ascending=[False, False, False, True]).to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
