#!/usr/bin/env python3
"""CONCEPT SET — S2 step 1 (define the PAIN concept set).

Rebuilds the 26-concept domain from the CLICS4 network restricted to links
attested in at least three language families, and checks that it equals the
concept list read by scripts 01–18.

  1. Access points (Methods): PAIN, ACHE, SORE, HURT. Only PAIN is a node of
     the three-family network; the others have no link attested in three or
     more families.
  2. Domain: every concept within colexification distance ≤ 2 of PAIN,
     PAIN included.

Inputs
  data/clics4-network-3-families.gml(.gz)   CLICS4 network, ≥ 3-family links
  data/pain_concepts.tsv                    concept list used by the pipeline

Output
  output/main/pain_concepts.tsv   ID / Concept, same format as the input list

Usage:
    python src/00_concept_set.py
"""
import igraph as ig
import pandas as pd

from config import CONCEPT_PATH, GLOSS_ATTR, MAIN_DIR, load_graph, load_concept_list

SEED_CONCEPT = "PAIN"
ACCESS_POINTS = ["PAIN", "ACHE", "SORE", "HURT"]


def get_depth2_concepts(g: ig.Graph, seed_gloss: str) -> list[str]:
    """Return sorted glosses reachable from seed within depth 2, including seed itself."""
    nodes = [v.index for v in g.vs if v[GLOSS_ATTR] == seed_gloss]
    if not nodes:
        raise ValueError(f"Node '{seed_gloss}' not found in network")
    seed_idx = nodes[0]

    depth1 = set(g.neighbors(seed_idx))
    depth2 = set()
    for n in depth1:
        depth2.update(g.neighbors(n))

    reachable = (depth1 | depth2) | {seed_idx}
    return sorted(g.vs[i][GLOSS_ATTR] for i in reachable)


def main() -> None:
    g = load_graph()
    labels = set(g.vs[GLOSS_ATTR])
    for gloss in ACCESS_POINTS:
        print(f"  {gloss:5s} in three-family network: {'yes' if gloss in labels else 'no'}")

    concepts = get_depth2_concepts(g, SEED_CONCEPT)
    print(f"Concepts within distance 2 of {SEED_CONCEPT}: {len(concepts)}")

    MAIN_DIR.mkdir(parents=True, exist_ok=True)
    out = MAIN_DIR / "pain_concepts.tsv"
    pd.DataFrame({"ID": range(1, len(concepts) + 1), "Concept": concepts}) \
        .to_csv(out, sep="\t", index=False)
    print(f"Saved: {out}")

    expected = load_concept_list()
    if concepts != expected:
        raise SystemExit(
            f"Concept set differs from {CONCEPT_PATH}: "
            f"only rebuilt {sorted(set(concepts) - set(expected))}, "
            f"only in file {sorted(set(expected) - set(concepts))}")
    print("Concept set = data/pain_concepts.tsv: OK")


if __name__ == "__main__":
    main()
