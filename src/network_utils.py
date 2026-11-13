"""Network construction and random-walk smoothing for the CLICS4 PAIN pipeline.

Implements Supplementary Table S2 steps 4–8 (network construction →
random-walk transformation → PAIN-concept extraction → PAIN-CC restriction →
community detection):

  subgraph_by_family_edges_clics4  — step 4: family edge filtering + weights
  random_walk_smoothing            — step 5: cumulative random-walk transform
  precompute_family_W_full         — step 5: full smoothed weight matrix
  build_paincc_subgraph            — steps 4–8, PAIN-CC restricted (MAIN)
  build_aligned_pain_subgraph      — steps 4–6 + 8, satellite-inclusive (secondary)
  build_prewalk_pain_subgraph      — raw pain subgraph, no walk (centrality baseline)
  build_nowalk_aligned             — no-walk PAIN-CC baseline (walk-step robustness)
  optimal_communities              — step 8: exact optimal modularity; one
                                     community when the maximum modularity ≤ 0

Function bodies are carried over verbatim from the analysis scripts that
produced the reported results; only import paths were changed.
"""
import igraph as ig
import numpy as np

from config import GLOSS_ATTR, WALK_T

Q_TOL = 1e-9


def optimal_communities(g, weights="weight"):
    """Exact optimal modularity partition (igraph/GLPK).

    When the maximum modularity is ≤ 0 the graph has no community structure,
    and the solver's choice among such partitions is arbitrary; the graph is
    then treated as a single community.
    """
    comms = g.community_optimal_modularity(weights=weights)
    if g.modularity(comms.membership, weights=weights) <= Q_TOL:
        return ig.VertexClustering(g, [0] * g.vcount())
    return comms


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — family edge filtering (CLICS4 `families`/`languages`, // separator)
# ─────────────────────────────────────────────────────────────────────────────

def subgraph_by_family_edges_clics4(g, family="Universal", weight_mode="original"):
    """Keep edges attested in the target family (glottocode); with
    weight_mode="local", re-weight each kept edge by the number of distinct
    languages *within that family* attesting the colexification
    (edge attribute `family_local_weight`)."""
    if family is None or family == "Universal":
        return g

    keep_eids = []
    local_w = []

    for e in g.es:
        fams = (e.attributes().get("families", "") or "")
        if not fams:
            continue
        # CLICS4 uses // as separator
        fam_set = {f.strip() for f in fams.split("//") if f.strip()}
        if family in fam_set:
            keep_eids.append(e.index)

            if weight_mode == "local":
                # `languages` and `families` are parallel //-separated lists
                langs = (e.attributes().get("languages", "") or "").split("//")
                efams = (e.attributes().get("families", "") or "").split("//")

                seen_langs = set()
                for lang, fam in zip(langs, efams):
                    if fam.strip() == family:
                        seen_langs.add(lang.strip())

                w = len(seen_langs) if seen_langs else 1
                local_w.append(int(w))

    if not keep_eids:
        raise ValueError(f"No edges found for family={family}")

    if weight_mode == "local":
        for eid, w in zip(keep_eids, local_w):
            g.es[eid]["family_local_weight"] = w

    g_fam0 = g.subgraph_edges(keep_eids, delete_vertices=False)
    g_fam = g_fam0.subgraph(g_fam0.vs.select(_degree_gt=0))
    return g_fam


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — cumulative random-walk smoothing
# W = 1 − Π_{k=1..T} (1 − P^k), elementwise, P row-stochastic
# ─────────────────────────────────────────────────────────────────────────────

def random_walk_smoothing(adj, steps=WALK_T):
    A = adj.astype(float)
    row_sums = A.sum(axis=1)
    row_sums[row_sums == 0.0] = 1.0
    P = A / row_sums[:, None]

    walk_sums = P.copy()
    walk_dist = P.copy()
    for _ in range(1, steps):
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            walk_dist = walk_dist @ P
        np.nan_to_num(walk_dist, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        walk_sums = walk_dist * (1.0 - walk_sums) + walk_sums
        np.nan_to_num(walk_sums, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return walk_sums


def precompute_family_W_full(g_fam, walk_steps=WALK_T, weight_attr="family_local_weight"):
    """Smoothed, symmetrised, ×100-quantised weight matrix of the FULL family
    graph (the walk runs before any PAIN extraction)."""
    if weight_attr and weight_attr in g_fam.es.attributes():
        adj = np.array(g_fam.get_adjacency(attribute=weight_attr).data, dtype=float)
    else:
        adj = np.array(g_fam.get_adjacency().data, dtype=float)

    W_full = random_walk_smoothing(adj, steps=walk_steps)
    W_full = (W_full + W_full.T) / 2.0
    W_full = np.nan_to_num(W_full, nan=0.0, posinf=0.0, neginf=0.0)
    W_full[W_full < 0] = 0.0
    np.fill_diagonal(W_full, 0.0)
    W_full = np.round(W_full * 100.0)
    return W_full


# ─────────────────────────────────────────────────────────────────────────────
# Steps 4–8 combined — MAIN pipeline graph (PAIN-CC restricted)
# ─────────────────────────────────────────────────────────────────────────────

def build_paincc_subgraph(g_full, family_code, concept_set, walk_steps=WALK_T,
                          detect_communities=True):
    """Family filter → random walk on the full family graph → post-walk PAIN
    extraction → PAIN-connected-component restriction → optimal-modularity
    community detection.

    detect_communities=False skips step 7 (no "community" attribute), for the
    robustness sweeps that detect communities on the graph themselves.

    Returns: (g_sub, n_pain_in_fam, n_paincc) or (None, n_pain_in_fam, 0).
    """
    # 1) family subgraph
    if family_code == "Universal":
        g_fam = g_full.copy()
        weight_attr = "language_count"
    else:
        try:
            g_fam = subgraph_by_family_edges_clics4(
                g_full, family=family_code, weight_mode="local"
            )
        except ValueError:
            return None, 0, 0
        weight_attr = "family_local_weight"

    if g_fam.vcount() == 0:
        return None, 0, 0

    # 2) pain_idx into g_fam
    labels = g_fam.vs[GLOSS_ATTR]
    pain_idx = [i for i, gl in enumerate(labels) if gl in concept_set]
    n_pain = len(pain_idx)
    if n_pain < 3:
        return None, n_pain, 0

    # 3) RW on the full family subgraph
    if weight_attr not in g_fam.es.attributes():
        weight_attr = None
    W_full = precompute_family_W_full(
        g_fam, walk_steps=walk_steps, weight_attr=weight_attr
    )

    # 4) slice pain submatrix
    W_pain = W_full[np.ix_(pain_idx, pain_idx)]
    # W_full is already symmetric and quantised to integers in [0, 100]
    # (see precompute_family_W_full); no further rescaling is applied here.
    W_pain = (W_pain + W_pain.T) / 2
    np.fill_diagonal(W_pain, 0)
    W_pain[~np.isfinite(W_pain)] = 0.0

    # 5) build igraph from W_pain
    g_sub = ig.Graph.Weighted_Adjacency(
        W_pain.tolist(), attr="weight", loops=False
    ).as_undirected(combine_edges="max")
    pain_labels = [labels[i] for i in pain_idx]
    g_sub.vs[GLOSS_ATTR] = pain_labels

    # drop zero-degree pain concepts after smoothing
    deg = g_sub.degree()
    keep = [i for i, d in enumerate(deg) if d > 0]
    if len(keep) < g_sub.vcount():
        g_sub = g_sub.subgraph(keep)

    if g_sub.vcount() < 2 or g_sub.ecount() == 0:
        return None, n_pain, 0

    # 6) restrict to PAIN-CC
    labels_sub = g_sub.vs[GLOSS_ATTR]
    seed_ids = [i for i, gl in enumerate(labels_sub) if gl == "PAIN"]
    if not seed_ids:
        return None, n_pain, 0
    cc_membership = g_sub.connected_components().membership
    primary_cc = cc_membership[seed_ids[0]]
    primary_ids = [i for i, m in enumerate(cc_membership) if m == primary_cc]
    g_sub = g_sub.subgraph(primary_ids)

    # families whose PAIN-CC has fewer than three concepts are excluded (Methods)
    if g_sub.vcount() < 3 or g_sub.ecount() == 0:
        return None, n_pain, g_sub.vcount()

    n_paincc = g_sub.vcount()

    # 7) community detection (exact optimal modularity — main analysis)
    if detect_communities:
        comms = optimal_communities(g_sub, weights="weight")
        g_sub.vs["community"] = comms.membership

    return g_sub, n_pain, n_paincc


# ─────────────────────────────────────────────────────────────────────────────
# Satellite-inclusive variant — secondary analysis
# ─────────────────────────────────────────────────────────────────────────────

def build_aligned_pain_subgraph(g_full, family_code, concept_set, walk_steps=WALK_T):
    """Same as build_paincc_subgraph but WITHOUT the PAIN-CC restriction:
    disconnected satellite components of the post-walk pain graph are retained.

    Returns: (g_sub, n_pain_in_fam) or (None, n_pain_in_fam).
    """
    # 1) family subgraph (drop degree-0 nodes)
    if family_code == "Universal":
        g_fam = g_full.copy()
        weight_attr = "language_count"
    else:
        try:
            g_fam = subgraph_by_family_edges_clics4(
                g_full, family=family_code, weight_mode="local"
            )
        except ValueError:
            return None, 0
        weight_attr = "family_local_weight"

    if g_fam.vcount() == 0:
        return None, 0

    # 2) pain_idx into g_fam
    labels = g_fam.vs[GLOSS_ATTR]
    pain_idx = [i for i, gl in enumerate(labels) if gl in concept_set]
    n_pain = len(pain_idx)
    if n_pain < 3:
        return None, n_pain

    # 3) RW on the full family subgraph
    if weight_attr not in g_fam.es.attributes():
        weight_attr = None  # fall back to unweighted
    W_full = precompute_family_W_full(
        g_fam, walk_steps=walk_steps, weight_attr=weight_attr
    )

    # 4) slice pain submatrix
    W_pain = W_full[np.ix_(pain_idx, pain_idx)]
    W_pain = (W_pain + W_pain.T) / 2
    np.fill_diagonal(W_pain, 0)
    # W_full is already quantised to integers in [0, 100]; no further rescaling.
    W_pain[~np.isfinite(W_pain)] = 0.0

    # 5) build igraph from W_pain
    g_sub = ig.Graph.Weighted_Adjacency(
        W_pain.tolist(), attr="weight", loops=False
    ).as_undirected(combine_edges="max")
    pain_labels = [labels[i] for i in pain_idx]
    g_sub.vs[GLOSS_ATTR] = pain_labels

    # drop zero-degree pain concepts after smoothing
    deg = g_sub.degree()
    keep = [i for i, d in enumerate(deg) if d > 0]
    if len(keep) < g_sub.vcount():
        g_sub = g_sub.subgraph(keep)

    if g_sub.vcount() < 2 or g_sub.ecount() == 0:
        return None, n_pain

    # 6) community detection (exact optimal modularity)
    comms = optimal_communities(g_sub, weights="weight")
    g_sub.vs["community"] = comms.membership

    return g_sub, n_pain


# ─────────────────────────────────────────────────────────────────────────────
# No-walk baselines
# ─────────────────────────────────────────────────────────────────────────────

def build_prewalk_pain_subgraph(g_full, family_code, concept_set):
    """Raw pain-induced subgraph in family. No random walk. Isolates dropped.
    Used for the pre-walk centrality baselines."""
    if family_code == "Universal":
        g_fam = g_full.copy()
        src_attrs = ["language_count", "weight"]
    else:
        try:
            g_fam = subgraph_by_family_edges_clics4(
                g_full, family=family_code, weight_mode="local"
            )
        except ValueError:
            return None
        src_attrs = ["family_local_weight", "weight"]

    if g_fam.vcount() == 0:
        return None

    labels = g_fam.vs[GLOSS_ATTR]
    pain_idx = [i for i, gl in enumerate(labels) if gl in concept_set]
    if len(pain_idx) < 2:
        return None

    g = g_fam.subgraph(pain_idx)
    attrs = g.es.attributes()
    src = next((a for a in src_attrs if a in attrs), None)
    if src is not None:
        g.es["weight"] = [float(w) if w else 0.0 for w in g.es[src]]
    else:
        g.es["weight"] = [1.0] * g.ecount()

    nz = [e.index for e in g.es if (e["weight"] or 0.0) > 0.0]
    if len(nz) < g.ecount():
        g = g.subgraph_edges(nz, delete_vertices=True)

    iso = [i for i, d in enumerate(g.degree()) if d == 0]
    if iso:
        g.delete_vertices(iso)

    if g.vcount() < 2 or g.ecount() == 0:
        return None
    return g


def build_nowalk_aligned(g_full, family_code, concept_set,
                         min_nodes=4, min_edges=3):
    """No-walk PAIN-CC baseline for the walk-step robustness sweep: induced
    pain subgraph from the family graph (family_local_weight, or
    language_count for Universal), no random walk, restricted to the connected
    component containing PAIN."""
    if family_code == "Universal":
        g_fam = g_full.copy()
        src_attrs = ["language_count", "weight"]
    else:
        try:
            g_fam = subgraph_by_family_edges_clics4(
                g_full, family=family_code, weight_mode="local"
            )
        except ValueError:
            return None
        src_attrs = ["family_local_weight", "weight"]

    if g_fam.vcount() == 0:
        return None

    labels = g_fam.vs[GLOSS_ATTR]
    pain_idx = [i for i, gl in enumerate(labels) if gl in concept_set]
    if len(pain_idx) < min_nodes:
        return None
    g = g_fam.subgraph(pain_idx)

    attrs = g.es.attributes()
    src = next((a for a in src_attrs if a in attrs), None)
    if src is not None:
        g.es["weight"] = [float(w) if w else 0.0 for w in g.es[src]]
    else:
        g.es["weight"] = [1.0] * g.ecount()

    nz = [e.index for e in g.es if (e["weight"] or 0.0) > 0.0]
    if len(nz) < g.ecount():
        g = g.subgraph_edges(nz, delete_vertices=True)

    iso = [i for i, d in enumerate(g.degree()) if d == 0]
    if iso:
        g.delete_vertices(iso)

    if g.vcount() == 0 or g.ecount() == 0:
        return None

    # restrict to the connected component containing PAIN (mirrors
    # build_paincc_subgraph step 6).  Families without a PAIN vertex are
    # skipped rather than falling back to the largest component.
    labels_sub = g.vs[GLOSS_ATTR]
    seed_ids = [i for i, gl in enumerate(labels_sub) if gl == "PAIN"]
    if not seed_ids:
        return None
    cc_membership = g.connected_components().membership
    primary_cc = cc_membership[seed_ids[0]]
    g = g.subgraph([i for i, m in enumerate(cc_membership) if m == primary_cc])

    if g.vcount() < min_nodes or g.ecount() < min_edges:
        return None
    return g
