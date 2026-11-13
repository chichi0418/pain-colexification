"""ARI evaluation, permutation testing, and per-family visualisation.

Implements Supplementary Table S2 steps 9–11:

  calculate_ari_for_subgraph      — step 9:  ARI vs. phy/nonphy ground truth,
                                    restricted to concepts present in both
  permutation_test                — step 10: 1,000-shuffle one-sided label-
                                    permutation null (NumPy RNG, seed 42)
  exact_permutation_p             — step 10: the same null enumerated exactly
                                    (reported p-value)
  min_attainable_p                — step 11: exact permutation floor
  visualize_local_with_permutation — per-family diagnostic panel (detected
                                    partition / ground truth / null histogram)

Function bodies are carried over verbatim from the analysis scripts that
produced the reported results; only import paths were changed.
"""
import random
from pathlib import Path

import igraph as ig
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no window popup
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from config import N_PERM, SEED
from network_utils import optimal_communities

_LAYOUT_SEED = 42

_GROUND_TRUTH_COLORS = {
    0: "#70F3FF",
    1: "#FFA400",
    2: "#FF3300",
    3: "#7B68EE",
    -1: "#BBBBBB",
}


def _load_label_df(label_csv_path):
    df = pd.read_csv(label_csv_path, sep=None, engine="python")
    df.columns = [c.lstrip("﻿") for c in df.columns]
    if "Gloss" not in df.columns and "Concept" in df.columns:
        df = df.rename(columns={"Concept": "Gloss"})
    df = df.dropna(subset=["Comm"]).reset_index(drop=True)
    return df


def calculate_ari_for_subgraph(g_sub, label_csv_path, gloss_attr="Gloss"):
    if g_sub is None or g_sub.vcount() == 0:
        return None

    label_df = _load_label_df(label_csv_path)
    comm_to_id = {c: i for i, c in enumerate(sorted(label_df["Comm"].unique()))}
    label_df["label_id"] = label_df["Comm"].map(comm_to_id)

    if "community" not in g_sub.vs.attributes():
        comms = optimal_communities(g_sub, weights="weight")
        g_sub.vs["community"] = comms.membership

    detected_comms = g_sub.vs["community"]

    glosses = g_sub.vs[gloss_attr]
    y_true, y_pred, matched_glosses = [], [], []

    for idx, gloss in enumerate(glosses):
        rows = label_df[label_df["Gloss"] == gloss]
        if len(rows) > 0:
            y_true.append(int(rows["label_id"].iloc[0]))
            y_pred.append(int(detected_comms[idx]))
            matched_glosses.append(gloss)

    if len(y_true) == 0:
        return None

    k_detected = len(set(y_pred))  # communities among matched nodes only
    ari = adjusted_rand_score(y_true, y_pred)

    comparison_df = pd.DataFrame({
        'Gloss': matched_glosses,
        'GroundTruth_Comm': [label_df[label_df["Gloss"] == g]["Comm"].iloc[0] for g in matched_glosses],
        'GroundTruth_ID': y_true,
        'Detected_Cluster': y_pred
    })

    return {
        'ari': ari,
        'k_detected': k_detected,
        'k_groundtruth': len(comm_to_id),
        'n_matched': len(y_true),
        'n_total': len(glosses),
        'comparison_df': comparison_df,
        'y_true': y_true,
        'y_pred': y_pred,
        'comm_to_id': comm_to_id
    }


def permutation_test(y_true, y_pred, obs_ari, n_perm=N_PERM, seed=SEED):
    """One-sided label-shuffling null: p = P(ARI_perm >= ARI_obs)."""
    rng = np.random.default_rng(seed)
    y_true_arr = np.array(y_true)
    perm_aris = [
        adjusted_rand_score(rng.permutation(y_true_arr), y_pred)
        for _ in range(n_perm)
    ]
    p_val = float(np.mean(np.array(perm_aris) >= obs_ari))
    return p_val, perm_aris


def _ari_from_contingency(table):
    """ARI from a contingency table (rows = ground-truth classes, cols = clusters)."""
    from math import comb
    n = sum(sum(r) for r in table)
    sum_ij = sum(comb(x, 2) for r in table for x in r)
    sum_a = sum(comb(sum(r), 2) for r in table)
    sum_b = sum(comb(sum(col), 2) for col in zip(*table))
    expected = sum_a * sum_b / comb(n, 2)
    max_idx = (sum_a + sum_b) / 2
    if max_idx == expected:
        return 1.0 if sum_ij == expected else 0.0
    return (sum_ij - expected) / (max_idx - expected)


def _exact_null(y_true, y_pred):
    """Exact label-permutation null of ARI as (aris, weights, total).

    Holding the detected partition (y_pred) and the class sizes of the binary
    ground truth fixed, every one of the C(n, n1) equally likely label
    assignments under the null is enumerated through its contingency table.
    """
    from itertools import product
    from math import comb, prod
    classes = sorted(set(y_true))
    if len(classes) != 2:
        raise ValueError("exact permutation null assumes a binary ground truth")
    n = len(y_true)
    n1 = sum(1 for y in y_true if y == classes[0])
    sizes = [sum(1 for c in y_pred if c == k) for k in sorted(set(y_pred))]

    aris, weights = [], []
    for counts in product(*(range(s + 1) for s in sizes)):
        if sum(counts) != n1:
            continue
        table = [list(counts), [s - c for s, c in zip(sizes, counts)]]
        aris.append(_ari_from_contingency(table))
        weights.append(prod(comb(s, c) for s, c in zip(sizes, counts)))
    return aris, weights, comb(n, n1)


def min_attainable_p(y_true, y_pred):
    """Smallest p-value the label-permutation test can attain, by exact enumeration.

    The minimum attainable p is the null probability of reaching the largest
    ARI any assignment can produce, i.e. the p-value obtained if the observed
    labels aligned as well as the detected communities allow.
    """
    aris, weights, total = _exact_null(y_true, y_pred)
    best = max(aris)
    hits = sum(w for a, w in zip(aris, weights) if a >= best - 1e-12)
    return hits / total


def exact_permutation_p(y_true, y_pred, obs_ari):
    """Exact one-sided p = P(ARI_perm >= ARI_obs) under the label-permutation null.

    Same null as permutation_test, enumerated rather than sampled, so it has no
    Monte Carlo error and can never fall below min_attainable_p (it equals it
    when the observed ARI is the maximum attainable ARI).
    """
    aris, weights, total = _exact_null(y_true, y_pred)
    hits = sum(w for a, w in zip(aris, weights) if a >= obs_ari - 1e-12)
    return hits / total


def visualize_local_with_permutation(
    g_local,
    ari_result,
    perm_aris,
    label_csv_path,
    family_name="",
    gloss_attr="Gloss",
    save_dir=None,
    p_val=None
):
    if g_local is None or ari_result is None:
        return

    ig.set_random_number_generator(random.Random(_LAYOUT_SEED))
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    layout = g_local.layout_fruchterman_reingold(weights="weight", niter=50)
    strengths = g_local.strength(weights="weight")
    s_min, s_max = min(strengths), max(strengths)
    sizes = [30] * len(strengths) if s_max == s_min else \
            [20 + (s - s_min) / (s_max - s_min) * 20 for s in strengths]

    # Edge widths normalised to max weight so the visual scales regardless of
    # the absolute weight range.
    _ws = g_local.es["weight"]
    _w_max = max(_ws) if _ws else 1.0
    edge_widths = [0.4 + 4.6 * (w / _w_max) for w in _ws]

    glosses = g_local.vs[gloss_attr]
    detected = g_local.vs["community"]
    k = len(set(detected))

    cmap = plt.get_cmap("tab20c")
    palette = [cmap(i / max(1, k)) for i in range(k)]
    vcol_det = [palette[c] for c in detected]
    ecol_det = [vcol_det[e.tuple[0]] for e in g_local.es]

    ig.plot(
        g_local, target=axes[0], layout=layout,
        vertex_color=vcol_det, vertex_size=sizes,
        vertex_frame_width=0.5, vertex_frame_color="black",
        vertex_label=glosses, vertex_label_size=10,
        edge_width=edge_widths,
        edge_color=ecol_det, margin=50, edge_curved=0.2
    )
    axes[0].set_title(
        f"{family_name} LOCAL — Detected (k={k}, ARI={ari_result['ari']:.3f})",
        fontsize=12, fontweight='bold'
    )

    label_df = _load_label_df(label_csv_path)
    comm_to_id = ari_result['comm_to_id']
    label_df["label_id"] = label_df["Comm"].map(comm_to_id)
    three_colors_map = _GROUND_TRUTH_COLORS

    true_labels = []
    for gloss in glosses:
        rows = label_df[label_df["Gloss"] == gloss]
        true_labels.append(-1 if len(rows) == 0 else int(rows["label_id"].iloc[0]))

    vcol_truth = [three_colors_map.get(c, "#BBBBBB") for c in true_labels]
    ecol_truth = [vcol_truth[e.tuple[0]] for e in g_local.es]

    ig.plot(
        g_local, target=axes[1], layout=layout,
        vertex_color=vcol_truth, vertex_size=sizes,
        vertex_frame_width=0.5, vertex_frame_color="black",
        vertex_label=glosses, vertex_label_size=10,
        edge_width=edge_widths,
        edge_color=ecol_truth, margin=50, edge_curved=0.2
    )
    axes[1].set_title(
        f"{family_name} LOCAL — Ground Truth",
        fontsize=12, fontweight='bold'
    )

    obs = ari_result['ari']
    if p_val is None:
        p_val = float(np.mean(np.array(perm_aris) >= obs))
    axes[2].hist(perm_aris, bins=40, color="steelblue", alpha=0.7, edgecolor="white")
    axes[2].axvline(obs, color="red", linewidth=2, label=f"Observed ARI = {obs:.3f}")
    axes[2].set_xlabel("ARI (permuted)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"Permutation Test (n=1000)\np = {p_val:.3f}", fontsize=12)
    axes[2].legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(
            Path(save_dir) / f"{family_name.replace(' ', '_')}_ari_comparison.png",
            dpi=150, bbox_inches="tight"
        )
    plt.close(fig)


def exact_test(y_true, y_pred, obs_ari):
    """Exact p, minimum attainable p and maximum attainable ARI in one pass.

    A single-class ground truth (possible after thresholding drops every
    concept of one class) makes every relabelling identical, so p = p_min = 1.
    """
    if len(set(y_true)) < 2:
        return {"p_exact": 1.0, "p_min": 1.0, "ari_max": obs_ari}
    aris, weights, total = _exact_null(y_true, y_pred)
    best = max(aris)
    return {
        "p_exact": sum(w for a, w in zip(aris, weights) if a >= obs_ari - 1e-12) / total,
        "p_min":   sum(w for a, w in zip(aris, weights) if a >= best - 1e-12) / total,
        "ari_max": best,
    }


def fmt_p(p):
    """4 decimals; 3 significant figures below .001, where 4 decimals would
    print 0."""
    return round(p, 4) if p >= 1e-3 else float(f"{p:.3g}")
