#!/usr/bin/env python3
"""ROBUSTNESS: THRESHOLD & TRANSFORM — Supplementary Table S2 step 16.

Tests sensitivity of the PAIN-CC ARI result to network-weight construction:
12 families × 10 conditions (six binary thresholds t ∈ {5, 10, 20, 30, 40, 50}
and four weight transforms: log1p, square-root, rank with tied weights sharing
their average rank, and min–max) on the post-walk PAIN-CC graph (T = 2). Each
variant is re-restricted to the PAIN-connected component (edges with weight
≤ 0 removed first); a variant is not evaluable when PAIN is removed, fewer
than 4 nodes or 3 edges remain, or only one label class remains. Communities
are detected with --algorithm:
  optimal (default) — exact optimal modularity, as in the main analysis
  leiden            — modularity objective, 10 iterations, resolution = 1
                      default, seed 42 (the original exploratory sweep)
Each fixed graph is partitioned once, in a worker process with a
per-condition time limit (--timeout); a timeout or error is recorded as such,
never replaced by another algorithm. Each condition gets an exact
label-permutation ARI test (p_val; the 1,000-draw Monte Carlo p is kept as
p_mc); the label permutations never re-run community detection.

Outputs (output/robustness/{algorithm}/threshold/):
  summary.csv            one row per evaluable condition (original columns)
  conditions.jsonl       per-condition log: status, Q, membership, timings
                         (re-runs reuse its completed conditions; --fresh resets)
  conditions.csv         the same without memberships
  failures.csv           timeouts / errors
  run_metadata.json      algorithm parameters, versions, command, timing totals
  {Family}/{Family}_ari_curve.png

Usage:
    python src/07_robustness_threshold.py
    python src/07_robustness_threshold.py --algorithm leiden --fresh
    python src/07_robustness_threshold.py --families Universal Indo-European --n-perm 100
"""

import io
import sys
import time
import contextlib
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    LABEL_CSV, ROBUSTNESS_DIR, WALK_T, GLOSS_ATTR,
    load_graph, load_concept_set, load_families,
)
from network_utils import build_paincc_subgraph
from ari_utils import calculate_ari_for_subgraph, exact_test, fmt_p, permutation_test
from robustness_utils import (
    ALGORITHMS,
    DEFAULT_TIMEOUT_S,
    ConditionStore,
    TimedDetector,
    apply_threshold_binary,
    apply_weight_transform,
    compute_graph_stats,
    detect_or_reuse,
    label_class_count,
    restrict_to_pain_cc,
    write_run_outputs,
)

# ── constants ─────────────────────────────────────────────────────────────────
FIXED_THRESHOLDS = [5, 10, 20, 30, 40, 50]
TRANSFORMS       = ["log1p", "sqrt", "rank", "minmax"]

MIN_NODES = 4
MIN_EDGES = 3
N_PERM    = 1000
ALPHA     = 0.05
ALPHA_EXP = 0.10
SEED      = 42



def out_dir_for(algorithm):
    return ROBUSTNESS_DIR / algorithm / "threshold"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_postwalk_graph(g_full, family_code, concept_set, walk_steps=WALK_T):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            g_sub, _, _ = build_paincc_subgraph(
                g_full, family_code, concept_set, walk_steps=walk_steps,
                detect_communities=False,
            )
        return g_sub
    except Exception as e:
        print(f"    pipeline error: {e}")
        return None


def run_variant(g, family_name, method, label, run, t_build, n_paincc_main):
    key = f"{family_name}|{method}|{label}"
    run["keys"].append(key)
    g, reason = restrict_to_pain_cc(g, gloss_attr=GLOSS_ATTR,
                                    min_nodes=MIN_NODES, min_edges=MIN_EDGES)
    if g is not None and label_class_count(g, LABEL_CSV, GLOSS_ATTR) < 2:
        g, reason = None, "single label class"
    if g is None:
        run["store"].save({"key": key, "status": "not_evaluable",
                           "error": reason, "t_build": t_build})
        return None

    rec, reused = detect_or_reuse(g, key, run["store"], run["detector"], N_PERM,
                                  gloss_attr=GLOSS_ATTR)
    if reused:
        run["store"].save({**rec, "reused": True})
        return rec["row"]
    rec["t_build"] = t_build
    rec["reused"] = False
    if rec["status"] != "ok":
        print(f"    community {rec['status']}: {rec['error']}")
        run["store"].save(rec)
        return None

    g = g.copy()
    g.vs["community"] = [rec["membership"][gl] for gl in g.vs[GLOSS_ATTR]]
    t_stats = time.perf_counter()
    row = _evaluate(g, family_name, method, label)
    if row is not None:
        row["n_nodes_paincc_main"] = n_paincc_main
        row["n_removed"] = n_paincc_main - g.vcount()
    rec["t_stats"] = round(time.perf_counter() - t_stats, 4)
    rec["row"] = row
    run["store"].save(rec)
    return row


def _evaluate(g, family_name, method, label):
    ari_result = calculate_ari_for_subgraph(g, LABEL_CSV, gloss_attr=GLOSS_ATTR)
    if ari_result is None:
        return None

    p_mc, _ = permutation_test(
        ari_result["y_true"], ari_result["y_pred"],
        ari_result["ari"], n_perm=N_PERM, seed=SEED,
    )
    exact = exact_test(ari_result["y_true"], ari_result["y_pred"], ari_result["ari"])
    p_val = exact["p_exact"]
    stats = compute_graph_stats(g)

    return {
        "Family":        family_name,
        "Method":        method,
        "Condition":     label,
        "n_nodes":       stats["n_nodes"],
        "n_edges":       stats["n_edges"],
        "density":       round(stats["density"], 6),
        "k_detected":    ari_result["k_detected"],
        "ARI":           round(ari_result["ari"], 4),
        "p_val":         fmt_p(p_val),
        "p_min":         fmt_p(exact["p_min"]),
        "ARI_max":       round(exact["ari_max"], 4),
        "p_mc":          round(max(p_mc, 1 / N_PERM), 4),
        "significant_05": bool(p_val < ALPHA),
        "significant_10": bool(p_val < ALPHA_EXP),
        "n_matched":     ari_result["n_matched"],
    }


# ── plotting ──────────────────────────────────────────────────────────────────

def save_threshold_ari_curve(family_rows, family, out_dir, algorithm):
    rows = [r for r in family_rows if r["Method"] == "threshold_binary"]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["Condition"].lstrip("t")))

    xs = [int(r["Condition"].lstrip("t")) for r in rows]
    aris = [r["ARI"] for r in rows]
    pvals = [r["p_val"] for r in rows]
    n_edges = [r["n_edges"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    bar_w = max(1, (max(xs) - min(xs)) / len(xs) * 0.6) if len(xs) > 1 else 2
    ax2.bar(xs, n_edges, color="orange", alpha=0.35, width=bar_w, label="Edges")
    ax2.set_ylabel("# Edges", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    ax1.plot(xs, aris, "b-o", linewidth=2, zorder=5, label="ARI")
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_xlabel("Threshold (binary, w >= t)")
    ax1.set_ylabel("ARI", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    for x, a, p in zip(xs, aris, pvals):
        if p < ALPHA:
            ax1.annotate("★", (x, a), textcoords="offset points",
                         xytext=(0, 10), ha="center",
                         fontsize=14, color="darkgreen")
        elif p < ALPHA_EXP:
            ax1.annotate("†", (x, a), textcoords="offset points",
                         xytext=(0, 10), ha="center",
                         fontsize=13, color="olive")

    ax1.set_title(f"{family} — ARI vs Threshold (binary, T = {WALK_T}, aligned; "
                  f"{algorithm})")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{family}_ari_curve.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global N_PERM
    parser = argparse.ArgumentParser(
        description="CLICS4 threshold + transform robustness (PAIN-CC, T=2)"
    )
    all_pairs = load_families()
    name_to_code = dict(all_pairs)

    parser.add_argument("--families", nargs="+",
                        default=[n for n, _ in all_pairs])
    parser.add_argument("--thresholds", nargs="+", type=int,
                        default=FIXED_THRESHOLDS)
    parser.add_argument("--transforms", nargs="+", default=TRANSFORMS,
                        choices=TRANSFORMS)
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="optimal",
                        help="community detection (default: optimal, as in the "
                             "main analysis; leiden reproduces the original sweep)")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help="seconds allowed per community-detection call")
    parser.add_argument("--fresh", action="store_true",
                        help="discard stored conditions and recompute all")
    args = parser.parse_args()

    families = []
    for f in args.families:
        if f not in name_to_code:
            parser.error(f"Unknown family: {f!r}. Choices: {list(name_to_code)}")
        families.append(f)

    N_PERM = args.n_perm
    OUT_DIR = out_dir_for(args.algorithm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    t_run = time.perf_counter()
    print(f"Community detection: {args.algorithm} → {OUT_DIR}")

    g_full = load_graph()
    concept_set = load_concept_set()
    run = {"store": ConditionStore(OUT_DIR / "conditions.jsonl", fresh=args.fresh),
           "detector": TimedDetector(args.algorithm, seed=SEED, timeout=args.timeout),
           "keys": []}

    all_rows = []
    total = len(families)

    for fam_idx, family_name in enumerate(families, 1):
        family_code = name_to_code[family_name]
        print(f"\n[{fam_idx}/{total}] {family_name} ({family_code})")
        fam_dir = OUT_DIR / family_name
        fam_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        g_sub = load_postwalk_graph(g_full, family_code, concept_set,
                                    walk_steps=WALK_T)
        t_family = time.perf_counter() - t0
        if g_sub is None:
            print(f"  SKIP: post-walk graph is None")
            continue

        family_rows = []

        print(f"  threshold sweep ...", end=" ", flush=True)
        kept = 0
        for t in sorted(set(args.thresholds)):
            t0 = time.perf_counter()
            g_t = apply_threshold_binary(g_sub, threshold=t)
            t_build = round(t_family + time.perf_counter() - t0, 4)
            r = run_variant(g_t, family_name, "threshold_binary", f"t{t}", run, t_build,
                            g_sub.vcount())
            if r is not None:
                family_rows.append(r)
                all_rows.append(r)
                kept += 1
        print(f"{kept}/{len(args.thresholds)} variants")

        print(f"  transforms ...", end=" ", flush=True)
        kept = 0
        for trans in args.transforms:
            t0 = time.perf_counter()
            try:
                g_t = apply_weight_transform(g_sub, transform=trans)
            except Exception as e:
                print(f"    transform {trans} failed: {e}")
                continue
            t_build = round(t_family + time.perf_counter() - t0, 4)
            r = run_variant(g_t, family_name, f"transform_{trans}", "full", run, t_build,
                            g_sub.vcount())
            if r is not None:
                family_rows.append(r)
                all_rows.append(r)
                kept += 1
        print(f"{kept}/{len(args.transforms)} variants")

        if family_rows:
            save_threshold_ari_curve(family_rows, family_name, fam_dir, args.algorithm)

        for r in family_rows:
            sig = "✓" if r["significant_05"] else ("~" if r["significant_10"] else "")
            print(f"    {r['Method']:20s} {r['Condition']:>6s}  "
                  f"n={r['n_nodes']:>3d} e={r['n_edges']:>3d}  "
                  f"ARI={r['ARI']:+.3f}{sig} p={r['p_val']:.4f}")

    if all_rows:
        cols = ["Family", "Method", "Condition", "n_nodes", "n_edges", "density",
                "k_detected", "ARI", "p_val", "p_min", "ARI_max", "p_mc",
                "significant_05", "significant_10", "n_matched",
                "n_nodes_paincc_main", "n_removed"]
        summary_df = pd.DataFrame(all_rows, columns=cols)
        summary_path = OUT_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved {len(summary_df)} rows → {summary_path}")

        sig = summary_df[summary_df["significant_05"]]
        if not sig.empty:
            print(f"\nSignificant (p < {ALPHA}):")
            print(sig[["Family", "Method", "Condition", "ARI", "p_val"]]
                  .to_string(index=False))
        else:
            print(f"\nNo results reached p < {ALPHA}.")
    else:
        print("\nNo results to save.")

    run["detector"].close()
    cond, fail = write_run_outputs(
        OUT_DIR, run["store"], run["keys"], args.algorithm, run["detector"].params,
        args.timeout, N_PERM, sys.argv, started_at, time.perf_counter() - t_run,
    )
    print(f"Conditions: {cond['status'].value_counts().to_dict()}; "
          f"failures/timeouts: {len(fail)} → {OUT_DIR / 'failures.csv'}")


if __name__ == "__main__":
    main()
