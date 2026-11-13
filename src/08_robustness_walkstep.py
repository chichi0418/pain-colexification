#!/usr/bin/env python3
"""ROBUSTNESS: WALK-STEP SENSITIVITY — Supplementary Table S2 step 17.

Tests sensitivity of the PAIN-CC ARI result to random-walk length: 12 families
× 11 settings (no-walk baseline + T = 1…10). Communities are detected with
--algorithm:
  optimal (default) — exact optimal modularity, as in the main analysis
  leiden            — modularity objective, 10 iterations, resolution = 1
                      default, seed 42 (the original exploratory sweep)
Each fixed graph is partitioned once, in a worker process with a
per-condition time limit (--timeout); a timeout or error is recorded as such,
never replaced by another algorithm. A condition whose PAIN-CC contains only
one label class is not evaluable. Each condition gets an exact
label-permutation ARI test (p_val; the 1,000-draw Monte Carlo p is kept as
p_mc); the label permutations never re-run community detection. Also detects
per-family density saturation steps (the basis for choosing T = 2 in the main
analysis).

Outputs (output/robustness/{algorithm}/walkstep/):
  summary.csv            one row per evaluable condition (original columns)
  saturation_summary.csv
  conditions.jsonl       per-condition log: status, Q, membership, timings
                         (re-runs reuse its completed conditions; --fresh resets)
  conditions.csv         the same without memberships
  failures.csv           timeouts / errors
  run_metadata.json      algorithm parameters, versions, command, timing totals
  {Family}/{Family}_ari_curve.png
  {Family}/{Family}_density_curve.png

Usage:
    python src/08_robustness_walkstep.py
    python src/08_robustness_walkstep.py --algorithm leiden --fresh
    python src/08_robustness_walkstep.py --families Universal Uralic --n-perm 100
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
    LABEL_CSV, ROBUSTNESS_DIR, GLOSS_ATTR,
    load_graph, load_concept_set, load_families,
)
from network_utils import build_paincc_subgraph, build_nowalk_aligned
from ari_utils import calculate_ari_for_subgraph, exact_test, fmt_p, permutation_test
from robustness_utils import (
    ALGORITHMS,
    DEFAULT_TIMEOUT_S,
    ConditionStore,
    TimedDetector,
    compute_graph_stats,
    detect_or_reuse,
    label_class_count,
    write_run_outputs,
)

# ── constants ─────────────────────────────────────────────────────────────────
WALK_STEPS   = list(range(1, 11))           # 1..10
POSTWALK_KEY = "postwalk_raw"
NOWALK_KEY   = "nowalk"

MIN_NODES = 4
MIN_EDGES = 3
N_PERM    = 1000
ALPHA     = 0.05
ALPHA_EXP = 0.10
SEED      = 42



def out_dir_for(algorithm):
    return ROBUSTNESS_DIR / algorithm / "walkstep"


# Saturation detection thresholds
SAT_REL = 0.95
SAT_DELTA = 0.01


# ── helpers ───────────────────────────────────────────────────────────────────

def call_aligned(g_full, family_code, concept_set, walk_steps):
    """Run the PAIN-CC pipeline silently; return (g_sub, n_pain) or (None, n_pain)."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            g_sub, n_pain, _ = build_paincc_subgraph(
                g_full, family_code, concept_set, walk_steps=walk_steps,
                detect_communities=False,
            )
        return g_sub, n_pain
    except Exception as e:
        print(f"    pipeline error: {e}")
        return None, 0


def run_variant(g, family_name, walk_steps, config, run, t_build):
    key = f"{family_name}|{config}|{walk_steps}"
    if label_class_count(g, LABEL_CSV, GLOSS_ATTR) < 2:
        not_evaluable(run, key, "single label class", t_build)
        return None
    run["keys"].append(key)
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
    row = _evaluate(g, family_name, walk_steps, config)
    rec["t_stats"] = round(time.perf_counter() - t_stats, 4)
    rec["row"] = row
    run["store"].save(rec)
    return row


def not_evaluable(run, key, reason, t_build):
    run["keys"].append(key)
    run["store"].save({"key": key, "status": "not_evaluable", "error": reason,
                       "t_build": t_build})


def _evaluate(g, family_name, walk_steps, config):
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
        "walk_steps":    walk_steps,
        "Config":        config,
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


def detect_saturation(rows_postwalk):
    if not rows_postwalk:
        return None
    rows_sorted = sorted(rows_postwalk, key=lambda r: r["walk_steps"])
    steps = [r["walk_steps"] for r in rows_sorted]
    dens = [r["density"] for r in rows_sorted]
    if not dens:
        return None
    max_d = max(dens)
    if max_d <= 0:
        return None
    for i, t in enumerate(steps):
        if i == len(steps) - 1:
            if dens[i] >= SAT_REL * max_d:
                return {"saturation_step": t,
                        "density_at_sat": dens[i],
                        "max_density": max_d}
            return None
        delta = abs(dens[i + 1] - dens[i])
        if dens[i] >= SAT_REL * max_d and delta < SAT_DELTA:
            return {"saturation_step": t,
                    "density_at_sat": dens[i],
                    "max_density": max_d}
    return None


# ── plotting ──────────────────────────────────────────────────────────────────

def save_ari_curve(family_rows, family, out_dir, algorithm):
    df = pd.DataFrame(family_rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    sub_pw = df[df["Config"] == POSTWALK_KEY].sort_values("walk_steps")
    if not sub_pw.empty:
        xs = sub_pw["walk_steps"].tolist()
        aris = sub_pw["ARI"].tolist()
        pvals = sub_pw["p_val"].tolist()
        ax.plot(xs, aris, "o-", color="steelblue", linewidth=2,
                markersize=7, label="postwalk_raw")
        for x, a, p in zip(xs, aris, pvals):
            if p < ALPHA:
                ax.annotate("★", (x, a), textcoords="offset points",
                            xytext=(0, 10), ha="center",
                            fontsize=13, color="darkgreen")
            elif p < ALPHA_EXP:
                ax.annotate("†", (x, a), textcoords="offset points",
                            xytext=(0, 10), ha="center",
                            fontsize=12, color="olive")

    sub_nw = df[df["Config"] == NOWALK_KEY]
    if not sub_nw.empty:
        ari_nw = sub_nw["ARI"].iloc[0]
        ax.axhline(ari_nw, color="gray", linestyle=":", linewidth=1.5,
                   label=f"nowalk (ARI={ari_nw:.3f})")

    ax.axhline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Random Walk Steps")
    ax.set_ylabel("ARI")
    ax.set_title(f"{family} — ARI vs Random Walk Steps (aligned; {algorithm})")
    ax.set_xticks(WALK_STEPS)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{family}_ari_curve.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_density_curve(family_rows, family, saturation, out_dir, algorithm):
    df = pd.DataFrame(family_rows)
    if df.empty:
        return
    sub_pw = df[df["Config"] == POSTWALK_KEY].sort_values("walk_steps")
    if sub_pw.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = sub_pw["walk_steps"].tolist()
    ds = sub_pw["density"].tolist()
    ax.plot(xs, ds, "o-", color="darkorange", linewidth=2,
            markersize=7, label="postwalk_raw density")

    sub_nw = df[df["Config"] == NOWALK_KEY]
    if not sub_nw.empty:
        d_nw = sub_nw["density"].iloc[0]
        ax.axhline(d_nw, color="gray", linestyle=":", linewidth=1.5,
                   label=f"nowalk (density={d_nw:.3f})")

    if saturation is not None:
        sat_t = saturation["saturation_step"]
        ax.axvline(sat_t, color="darkgreen", linestyle="--", linewidth=1.5,
                   label=f"saturation at step {sat_t}")

    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Random Walk Steps")
    ax.set_ylabel("Graph density")
    ax.set_title(f"{family} — Density vs Random Walk Steps (aligned; {algorithm})")
    ax.set_xticks(WALK_STEPS)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(out_dir) / f"{family}_density_curve.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global N_PERM
    parser = argparse.ArgumentParser(
        description="CLICS4 walk-step robustness (PAIN-CC)"
    )
    all_pairs = load_families()
    name_to_code = dict(all_pairs)

    parser.add_argument("--families", nargs="+",
                        default=[n for n, _ in all_pairs])
    parser.add_argument("--walk-steps", nargs="+", type=int,
                        default=WALK_STEPS, dest="walk_steps")
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
    sat_rows = []
    total = len(families)

    for fam_idx, family_name in enumerate(families, 1):
        family_code = name_to_code[family_name]
        print(f"\n[{fam_idx}/{total}] {family_name} ({family_code})")
        family_rows = []

        # nowalk baseline
        print("  nowalk ...", end=" ", flush=True)
        t0 = time.perf_counter()
        g_nw = build_nowalk_aligned(g_full, family_code, concept_set,
                                    min_nodes=MIN_NODES, min_edges=MIN_EDGES)
        t_build = round(time.perf_counter() - t0, 4)
        if g_nw is not None:
            r = run_variant(g_nw, family_name, walk_steps=0, config=NOWALK_KEY,
                            run=run, t_build=t_build)
            if r is not None:
                family_rows.append(r)
                all_rows.append(r)
                print(f"ARI={r['ARI']:.4f} p={r['p_val']:.4f} "
                      f"density={r['density']:.3f}")
            else:
                print("skipped (ARI None)")
        else:
            not_evaluable(run, f"{family_name}|{NOWALK_KEY}|0",
                          "nowalk graph too small", t_build)
            print("skipped (nowalk graph too small)")

        # postwalk_raw across walk steps
        for ws in sorted(set(args.walk_steps)):
            print(f"  steps={ws} ...", end=" ", flush=True)
            t0 = time.perf_counter()
            g_sub, n_pain = call_aligned(g_full, family_code, concept_set, ws)
            t_build = round(time.perf_counter() - t0, 4)
            key = f"{family_name}|{POSTWALK_KEY}|{ws}"
            if g_sub is None:
                not_evaluable(run, key, f"no PAIN-CC graph (n_pain={n_pain})", t_build)
                print(f"skip (n_pain={n_pain})")
                continue
            if g_sub.vcount() < MIN_NODES or g_sub.ecount() < MIN_EDGES:
                not_evaluable(run, key, "graph too small", t_build)
                print(f"skip (n={g_sub.vcount()}, e={g_sub.ecount()})")
                continue
            r = run_variant(g_sub.copy(), family_name, walk_steps=ws,
                            config=POSTWALK_KEY, run=run, t_build=t_build)
            if r is not None:
                family_rows.append(r)
                all_rows.append(r)
                sig = "✓" if r["significant_05"] else ("~" if r["significant_10"] else "")
                print(f"ARI={r['ARI']:.3f}{sig} p={r['p_val']:.4f} "
                      f"density={r['density']:.3f}")
            else:
                print("ARI None")

        pw_rows = [r for r in family_rows if r["Config"] == POSTWALK_KEY]
        sat = detect_saturation(pw_rows)
        if sat is not None:
            sat_rows.append({"Family": family_name, **sat})
            print(f"  → saturation at step {sat['saturation_step']} "
                  f"(density={sat['density_at_sat']:.3f}, "
                  f"max={sat['max_density']:.3f})")
        else:
            print("  → no saturation detected")

        if family_rows:
            fam_dir = OUT_DIR / family_name
            fam_dir.mkdir(parents=True, exist_ok=True)
            save_ari_curve(family_rows, family_name, fam_dir, args.algorithm)
            save_density_curve(family_rows, family_name, sat, fam_dir, args.algorithm)

    if all_rows:
        cols = ["Family", "walk_steps", "Config", "n_nodes", "n_edges",
                "density", "k_detected", "ARI", "p_val", "p_min", "ARI_max", "p_mc",
                "significant_05", "significant_10", "n_matched"]
        summary_df = pd.DataFrame(all_rows, columns=cols)
        summary_path = OUT_DIR / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nSaved {len(summary_df)} rows → {summary_path}")

        sig = summary_df[summary_df["significant_05"]]
        if not sig.empty:
            print(f"\nSignificant (p < {ALPHA}):")
            print(sig[["Family", "walk_steps", "Config", "ARI", "p_val"]]
                  .to_string(index=False))
        else:
            print(f"\nNo results reached p < {ALPHA}.")
    else:
        print("\nNo results to save.")

    if sat_rows:
        sat_df = pd.DataFrame(sat_rows,
                              columns=["Family", "saturation_step",
                                       "density_at_sat", "max_density"])
        sat_df[["density_at_sat", "max_density"]] = (
            sat_df[["density_at_sat", "max_density"]].round(4))
        sat_path = OUT_DIR / "saturation_summary.csv"
        sat_df.to_csv(sat_path, index=False)
        print(f"Saved saturation summary → {sat_path}")

    run["detector"].close()
    cond, fail = write_run_outputs(
        OUT_DIR, run["store"], run["keys"], args.algorithm, run["detector"].params,
        args.timeout, N_PERM, sys.argv, started_at, time.perf_counter() - t_run,
    )
    print(f"Conditions: {cond['status'].value_counts().to_dict()}; "
          f"failures/timeouts: {len(fail)} → {OUT_DIR / 'failures.csv'}")


if __name__ == "__main__":
    main()
