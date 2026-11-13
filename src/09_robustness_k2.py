#!/usr/bin/env python3
"""ROBUSTNESS: CONSTRAINED k = 2 — Supplementary Table S2 step 18 (Supplementary Table S10).

Forces exactly two communities — matching the binary physical/non-physical
ground truth — under three independent algorithms (walktrap, fast-greedy,
leading eigenvector; seed 42 where applicable) across six graph configurations
(postwalk_raw, thresholds t10/t20/t30, log1p transform, no-walk), all
PAIN-CC restricted; threshold/transform variants are re-restricted to the
PAIN-connected component exactly as in script 07, and a configuration whose
PAIN-CC holds only one label class is skipped. Each combination gets an exact label-permutation ARI test
(the 1,000-draw Monte Carlo p is kept as p_mc) and is compared against the
unconstrained baselines that scripts 07–08 produced with --baseline-algorithm
(optimal by default, or leiden). ARI, k, p and significance of a baseline are
all read from the same row, i.e. from one partition.

The k = 2 partitions do not depend on the baseline algorithm; k2_summary.csv
is therefore identical in both output directories.

Outputs (output/robustness/{baseline_algorithm}/k2/):
  k2_summary.csv
  comparison_table.csv

Usage:
    python src/09_robustness_k2.py                            (requires 07 and 08)
    python src/09_robustness_k2.py --baseline-algorithm leiden
"""

import io
import contextlib
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    LABEL_CSV, ROBUSTNESS_DIR, WALK_T, GLOSS_ATTR,
    load_graph, load_concept_set, load_families,
)
from network_utils import build_paincc_subgraph, build_nowalk_aligned
from ari_utils import calculate_ari_for_subgraph, exact_test, fmt_p, permutation_test
from robustness_utils import (
    ALGORITHMS,
    apply_threshold_binary,
    apply_weight_transform,
    compute_graph_stats,
    label_class_count,
    restrict_to_pain_cc,
    run_k2_detection,
)

# ── constants ─────────────────────────────────────────────────────────────────
GRAPH_CONFIGS = [
    ("postwalk_raw",    None,        None),
    ("threshold_t10",   "threshold", 10),
    ("threshold_t20",   "threshold", 20),
    ("threshold_t30",   "threshold", 30),
    ("transform_log1p", "transform", "log1p"),
    ("nowalk",          "nowalk",    None),
]

K2_ALGORITHMS = ["walktrap", "fastgreedy", "leading_eigenvector"]
N_PERM        = 1000
SEED          = 42
MIN_NODES     = 4
ALPHA         = 0.05
ALPHA_EXP     = 0.10



def out_dir_for(baseline_algorithm):
    return ROBUSTNESS_DIR / baseline_algorithm / "k2"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_postwalk_graph(g_full, family_code, concept_set, walk_steps=WALK_T):
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            g_sub, _, _ = build_paincc_subgraph(
                g_full, family_code, concept_set, walk_steps=walk_steps
            )
        return g_sub
    except Exception as e:
        print(f"    pipeline error: {e}")
        return None


def build_graph_for_config(config_type, config_val, g_sub, g_full,
                           family_code, concept_set):
    if config_type is None:                    # postwalk_raw
        g = g_sub.copy() if g_sub is not None else None
    elif config_type == "threshold":
        g, _ = restrict_to_pain_cc(apply_threshold_binary(g_sub, threshold=config_val),
                                   gloss_attr=GLOSS_ATTR)
    elif config_type == "transform":
        g, _ = restrict_to_pain_cc(apply_weight_transform(g_sub, transform=config_val),
                                   gloss_attr=GLOSS_ATTR)
    elif config_type == "nowalk":
        g = build_nowalk_aligned(g_full, family_code, concept_set)
    else:
        return None
    if g is None or g.vcount() < MIN_NODES:
        return None
    if label_class_count(g, LABEL_CSV, GLOSS_ATTR) < 2:
        return None
    return g


def run_k2_variant(g, config_name, algorithm, family, n_perm, seed):
    try:
        comms = run_k2_detection(g, algorithm, seed=seed)
    except Exception:
        return None

    g = g.copy()
    g.vs["community"] = comms.membership
    k_detected = len(set(comms.membership))

    result = calculate_ari_for_subgraph(g, LABEL_CSV, gloss_attr=GLOSS_ATTR)
    if result is None:
        return None

    p_mc, _ = permutation_test(result["y_true"], result["y_pred"],
                               result["ari"], n_perm, seed)
    exact = exact_test(result["y_true"], result["y_pred"], result["ari"])
    p_val = exact["p_exact"]
    stats = compute_graph_stats(g)

    return {
        "family":         family,
        "config":         config_name,
        "algorithm":      algorithm,
        "k_detected":     k_detected,
        "stats":          stats,
        "ari":            result["ari"],
        "p_val":          p_val,
        "p_min":          exact["p_min"],
        "ari_max":        exact["ari_max"],
        "p_mc":           p_mc,
        "significant_05": bool(p_val < ALPHA),
        "significant_10": bool(p_val < ALPHA_EXP),
        "n_matched":      result["n_matched"],
    }


# ── output ────────────────────────────────────────────────────────────────────

def build_k2_summary_df(k2_rows):
    cols = ["Family", "Config", "Algorithm",
            "n_nodes", "n_edges", "density",
            "k_detected", "ARI", "p_val", "p_min", "ARI_max", "p_mc",
            "significant_05", "significant_10", "n_matched"]
    rows = []
    for r in k2_rows:
        s = r["stats"]
        rows.append({
            "Family":        r["family"],
            "Config":        r["config"],
            "Algorithm":     r["algorithm"],
            "n_nodes":       s["n_nodes"],
            "n_edges":       s["n_edges"],
            "density":       round(s["density"], 4),
            "k_detected":    r["k_detected"],
            "ARI":           round(r["ari"], 4),
            "p_val":         fmt_p(r["p_val"]),
            "p_min":         fmt_p(r["p_min"]),
            "ARI_max":       round(r["ari_max"], 4),
            "p_mc":          round(max(r["p_mc"], 1 / N_PERM), 4),
            "significant_05": r["significant_05"],
            "significant_10": r["significant_10"],
            "n_matched":     r["n_matched"],
        })
    return pd.DataFrame(rows, columns=cols)


def load_unconstrained(baseline_algorithm):
    """Unconstrained baselines; every field of a key comes from one row."""
    walkstep_csv = ROBUSTNESS_DIR / baseline_algorithm / "walkstep" / "summary.csv"
    threshold_csv = ROBUSTNESS_DIR / baseline_algorithm / "threshold" / "summary.csv"
    for path in (walkstep_csv, threshold_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing — run scripts 07 and 08 with "
                f"--algorithm {baseline_algorithm} first")
    out = {}
    if walkstep_csv.exists():
        df = pd.read_csv(walkstep_csv)
        for _, r in df.iterrows():
            fam = r["Family"]
            cfg = r["Config"]
            if cfg == "postwalk_raw" and int(r["walk_steps"]) == WALK_T:
                key = (fam, "postwalk_raw")
            elif cfg == "nowalk":
                key = (fam, "nowalk")
            else:
                continue
            out[key] = (float(r["ARI"]), float(r["k_detected"]),
                        float(r["p_val"]),
                        bool(r["significant_05"]), bool(r["significant_10"]))

    if threshold_csv.exists():
        df = pd.read_csv(threshold_csv)
        for _, r in df.iterrows():
            fam = r["Family"]
            method = r["Method"]
            cond = r["Condition"]
            if method == "threshold_binary" and cond in ("t10", "t20", "t30"):
                cfg = f"threshold_{cond}"
            elif method == "transform_log1p":
                cfg = "transform_log1p"
            else:
                continue
            out[(fam, cfg)] = (float(r["ARI"]), float(r["k_detected"]),
                               float(r["p_val"]),
                               bool(r["significant_05"]), bool(r["significant_10"]))
    return out


def build_comparison_df(k2_rows, uncon_map, baseline_algorithm):
    k2_idx = {(r["family"], r["config"], r["algorithm"]): r for r in k2_rows}
    seen = set()
    rows = []

    for r in k2_rows:
        key = (r["family"], r["config"])
        if key in seen:
            continue
        seen.add(key)
        family, config = key
        ari_u, k_u, p_u, sig05_u, sig10_u = uncon_map.get(
            key, (float("nan"), float("nan"), float("nan"), False, False)
        )

        row = {
            "Family": family,
            "Config": config,
            "Baseline_Algorithm": baseline_algorithm,
            "ARI_unconstrained": round(ari_u, 4) if not np.isnan(ari_u) else float("nan"),
            "k_unconstrained":   k_u,
            "p_unconstrained":   fmt_p(p_u) if not np.isnan(p_u) else float("nan"),
            "sig_unconstrained": sig05_u,
        }
        for alg in K2_ALGORITHMS:
            short = "eigenvector" if alg == "leading_eigenvector" else alg
            k2r = k2_idx.get((family, config, alg))
            if k2r:
                row[f"ARI_{short}_k2"] = round(k2r["ari"], 4)
                row[f"p_{short}_k2"]   = fmt_p(k2r["p_val"])
                row[f"sig_{short}_k2"] = k2r["significant_05"]
                row[f"k_{short}_k2"]   = k2r["k_detected"]
            else:
                row[f"ARI_{short}_k2"] = float("nan")
                row[f"p_{short}_k2"]   = float("nan")
                row[f"sig_{short}_k2"] = False
                row[f"k_{short}_k2"]   = float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def print_terminal_summary(comp_df):
    if comp_df.empty:
        return
    newly = comp_df[comp_df["sig_walktrap_k2"] & ~comp_df["sig_unconstrained"]]
    lost  = comp_df[~comp_df["sig_walktrap_k2"] & comp_df["sig_unconstrained"]]

    print("\n── K=2 Analysis Summary (aligned) ────────────────────────────")
    if not newly.empty:
        print("\nNewly significant under walktrap k=2:")
        print(newly[["Family", "Config",
                     "ARI_unconstrained", "p_unconstrained",
                     "ARI_walktrap_k2",   "p_walktrap_k2"]]
              .to_string(index=False))
    else:
        print("\nNo newly significant (walktrap k=2).")

    if not lost.empty:
        print("\nWARNING — Lost significance with walktrap k=2:")
        print(lost[["Family", "Config",
                    "ARI_unconstrained", "p_unconstrained",
                    "ARI_walktrap_k2",   "p_walktrap_k2"]]
              .to_string(index=False))
    else:
        print("\nNo significance lost with walktrap k=2.")
    print("──────────────────────────────────────────────────────────────\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    global N_PERM
    parser = argparse.ArgumentParser(
        description="CLICS4 k=2 forced clustering analysis (PAIN-CC)"
    )
    all_pairs = load_families()
    name_to_code = dict(all_pairs)

    parser.add_argument("--families", nargs="+",
                        default=[n for n, _ in all_pairs])
    parser.add_argument("--n-perm", type=int, default=N_PERM)
    parser.add_argument("--baseline-algorithm", choices=ALGORITHMS,
                        default="optimal",
                        help="which unconstrained sweep (scripts 07-08) to compare against")
    args = parser.parse_args()

    families = []
    for f in args.families:
        if f not in name_to_code:
            parser.error(f"Unknown family: {f!r}")
        families.append(f)
    N_PERM = args.n_perm

    uncon = load_unconstrained(args.baseline_algorithm)
    OUT_DIR = out_dir_for(args.baseline_algorithm)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    g_full = load_graph()
    concept_set = load_concept_set()

    k2_rows = []
    total = len(families)

    for fam_idx, family_name in enumerate(families, 1):
        family_code = name_to_code[family_name]
        print(f"\n[{fam_idx}/{total}] {family_name} ({family_code})")
        g_sub = load_postwalk_graph(g_full, family_code, concept_set,
                                    walk_steps=WALK_T)
        if g_sub is None:
            print("  SKIP: post-walk graph is None")
            continue

        for cfg_name, cfg_type, cfg_val in GRAPH_CONFIGS:
            g = build_graph_for_config(cfg_type, cfg_val, g_sub, g_full,
                                       family_code, concept_set)
            if g is None:
                print(f"  {cfg_name}: skipped (not evaluable: PAIN-CC too small, PAIN removed or one label class)")
                continue
            lines = []
            for alg in K2_ALGORITHMS:
                r = run_k2_variant(g, cfg_name, alg, family_name, N_PERM, SEED)
                if r is None:
                    continue
                k2_rows.append(r)
                tag = ("*" if r["significant_05"] else
                       "†" if r["significant_10"] else "")
                lines.append(f"{alg}(k={r['k_detected']}) "
                             f"ARI={r['ari']:+.4f} p={r['p_val']:.4f} {tag}")
            if lines:
                print(f"  {cfg_name}:")
                for l in lines:
                    print(f"    {l}")

    if not k2_rows:
        print("\nNo results.")
        return

    k2_df = build_k2_summary_df(k2_rows)
    k2_path = OUT_DIR / "k2_summary.csv"
    k2_df.to_csv(k2_path, index=False)
    print(f"\nSaved {len(k2_df)} rows → {k2_path}")

    comp_df = build_comparison_df(k2_rows, uncon, args.baseline_algorithm)
    comp_path = OUT_DIR / "comparison_table.csv"
    comp_df.to_csv(comp_path, index=False)
    print(f"Saved {len(comp_df)} rows → {comp_path}")

    print_terminal_summary(comp_df)


if __name__ == "__main__":
    main()
