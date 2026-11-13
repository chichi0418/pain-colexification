#!/usr/bin/env python3
"""ROBUSTNESS: LEIDEN vs EXACT OPTIMAL MODULARITY — algorithm comparison.

Compares, condition by condition, the threshold/transform (script 07) and
walk-step (script 08) sweeps run with --algorithm optimal and --algorithm
leiden. Both sweeps must have been run; this script reads their per-condition
logs (conditions.jsonl) and never re-runs the sweeps.

For every condition it reports whether the two runs partitioned the same graph
(node/edge counts and an exact node/edge/weight fingerprint), each algorithm's
modularity Q, community count, label ARI, exact p, minimum attainable p,
Monte Carlo p and nominal significance, the ARI between the two partitions,
and detection runtime. Q_diff = Q_optimal − Q_leiden must be ≥ 0; a negative
value is flagged as a solver problem.

Equal Q does not imply equal partitions. Where Leiden reached the optimal Q
with a different partition (same_Q_diff_partition), the optimum is not unique;
for those conditions the graph is rebuilt and exact optimisation is re-run
under N_ORDERINGS random vertex orderings to show which optimal partitions the
solver can return and their label ARIs. No result is selected by label ARI.

Also checks that the optimal walk-step row at T = 2 reproduces the main
analysis (output/main/summary.csv).

Outputs (output/robustness/comparison/):
  leiden_vs_optimal.csv        one row per condition
  degeneracy_check.csv         re-optimisation under vertex reorderings
  main_consistency.csv         optimal T = 2 vs main analysis
  comparison_overview.csv      counts per family and analysis
  threshold_overview.png       label ARI per condition, both algorithms
  walkstep_overview.png

Usage:
    python src/18_compare_leiden_optimal.py      (requires 07 and 08 run with both algorithms)
"""

import io
import json
import random
import contextlib

import igraph as ig
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from config import (
    LABEL_CSV, MAIN_DIR, ROBUSTNESS_DIR, WALK_T, GLOSS_ATTR,
    load_graph, load_concept_set, load_families,
)
from network_utils import build_paincc_subgraph, build_nowalk_aligned, optimal_communities
from ari_utils import calculate_ari_for_subgraph, exact_test, fmt_p
from robustness_utils import (
    apply_threshold_binary, apply_weight_transform, graph_fingerprint,
    restrict_to_pain_cc,
)

OUT_DIR = ROBUSTNESS_DIR / "comparison"
ALPHA = 0.05
Q_TOL = 1e-9
N_ORDERINGS = 20
SEED = 42

ANALYSES = {
    "threshold": ["t5", "t10", "t20", "t30", "t40", "t50",
                  "log1p", "sqrt", "rank", "minmax"],
    "walkstep": ["nowalk"] + [f"T{t}" for t in range(1, 11)],
}

OPT_COLOR = "#2a78d6"   # categorical slot 1
LEI_COLOR = "#eb6834"   # categorical slot 2
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ── loading ───────────────────────────────────────────────────────────────────

def load_records(algorithm, analysis):
    path = ROBUSTNESS_DIR / algorithm / analysis / "conditions.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run script {'07' if analysis == 'threshold' else '08'} "
            f"with --algorithm {algorithm}")
    recs = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[r["key"]] = r
    return recs


def condition_label(analysis, key):
    fam, a, b = key.split("|")
    if analysis == "threshold":
        return fam, (b if a == "threshold_binary" else a.replace("transform_", ""))
    return fam, ("nowalk" if a == "nowalk" else f"T{b}")


# ── per-condition comparison ──────────────────────────────────────────────────

def _side(rec, prefix):
    out = {f"status_{prefix}": rec["status"] if rec else "missing"}
    row = (rec or {}).get("row") or {}
    ok = rec is not None and rec["status"] == "ok" and bool(row)
    out.update({
        f"Q_{prefix}":         round(rec["Q"], 6) if ok else np.nan,
        f"k_{prefix}":         rec.get("k_all") if ok else np.nan,
        f"k_matched_{prefix}": row.get("k_detected", np.nan),
        f"label_ARI_{prefix}": row.get("ARI", np.nan),
        f"p_exact_{prefix}":   row.get("p_val", np.nan),
        f"p_min_{prefix}":     row.get("p_min", np.nan),
        f"p_mc_{prefix}":      row.get("p_mc", np.nan),
        f"sig_05_{prefix}":    row.get("significant_05", np.nan),
        f"runtime_s_{prefix}": rec.get("t_detect", np.nan) if rec else np.nan,
    })
    return out


def compare_analysis(analysis):
    opt = load_records("optimal", analysis)
    lei = load_records("leiden", analysis)
    rows = []
    for key in list(dict.fromkeys(list(opt) + list(lei))):
        o, l = opt.get(key), lei.get(key)
        fam, cond = condition_label(analysis, key)
        row = {"Analysis": analysis, "Family": fam, "Condition": cond, "key": key}
        ref = o or l
        row["n_nodes"] = ref.get("n_nodes")
        row["n_edges"] = ref.get("n_edges")
        both_built = (o and l and o.get("graph_fingerprint")
                      and l.get("graph_fingerprint"))
        row["same_graph"] = (bool(both_built)
                             and o["graph_fingerprint"] == l["graph_fingerprint"]
                             and o["n_nodes"] == l["n_nodes"]
                             and o["n_edges"] == l["n_edges"])
        row["weight_hash"] = ref.get("graph_fingerprint")
        row.update(_side(o, "optimal"))
        row.update(_side(l, "leiden"))

        both_ok = (row["status_optimal"] == "ok" and row["status_leiden"] == "ok"
                   and not np.isnan(row["Q_optimal"]) and not np.isnan(row["Q_leiden"]))
        if both_ok and row["same_graph"]:
            glosses = sorted(o["membership"])
            mo = [o["membership"][g] for g in glosses]
            ml = [l["membership"][g] for g in glosses]
            p_ari = adjusted_rand_score(mo, ml)
            q_diff = o["Q"] - l["Q"]
            same = p_ari > 1 - 1e-12
            row.update({
                "partition_ARI": round(p_ari, 4),
                "same_partition": same,
                "Q_diff": round(q_diff, 6),
                "optimal_below_leiden": q_diff < -Q_TOL,
                "same_Q_diff_partition": abs(q_diff) <= Q_TOL and not same,
                "sig_changed": bool(row["sig_05_optimal"]) != bool(row["sig_05_leiden"]),
            })
        else:
            row.update({"partition_ARI": np.nan, "same_partition": np.nan,
                        "Q_diff": np.nan, "optimal_below_leiden": np.nan,
                        "same_Q_diff_partition": np.nan, "sig_changed": np.nan})
        rows.append(row)
    return pd.DataFrame(rows)


# ── degeneracy check ──────────────────────────────────────────────────────────

def rebuild_graph(analysis, key, g_full, concept_set, name_to_code):
    fam, a, b = key.split("|")
    code = name_to_code[fam]
    with contextlib.redirect_stdout(io.StringIO()):
        if analysis == "walkstep" and a == "nowalk":
            return build_nowalk_aligned(g_full, code, concept_set)
        steps = int(b) if analysis == "walkstep" else WALK_T
        g_sub, _, _ = build_paincc_subgraph(g_full, code, concept_set,
                                            walk_steps=steps, detect_communities=False)
    if analysis == "walkstep":
        return g_sub
    if a == "threshold_binary":
        g_var = apply_threshold_binary(g_sub, threshold=int(b.lstrip("t")))
    else:
        g_var = apply_weight_transform(g_sub, transform=a.replace("transform_", ""))
    return restrict_to_pain_cc(g_var, gloss_attr=GLOSS_ATTR)[0]


def label_stats(g, membership_by_gloss):
    g = g.copy()
    g.vs["community"] = [membership_by_gloss[gl] for gl in g.vs[GLOSS_ATTR]]
    res = calculate_ari_for_subgraph(g, LABEL_CSV, gloss_attr=GLOSS_ATTR)
    ex = exact_test(res["y_true"], res["y_pred"], res["ari"])
    return res["ari"], ex["p_exact"], res["k_detected"]


def degeneracy_check(comp, g_full, concept_set, name_to_code):
    flagged = comp[comp["same_Q_diff_partition"] == True]
    rng = random.Random(SEED)
    rows = []
    for _, r in flagged.iterrows():
        g = rebuild_graph(r["Analysis"], r["key"], g_full, concept_set, name_to_code)
        if graph_fingerprint(g, gloss_attr=GLOSS_ATTR) != r["weight_hash"]:
            raise RuntimeError(f"rebuilt graph differs from stored graph: {r['key']}")
        glosses = g.vs[GLOSS_ATTR]
        found = {}
        for i in range(N_ORDERINGS + 1):
            perm = list(range(g.vcount()))
            if i > 0:
                rng.shuffle(perm)
            gp = g.permute_vertices(perm)          # attributes travel with vertices
            memb = optimal_communities(gp, weights="weight").membership
            by_gloss = dict(zip(gp.vs[GLOSS_ATTR], memb))
            canon = tuple(pd.factorize(np.array([by_gloss[gl] for gl in sorted(glosses)]))[0])
            if canon not in found:
                q = g.modularity([by_gloss[gl] for gl in glosses], weights="weight")
                ari, p, k = label_stats(g, by_gloss)
                found[canon] = {"first_ordering": i, "count": 0, "Q": q,
                                "label_ARI": ari, "p_exact": p, "k_matched": k}
            found[canon]["count"] += 1
        for j, (canon, f) in enumerate(found.items()):
            rows.append({
                "Analysis": r["Analysis"], "Family": r["Family"],
                "Condition": r["Condition"], "optimum": j + 1,
                "n_distinct_optima_found": len(found),
                "orderings_returning_it": f"{f['count']}/{N_ORDERINGS + 1}",
                "is_reported_optimal": f["first_ordering"] == 0,
                "Q": round(f["Q"], 6), "k_matched": f["k_matched"],
                "label_ARI": round(f["label_ARI"], 4),
                "p_exact": fmt_p(f["p_exact"]),
                "sig_05": f["p_exact"] < ALPHA,
                "label_ARI_leiden": r["label_ARI_leiden"],
                "p_exact_leiden": r["p_exact_leiden"],
            })
    return pd.DataFrame(rows)


# ── main-analysis consistency ─────────────────────────────────────────────────

def main_consistency():
    main = pd.read_csv(MAIN_DIR / "summary.csv")
    ws = pd.read_csv(ROBUSTNESS_DIR / "optimal" / "walkstep" / "summary.csv")
    ws = ws[(ws["Config"] == "postwalk_raw") & (ws["walk_steps"] == WALK_T)]
    m = ws.merge(main, on="Family", how="outer")
    out = pd.DataFrame({
        "Family": m["Family"],
        "nodes_sweep": m["n_nodes"], "nodes_main": m["PainCCNodes"],
        "k_sweep": m["k_detected"], "k_main": m["k_Local"],
        "ARI_sweep": m["ARI"], "ARI_main": m["ARI_Local"].round(4),
        "p_sweep": m["p_val"], "p_main": m["p_Exact"],
    })
    out["identical"] = ((out["nodes_sweep"] == out["nodes_main"])
                        & (out["k_sweep"] == out["k_main"])
                        & ((out["ARI_sweep"] - out["ARI_main"]).abs() < 1e-4)
                        & ((out["p_sweep"] - out["p_main"]).abs() < 1e-4))
    return out


# ── overview ──────────────────────────────────────────────────────────────────

def overview(comp):
    rows = []
    for (analysis, fam), sub in comp.groupby(["Analysis", "Family"], sort=False):
        both = sub[(sub["status_optimal"] == "ok") & (sub["status_leiden"] == "ok")
                   & sub["label_ARI_optimal"].notna()]
        rows.append({
            "Analysis": analysis, "Family": fam,
            "conditions_evaluable": len(both),
            "sig_optimal": int(both["sig_05_optimal"].astype(bool).sum()),
            "sig_leiden": int(both["sig_05_leiden"].astype(bool).sum()),
            "same_partition": int(both["same_partition"].astype(bool).sum()),
            "same_Q_diff_partition": int(both["same_Q_diff_partition"].astype(bool).sum()),
            "Q_optimal_higher": int((both["Q_diff"] > Q_TOL).sum()),
            "sig_changed": int(both["sig_changed"].astype(bool).sum()),
            "max_abs_label_ARI_diff": round(float(
                (both["label_ARI_optimal"] - both["label_ARI_leiden"]).abs().max()), 4)
                if len(both) else np.nan,
        })
    return pd.DataFrame(rows)


def plot_overview(comp, analysis, families, path):
    conds = ANALYSES[analysis]
    sub = comp[comp["Analysis"] == analysis]
    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True, sharey=True)
    x = np.arange(len(conds))
    for ax, fam in zip(axes.flat, families):
        f = sub[sub["Family"] == fam].set_index("Condition").reindex(conds)
        for prefix, color, ls, marker, dx in (("optimal", OPT_COLOR, "-", "o", -0.08),
                                               ("leiden", LEI_COLOR, "--", "s", 0.08)):
            y = f[f"label_ARI_{prefix}"].astype(float).values
            sig = (f[f"sig_05_{prefix}"] == True).values
            ax.plot(x + dx, y, ls, color=color, linewidth=2, zorder=2)
            ax.scatter((x + dx)[sig], y[sig], marker=marker, s=40, color=color,
                       edgecolors="white", linewidths=1, zorder=3)
            ax.scatter((x + dx)[~sig], y[~sig], marker=marker, s=40,
                       facecolors="white", edgecolors=color, linewidths=1.5, zorder=3)
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_title(fam, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(INK_2)
        ax.tick_params(colors=INK_2, labelsize=8)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(conds, rotation=60, ha="right")
    for ax in axes[:, 0]:
        ax.set_ylabel("Label ARI", color=INK_2)
    handles = [
        plt.Line2D([], [], color=OPT_COLOR, ls="-", marker="o", label="Exact optimal modularity"),
        plt.Line2D([], [], color=LEI_COLOR, ls="--", marker="s", label="Leiden (seed 42)"),
        plt.Line2D([], [], ls="", marker="o", color=INK_2, label="filled: exact p < .05"),
        plt.Line2D([], [], ls="", marker="o", markerfacecolor="white",
                   markeredgecolor=INK_2, label="open: exact p ≥ .05"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    comp = pd.concat([compare_analysis(a) for a in ANALYSES], ignore_index=True)

    bad = comp[comp["optimal_below_leiden"] == True]
    if not bad.empty:
        print("WARNING — optimal Q below Leiden Q:")
        print(bad[["Analysis", "Family", "Condition", "Q_optimal", "Q_leiden"]].to_string())

    cols_out = [c for c in comp.columns if c != "key"]
    comp[cols_out].to_csv(OUT_DIR / "leiden_vs_optimal.csv", index=False)
    print(f"Saved {len(comp)} conditions → {OUT_DIR / 'leiden_vs_optimal.csv'}")

    all_pairs = load_families()
    name_to_code = dict(all_pairs)
    g_full = load_graph()
    concept_set = load_concept_set()
    deg = degeneracy_check(comp, g_full, concept_set, name_to_code)
    deg.to_csv(OUT_DIR / "degeneracy_check.csv", index=False)
    print(f"Saved degeneracy check ({deg[['Analysis', 'Family', 'Condition']].drop_duplicates().shape[0]}"
          f" conditions) → {OUT_DIR / 'degeneracy_check.csv'}")

    mc = main_consistency()
    mc.to_csv(OUT_DIR / "main_consistency.csv", index=False)
    print(f"Optimal T = {WALK_T} reproduces main analysis: "
          f"{int(mc['identical'].sum())}/{len(mc)} families")

    ov = overview(comp)
    ov.to_csv(OUT_DIR / "comparison_overview.csv", index=False)
    print(ov.to_string(index=False))

    families = [n for n, _ in all_pairs]
    for analysis in ANALYSES:
        plot_overview(comp, analysis, families, OUT_DIR / f"{analysis}_overview.png")
    print(f"Saved figures → {OUT_DIR}")


if __name__ == "__main__":
    main()
