#!/usr/bin/env python3
"""ROBUSTNESS SUMMARY TABLES — consolidates the sweeps of scripts 07–09.

Reads the raw robustness CSVs of one community-detection algorithm
(output/robustness/{algorithm}/) plus the main-analysis summary
(output/main/summary.csv, always exact optimal modularity) and produces the
consolidated tables behind the supplementary robustness tables: per-family
threshold summary, walk-step summary, k = 2 comparison, and the cross-family
robustness overview. Every table carries an Algorithm column naming the
algorithm of its robustness sweeps; numbers from different algorithms are
never mixed in one table. Significance counts use the exact permutation p
(significant_05 / significant_10) over evaluable conditions; Not_Evaluable
counts the conditions of the sweep that could not be analysed (reasons in
the sweep's conditions.csv).

Outputs (output/main/robustness/{algorithm}/):
  threshold_summary_table.csv
  walkstep_summary_table.csv
  k2_comparison_table.csv
  cross_family_robustness_table.csv

Usage:
    python src/10_robustness_summary.py                     (requires 01, 07, 08, 09)
    python src/10_robustness_summary.py --algorithm leiden
"""

import argparse

import pandas as pd

from ari_utils import fmt_p
from config import MAIN_DIR, ROBUSTNESS_DIR
from robustness_utils import ALGORITHMS

# ── Paths ─────────────────────────────────────────────────────────────────────
MAIN_CSV   = MAIN_DIR / "summary.csv"
ROB_DIR    = None   # output/robustness/{algorithm}, set in main()
OUT_DIR    = None   # output/main/robustness/{algorithm}, set in main()
ALGORITHM  = None

FAMILY_ORDER = [
    "Afro-Asiatic", "Austroasiatic", "Austronesian", "Indo-European",
    "Nakh-Daghestanian", "Pano-Tacanan", "Sino-Tibetan", "Tai-Kadai",
    "Tupian", "Turkic", "Uralic", "Universal",
]


def not_evaluable_counts(analysis):
    """Per-family number of not-evaluable conditions and their reasons."""
    cond = pd.read_csv(ROB_DIR / analysis / "conditions.csv")
    cond = cond[cond["status"] == "not_evaluable"].copy()
    cond["Family"] = cond["key"].str.split("|").str[0]
    out = {}
    for fam, sub in cond.groupby("Family"):
        reasons = sub["error"].str.replace(r" \(.*\)", "", regex=True).value_counts()
        out[fam] = (len(sub), "; ".join(f"{r} ×{n}" for r, n in reasons.items()))
    return out


# ── Table 1: Threshold / transform summary ───────────────────────────────────

def build_threshold_summary() -> pd.DataFrame:
    df = pd.read_csv(ROB_DIR / "threshold" / "summary.csv")
    ne = not_evaluable_counts("threshold")

    rows = []
    for fam in FAMILY_ORDER:
        sub = df[df["Family"] == fam]
        n_ne, why = ne.get(fam, (0, ""))
        if sub.empty:
            if n_ne:
                rows.append({"Family": fam, "Significant_p05": "0/0",
                             "Marginal_p10": "0/0", "Best_ARI": float("nan"),
                             "Best_Condition": "", "Not_Evaluable": n_ne,
                             "Not_Evaluable_Reasons": why})
            continue
        total = len(sub)
        sig   = sub["significant_05"].sum()
        marg  = sub["significant_10"].sum()
        best_idx = sub["ARI"].idxmax()
        best_ari = sub.loc[best_idx, "ARI"]
        # build condition label from Method + Condition
        method    = sub.loc[best_idx, "Method"]
        condition = sub.loc[best_idx, "Condition"]
        best_cond = f"{method}_{condition}" if method != "full" else "full"
        rows.append({
            "Family":          fam,
            "Significant_p05": f"{sig}/{total}",
            "Marginal_p10":    f"{marg}/{total}",
            "Best_ARI":        round(best_ari, 4),
            "Best_Condition":  best_cond,
            "Not_Evaluable":   n_ne,
            "Not_Evaluable_Reasons": why,
        })

    out = pd.DataFrame(rows)
    out.insert(1, "Algorithm", ALGORITHM)
    path = OUT_DIR / "threshold_summary_table.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")
    return out


# ── Table 2: Walk-step sensitivity summary ───────────────────────────────────

def build_walkstep_summary() -> pd.DataFrame:
    df = pd.read_csv(ROB_DIR / "walkstep" / "summary.csv")
    ne = not_evaluable_counts("walkstep")

    rows = []
    for fam in FAMILY_ORDER:
        sub = df[df["Family"] == fam]
        if sub.empty:
            continue
        total = len(sub)
        sig   = sub["significant_05"].sum()
        best_idx  = sub["ARI"].idxmax()
        best_ari  = sub.loc[best_idx, "ARI"]
        best_step = sub.loc[best_idx, "walk_steps"]
        best_conf = sub.loc[best_idx, "Config"]
        best_cond = f"steps={best_step}, {best_conf}"
        n_ne, why = ne.get(fam, (0, ""))
        rows.append({
            "Family":          fam,
            "Significant_p05": f"{sig}/{total}",
            "Best_ARI":        round(best_ari, 4),
            "Best_Condition":  best_cond,
            "Not_Evaluable":   n_ne,
            "Not_Evaluable_Reasons": why,
        })

    out = pd.DataFrame(rows)
    out.insert(1, "Algorithm", ALGORITHM)
    path = OUT_DIR / "walkstep_summary_table.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")
    return out


# ── Table 3: k=2 constrained comparison ──────────────────────────────────────

def build_k2_comparison() -> pd.DataFrame:
    """Use postwalk_raw config, report unconstrained vs walktrap k=2."""
    df = pd.read_csv(ROB_DIR / "k2" / "comparison_table.csv")
    # Focus on the primary postwalk_raw configuration
    sub = df[df["Config"] == "postwalk_raw"].copy()

    rows = []
    for fam in FAMILY_ORDER:
        row = sub[sub["Family"] == fam]
        if row.empty:
            # fall back to any config
            row = df[df["Family"] == fam].head(1)
        if row.empty:
            continue
        r = row.iloc[0]

        ari_unc = r["ARI_unconstrained"]
        k_unc   = r["k_unconstrained"]
        p_unc   = r["p_unconstrained"]
        sig_unc = r["sig_unconstrained"]

        ari_wt  = r["ARI_walktrap_k2"]
        p_wt    = r["p_walktrap_k2"]
        sig_wt  = r["sig_walktrap_k2"]

        # Determine direction of change
        if pd.isna(ari_wt):
            change = "walktrap failed"
        elif abs(float(ari_wt) - float(ari_unc)) < 0.001:
            change = "no change"
        elif float(ari_wt) > float(ari_unc):
            if not sig_unc and sig_wt:
                change = "newly significant"
            else:
                change = "ARI increased"
        else:
            change = "worse"

        rows.append({
            "Family":             fam,
            "ARI_unconstrained":  round(float(ari_unc), 4),
            "k_unconstrained":    int(float(k_unc)) if not pd.isna(k_unc) else "",
            "p_unconstrained":    fmt_p(float(p_unc)),
            "sig_unconstrained":  sig_unc,
            "ARI_walktrap_k2":    round(float(ari_wt), 4) if not pd.isna(ari_wt) else "",
            "p_walktrap_k2":      fmt_p(float(p_wt)) if not pd.isna(p_wt) else "",
            "sig_walktrap_k2":    sig_wt if not pd.isna(ari_wt) else "",
            "Change":             change,
        })

    out = pd.DataFrame(rows)
    out.insert(1, "Algorithm", ALGORITHM)
    path = OUT_DIR / "k2_comparison_table.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")
    return out


# ── Table 4: Cross-family robustness summary ─────────────────────────────────

def _overall_evaluation(main_sig: bool, main_p: float,
                        threshold_sig: int, threshold_total: int,
                        walkstep_sig: int, walkstep_total: int,
                        k2_change: str) -> str:
    """Assign a qualitative robustness label."""
    # Proportion of conditions that are significant
    t_rate = threshold_sig / threshold_total if threshold_total > 0 else 0.0
    w_rate = walkstep_sig  / walkstep_total  if walkstep_total  > 0 else 0.0

    if not main_sig and t_rate == 0 and w_rate == 0:
        return "Structural mismatch"
    if t_rate >= 0.9 and w_rate >= 0.9 and main_sig:
        return "Strongest support, fully robust"
    if t_rate >= 0.5 or w_rate >= 0.5:
        return "Conditionally significant"
    if main_p < 0.10:
        return "Robust but near-significant"
    return "Non-significant (low power)"


def build_cross_family_table(thresh_df: pd.DataFrame,
                             walk_df:  pd.DataFrame,
                             k2_df:    pd.DataFrame) -> pd.DataFrame:
    main = pd.read_csv(MAIN_CSV)
    thresh = pd.read_csv(ROB_DIR / "threshold" / "summary.csv")
    walk   = pd.read_csv(ROB_DIR / "walkstep"  / "summary.csv")

    rows = []
    for fam in FAMILY_ORDER:
        m = main[main["Family"] == fam]
        if m.empty:
            continue
        m = m.iloc[0]

        # Main analysis
        ari_main = m["ARI_Local"]
        p_main   = m["p_Exact"]   # exact p, as in Table 3
        sig_main = p_main < 0.05

        # Threshold counts
        t_sub = thresh[thresh["Family"] == fam]
        t_sig  = int(t_sub["significant_05"].sum())
        t_tot  = len(t_sub)

        # Walk-step counts
        w_sub  = walk[walk["Family"] == fam]
        w_sig  = int(w_sub["significant_05"].sum())
        w_tot  = len(w_sub)

        # k2 change description
        k2_row = k2_df[k2_df["Family"] == fam]
        k2_change = k2_row["Change"].iloc[0] if not k2_row.empty else "—"

        # Threshold verdict string
        t_row = thresh_df[thresh_df["Family"] == fam]
        t_verdict = t_row["Significant_p05"].iloc[0] if not t_row.empty else "—"

        # Walk verdict string
        w_row = walk_df[walk_df["Family"] == fam]
        w_verdict = w_row["Significant_p05"].iloc[0] if not w_row.empty else "—"

        overall = _overall_evaluation(
            sig_main, p_main, t_sig, t_tot, w_sig, w_tot, k2_change)

        rows.append({
            "Family":              fam,
            "Main_ARI":            round(float(ari_main), 4),
            "Main_p":              fmt_p(float(p_main)),
            "Main_significant":    sig_main,
            "Threshold":           f"{t_verdict} ({'✓' if t_sig > 0 else '✗'})",
            "K2":                  k2_change,
            "Walk_Steps":          f"{w_verdict} ({'✓' if w_sig > 0 else '✗'})",
            "Overall_Evaluation":  overall,
        })

    out = pd.DataFrame(rows)
    out.insert(1, "Algorithm", ALGORITHM)
    path = OUT_DIR / "cross_family_robustness_table.csv"
    out.to_csv(path, index=False)
    print(f"Saved: {path}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    global ROB_DIR, OUT_DIR, ALGORITHM
    parser = argparse.ArgumentParser(description="CLICS4 robustness summary tables")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="optimal",
                        help="community detection of the robustness sweeps to summarise")
    args = parser.parse_args()
    ALGORITHM = args.algorithm
    ROB_DIR = ROBUSTNESS_DIR / ALGORITHM
    OUT_DIR = MAIN_DIR / "robustness" / ALGORITHM
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"CLICS4 PAIN-CC: ROBUSTNESS SUMMARY TABLES ({ALGORITHM})")
    print("=" * 60)

    thresh_df = build_threshold_summary()
    walk_df   = build_walkstep_summary()
    k2_df     = build_k2_comparison()
    summary   = build_cross_family_table(thresh_df, walk_df, k2_df)

    print("\nCross-family robustness overview:")
    print(summary[["Family", "Main_ARI", "Main_p", "Overall_Evaluation"]].to_string(index=False))
    print(f"\nOutputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
