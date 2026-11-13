#!/usr/bin/env python3
"""EFFECT SIZES — S2 step 12 (physical vs. non-physical comparison).

Reads the per-family post-walk centrality CSVs written by script 02 and
rewrites output/main/centrality_summary_postwalk.csv in place with:

  Top-value columns (paired with the existing name columns)
    Top_by_strength_value / Top2_by_strength_value   top-1 / top-2 Strength score
    Top_by_betw_value      / Top2_by_betw_value       top-1 / top-2 Betweenness score

  Effect sizes (phy vs nonphy)
    cohens_d_strength, hedges_g_strength              d (pooled SD) and small-sample g
    cohens_d_betweenness, hedges_g_betweenness        same on Betweenness (no CI)

  Uncertainty (Strength only)
    hedges_g_strength_ci_low / _ci_high              percentile 95% bootstrap CI
                                                     (B = 10,000, seed = 0)

  Small-sample flags
    min_group_n = min(n_phy, n_nonphy)
    d_valid     = True iff min_group_n >= 2

When min_group_n < 2 the non-physical group has no variance, so the pooled SD
reduces to the physical-group SD and ``d`` is really a z-score of a single
point. The value is kept but ``d_valid`` is False and the CI is left blank.

Group means, effect sizes and CIs are stored at full precision; values are
rounded once, when the manuscript tables (Table 2, Supplementary Table S6)
are made.

Pure standard library (no numpy/pandas/scipy).

Usage:
    python src/03_effect_sizes.py       (requires script 02 to have run)
"""
import csv
import math
import random
from pathlib import Path

from config import MAIN_DIR

SUMMARY_CSV = MAIN_DIR / "centrality_summary_postwalk.csv"
FAM_DIR = MAIN_DIR / "families"

B_BOOT = 10000
SEED = 0
CI_LOW_PCT, CI_HIGH_PCT = 2.5, 97.5


def _var(xs):
    """Sample variance (ddof=1); 0.0 for n < 2."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((v - m) ** 2 for v in xs) / (n - 1)


def cohens_d(a, b):
    """Cohen's d with pooled SD. NaN if the pooled SD is 0 or df <= 0."""
    na, nb = len(a), len(b)
    if na + nb - 2 <= 0:
        return float("nan")
    pooled = math.sqrt((_var(a) * (na - 1) + _var(b) * (nb - 1)) / (na + nb - 2))
    if pooled == 0:
        return float("nan")
    return (sum(a) / na - sum(b) / nb) / pooled


def hedges_g(a, b):
    """Small-sample-corrected d: g = d * (1 - 3 / (4N - 9)), N = na + nb."""
    d = cohens_d(a, b)
    if math.isnan(d):
        return float("nan")
    n = len(a) + len(b)
    j = 1.0 - 3.0 / (4.0 * n - 9.0)
    return d * j


def _percentile(sorted_vals, pct):
    """Linear-interpolation percentile on a pre-sorted list."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = int(math.floor(rank))
    hi = int(math.ceil(rank))
    if lo == hi:
        return sorted_vals[lo]
    frac = rank - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def bootstrap_g_ci(a, b, rng, n_boot=B_BOOT):
    """Stratified percentile bootstrap CI for Hedges' g. Blank if min group < 2."""
    if min(len(a), len(b)) < 2:
        return None, None
    reps = []
    na, nb = len(a), len(b)
    for _ in range(n_boot):
        ra = [a[rng.randrange(na)] for _ in range(na)]
        rb = [b[rng.randrange(nb)] for _ in range(nb)]
        g = hedges_g(ra, rb)
        if not math.isnan(g):
            reps.append(g)
    if len(reps) < n_boot // 2:  # too many degenerate resamples to trust
        return None, None
    reps.sort()
    return _percentile(reps, CI_LOW_PCT), _percentile(reps, CI_HIGH_PCT)


def _fmt(x, nd=None):
    """Full precision by default; tables are rounded once, when made."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return str(x) if nd is None else f"{round(x, nd)}"


def _show(v):
    return f"{float(v):+.3f}" if v != "" else "—"


def top_values(rows, metric):
    """Top-1 / top-2 positive values of a metric, matching the name columns'
    'exclude zeros' rule. Returns ('' , '') style blanks when absent."""
    vals = sorted((float(r[metric]) for r in rows if float(r[metric]) > 0), reverse=True)
    v1 = vals[0] if len(vals) >= 1 else ""
    v2 = vals[1] if len(vals) >= 2 else ""
    return v1, v2


def main():
    rng = random.Random(SEED)

    with open(SUMMARY_CSV, newline="") as fh:
        summary = list(csv.DictReader(fh))

    out_rows = []
    for row in summary:
        fam = row["Family"]
        cent_path = FAM_DIR / fam / f"{fam}_centrality_postwalk.csv"
        with open(cent_path, newline="") as fh:
            cent = list(csv.DictReader(fh))

        phy_s = [float(r["Strength"]) for r in cent if r["Comm"] == "phy"]
        non_s = [float(r["Strength"]) for r in cent if r["Comm"] == "nonphy"]
        phy_b = [float(r["Betweenness"]) for r in cent if r["Comm"] == "phy"]
        non_b = [float(r["Betweenness"]) for r in cent if r["Comm"] == "nonphy"]

        min_n = min(len(phy_s), len(non_s))
        d_s = cohens_d(phy_s, non_s)
        g_s = hedges_g(phy_s, non_s)
        d_b = cohens_d(phy_b, non_b)
        g_b = hedges_g(phy_b, non_b)
        ci_lo, ci_hi = bootstrap_g_ci(phy_s, non_s, rng)

        s1, s2 = top_values(cent, "Strength")
        b1, b2 = top_values(cent, "Betweenness")
        # keep the "—" convention where the paired name column is "—"
        b1_out = "" if row.get("Top_by_betw", "") in ("", "—") else b1
        b2_out = "—" if row.get("Top2_by_betw", "") in ("", "—") else b2
        s2_out = "—" if row.get("Top2_by_strength", "") in ("", "—") else s2

        out_rows.append({
            "Family": fam,
            "Phys.": row["Phys."],
            "Nonphys.": row["Nonphys."],
            "Phys. conc.": row["Phys. conc."],
            "Nonphys. conc.": row["Nonphys. conc."],
            "Top_by_strength": row["Top_by_strength"],
            "Top_by_strength_value": (str(int(round(s1))) if s1 != "" else ""),
            "Top2_by_strength": row["Top2_by_strength"],
            "Top2_by_strength_value": (str(int(round(s2))) if isinstance(s2, float) and s2_out != "—" else s2_out),
            "Top_by_betw": row["Top_by_betw"],
            "Top_by_betw_value": (_fmt(b1, 2) if b1_out != "" else ""),
            "Top2_by_betw": row["Top2_by_betw"],
            "Top2_by_betw_value": (_fmt(b2, 2) if isinstance(b2, float) and b2_out != "—" else "—"),
            "Density": row["Density"],
            "cohens_d_strength": _fmt(d_s),
            "hedges_g_strength": _fmt(g_s),
            "hedges_g_strength_ci_low": _fmt(ci_lo),
            "hedges_g_strength_ci_high": _fmt(ci_hi),
            "cohens_d_betweenness": _fmt(d_b),
            "hedges_g_betweenness": _fmt(g_b),
            "min_group_n": min_n,
            "d_valid": min_n >= 2,
        })

    cols = ["Family", "Phys.", "Nonphys.", "Phys. conc.", "Nonphys. conc.",
            "Top_by_strength", "Top_by_strength_value",
            "Top2_by_strength", "Top2_by_strength_value",
            "Top_by_betw", "Top_by_betw_value",
            "Top2_by_betw", "Top2_by_betw_value",
            "Density",
            "cohens_d_strength", "hedges_g_strength",
            "hedges_g_strength_ci_low", "hedges_g_strength_ci_high",
            "cohens_d_betweenness", "hedges_g_betweenness",
            "min_group_n", "d_valid"]

    with open(SUMMARY_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Rewrote {SUMMARY_CSV}")
    print(f"  B={B_BOOT}, seed={SEED}, percentile {CI_LOW_PCT}-{CI_HIGH_PCT}% CI on Hedges' g (Strength)")
    hdr = f"{'Family':18s} {'d_S':>7s} {'g_S':>7s} {'CI':>18s} {'d_B':>7s} {'g_B':>7s} {'minN':>4s} valid"
    print(hdr)
    for r in out_rows:
        ci = f"[{_show(r['hedges_g_strength_ci_low'])},{_show(r['hedges_g_strength_ci_high'])}]"
        print(f"{r['Family']:18s} {_show(r['cohens_d_strength']):>7s} {_show(r['hedges_g_strength']):>7s} "
              f"{ci:>18s} {_show(r['cohens_d_betweenness']):>7s} {_show(r['hedges_g_betweenness']):>7s} "
              f"{str(r['min_group_n']):>4s} {r['d_valid']}")


if __name__ == "__main__":
    main()
