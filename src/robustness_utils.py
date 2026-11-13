"""Graph-variant construction and community detection for the robustness sweeps.

Implements the shared machinery of Supplementary Table S2 steps 16–18:

  apply_threshold_binary   — binary edge thresholds t ∈ {5,10,20,30,40,50} (step 16)
  apply_weight_transform   — log1p / sqrt / rank / min–max weight transforms (step 16)
  restrict_to_pain_cc      — after a threshold or transform: drop edges with
                             weight ≤ 0 and isolates, keep the component
                             containing PAIN, check the minimum size
  label_class_count        — number of phy/nonphy classes among a graph's concepts
  run_community_detection  — exact optimal modularity, or Leiden (modularity
                             objective, 10 iterations, resolution = 1 default,
                             seed 42) — selected per sweep in scripts 07–08
  run_k2_detection         — walktrap / fast-greedy / leading-eigenvector cut
                             to exactly 2 communities (step 18)
  compute_graph_stats      — node/edge/density bookkeeping per variant
  TimedDetector            — algorithm-selectable (optimal / Leiden) detection
                             in a worker process with a per-condition time limit
  ConditionStore           — per-condition JSONL log that makes sweeps resumable
  detect_or_reuse          — detect once per fixed graph, or reuse a stored result

Every threshold/transform variant is re-restricted to the PAIN-connected
component, so that, like the main analysis, it is analysed on the concepts
connected to PAIN. Rank weights give tied weights their average rank.
"""
import hashlib
import inspect
import json
import multiprocessing as mp
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import igraph as ig
import numpy as np

from network_utils import optimal_communities

MIN_NODES = 4
MIN_EDGES = 3


def compute_graph_stats(g, weight_attr="weight"):
    """Return basic graph statistics as a dict."""
    n_nodes = g.vcount()
    n_edges = g.ecount()
    density = g.density()
    n_components = len(g.clusters())
    avg_degree = float(np.mean(g.degree())) if n_nodes > 0 else 0.0
    if weight_attr in g.es.attributes() and n_nodes > 0:
        avg_strength = float(np.mean(g.strength(weights=weight_attr)))
    else:
        avg_strength = avg_degree
    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": density,
        "n_components": n_components,
        "avg_degree": avg_degree,
        "avg_strength": avg_strength,
    }


def run_community_detection(g, method="leiden", weight_attr="weight", seed=42):
    """Detect communities using the specified method."""
    ig.set_random_number_generator(random.Random(seed))
    if method == "leiden":
        return g.community_leiden(
            weights=weight_attr,
            objective_function="modularity",
            n_iterations=10,
        )
    elif method == "walktrap":
        return g.community_walktrap(weights=weight_attr).as_clustering()
    elif method == "optimal":
        return optimal_communities(g, weights=weight_attr)
    else:
        raise ValueError(f"Unknown community method: {method}")


def run_k2_detection(g, algorithm, weight_attr="weight", seed=42):
    """Force exactly two communities under an independent algorithm."""
    ig.set_random_number_generator(random.Random(seed))
    w = g.es[weight_attr] if weight_attr in g.es.attributes() else None
    if algorithm == "walktrap":
        return g.community_walktrap(weights=w).as_clustering(2)
    elif algorithm == "fastgreedy":
        return g.community_fastgreedy(weights=w).as_clustering(2)
    elif algorithm == "leading_eigenvector":
        return g.community_leading_eigenvector(clusters=2, weights=w)
    else:
        raise ValueError(f"Unknown k2 algorithm: {algorithm}")


def apply_threshold_binary(g_sub, threshold, weight_attr="weight"):
    """
    Keep edges with weight >= threshold, binarize all kept edges to weight=1.0.
    Returns None if no edge survives; size and PAIN-CC are checked by
    restrict_to_pain_cc.
    """
    keep_eids = [e.index for e in g_sub.es if e[weight_attr] >= threshold]
    if not keep_eids:
        return None
    g_new = g_sub.subgraph_edges(keep_eids, delete_vertices=True)
    g_new.es[weight_attr] = [1.0] * g_new.ecount()
    return g_new


def restrict_to_pain_cc(g, weight_attr="weight", gloss_attr="label",
                        min_nodes=MIN_NODES, min_edges=MIN_EDGES):
    """Re-apply the PAIN-CC restriction of the main analysis to a variant.

    Edges with weight <= 0 (e.g. the minimum edge under min–max scaling)
    carry no modularity weight and are removed before connectivity is
    judged. Returns (graph, None), or (None, reason) when the variant is
    not evaluable.
    """
    if g is None or g.ecount() == 0:
        return None, "no edges left"
    keep = [e.index for e in g.es if e[weight_attr] > 0]
    g = g.subgraph_edges(keep, delete_vertices=True)
    labels = g.vs[gloss_attr]
    if "PAIN" not in labels:
        return None, "PAIN removed"
    cc = g.connected_components().membership
    pain_cc = cc[labels.index("PAIN")]
    g = g.subgraph([i for i, c in enumerate(cc) if c == pain_cc])
    if g.vcount() < min_nodes or g.ecount() < min_edges:
        return None, f"PAIN-CC too small ({g.vcount()} nodes, {g.ecount()} edges)"
    return g, None


def label_class_count(g, label_csv, gloss_attr="label"):
    """Number of distinct phy/nonphy classes among the graph's labelled concepts."""
    from ari_utils import _load_label_df
    df = _load_label_df(label_csv)
    by_gloss = dict(zip(df["Gloss"], df["Comm"]))
    return len({by_gloss[gl] for gl in g.vs[gloss_attr] if gl in by_gloss})


def apply_weight_transform(g_sub, transform, weight_attr="weight"):
    """
    Apply weight transformation on a copy of g_sub.
    Supported transforms: 'log1p', 'sqrt', 'rank', 'minmax'.
    """
    g_new = g_sub.copy()
    ws = np.array(g_new.es[weight_attr], dtype=float)

    if transform == "log1p":
        ws = np.log1p(ws)
    elif transform == "sqrt":
        ws = np.sqrt(ws)
    elif transform == "rank":
        from scipy.stats import rankdata
        ws = rankdata(ws, method="average") / float(len(ws))  # ties share a rank
    elif transform == "minmax":
        w_min, w_max = ws.min(), ws.max()
        if w_max > w_min:
            ws = (ws - w_min) / (w_max - w_min)
        else:
            ws = np.ones_like(ws)
    else:
        raise ValueError(f"Unknown transform: {transform}")

    g_new.es[weight_attr] = ws.tolist()
    return g_new


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm-selectable sweeps (scripts 07–08): timing, time limit, resume
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHMS = ("optimal", "leiden")
DEFAULT_TIMEOUT_S = 600


def algorithm_parameters(algorithm, seed=42):
    """Parameters actually passed to (or defaulted by) the detection call.

    Leiden defaults (resolution, beta) are read from the installed igraph
    signature rather than restated, so the record reflects what really ran.
    """
    if algorithm == "leiden":
        sig = inspect.signature(ig.Graph.community_leiden).parameters
        return {
            "function": "igraph.Graph.community_leiden",
            "objective_function": "modularity",
            "n_iterations": 10,
            "resolution": sig["resolution"].default,
            "beta": sig["beta"].default,
            "weights": "weight",
            "seed": seed,
            "rng": "random.Random(seed), reset before every call",
        }
    if algorithm == "optimal":
        return {
            "function": "igraph.Graph.community_optimal_modularity",
            "method": "exact modularity maximisation (integer programming, GLPK)",
            "nonpositive_Q": "single community",
            "weights": "weight",
            "seed": None,
        }
    raise ValueError(f"Unknown algorithm: {algorithm}")


def software_versions():
    import matplotlib, pandas, scipy, sklearn
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "igraph": ig.__version__,
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
    }


def graph_fingerprint(g, gloss_attr="label", weight_attr="weight"):
    """Order-independent hash of node labels, edges and exact edge weights."""
    labels = g.vs[gloss_attr]
    edges = sorted(
        tuple(sorted((labels[e.source], labels[e.target]))) + (repr(float(e[weight_attr])),)
        for e in g.es
    )
    payload = json.dumps({"nodes": sorted(labels), "edges": edges})
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _detect_worker(g, algorithm, seed):
    t0 = time.perf_counter()
    comms = run_community_detection(g, method=algorithm, seed=seed)
    elapsed = time.perf_counter() - t0
    membership = list(comms.membership)
    return membership, g.modularity(membership, weights="weight"), elapsed


class TimedDetector:
    """Runs one detection call per graph in a persistent worker process.

    A call exceeding `timeout` seconds kills the worker and is reported as
    status "timeout"; there is no fallback to another algorithm.
    """

    def __init__(self, algorithm, seed=42, timeout=DEFAULT_TIMEOUT_S):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        self.algorithm = algorithm
        self.seed = seed
        self.timeout = timeout
        self.params = algorithm_parameters(algorithm, seed)
        self._ctx = mp.get_context("spawn")
        self._pool = None

    def detect(self, g):
        if self._pool is None:
            self._pool = self._ctx.Pool(1)
        t0 = time.perf_counter()
        job = self._pool.apply_async(_detect_worker, (g, self.algorithm, self.seed))
        try:
            membership, q, elapsed = job.get(timeout=self.timeout)
        except mp.TimeoutError:
            self.close()
            return {"status": "timeout",
                    "error": f"exceeded {self.timeout} s",
                    "t_detect_wall": round(time.perf_counter() - t0, 4)}
        except Exception as e:
            return {"status": "error", "error": repr(e),
                    "t_detect_wall": round(time.perf_counter() - t0, 4)}
        return {"status": "ok", "membership": membership, "Q": q,
                "t_detect": round(elapsed, 6),
                "t_detect_wall": round(time.perf_counter() - t0, 4)}

    def close(self):
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None


class ConditionStore:
    """Append-only JSONL log with one record per condition (latest wins)."""

    def __init__(self, path, fresh=False):
        self.path = Path(path)
        self.records = {}
        if fresh and self.path.exists():
            self.path.unlink()
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.records[rec["key"]] = rec

    def reusable(self, key, fingerprint, n_perm, params):
        rec = self.records.get(key)
        if (rec and rec["status"] == "ok"
                and rec["graph_fingerprint"] == fingerprint
                and rec["n_perm"] == n_perm and rec["params"] == params):
            return rec
        return None

    def save(self, rec):
        rec = {**rec, "saved_at": datetime.now(timezone.utc).isoformat()}
        self.records[rec["key"]] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()


def detect_or_reuse(g, key, store, detector, n_perm, gloss_attr="label"):
    """Detect communities once for this fixed graph, or reuse a stored result.

    Returns (record, reused). A reused record carries the summary row computed
    when it was first run, so permutation tests are never redone either.
    """
    fp = graph_fingerprint(g, gloss_attr=gloss_attr)
    rec = store.reusable(key, fp, n_perm, detector.params)
    if rec is not None:
        return rec, True
    det = detector.detect(g)
    rec = {"key": key, "algorithm": detector.algorithm, "params": detector.params,
           "n_perm": n_perm, "graph_fingerprint": fp,
           "n_nodes": g.vcount(), "n_edges": g.ecount()}
    membership = det.pop("membership", None)
    rec.update(det)
    if membership is not None:
        rec["membership"] = dict(zip(g.vs[gloss_attr], membership))
        rec["k_all"] = len(set(membership))
    return rec, False


def write_run_outputs(out_dir, store, keys, algorithm, params, timeout, n_perm, argv,
                      started_at, t_total):
    """conditions.csv (status, Q, timings), failures.csv and run_metadata.json."""
    import pandas as pd
    cols = ["key", "status", "error", "n_nodes", "n_edges", "graph_fingerprint",
            "k_all", "Q", "t_build", "t_detect", "t_detect_wall", "t_stats", "reused"]
    rows = [{c: store.records[k].get(c) for c in cols} for k in keys if k in store.records]
    cond = pd.DataFrame(rows, columns=cols).astype(
        {"n_nodes": "Int64", "n_edges": "Int64", "k_all": "Int64"})
    cond.to_csv(Path(out_dir) / "conditions.csv", index=False)
    fail = cond[cond["status"].isin(["timeout", "error"])]
    fail.to_csv(Path(out_dir) / "failures.csv", index=False)
    meta = {
        "algorithm": algorithm,
        "parameters": params,
        "timeout_s_per_condition": timeout,
        "n_perm": n_perm,
        "command": " ".join(argv),
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "wall_time_s": round(t_total, 2),
        "status_counts": cond["status"].value_counts().to_dict(),
        "timing_totals_s": {c: round(float(pd.to_numeric(cond[c], errors="coerce").sum()), 4)
                            for c in ("t_build", "t_detect", "t_detect_wall", "t_stats")},
        "software": software_versions(),
    }
    (Path(out_dir) / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    return cond, fail
