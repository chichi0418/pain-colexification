"""PUBLICATION FIGURES — per-family network panels.

Renders the network figures for the 11 language families + Universal
(12 panels, 4×3 grid), from the satellite-inclusive post-walk pipeline
(build_aligned_pain_subgraph): the component containing PAIN is the core
(main-text network figure), the remaining components are the satellites
(supplementary satellite-components figure).

  fig1_network_no_satellite   — PAIN-CC only per family (main text Figure 1)
  fig3_network_satellite_only — satellite components only (supplementary)

The manuscript's Figure 2 (hierarchical clustering dendrogram) is NOT drawn
here — it is output/main/cross_family/dendrogram.pdf, produced by script 04.

Each figure is exported as PDF, EPS, TIFF (300 dpi, LZW) and PNG (200 dpi
preview). No titles/captions are embedded inside the figure files (journal
guidelines require legends to live in the manuscript text); family names are
shown inside each panel as identifiers only.

Outputs: output/figures/{pdf,eps,tiff,png}/

Usage:
    python src/15_figures.py
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, replace as dataclass_replace
from pathlib import Path

import igraph as ig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FIGURES_DIR, WALK_T, GLOSS_ATTR,
    load_graph, load_concept_set, load_families,
)
from network_utils import build_aligned_pain_subgraph


# ─────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────
RANDOM_SEED = 100

OUT_DIR = FIGURES_DIR

# 11 families + Universal (alphabetical, Universal last — mirrors clics3).
FAMILIES = [
    "Afro-Asiatic",
    "Austroasiatic",
    "Austronesian",
    "Indo-European",
    "Nakh-Daghestanian",
    "Pano-Tacanan",
    "Sino-Tibetan",
    "Tai-Kadai",
    "Tupian",
    "Turkic",
    "Uralic",
    "Universal",
]
GRID_ROWS, GRID_COLS = 4, 3        # 12 panels, no blanks
PANEL_INCH = (5.0, 5.0)            # larger panels for readable nodes/labels

PALETTE_NAME = "tab10"             # saturated, clearly distinguishable communities
EDGE_COLOR = "#7d7d7d"             # single neutral grey edge color (PAIN-style)
SATELLITE_COLOR = "#5b8db8"        # one colour for all satellite nodes

LABEL_FONTSIZE = 8.5               # readable at publication size
PANEL_TITLE_FONTSIZE = 17.25       # family name above each panel (15 * 1.15)
PANEL_LETTER_FONTSIZE = 18.15      # (a), (b), ... panel identifiers (11 * 1.65)
EDGE_ALPHA = 0.7

DPI_PNG = 200
DPI_TIFF = 300

# Vector-friendly defaults: real text in PDF/EPS instead of outlined glyphs
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman"]
matplotlib.rcParams["mathtext.fontset"] = "stix"


# ─────────────────────────────────────────────────────────────────────────
# Style dataclasses (kept separate per figure type, per spec)
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NoSatelliteStyle:
    node_size_range: tuple = (260, 640)
    edge_lw_range: tuple = (0.8, 3.0)
    edge_alpha: float = EDGE_ALPHA
    edge_color: str = EDGE_COLOR
    label_fontsize: float = LABEL_FONTSIZE - 0.5
    label_min_fontsize: float = 6.0    # auto-shrink floor for very long labels
    fr_iter: int = 800


@dataclass(frozen=True)
class SatelliteOnlyStyle:
    # Satellite panels are sparse (2-6 nodes) with lots of whitespace, so nodes
    # and labels are drawn larger than the dense PAIN-component panels (Fig 1).
    node_size_range: tuple = (780, 1550)
    edge_lw_range: tuple = (1.0, 3.0)
    edge_alpha: float = EDGE_ALPHA
    edge_color: str = EDGE_COLOR
    label_fontsize: float = 12.5
    label_min_fontsize: float = 9.0
    fr_iter: int = 500
    color: str = SATELLITE_COLOR


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────
def reseed(tag: str) -> int:
    """Deterministic per-call reseed of all RNGs igraph layout depends on."""
    s = (RANDOM_SEED + zlib.crc32(tag.encode())) & 0xFFFFFFFF
    random.seed(s)
    np.random.seed(s)
    ig.set_random_number_generator(random.Random(s))
    return s


def ensure_output_dirs() -> None:
    for sub in ("pdf", "eps", "tiff", "png"):
        (OUT_DIR / sub).mkdir(parents=True, exist_ok=True)


def export_figure(fig: plt.Figure, stem: str) -> None:
    """Write PDF + EPS + TIFF + PNG for a single figure."""
    fig.savefig(OUT_DIR / "pdf" / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT_DIR / "eps" / f"{stem}.eps", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(
        OUT_DIR / "tiff" / f"{stem}.tiff",
        dpi=DPI_TIFF, bbox_inches="tight", pad_inches=0.05,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(OUT_DIR / "png" / f"{stem}.png",
                dpi=DPI_PNG, bbox_inches="tight", pad_inches=0.05)
    print(f"  wrote {stem}.{{pdf,eps,tiff,png}}")


def fr_layout(g: ig.Graph, niter: int, tag: str) -> np.ndarray:
    """Fruchterman-Reingold with explicit, seeded initial coordinates."""
    s = reseed(tag)
    rng = np.random.default_rng(s)
    init_coords = rng.uniform(-1.0, 1.0, size=(g.vcount(), 2)).tolist()
    weights = "weight" if "weight" in g.es.attributes() else None
    layout = g.layout_fruchterman_reingold(
        weights=weights, niter=niter, seed=init_coords,
    )
    return np.array(list(layout))


def normalise_to_box(coords: np.ndarray, x0: float, x1: float, y0: float, y1: float,
                     pad: float = 0.08) -> np.ndarray:
    if len(coords) == 0:
        return coords
    c_min = coords.min(axis=0)
    c_range = coords.max(axis=0) - c_min
    c_range[c_range == 0] = 1.0
    norm = (coords - c_min) / c_range
    out = np.empty_like(coords)
    out[:, 0] = x0 + pad * (x1 - x0) + norm[:, 0] * (1 - 2 * pad) * (x1 - x0)
    out[:, 1] = y0 + pad * (y1 - y0) + norm[:, 1] * (1 - 2 * pad) * (y1 - y0)
    return out


def scale_range(values, lo: float, hi: float) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return v
    span = max(v.max() - v.min(), 1e-9)
    return lo + (v - v.min()) / span * (hi - lo)


PT_PER_INCH = 72.0
CHAR_WIDTH_FACTOR = 0.72   # avg bold DejaVu-Sans glyph width / fontsize (measured empirically)


def label_radius_box(label: str, fontsize_pt: float, node_size_pts2: float,
                     panel_inch: float) -> float:
    """Effective half-extent (in [0,1] box-fraction units) a node + its centred
    label occupies, so overlap checks can push apart long-label pairs (e.g.
    "DAMAGE OR INJURE") that a node-count-only heuristic would miss."""
    label_halfwidth_in = (len(label) * fontsize_pt * CHAR_WIDTH_FACTOR / PT_PER_INCH) / 2
    circle_radius_in = (np.sqrt(node_size_pts2 / np.pi)) / PT_PER_INCH
    return max(label_halfwidth_in, circle_radius_in) / panel_inch


def node_vertical_halfextent_box(fontsize_pt: float, node_size_pts2: float,
                                 panel_inch: float) -> float:
    """Vertical half-extent (box-fraction units) of a node + its centred label.

    Unlike label_radius_box this ignores label *width* — a centred label extends
    only ~fontsize/2 vertically regardless of how wide it is. Used to fit stacked
    satellite components into their vertical slots without over-shrinking a
    component just because it carries a wide-but-short label (e.g. the
    "DAMAGE OR INJURE" satellite in Fig 3 panels e/f)."""
    label_halfheight_in = (fontsize_pt / PT_PER_INCH) / 2
    circle_radius_in = (np.sqrt(node_size_pts2 / np.pi)) / PT_PER_INCH
    return max(label_halfheight_in, circle_radius_in) / panel_inch


def resolve_overlaps(pos: np.ndarray, x0: float, x1: float, y0: float, y1: float,
                     radii: np.ndarray, gap: float = 0.012,
                     iters: int = 400) -> np.ndarray:
    """Push apart any node pair whose (label-aware) radii overlap — a
    force-directed de-overlap pass — so dense communities and long labels
    (e.g. "DAMAGE OR INJURE", "BAD LUCK") don't collapse on top of each
    other. Keeps every node inside the [x0,x1] x [y0,y1] box."""
    pos = pos.astype(float).copy()
    n = len(pos)
    if n < 2:
        return pos
    rng = np.random.default_rng(0)
    for _ in range(iters):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                min_sep = radii[i] + radii[j] + gap
                d = pos[j] - pos[i]
                dist = float(np.hypot(d[0], d[1]))
                if dist < 1e-9:                       # identical -> nudge apart
                    d = rng.uniform(-1, 1, 2)
                    dist = float(np.hypot(d[0], d[1])) or 1.0
                if dist < min_sep:
                    shift = (min_sep - dist) / 2.0
                    u = d / dist
                    pos[i] -= u * shift
                    pos[j] += u * shift
                    moved = True
        pos[:, 0] = np.clip(pos[:, 0], x0, x1)
        pos[:, 1] = np.clip(pos[:, 1], y0, y1)
        if not moved:
            break
    return pos


# ─────────────────────────────────────────────────────────────────────────
# Per-family pipeline (aligned-T2; cached so we run extraction only once)
# ─────────────────────────────────────────────────────────────────────────
def build_family_data_aligned(g_full: ig.Graph, family_code: str,
                              concept_set: set) -> dict:
    """Run the aligned-T2 pipeline for one family and split into core/satellites.

    Returns dict with g_pain (PAIN component), membership (community labels for
    g_pain remapped to 0..k-1), and satellite_graphs (components without PAIN).
    Returns all-None / empty values if the family yields no valid subgraph.
    """
    empty = dict(g_pain=None, membership=None, satellite_graphs=[])
    try:
        g_sub, n_pain = build_aligned_pain_subgraph(
            g_full, family_code, concept_set, walk_steps=WALK_T
        )
    except Exception as exc:  # noqa: BLE001 — report and skip, like the ARI runner
        print(f"    [WARN] {family_code}: {exc}")
        return empty
    if g_sub is None or g_sub.vcount() == 0:
        return empty

    comps = g_sub.connected_components()
    labels = g_sub.vs[GLOSS_ATTR]
    pain_vertices = [i for i, gl in enumerate(labels) if gl == "PAIN"]
    pain_cid = comps.membership[pain_vertices[0]] if pain_vertices else None

    g_pain, satellite_graphs = None, []
    for cid, comp in enumerate(comps):
        sub = g_sub.subgraph(comp)
        if cid == pain_cid:
            g_pain = sub
        else:
            satellite_graphs.append(sub)

    membership = None
    if g_pain is not None and "community" in g_pain.vs.attributes():
        raw = g_pain.vs["community"]
        remap = {c: i for i, c in enumerate(sorted(set(raw)))}
        membership = [remap[c] for c in raw]

    return dict(g_pain=g_pain, membership=membership,
                satellite_graphs=satellite_graphs)


# ─────────────────────────────────────────────────────────────────────────
# Panel renderers
# ─────────────────────────────────────────────────────────────────────────
def _empty_panel(ax: plt.Axes, family: str, message: str = "—") -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.5, 0.5, message, ha="center", va="center",
            fontsize=10, color="#888888")
    ax.set_title(family, fontsize=PANEL_TITLE_FONTSIZE)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("#cccccc")


def _label_fontsize(label: str, base: float, floor: float) -> float:
    """Shrink very long labels so they don't overflow the node area."""
    n = len(label)
    if n <= 8:
        return base
    if n <= 14:
        return max(floor, base - 1.0)
    if n <= 22:
        return max(floor, base - 2.0)
    return floor


def _draw_graph_in_box(ax: plt.Axes, g: ig.Graph, coords: np.ndarray,
                       node_colors, style, x0=0.02, x1=0.98, y0=0.02, y1=0.98) -> None:
    """Draw an igraph onto an axis whose limits are [0,1] x [0,1].

    Edges are a single neutral grey (PAIN-article style); nodes carry the
    community colour and are stroked with a thin black border. Labels sit on
    top of the nodes with a small white halo for legibility, and very long
    labels are auto-shrunk so they don't overflow.
    """
    import matplotlib.patheffects as pe

    if g.vcount() == 0:
        return
    pos = normalise_to_box(coords, x0, x1, y0, y1) if g.vcount() > 1 else \
        np.array([[(x0 + x1) / 2, (y0 + y1) / 2]])

    labels = g.vs[GLOSS_ATTR]
    fontsizes = [_label_fontsize(lab, style.label_fontsize, style.label_min_fontsize)
                for lab in labels]

    # Node sizes are needed before the de-overlap pass (label-aware radii
    # depend on them), so compute them here and reuse below when drawing.
    if "weight" in g.es.attributes() and g.ecount() > 0:
        strengths = g.strength(weights="weight")
    else:
        strengths = g.degree()
    sizes = scale_range(strengths, *style.node_size_range) if g.vcount() > 1 else \
        [np.mean(style.node_size_range)]

    # De-overlap: spread nodes that the layout collapsed together so their
    # labels stay legible. Per-pair separation accounts for each node's
    # label width (long labels like "DAMAGE OR INJURE" need more room than
    # a node-count-only heuristic gives them), not just node count. Nodes
    # are kept inside an inset box so edge labels are not clipped by the panel.
    if g.vcount() > 1:
        panel_inch = PANEL_INCH[0]
        radii = np.array([
            label_radius_box(lab, fs, sz, panel_inch)
            for lab, fs, sz in zip(labels, fontsizes, sizes)
        ])
        ix = 0.04 * (x1 - x0)   # horizontal margin for (wide) labels
        iy = 0.025 * (y1 - y0)
        pos = resolve_overlaps(pos, x0 + ix, x1 - ix, y0 + iy, y1 - iy, radii)

    weights = g.es["weight"] if "weight" in g.es.attributes() else [1] * g.ecount()
    edge_lw = scale_range(weights, *style.edge_lw_range) if g.ecount() else []

    # edges (drawn first, single neutral color)
    for ei, e in enumerate(g.es):
        x_a, y_a = pos[e.source]
        x_b, y_b = pos[e.target]
        ax.plot([x_a, x_b], [y_a, y_b],
                color=style.edge_color,
                alpha=style.edge_alpha,
                linewidth=edge_lw[ei],
                solid_capstyle="round", zorder=1)

    # nodes
    halo = [pe.withStroke(linewidth=3.0, foreground="white")]
    for i in range(g.vcount()):
        x, y = pos[i]
        ax.scatter(x, y, s=sizes[i], c=[node_colors[i]],
                   edgecolors="black", linewidths=0.7, zorder=3)
        ax.text(x, y, labels[i], ha="center", va="center",
                fontsize=fontsizes[i], fontweight="bold",
                color="black", path_effects=halo, zorder=4)


def render_pain_panel(ax: plt.Axes, family: str, data: dict,
                      style: NoSatelliteStyle) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(family, fontsize=PANEL_TITLE_FONTSIZE)

    g_pain = data["g_pain"]
    membership = data["membership"]
    if g_pain is None or g_pain.vcount() == 0 or membership is None:
        _empty_panel(ax, family, message="(no PAIN component)")
        ax.axis("off")
        return

    cmap = plt.get_cmap(PALETTE_NAME)
    n_clusters = max(len(set(membership)), 1)
    palette = [cmap(i / max(n_clusters, 1)) for i in range(n_clusters)]
    node_colors = [palette[m] for m in membership]

    if g_pain.vcount() == 1:
        coords = np.array([[0.5, 0.5]])
    else:
        coords = fr_layout(g_pain, style.fr_iter, tag=f"figA::{family}")
    # Reserve a bottom band (y0=0.09) so a graph reaching the panel floor —
    # e.g. Universal's BOIL (OF SKIN) — can't collide with the enlarged
    # panel-letter row drawn at the bottom of the axes.
    _draw_graph_in_box(ax, g_pain, coords, node_colors, style, y0=0.09)


def render_satellite_panel(ax: plt.Axes, family: str, data: dict,
                           style: SatelliteOnlyStyle) -> None:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title(family, fontsize=PANEL_TITLE_FONTSIZE)

    sats = data["satellite_graphs"]
    if not sats:
        _empty_panel(ax, family, message="(no satellites)")
        ax.axis("off")
        return

    n = len(sats)
    pad_y = 0.06                                # headroom from panel-letter row
    gap = 0.035                                 # vertical gap between components
    slot_h = (1 - 2 * pad_y - gap * (n - 1)) / n
    for idx, g_sat in enumerate(sats):
        y0 = pad_y + idx * (slot_h + gap)
        y1 = y0 + slot_h

        # very subtle separator line between stacked components (white background)
        if idx > 0:
            ax.axhline(y=y0 - gap / 2, color="#dddddd",
                       linewidth=0.4, zorder=0)

        node_colors = [style.color] * g_sat.vcount()

        # Shrink this component's nodes/labels only as much as its *vertical*
        # footprint requires to fit inside its stacked slot — otherwise a
        # component with a big node bleeds into the neighbouring component or
        # the panel-letter row below. Sparse panels (n=1) are never shrunk.
        #
        # Crucially we size against the vertical half-extent (node radius or
        # label *height*), not the full label width: a wide-but-short label
        # like "DAMAGE OR INJURE" needs horizontal room — handled by the 2D
        # de-overlap pass across the full-width box — not vertical shrinking.
        # This keeps the satellite nodes large and the network readable in the
        # multi-component panels (e, f) while still guaranteeing containment.
        #
        # normalise_to_box can place a node's *centre* anywhere across the
        # slot, so the inset reserved around the slot must itself be >= that
        # node's vertical half-extent. We solve for the largest scale that
        # leaves >= min_content_frac of the slot for actual layout spread,
        # then size the inset from the *scaled* extent so every node is
        # provably contained inside [y0, y1] wherever the layout puts it.
        comp_labels = g_sat.vs[GLOSS_ATTR]
        fontsizes0 = [_label_fontsize(lab, style.label_fontsize, style.label_min_fontsize)
                     for lab in comp_labels]
        max_r = max(
            (node_vertical_halfextent_box(fs, style.node_size_range[1], PANEL_INCH[0])
             for fs in fontsizes0),
            default=0.0,
        )
        base_iy = 0.015
        min_content_frac = 0.4
        if max_r > 0:
            budget = slot_h * (1 - min_content_frac) / 2 - base_iy
            scale = float(np.clip(budget / max_r, 0.45, 1.0))
        else:
            scale = 1.0
        iy_dynamic = min(base_iy + scale * max_r, slot_h / 2 - 0.01)
        comp_style = style if scale >= 0.999 else dataclass_replace(
            style,
            node_size_range=tuple(s * scale * scale for s in style.node_size_range),
            label_fontsize=style.label_fontsize * scale,
            label_min_fontsize=style.label_min_fontsize * scale,
        )

        # Horizontal inset by the widest (scaled) label so a wide label like
        # "DAMAGE OR INJURE" — now kept at full node size — can't overflow the
        # panel's left/right edge into the inter-panel gutter or a neighbour.
        comp_fontsizes = [
            _label_fontsize(lab, comp_style.label_fontsize, comp_style.label_min_fontsize)
            for lab in comp_labels
        ]
        ix_label = max(
            (label_radius_box(lab, fs, comp_style.node_size_range[1], PANEL_INCH[0])
             for lab, fs in zip(comp_labels, comp_fontsizes)),
            default=0.0,
        )
        ix_label = min(ix_label, 0.30)   # keep the usable box from collapsing
        bx0, bx1 = 0.04 + ix_label, 0.96 - ix_label

        if g_sat.vcount() == 1:
            coords = np.array([[0.5, (y0 + y1) / 2]])
        else:
            coords = fr_layout(g_sat, style.fr_iter, tag=f"figC::{family}::sat{idx}")
        _draw_graph_in_box(ax, g_sat, coords, node_colors, comp_style,
                           x0=bx0, x1=bx1, y0=y0 + iy_dynamic, y1=y1 - iy_dynamic)


# ─────────────────────────────────────────────────────────────────────────
# Figure builders
# ─────────────────────────────────────────────────────────────────────────
def _make_grid_figure(family_data: dict, render_fn, style):
    rows, cols = GRID_ROWS, GRID_COLS
    fig_w = PANEL_INCH[0] * cols
    fig_h = PANEL_INCH[1] * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(-1)

    for i, family in enumerate(FAMILIES):
        render_fn(axes[i], family, family_data[family], style)
        label = f"({chr(ord('a') + i)})"
        axes[i].text(0.5, 0.01, label, transform=axes[i].transAxes,
                     fontsize=PANEL_LETTER_FONTSIZE, fontweight="bold",
                     va="bottom", ha="center", zorder=10)
    for j in range(len(FAMILIES), len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.96, bottom=0.02,
                        wspace=0.04, hspace=0.16)
    return fig


def make_fig_no_satellite(family_data: dict) -> None:
    style = NoSatelliteStyle()
    print("Fig 1: no-satellite combined network")
    fig = _make_grid_figure(family_data, render_pain_panel, style)
    export_figure(fig, "fig1_network_no_satellite")
    plt.close(fig)


def make_fig_satellite_only(family_data: dict) -> None:
    style = SatelliteOnlyStyle()
    print("Fig 3: satellite-only combined network")
    fig = _make_grid_figure(family_data, render_satellite_panel, style)
    export_figure(fig, "fig3_network_satellite_only")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────
def main() -> None:
    ensure_output_dirs()
    print(f"Output dir: {OUT_DIR}")

    concept_set = load_concept_set()

    # family name → glottocode mapping (Universal stays "Universal")
    family_map = dict(load_families())

    # Top-level seeding (build_aligned_pain_subgraph is deterministic; every
    # layout call is reseeded individually via reseed()).
    reseed("__init__")

    g_full = load_graph()

    family_data = {}
    for family in FAMILIES:
        print(f"  pipeline: {family}")
        family_code = family_map.get(family, family)
        family_data[family] = build_family_data_aligned(
            g_full, family_code, concept_set
        )

    make_fig_no_satellite(family_data)
    make_fig_satellite_only(family_data)

    print("Done.")


if __name__ == "__main__":
    main()
