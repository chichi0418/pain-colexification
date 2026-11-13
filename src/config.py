"""Shared configuration for the CLICS4 PAIN-colexification analysis pipeline.

Central definition of data paths, output paths, and the analysis constants
used across every numbered script (walk length T, permutation count, seeds).
All scripts resolve paths relative to this file, so the pipeline can be run
from any working directory:

    python src/01_main_ari_paincc.py
"""
import gzip
import shutil
from pathlib import Path

import igraph as ig
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
PUBLIC_ROOT = SRC_DIR.parent
DATA_DIR = PUBLIC_ROOT / "data"
OUTPUT_DIR = PUBLIC_ROOT / "output"

# ── Data inputs ───────────────────────────────────────────────────────────────
GML_GZ_PATH = DATA_DIR / "clics4-network-3-families.gml.gz"
GML_PATH = str(DATA_DIR / "clics4-network-3-families.gml")
CONCEPT_PATH = str(DATA_DIR / "pain_concepts.tsv")
LABEL_CSV = str(DATA_DIR / "pain_phy_nonphy_labels.csv")
RATER_CSV = DATA_DIR / "rater_labels.csv"   # individual labels of the four raters (S2 step 3)
FAMILIES_CSV = DATA_DIR / "major_families.csv"
FAMILY_SELECTION_CSV = DATA_DIR / "family_selection.csv"  # 93 candidate families (S2 step 2)
GEOGRAPHY_CSV = DATA_DIR / "family_pairwise_geography.csv"
# Jackson et al. (2019) OSF file carrying both family-distance definitions
# (`values.distance` mean all-pairs Haversine; `values.chulldist` convex-hull
# centroid). Used only for the distance-definition robustness check; optional.
JACKSON_GEO_CSV = DATA_DIR / "jackson2019_geography_integrated_file.csv"

# ── Output trees ──────────────────────────────────────────────────────────────
MAIN_DIR = OUTPUT_DIR / "main"              # primary analysis (PAIN-CC)
SATELLITE_DIR = OUTPUT_DIR / "satellite"    # secondary analysis (satellite-inclusive)
ROBUSTNESS_DIR = OUTPUT_DIR / "robustness"  # raw robustness sweeps (PAIN-CC)
FIGURES_DIR = OUTPUT_DIR / "figures"        # publication figure exports

# ── Analysis constants ────────────────────────────────────────────────────────
WALK_T = 2          # random-walk steps (main analysis)
N_PERM = 1000       # ARI permutation-test iterations
SEED = 42           # RNG seed (igraph community detection + permutation null)
GLOSS_ATTR = "label"  # vertex attribute carrying the concept gloss


# ── Shared loaders ────────────────────────────────────────────────────────────

def ensure_gml() -> str:
    """Decompress the network file next to its .gz on first use."""
    gml = Path(GML_PATH)
    if not gml.exists():
        if not GML_GZ_PATH.exists():
            raise FileNotFoundError(
                f"Neither {gml} nor {GML_GZ_PATH} found — the network file is missing."
            )
        print(f"Decompressing {GML_GZ_PATH.name} (one-time) ...")
        with gzip.open(GML_GZ_PATH, "rb") as fin, open(gml, "wb") as fout:
            shutil.copyfileobj(fin, fout)
    return str(gml)


def load_graph() -> ig.Graph:
    """Load the CLICS4 colexification network as an undirected igraph Graph."""
    path = ensure_gml()
    print(f"Loading {path} ...")
    g = ig.Graph.Read_GML(path).as_undirected()
    print(f"  full graph: v={g.vcount()}, e={g.ecount()}")
    return g


def load_concept_set() -> set:
    df = pd.read_csv(CONCEPT_PATH, sep="\t", header=0)
    col = "Gloss" if "Gloss" in df.columns else "Concept"
    return set(df[col])


def load_concept_list() -> list:
    df = pd.read_csv(CONCEPT_PATH, sep="\t")
    return df["Concept"].tolist()


def load_families():
    """Return [(FamilyName, family_code)] — code is the Glottolog glottocode,
    or the display name itself for the pooled Universal network."""
    df = pd.read_csv(FAMILIES_CSV)
    df["Family"] = df.apply(
        lambda r: r["FamilyName"] if pd.isna(r["Family"]) else r["Family"], axis=1
    )
    return list(zip(df["FamilyName"], df["Family"]))


def load_label_map() -> dict:
    df = pd.read_csv(LABEL_CSV, sep=None, engine="python")
    df.columns = [c.lstrip("﻿") for c in df.columns]
    if "Gloss" not in df.columns and "Concept" in df.columns:
        df = df.rename(columns={"Concept": "Gloss"})
    return dict(zip(df["Gloss"], df["Comm"]))
