# CLICS4 PAIN Colexification — Analysis Code

Analysis code and data for the study of cross-linguistic semantic structure of
PAIN, based on the CLICS⁴ colexification database. The pipeline builds
family-specific and Universal colexification networks for 26 PAIN-related
concepts, detects communities, evaluates their alignment with a
physical/non-physical ground-truth partition (Adjusted Rand Index with
permutation tests), and runs the robustness analyses reported in the
manuscript's supplementary materials.

## Setup

Python 3.9 with the pinned package versions in `requirements.txt`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Exact community detection (`community_optimal_modularity`) requires igraph
built with **GLPK** support; this is included in the standard `python-igraph`
1.0.0 wheels.

The network file ships compressed (`data/clics4-network-3-families.gml.gz`,
22 MB; ~101 MB decompressed). The scripts decompress it automatically on first
run — no manual step needed.

## Running the pipeline

Run the numbered scripts in order (each script's docstring states its inputs,
outputs, and prerequisites). All outputs land under `output/`.

```bash
# Inputs: concept set, family selection, inter-rater agreement (checks)
python src/00_concept_set.py           # step 1: 26 concepts within distance 2 of PAIN
python src/00_family_selection.py      # step 2: 93 → 51 → 24 → 11 families (Methods)
python src/00_rater_agreement.py       # step 3: Fleiss' kappa, majority vote (Methods, S1)

# Primary analysis (PAIN-connected component, T = 2)
python src/01_main_ari_paincc.py       # steps 4–11: networks → ARI → exact p (Table 3)
python src/02_centrality_paincc.py     # step 12: strength/betweenness centrality
python src/03_effect_sizes.py          # step 12: Cohen's d / Hedges' g + CIs (Table 2, S6)
python src/04_family_clustering.py     # step 13: clustering (Fig. 2, Table 1, S3)
python src/05_dendrogram_k4.py         # k = 4 alternative (S5, Fig. S2)
python src/06_geography_control.py     # step 15: distance vs similarity (S4B–C, Fig. S3)
python src/17_cluster_similarity_geography.py  # steps 14–15: within/between-cluster
                                       #   rho (S4A) + cluster compactness test
                                       #   (needs 04 and 06; run after 13 to add the
                                       #   satellite k = 4 section, not in the manuscript)

# Robustness — formal sensitivity analyses, exact optimal modularity as in 01
python src/07_robustness_threshold.py  # step 16: thresholds + transforms (S9)
python src/08_robustness_walkstep.py   # step 17: walk steps + saturation (S11)
python src/09_robustness_k2.py         # step 18: k = 2, 3 algorithms (S10)
python src/10_robustness_summary.py    # summary tables (S8, S9, S11)

# Robustness — original exploratory Leiden sweeps (algorithm comparison)
python src/07_robustness_threshold.py --algorithm leiden
python src/08_robustness_walkstep.py  --algorithm leiden
python src/09_robustness_k2.py        --baseline-algorithm leiden
python src/10_robustness_summary.py   --algorithm leiden
python src/18_compare_leiden_optimal.py  # per-condition Leiden vs optimal comparison

# Secondary analysis (satellite-inclusive, i.e. no PAIN-CC restriction)
python src/11_satellite_ari.py         # step 19 (S12)
python src/12_satellite_centrality.py
python src/13_satellite_clustering.py
python src/14_satellite_geography.py

# Publication figures
python src/15_figures.py               # Fig. 1 + Fig. S1 (satellite components)

# Label-assignment sensitivity (Supplementary Tables S13–S14): PAIN × PAINFUL labels
python src/16_ablation_pain_labels.py  # reruns 01 + 11 under 4 labelings → output/ablation/SUMMARY.md
```

Scripts 01 and 11 accept `--labels LABELS.csv --outdir DIR`; without them they
use the manuscript labels and default output directories.

Shared building blocks live in four modules: `config.py` (paths, constants,
loaders), `network_utils.py` (network construction + random-walk smoothing),
`ari_utils.py` (ARI + permutation test + diagnostics), and
`robustness_utils.py` (graph variants, optimal/Leiden/k = 2 detection, and the
timed, time-limited, resumable per-condition runner used by 07–08).

Scripts 07 and 08 write to `output/robustness/{optimal,leiden}/`. Each fixed
graph is partitioned once, in a worker process limited to `--timeout` seconds
(default 600); a timeout or error is logged in `failures.csv` and is never
replaced by another algorithm. Completed conditions are stored in
`conditions.jsonl` and reused on re-runs (`--fresh` recomputes everything);
`run_metadata.json` records the algorithm parameters, software versions,
command and timing totals.

## Pipeline steps → scripts

Supplementary Table S2 lists the analytical workflow in 20 steps; this is
where each one lives in the code. Steps 1–3 produced the input files that
scripts 01–18 read; the three `00_*` scripts rebuild them (step 1) or re-apply
the stated rules to the included inputs (steps 2–3: per-family counts, the
four raters' labels) and check that the results match.

| S2 step | Procedure | Implementation |
|---|---|---|
| 1. Define the PAIN concept set | Links attested in ≥ 3 families; concepts within colexification distance 2 of PAIN; 26 concepts | `00_concept_set.py` (result: `data/pain_concepts.tsv`) |
| 2. Select language families | > 1 edge; edges/nodes > 0.6667; > 6 internal edges; 11 families + Universal (both thresholds are medians: of edges/nodes over the 51 families, and of edges over the 24) | `00_family_selection.py` (counts: `data/family_selection.csv`; result: `data/major_families.csv`) |
| 3. Annotate semantic categories | Four raters; Fleiss' κ; majority vote (4 concepts with 3–1 splits) | `00_rater_agreement.py` (ratings: `data/rater_labels.csv`; final labels: `data/pain_phy_nonphy_labels.csv`) |
| 4. Construct weighted networks | Family edges weighted by unique-language count (`family_local_weight`); Universal by pooled `language_count` | `network_utils.subgraph_by_family_edges_clics4` (called from `01`) |
| 5. Random-walk transformation | T = 2; P = D⁻¹A; W = 1 − Π_{k=1..T}(1 − Pᵏ), elementwise, on the **full** family network; symmetrised, diagonal zeroed, ×100 and rounded to integers in [0, 100] | `network_utils.random_walk_smoothing` / `precompute_family_W_full` |
| 6. Extract PAIN networks | 26-concept submatrix taken **after** the transformation (all steps of 5 are elementwise, so their order relative to extraction does not matter) | `network_utils.build_paincc_subgraph` steps 4–5 |
| 7. Retain the PAIN-CC | Keep the connected component containing PAIN; families with < 3 PAIN concepts, no PAIN-CC, or a PAIN-CC of < 3 concepts are excluded | `network_utils.build_paincc_subgraph` step 6 |
| 8. Identify communities | Exact optimal modularity (igraph/GLPK), k free; if the maximum modularity is ≤ 0 the graph is treated as one community | `network_utils.optimal_communities` |
| 9. Measure category alignment | ARI over concepts present in both partitions | `ari_utils.calculate_ari_for_subgraph` (via `01`) |
| 10. Test category alignment | One-sided label-permutation null, enumerated exactly → `p_Exact` (Table 3); 1,000-shuffle Monte Carlo estimate (NumPy seed 42, floored at .001) kept as `p_Local` | `ari_utils.exact_permutation_p`, `ari_utils.permutation_test` (via `01`) |
| 11. Minimum attainable p | Exact probability of the maximum attainable ARI → `p_MinAttainable` | `ari_utils.min_attainable_p` (via `01`) |
| 12. Compare centrality | Strength (and betweenness) on the post-walk PAIN-CC; Cohen's d / Hedges' g; stratified bootstrap CI, B = 10,000, seed 0 | `02_centrality_paincc.py`, `03_effect_sizes.py` |
| 13. Cluster networks | Spearman ρ between strength profiles; average linkage on 1 − ρ; k = 2–6 assessed by merge-height gap and silhouette | `04_family_clustering.py` (k = 4 alternative: `05_dendrogram_k4.py`) |
| 14. Describe cluster structure | Within/between-cluster ρ; per-family silhouette | `17_cluster_similarity_geography.py` |
| 15. Assess geography | Distance–similarity Spearman (55 pairs); cluster compactness against all 462 same-size subsets | `06_geography_control.py`, `17_cluster_similarity_geography.py` |
| 16. Edge-weight sensitivity | 6 binary thresholds + 4 transforms (rank: tied weights share their average rank); each variant re-restricted to the PAIN-CC after dropping edges with weight ≤ 0; not evaluable if PAIN is removed, < 4 nodes or 3 edges remain, or one label class remains | `07_robustness_threshold.py` + `robustness_utils.restrict_to_pain_cc` |
| 17. Walk-length sensitivity | No walk + T = 1–10; not evaluable if the PAIN-CC holds one label class; density saturation | `08_robustness_walkstep.py` |
| 18. Two-community solutions | Walktrap / Fast Greedy / Leading Eigenvector, cut or capped at 2 communities, seed 42 | `09_robustness_k2.py` |
| 19. Satellite-inclusive networks | As 4–11, keeping components disconnected from PAIN | `11_satellite_ari.py` (12–14: centrality, clustering, geography) |
| 20. Label-assignment sensitivity | PAIN × PAINFUL labels in a 2 × 2 design, PAIN-CC and satellite-inclusive | `16_ablation_pain_labels.py` |

Steps 16–17 use exact optimal modularity by default; `--algorithm leiden`
reruns them with Leiden (seed 42) for the algorithm check reported in §2.4
(identical partitions in 215 of 222 evaluable conditions; script 18).

## Manuscript figures and tables → outputs

Paths are relative to `output/`; the CSV summaries are also in `expected_output/`.

| Manuscript item | Output file | Script |
|---|---|---|
| Table 1 (cluster signatures) | `main/cross_family/cluster_signatures_k2.csv` | 04 |
| Table 2 (strength by category) | `main/centrality_summary_postwalk.csv` | 02 + 03 |
| Table 3 (ARI, exact p, minimum attainable p) | `main/summary.csv` | 01 |
| Figure 1 (PAIN-CC networks) | `figures/pdf/fig1_network_no_satellite.pdf` | 15 |
| Figure 2 (k = 2 dendrogram, ρ with Universal) | `main/cross_family/dendrogram.pdf`, `main/cross_family/similarity_matrix.csv` | 04 |
| §3.2 k selection (merge-height gaps, silhouettes) | printed by 04; `main/cross_family/k_selection_diagnostics.png` | 04 |
| Methods: concept set (26 concepts; only PAIN in the three-family network) | `main/pain_concepts.tsv` | 00 |
| Methods: family selection (93 → 51 → 24 → 11) | `main/family_selection.csv` | 00 |
| Methods: inter-rater agreement (κ, P̄, % unanimous) | `main/rater_agreement.json` | 00 |
| Supplementary Table S1 (annotations; † = 3–1 split) | `data/pain_phy_nonphy_labels.csv`; splits in `main/rater_agreement.json` | 00 |
| Supplementary Table S3 (cluster contrast) | `main/cross_family/cluster_contrast.csv` | 04 |
| Supplementary Table S4A–C (similarity, geography) | `main/cross_family/cluster_similarity/`, `main/cross_family/geography_control/geography_control_pairs.csv` | 06, 17 |
| Supplementary Table S5 (k = 4 clusters) | `main/cross_family/cluster_signatures_k4.csv` | 05 |
| Supplementary Table S6 (betweenness) | `main/centrality_summary_postwalk.csv` | 02 + 03 |
| Supplementary Table S7 (condition counts) | `robustness/optimal/{threshold,walkstep}/summary.csv`, `robustness/optimal/k2/k2_summary.csv` | 07, 08, 09 |
| Supplementary Table S8 (robustness overview) | `main/robustness/optimal/cross_family_robustness_table.csv` | 10 |
| Supplementary Table S9 (threshold/transform) | `main/robustness/optimal/threshold_summary_table.csv`, `robustness/optimal/threshold/summary.csv` | 07 + 10 |
| Supplementary Table S10 (k = 2) | `robustness/optimal/k2/comparison_table.csv` (`postwalk_raw` rows) | 09 |
| Supplementary Table S11 (walk steps) | `main/robustness/optimal/walkstep_summary_table.csv`, `robustness/optimal/walkstep/saturation_summary.csv` | 08 + 10 |
| Supplementary Table S12 (satellite-inclusive) | `satellite/summary.csv` | 11 |
| Supplementary Tables S13–S14 (label assignments) | `ablation/comparison_main.csv`, `ablation/comparison_satellite.csv`, `ablation/SUMMARY.md` | 16 |
| Supplementary Figure S1 (satellite components) | `figures/pdf/fig3_network_satellite_only.pdf` | 15 |
| Supplementary Figure S2 (k = 4 dendrogram) | `main/cross_family/dendrogram_k4.pdf` | 05 |
| Supplementary Figure S3 (geography scatter) | `main/cross_family/geography_control/geography_control_scatter_paper.png` | 06 |
| §2.4 Leiden vs optimal (215 of 222) | `robustness/comparison/comparison_overview.csv` (`same_partition`) | 18 |

The "Overall assessment" column of Supplementary Table S8 is the authors'
summary of the other columns, not the `Overall_Evaluation` label written by
script 10. Supplementary Table S2 is a description of the workflow and has no
output file; the software versions in §2.4 correspond to `requirements.txt`.

## Expected outputs

`expected_output/` holds the summary CSVs produced by the runs reported in the
manuscript. After running the pipeline, your `output/` summaries should match
them. Point estimates from the main analysis are exactly reproducible: the
main community detection is exact (optimal modularity, no randomness) and all
stochastic steps (permutation tests, Leiden, bootstrap) are seeded.
The `summary.csv` files of scripts 01 and 11 (ARI, Monte Carlo p) and the
centrality summaries of scripts 02–03 (group means, Cohen's d, Hedges' g, CIs)
are stored at full precision; values are rounded once, when the manuscript
tables are made.

Robustness results are kept per algorithm under
`expected_output/robustness/{optimal,leiden}/` and
`expected_output/main/robustness/{optimal,leiden}/`. They replace the earlier
flat-path Leiden results (`robustness/{threshold,walkstep,k2}/`, still in the
git history), which analysed thresholded and min–max networks without
re-restricting them to the PAIN-connected component, ranked tied weights
arbitrarily, and counted single-label-class networks as p = 1.
`robustness/comparison/leiden_vs_optimal.csv` contains detection runtimes,
which vary between machines.

## Reproducibility notes

- **Seeds.** igraph RNG: `igraph.set_random_number_generator(random.Random(42))`
  before every randomized community-detection call; permutation RNG:
  `numpy.random.default_rng(42)`; bootstrap RNG (script 03): `random.Random(0)`.
  Figure layouts use their own fixed seeds and affect only node positions.
- **Main vs. robustness detection.** The main analysis and, by default, the
  threshold/transform and walk-step sweeps use exact optimal modularity
  (deterministic). Leiden (modularity objective, 10 iterations, resolution = 1
  default, β = 0.01 default, seed 42) produced the original exploratory sweeps
  and is kept for the algorithm comparison (script 18);
  walktrap/fast-greedy/leading-eigenvector are used only in the constrained
  k = 2 module.
- **Degenerate optima.** When the maximum modularity of a graph is ≤ 0 (no
  community structure; e.g. Turkic at thresholds t5 and t10), the graph is
  treated as a single community (`network_utils.optimal_communities`). When
  several partitions share a positive maximum (e.g. Uralic t5), the partition
  returned by the solver is reported; `robustness/comparison/degeneracy_check.csv`
  shows that GLPK returns the same one under 20 vertex reorderings.
- **Reading `p_Exact` and `p_MinAttainable`.** The permutation null only reshuffles the
  phy/nonphy labels, so the detected partition and the two class sizes are
  fixed and the null contains exactly C(*n*, *n*₁) equally likely assignments
  (*n* = PAIN-CC nodes, *n*₁ = physical concepts). `min_attainable_p`
  enumerates that set exactly — every contingency table (*a*₁ … *a_k*) of
  physical labels across the detected communities, weighted by
  Π_j C(*s_j*, *a_j*) — and returns the null probability of the best ARI the
  partition allows. It is the smallest p the test can reach for that family,
  so a p-value near its floor means limited power, not weak alignment: Turkic
  (*n* = 6, 2/4 split) reaches ARI = 1.0 with a floor of 1/C(6,2) = 0.0667,
  whereas Universal (*n* = 26) has a floor of 3.2 × 10⁻⁷.
  `exact_permutation_p` sums the same weights over every table with
  ARI ≥ the observed ARI, giving the exact one-sided p reported in the
  manuscript. Neither carries Monte-Carlo error, and `p_Exact` equals
  `p_MinAttainable` whenever the observed ARI is the maximum the partition
  allows (Turkic 0.0667, Uralic 0.0952, Tai-Kadai 0.4286). The 1,000-draw
  estimate `p_Local` can fall below the floor through sampling variation
  (Turkic 0.061, Tai-Kadai 0.396), which is why the primary analysis
  reports `p_Exact` instead.
- **p-values in the robustness sweeps.** Scripts 07–09 report both the exact
  permutation p (`p_val`) and the 1,000-draw Monte Carlo estimate (`p_mc`);
  `significant_05` / `significant_10` and the counts in script 10 use the
  exact p, over evaluable conditions only (`Not_Evaluable` gives the rest).
- **Primary vs. secondary scope.** Scripts 01–10 analyze the PAIN-connected
  component (primary). Scripts 11–14 retain disconnected satellite components
  (secondary, less conservative); the manuscript reports both and discusses the
  contrast.
- **Label ablation (Supplementary Tables S13–S14).** Script 16 tests how sensitive ARI is to the manually
  assigned labels of PAIN and PAINFUL: A0 (PAIN phy / PAINFUL nonphy, the
  manuscript labels), A1 (phy/phy), A2 (nonphy/phy), A3 (nonphy/nonphy). The
  four label files are in `data/labels/ablation/`; A0 is identical to
  `data/pain_phy_nonphy_labels.csv`. Results: `expected_output/ablation/SUMMARY.md`.
- **Runtime.** Scripts 01–06 take minutes each (dominated by loading the
  101 MB network). On the PAIN-CC graphs (≤ 26 nodes) exact optimal modularity
  takes at most ~0.02 s per condition; scripts 07 and 08 take under a minute
  each with either algorithm, dominated by the permutation tests.

## Data provenance and licenses

The analysis code is released under the MIT License (see `LICENSE`). The MIT
License does not cover the files under `data/`, which are redistributed
research data under their own terms:

| File | Content | Source / license |
|---|---|---|
| `clics4-network-3-families.gml.gz` | CLICS⁴ colexification network, restricted to links attested in ≥ 3 language families | Derived from [CLICS⁴](https://clics.clld.org/) (Tjuka et al. 2025), CC-BY 4.0 |
| `pain_concepts.tsv` | The 26 PAIN-related target concepts (colexification distance ≤ 2 from PAIN) | Derived from CLICS⁴, CC-BY 4.0 |
| `pain_phy_nonphy_labels.csv` | Ground-truth physical/non-physical annotation per concept | This study |
| `rater_labels.csv` | Each rater's physical/non-physical label per concept (raters anonymized as R1–R4) and the majority-vote final label (script 00) | This study |
| `labels/ablation/*.csv` | The ground truth with the four PAIN × PAINFUL label combinations (script 16) | This study |
| `family_selection.csv` | The 93 families with at least one PAIN-related colexification: per family, the network edges linking two of the 26 concepts (`PainEdges`) and the concepts on them (`PainNodes`), used for the family filters (script 00) | Derived from CLICS⁴, CC-BY 4.0; glottocodes from Glottolog, CC-BY 4.0 |
| `major_families.csv` | The 11 analyzed families: display name ↔ Glottolog glottocode | Glottocodes from [Glottolog](https://glottolog.org/), CC-BY 4.0 |
| `family_pairwise_geography.csv` | Family-pair geographic distances (mean all-pairs Haversine, `distance` column) | Derived from Jackson et al. (2019) OSF materials |
| `jackson2019_geography_integrated_file.csv` | Jackson et al. (2019) family-pair distances incl. convex-hull-centroid variant | Jackson et al. (2019) OSF materials |

Please cite CLICS⁴, Glottolog, and Jackson et al. (2019) when reusing the
respective data files:

- Jackson JC, Watts J, Henry TR, List J-M, Forkel R, Mucha PJ, Greenhill SJ,
  Gray RD, Lindquist KA (2019). *Emotion semantics show both cultural variation
  and universal structure.* Science 366(6472):1517–1522.
- Tjuka A, Forkel R, Rzymski C, List J-M (2025). *Advancing the Database of
  Cross-Linguistic Colexifications with New Workflows and Data.* Proceedings of
  the 16th International Conference on Computational Semantics (IWCS), 1–15.
  https://aclanthology.org/2025.iwcs-main.1/
- Forkel R, Hammarström H (2022). *Glottocodes: Identifiers linking families,
  languages and dialects to comprehensive reference information.* Semantic Web
  13:917–924.

## Citation

If you use this code or the derived data, please cite the accompanying
article (citation to be added on publication) and this repository
(`CITATION.cff`; archived on Zenodo).

## Method references

- Brandes U, et al. (2008). *On modularity clustering.* IEEE TKDE 20(2):172–188. (optimal modularity)
- Traag VA, Waltman L, van Eck NJ (2019). *From Louvain to Leiden.* Sci Rep 9:5233. (Leiden)
- Pons P, Latapy M (2005). *Computing communities in large networks using random walks.* ISCIS 2005. (walktrap)
- Clauset A, Newman MEJ, Moore C (2004). *Finding community structure in very large networks.* Phys Rev E 70:066111. (fast-greedy)
- Newman MEJ (2006). *Finding community structure in networks using the eigenvectors of matrices.* Phys Rev E 74:036104. (leading eigenvector)
- Jackson JC, et al. (2019). *Emotion semantics show both cultural variation and universal structure.* Science 366:1517–1522. (random-walk transform)
- Hubert L, Arabie P (1985). *Comparing partitions.* J Classification 2:193–218. (ARI)
