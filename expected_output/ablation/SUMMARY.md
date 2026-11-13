# PAIN / PAINFUL label ablation

Sensitivity of community–label alignment (ARI) to the manually assigned phy/nonphy labels of PAIN and PAINFUL. All other labels, the networks, community detection, and the permutation test (1000 permutations, seed 42) are identical across conditions.

| Condition | PAIN | PAINFUL | Note |
|---|---|---|---|
| A0_pain-phy_painful-nonphy | phy | nonphy | manuscript labels (baseline) |
| A1_pain-phy_painful-phy | phy | phy |  |
| A2_pain-nonphy_painful-phy | nonphy | phy |  |
| A3_pain-nonphy_painful-nonphy | nonphy | nonphy |  |

## Method

**Rationale.** The phy/nonphy ground truth is a manual annotation. PAIN and PAINFUL are the concepts whose labels are most open to dispute: PAIN is the seed concept of the whole domain, and PAINFUL is defined as causing distress "either physical or mental". The ablation asks whether the manuscript's conclusions depend on how these two concepts are labelled.

**Design.** A 2 × 2 crossing of PAIN ∈ {phy, nonphy} and PAINFUL ∈ {phy, nonphy}. Each condition is a copy of the manuscript label file (`data/pain_phy_nonphy_labels.csv`) with only these two rows changed (`data/labels/ablation/{condition}.csv`); the other 24 labels are untouched. The phy/nonphy class sizes are therefore 17/9 (A0, A2), 18/8 (A1) and 16/10 (A3).

**Pipeline.** Unchanged from the manuscript. For each of the 11 families and the Universal network, script 01 (primary) builds the family network, applies random-walk smoothing (T = 2) on the full family network, extracts the 26 PAIN concepts, restricts to the PAIN-connected component (PAIN-CC), and detects communities with exact optimal modularity. Script 11 (secondary) is identical but keeps satellite components disconnected from PAIN. Because networks and community detection do not use the labels, the detected communities are the same in all four conditions; only the ground truth they are compared with changes.

**Evaluation.** ARI between detected communities and the phy/nonphy labels over the concepts present in both. Significance by a one-sided label-permutation test (1000 permutations, NumPy seed 42, p = P(ARI_perm ≥ ARI_obs), floored at 0.001), with the same seed in every condition; significant = p < 0.05. No multiple-comparison correction, matching the manuscript.

**Reading the results.**

- Where PAINFUL is absent from a family's network, A0 = A1 and A2 = A3 by construction; only the PAIN label can matter there.
- Several families have few matched concepts (6–8), so a single label change moves ARI substantially and their minimum attainable p can be close to or above 0.05 (see `p_MinAttainable` in `comparison_main.csv`).
- Significance changes with p close to 0.05 should be read as borderline, not as a qualitative shift.
- Supplementary Tables S13–S14 report these Monte Carlo p-values, except where the labelled network is identical to the primary analysis (A0, and A1 without PAINFUL, in the PAIN-CC analysis; Universal A0 in the satellite-inclusive analysis), where the exact p from Table 3 (`p_Exact` in `comparison_main.csv`) is reported instead.

**Reproduce.** `python src/16_ablation_pain_labels.py` (about 1 min); `--compare-only` rebuilds the tables from existing outputs. A0 reproduces `expected_output/main/summary.csv` and `expected_output/satellite/summary.csv` byte for byte.

## Results

Cells: ARI (permutation p). `*` p < 0.05; `†` significance differs from A0.
`main` = PAIN-CC analysis (script 01); `satellite` = satellite-inclusive analysis (script 11).

### main

| Family | Matched | PAINFUL in network | A0 | A1 | A2 | A3 |
|---|---|---|---|---|---|---|
| Austronesian | 14 | yes | +0.228 (p=0.057) | +0.325 (p=0.011)* † | +0.309 (p=0.013)* † | +0.173 (p=0.084) |
| Tai-Kadai | 14 | no | +0.018 (p=0.396) | +0.018 (p=0.396) | -0.009 (p=0.824) | -0.009 (p=0.824) |
| Austroasiatic | 11 | no | -0.009 (p=1.000) | -0.009 (p=1.000) | -0.043 (p=1.000) | -0.043 (p=1.000) |
| Sino-Tibetan | 11 | yes | +0.121 (p=0.257) | +0.306 (p=0.045)* † | +0.341 (p=0.054) | +0.096 (p=0.239) |
| Nakh-Daghestanian | 10 | no | -0.028 (p=1.000) | -0.028 (p=1.000) | -0.109 (p=1.000) | -0.109 (p=1.000) |
| Indo-European | 14 | yes | +0.342 (p=0.011)* | +0.495 (p=0.001)* | +0.388 (p=0.006)* | +0.214 (p=0.035)* |
| Tupian | 10 | yes | +0.103 (p=0.309) | +0.242 (p=0.065) | +0.103 (p=0.285) | +0.074 (p=0.509) |
| Uralic | 7 | no | +0.442 (p=0.098) | +0.442 (p=0.098) | +0.176 (p=0.310) | +0.176 (p=0.310) |
| Turkic | 6 | no | +1.000 (p=0.061) | +1.000 (p=0.061) | +0.324 (p=0.394) | +0.324 (p=0.394) |
| Pano-Tacanan | 8 | yes | -0.061 (p=1.000) | -0.103 (p=1.000) | -0.061 (p=1.000) | +0.138 (p=0.488) |
| Afro-Asiatic | 7 | no | -0.077 (p=1.000) | -0.077 (p=1.000) | +0.054 (p=0.434) | +0.054 (p=0.434) |
| Universal | 26 | yes | +0.139 (p=0.015)* | +0.167 (p=0.010)* | +0.163 (p=0.011)* | +0.127 (p=0.030)* |
| **Significant (p < 0.05)** | | | **2/12** | **4/12** | **3/12** | **2/12** |

PAINFUL is absent from the network of: Tai-Kadai, Austroasiatic, Nakh-Daghestanian, Uralic, Turkic, Afro-Asiatic. For these families A0 = A1 and A2 = A3 by construction.

Significance changes relative to the baseline (A0):

- Austronesian · A1_pain-phy_painful-phy: becomes significant (ARI +0.325, p = 0.011)
- Sino-Tibetan · A1_pain-phy_painful-phy: becomes significant (ARI +0.306, p = 0.045)
- Austronesian · A2_pain-nonphy_painful-phy: becomes significant (ARI +0.309, p = 0.013)

### satellite

| Family | Matched | PAINFUL in network | A0 | A1 | A2 | A3 |
|---|---|---|---|---|---|---|
| Austronesian | 20 | yes | +0.260 (p=0.001)* | +0.340 (p=0.001)* | +0.302 (p=0.001)* | +0.206 (p=0.015)* |
| Tai-Kadai | 16 | no | +0.112 (p=0.010)* | +0.112 (p=0.010)* | +0.076 (p=0.118) † | +0.076 (p=0.118) † |
| Austroasiatic | 15 | no | +0.071 (p=0.238) | +0.071 (p=0.238) | +0.028 (p=0.339) | +0.028 (p=0.339) |
| Sino-Tibetan | 15 | yes | +0.039 (p=0.346) | +0.137 (p=0.091) | +0.150 (p=0.100) | +0.030 (p=0.382) |
| Nakh-Daghestanian | 17 | no | +0.278 (p=0.004)* | +0.278 (p=0.004)* | +0.161 (p=0.041)* | +0.161 (p=0.041)* |
| Indo-European | 16 | yes | +0.286 (p=0.005)* | +0.380 (p=0.001)* | +0.321 (p=0.002)* | +0.208 (p=0.021)* |
| Tupian | 12 | yes | +0.059 (p=0.365) | +0.144 (p=0.056) | +0.059 (p=0.351) | +0.050 (p=0.479) |
| Uralic | 13 | no | +0.162 (p=0.028)* | +0.162 (p=0.028)* | +0.071 (p=0.265) † | +0.071 (p=0.265) † |
| Turkic | 11 | no | +0.412 (p=0.006)* | +0.412 (p=0.006)* | +0.231 (p=0.080) † | +0.231 (p=0.080) † |
| Pano-Tacanan | 12 | yes | +0.076 (p=0.217) | +0.118 (p=0.190) | +0.076 (p=0.226) | +0.123 (p=0.177) |
| Afro-Asiatic | 9 | no | -0.069 (p=1.000) | -0.069 (p=1.000) | -0.011 (p=0.669) | -0.011 (p=0.669) |
| Universal | 26 | yes | +0.139 (p=0.015)* | +0.167 (p=0.010)* | +0.163 (p=0.011)* | +0.127 (p=0.030)* |
| **Significant (p < 0.05)** | | | **7/12** | **7/12** | **4/12** | **4/12** |

PAINFUL is absent from the network of: Tai-Kadai, Austroasiatic, Nakh-Daghestanian, Uralic, Turkic, Afro-Asiatic. For these families A0 = A1 and A2 = A3 by construction.

Significance changes relative to the baseline (A0):

- Tai-Kadai · A2_pain-nonphy_painful-phy: loses significance (ARI +0.076, p = 0.118)
- Uralic · A2_pain-nonphy_painful-phy: loses significance (ARI +0.071, p = 0.265)
- Turkic · A2_pain-nonphy_painful-phy: loses significance (ARI +0.231, p = 0.080)
- Tai-Kadai · A3_pain-nonphy_painful-nonphy: loses significance (ARI +0.076, p = 0.118)
- Uralic · A3_pain-nonphy_painful-nonphy: loses significance (ARI +0.071, p = 0.265)
- Turkic · A3_pain-nonphy_painful-nonphy: loses significance (ARI +0.231, p = 0.080)
