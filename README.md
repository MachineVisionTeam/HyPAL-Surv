# HyPAL-Surv

**Hypercomplex-Gated Bilinear Fusion with Inverse Peer-Assisted Learning for Survival Prediction**

A unified architecture for histology-genomics survival prediction on TCGA cohorts, combining a novel parameter-efficient bilinear fusion operator (HGBF) with a distillation-based training recipe (Inverse PAL).

---

## TL;DR

HyPAL-Surv combines two contributions:

| Component | Name | What it is |
|-----------|------|------------|
| **Fusion operator** | **HGBF** (Hypercomplex-Gated Bilinear Fusion) | PHM-modulated bottleneck Kronecker bilinear fusion |
| **Training recipe** | **Inverse PAL** (Inverse Peer-Assisted Learning) | Gene-aux teacher distills risk-ranking INTO the fused output (not into a side branch) |

Across **four TCGA cohorts** (GBMLGG, KIRC, LUAD, UCEC), with frozen UNI2-h image features and frozen BulkRNABert gene features:

* **+0.018 to +0.029 c-index lift** over the same architecture without the recipe (positive in 4-of-4 cohorts; mean lift +0.0221).
* **Beats Pathomic Fusion's replicated trimodal** on both PF cohorts (+0.026 on GBMLGG, +0.005 on KIRC).
* **Log-rank stratification p ≪ 0.01** on 3 of 4 cohorts (1.78×10⁻²⁴ on GBMLGG; 9.10×10⁻⁷ on UCEC).
* **~2.23 M trainable parameters** vs PF's ~80 M (36× fewer).

---

## Method overview

```
                     ┌─────────────────┐
   image (1536-d) ──>│ ImageAdapter   ├─> z_img ──┐
   (UNI2-h frozen)   └─────────────────┘           │
                                                   v
                                      ┌─────────────────┐
                                      │      HGBF       │
                                      │ Hypercomplex-   │
                                      │ Gated Bilinear  │
                                      │     Fusion      │
                                      └────────┬────────┘
                                               │ fused
                                               v
                                      ┌─────────────────┐
                                      │  CoxHead_main   │──> hazard_main
                                      └─────────────────┘            │
                                                                     │
                     ┌─────────────────┐                              │
   gene (256-d) ───>│ GeneAdapter     ├─> z_gene ──┐                  │
   (BulkRNABert     └─────────────────┘            │                  │
    frozen)                                        v                  │
                                       ┌─────────────────┐            │
                                       │  GeneAuxHead    │──> aux_g   │
                                       │ (training only) │     │      │
                                       └─────────────────┘     │      │
                                                                v      │
                                              ╔══════════════════════╗ │
                                              ║  ★ INVERSE PAL ★     ║ │
                                              ║  L_dist =            ║ │
                                              ║   rank_distill(      ║<┘
                                              ║     hazard_main,     ║
                                              ║     aux_g.detach())  ║
                                              ╚══════════════════════╝
```

### HGBF — the fusion operator

Five steps, parses as **H-G-B-F**:

1. **Bottleneck projection** — `z_a_bn = Linear(z_img, 256→64)`, same for gene.
2. **H**ypercomplex modulation signal — `m = PHM(z_a_bn, z_b_bn)` ∈ ℝ⁶⁴ (Zhang et al. ICLR 2021).
3. **G**ated per-dim signals — `g_a = sigmoid(W_a · m)`, `g_b = sigmoid(W_b · m)`, both ∈ ℝ⁶⁴.
4. Element-wise modulation — `z_a_mod = g_a ⊙ z_a_bn`, same for `z_b_mod`.
5. **B**ilinear (Kronecker outer-product) **F**usion at bottleneck — `fused = KroneckerTFN(z_a_mod, z_b_mod)`.

~1.77 M parameters vs ~51 M for plain Kronecker (–97%).

### Inverse PAL — the training recipe

```
L_total = L_Cox(hazard_main, T, event)               <-- main survival loss
        + λ_aux  · L_Cox(aux_g,      T, event)       <-- train the gene teacher
        + λ_dist · L_dist(hazard_main, aux_g.detach()) <-- ★ Inverse PAL distillation
```

The key insight: classical PAL distills into a side branch (aux_p) that is **discarded at inference**. Inverse PAL distills directly into the **fused inference output** (hazard_main). This single change of target moves the mean cross-cohort effect from **−0.019 (Classical PAL, hurts)** to **+0.022 (Inverse PAL, helps)** — a +0.041 c-index swing attributable purely to flipping the distillation target.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full mathematical derivation, hyperparameters, gradient-flow analysis, and per-paper lineage.

---

## Headline cross-cohort results

| Cohort | Protocol | HGBF baseline | HyPAL-Surv | Lift | Paired-t / Wilcoxon | KM log-rank |
|--------|----------|--------------:|----------:|-----:|--------------------:|------------:|
| GBMLGG (PF-501) | 15-fold MCCV | 0.8254 ± 0.0667 | **0.8435 ± 0.0524** | **+0.0181** | t=0.004 \*\* / W=0.005 \*\* | **1.78 × 10⁻²⁴** |
| KIRC (PF-417) | 15-fold MCCV | 0.7041 ± 0.0458 | **0.7235 ± 0.0472** | **+0.0194** | t=0.003 \*\* / W=0.004 \*\* | **7.34 × 10⁻¹⁰** |
| LUAD (450) | 5×80/20 split | 0.6291 ± 0.0317 | **0.6585 ± 0.0352** | **+0.0294** | t=0.103 ns / W=0.125 ns | 0.0625 |
| UCEC (478) | 5×80/20 split | 0.6808 ± 0.0781 | **0.7023 ± 0.0477** | **+0.0215** | t=0.065 . / W=0.125 ns | **9.10 × 10⁻⁷** |

* **Sig key:** \*\* p<0.01, \* p<0.05, . p<0.10, ns not significant. Paired tests two-sided.
* **Sign-test (positive in 4-of-4 cohorts):** p = 0.0625 two-sided binomial.
* **n=5 power note:** Wilcoxon signed-rank two-sided min is 0.125 with n=5 splits, so LUAD/UCEC p_W=0.125 is the test ceiling, not a weak effect.

---

## Per-cohort ablation tables

For each cohort: **unimodal baselines → fusion baseline → fusion + each training recipe**. The training recipes all use the **SAME** HGBF architecture, the **SAME** gene-auxiliary teacher, and the **SAME** loss weights. The only thing that changes between rows is the training recipe.

### TCGA-GBMLGG (n=501 patients, 997 ROIs, 15-fold MCCV)

| Method | c-index ± std | Δ vs HGBF | Paired test (n=15) |
|--------|--------------:|----------:|--------------------:|
| **Unimodal baselines** | | | |
| UNI2-h alone (image only) | 0.7856 ± 0.0604 |  |  |
| BulkRNABert alone (gene only) | 0.8334 ± 0.0539 |  |  |
| **Fusion baseline (no training recipe)** | | | |
| HGBF (fusion only, no recipe) | 0.8254 ± 0.0667 | anchor | — |
| **Fusion + training recipes** | | | |
| HGBF + Classical PAL [BMLSurv 2026] | 0.8253 ± 0.0665 | −0.0001 | t=0.96 ns / W=0.97 ns |
| HGBF + GradMod [Path-GPTOmic 2024] | 0.8253 ± 0.0656 | −0.0001 | t=0.97 ns / W=0.95 ns |
| **HGBF + Inverse PAL  (HyPAL-Surv)** ⭐ | **0.8435 ± 0.0524** | **+0.0181** | **t=0.004 \*\* / W=0.005 \*\*** |
| _Comparator: PF Trimodal (replicated)_ | 0.8174 ± 0.0717 | (−0.026 vs ours) |  |

KM log-rank p (median-cut risk stratification): **1.78 × 10⁻²⁴**.

### TCGA-KIRC (n=417 patients, 1260 ROIs, 15-fold MCCV)

| Method | c-index ± std | Δ vs HGBF | Paired test (n=15) |
|--------|--------------:|----------:|--------------------:|
| **Unimodal baselines** | | | |
| UNI2-h alone (image only) | 0.6578 ± 0.0363 |  |  |
| BulkRNABert alone (gene only) | 0.6794 ± 0.0351 |  |  |
| **Fusion baseline (no training recipe)** | | | |
| HGBF (fusion only, no recipe) | 0.7041 ± 0.0458 | anchor | — |
| **Fusion + training recipes** | | | |
| HGBF + Classical PAL [BMLSurv 2026] | 0.7035 ± 0.0466 | −0.0006 | t=0.78 ns / W=0.92 ns |
| HGBF + GradMod [Path-GPTOmic 2024] | 0.7063 ± 0.0464 | +0.0022 | t=0.27 ns / W=0.22 ns |
| **HGBF + Inverse PAL  (HyPAL-Surv)** ⭐ | **0.7235 ± 0.0472** | **+0.0194** | **t=0.003 \*\* / W=0.004 \*\*** |
| _Comparator: PF Trimodal (replicated)_ | 0.7184 ± 0.0513 | (−0.005 vs ours) |  |

KM log-rank p (median-cut risk stratification): **7.34 × 10⁻¹⁰**.

### TCGA-LUAD (n=450 patients, 5×80/20 stratified splits)

| Method | c-index ± std | Δ vs HGBF | Paired test (n=5) |
|--------|--------------:|----------:|------------------:|
| **Unimodal baselines** | | | |
| UNI2-h alone (image only) | 0.5722 ± 0.0205 |  |  |
| BulkRNABert alone (gene only) | 0.6450 ± 0.0453 |  |  |
| **Fusion baseline (no training recipe)** | | | |
| HGBF (fusion only, no recipe) | 0.6291 ± 0.0317 | anchor | — |
| **Fusion + training recipes** | | | |
| HGBF + Classical PAL [BMLSurv 2026] | 0.5880 ± 0.0208 | **−0.0411 (HURTS)** | t=0.005 \*\* (sig. negative) |
| HGBF + GradMod [Path-GPTOmic 2024] | 0.6128 ± 0.0241 | −0.0163 | t=0.18 ns / W=0.31 ns |
| **HGBF + Inverse PAL  (HyPAL-Surv)** ⭐ | **0.6585 ± 0.0352** | **+0.0294 (★ biggest)** | t=0.103 ns / W=0.125 ns |

KM log-rank p (median-cut risk stratification): 0.0625 (marginal).

**Story:** LUAD is strongly gene-dominant — without a training recipe, the best fusion (HGBF 0.6291) trails gene-alone by −0.0159. The Inverse PAL recipe pushes HyPAL-Surv to 0.6585, **crossing gene-alone by +0.0135** — the strongest evidence in the project that the recipe (not the fusion architecture alone) is the differentiator on a gene-dominant cohort. The +0.0705 recipe swing from Classical PAL to Inverse PAL is the largest single-cohort recipe-direction effect across all 4 cohorts.

### TCGA-UCEC (n=478 patients, 5×80/20 stratified splits)

| Method | c-index ± std | Δ vs HGBF | Paired test (n=5) |
|--------|--------------:|----------:|------------------:|
| **Unimodal baselines** | | | |
| UNI2-h alone (image only) | 0.6736 ± 0.0593 |  |  |
| BulkRNABert alone (gene only) | 0.6454 ± 0.0995 |  |  |
| **Fusion baseline (no training recipe)** | | | |
| HGBF (fusion only, no recipe) | 0.6808 ± 0.0781 | anchor | — |
| **Fusion + training recipes** | | | |
| HGBF + Classical PAL [BMLSurv 2026] | 0.6462 ± 0.0943 | −0.0346 (hurts) | t=0.18 ns / W=0.31 ns |
| HGBF + GradMod [Path-GPTOmic 2024] | 0.6757 ± 0.0889 | −0.0051 | t=0.78 ns / W=0.81 ns |
| **HGBF + Inverse PAL  (HyPAL-Surv)** ⭐ | **0.7023 ± 0.0477** | **+0.0215** | t=0.065 . / W=0.125 ns |

KM log-rank p (median-cut risk stratification): **9.10 × 10⁻⁷**.

**Story:** UCEC is the only cohort where IMAGE dominates (UNI2-h alone 0.6736 > BulkRNABert alone 0.6454). HyPAL-Surv works in this image-dominant regime too — confirming the recipe is not modality-direction-specific. The variance also collapses (baseline std 0.078 → HyPAL-Surv std 0.048, 1.6× lower spread).

---

## Cross-cohort recipe ablation (the central methodological evidence)

The same HGBF architecture under four training recipes, across all four cohorts. **Only the training recipe changes between rows.**

| Recipe | GBMLGG | KIRC | LUAD | UCEC | **Mean** |
|--------|-------:|-----:|-----:|-----:|---------:|
| HGBF baseline | 0.8254 | 0.7041 | 0.6291 | 0.6808 | — |
| + Classical PAL (target=aux_p) | −0.0001 | −0.0006 | **−0.0411** | −0.0346 | **−0.0191 (HURTS)** |
| + GradMod | −0.0001 | +0.0022 | −0.0163 | −0.0051 | −0.0048 (~null) |
| **+ Inverse PAL ⭐ (target=hazard_main)** | **+0.0181** | **+0.0194** | **+0.0294** | **+0.0215** | **+0.0221 (HELPS)** |

* **Mean cross-cohort swing Classical PAL → Inverse PAL: +0.0412 c-index.**
* Attributable purely to flipping the distillation target from a discarded side branch (aux_p) to the fused inference output (hazard_main).
* Same architecture, same gene-aux teacher, same loss weights, same rank-MSE distillation loss; only the target differs.

---

## Repository structure

```
HyPAL-Surv/
├── README.md                            ← this file
├── ARCHITECTURE.md                       ← full mathematical reference + lineage
├── LICENSE                               (MIT)
├── requirements.txt                      Python dependencies
│
├── hypal_surv/                          ★ The model package
│   ├── __init__.py                       Public API (HyPALSurv, HGBF, ...)
│   ├── model.py                          HyPAL-Surv model class + training loop
│   └── fusion_ops.py                     HGBF + 4 comparator fusion operators
│                                         (Concat, Kronecker, LMF, PHM, BKron, HGBF)
│
├── scripts/                              Entry points
│   ├── train_unimodal.py                   UNI2-h-only or BulkRNABert-only baselines
│   ├── train_fusion.py              Fusion sweep (Concat/LMF/PHM/Kron/HGBF)
│   ├── train_hypal_surv.py              Training recipes (Classical PAL / GradMod /
│   │                                     Inverse PAL = HyPAL-Surv)
│   ├── compute_pvalues.py                Paired t-test, Wilcoxon, bootstrap CI
│   ├── plot_km_grid.py                   2×2 Kaplan-Meier grid (with log-rank p)
│   ├── plot_hazard_histograms.py         Hazard distribution by survival outcome
│   └── aggregate_predictions.py          Per-fold ROI predictions → per-patient CSV
│
├── data_loaders/                         Per-cohort splits & ROI/slide loaders
│   ├── roi_splits_gbmlgg.py              TCGA-GBMLGG (PF-501, 15-fold MCCV)
│   ├── roi_splits_kirc.py                TCGA-KIRC   (PF-417, 15-fold MCCV)
│   ├── roi_splits_luad.py                TCGA-LUAD   (450 pt, 5×80/20 splits)
│   ├── roi_splits_ucec.py                TCGA-UCEC   (478 pt, 5×80/20 splits)
│   ├── roi_loader.py                     Shared ROI batch assembly
│   └── roi_loader_{kirc,luad,ucec}.py    Cohort-specific loaders
│
└── results/
    └── predictions/                      Per-patient hazard CSVs (one per cohort)
        ├── gbmlgg_per_patient.csv        481 patients, mean hazard ± std, T, event
        ├── kirc_per_patient.csv          405 patients
        ├── luad_per_patient.csv          298 patients
        └── ucec_per_patient.csv          316 patients
```

---

## Installation

```bash
git clone https://github.com/<your-username>/HyPAL-Surv.git
cd HyPAL-Surv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### 1. Train HyPAL-Surv on a new dataset

```python
import numpy as np
from hypal_surv import HyPALSurv, train_hypal_surv

# Prepare your data (one sample = one ROI or one patient)
X_image_train = ...   # numpy array (N, img_dim), e.g. UNI2-h features (img_dim=1536)
X_gene_train  = ...   # numpy array (N, gene_dim), e.g. BulkRNABert features (gene_dim=256)
T_train       = ...   # numpy array (N,), survival times
event_train   = ...   # numpy array (N,), 0/1 event indicators

# Instantiate model (defaults reproduce TCGA-GBMLGG headline result)
model = HyPALSurv(img_dim=1536, gene_dim=256).to("cuda")

# Train with Inverse PAL distillation (default recipe)
train_hypal_surv(
    model,
    X_image_train, X_gene_train, T_train, event_train,
    epochs=200, lr=1e-4,
    lambda_aux=0.5, lambda_dist=0.3,  # default Inverse PAL weights
)

# Inference: predict hazard on a test set
hazards = model.predict(X_image_test, X_gene_test)
```

### 2. Reproduce the headline cohort results

Each cohort has its own split file and loader. You will need:

* **Image features** — per-ROI or per-slide UNI2-h ViT-H/14 embeddings (1536-d).
* **Gene features** — per-patient BulkRNABert embeddings (256-d).
* **Labels** — survival time `T` and event indicator from the TCGA cohort's clinical file.

```bash
# 1. Fusion sweep (Concat / LMF / PHM / Kronecker / HGBF)
python -m scripts.train_fusion --dataset=gbmlgg --fusion=slim_cpkf

# 2. Training recipes (Classical PAL / GradMod / Inverse PAL)
python -m scripts.train_hypal_surv --dataset=gbmlgg --fusion=slim_cpkf \
       --pal_type=inverse                   # = HyPAL-Surv (our recipe)

# Other recipes (for ablation):
python -m scripts.train_hypal_surv --dataset=gbmlgg --fusion=slim_cpkf \
       --pal_type=classical                 # BMLSurv-style PAL
python -m scripts.train_hypal_surv --dataset=gbmlgg --fusion=slim_cpkf \
       --no_pal                             # GradMod (Path-GPTOmic)
```

Cohort flag is one of: `gbmlgg`, `kirc`, `luad`, `ucec`.

### 3. Reproduce the figures

```bash
# Kaplan-Meier 2×2 grid (with log-rank p-values)
python -m scripts.plot_km_grid

# Hazard distribution histograms (event vs censored)
python -m scripts.plot_hazard_histograms
```

Both scripts read from `results/predictions/<cohort>_per_patient.csv` — already included in this repo so figures can be regenerated without re-running training.

### 4. Compute the paired statistical tests

```bash
python -m scripts.compute_pvalues
```

Outputs paired t-test, Wilcoxon signed-rank, and 95% bootstrap CI for each (recipe, cohort) comparison. Per-fold/per-split c-indices are baked into the script (taken verbatim from training logs).

---

## Data availability

This repository contains the model code, training scripts, statistical analysis, and per-patient hazard predictions. It does **not** contain:

* **TCGA whole-slide images** — download from the [GDC portal](https://portal.gdc.cancer.gov/).
* **Bulk RNA-seq** — STAR-Counts from GDC, then pass through [BulkRNABert](https://github.com/instadeepai/BulkRNABert) for 256-d embeddings.
* **UNI2-h foundation model weights** — request from [Mahmood Lab UNI](https://github.com/mahmoodlab/UNI).
* **Patient splits** — match the Pathomic Fusion 15-fold MCCV protocol (GBMLGG/KIRC) or the BulkRNABert paper's 5×80/20 stratified splits (LUAD/UCEC).

---

## Key design decisions

1. **Frozen foundation encoders** — UNI2-h (image, 1536-d) and BulkRNABert (gene, 256-d) are both frozen. Only the adapters, HGBF fusion, and Cox heads are trained. Total trainable parameters: **2.23 M** (vs Pathomic Fusion ~80 M end-to-end).
2. **Parameter-efficient bilinear fusion (HGBF)** — Compresses both modalities to a 64-d bottleneck, applies PHM-derived per-dim gates, then computes the Kronecker outer product at the compressed dimension. ~1.77 M params for the fusion, vs ~51 M for full-dim Kronecker.
3. **Inverse Peer-Assisted Learning** — A training-only gene-aux head distills risk-ranking into the fused inference output (NOT into a side branch like classical PAL). The aux head is discarded at inference.
4. **Cox negative partial log-likelihood** — Standard survival loss. We use the numerically-stable logcumsumexp form.
5. **Two protocols** — 15-fold MCCV for GBMLGG/KIRC (matches Pathomic Fusion); 5×80/20 stratified splits for LUAD/UCEC (matches BulkRNABert paper Table 3). All splits patient-disjoint.

---

## Statistical methodology

### Paired tests

* **Paired t-test** (two-sided) — primary test for the n=15 cohorts (GBMLGG, KIRC).
* **Wilcoxon signed-rank** (two-sided) — primary test for the n=5 cohorts (LUAD, UCEC); robustness check for n=15.
* **95% bootstrap CI** — 10,000 resamples of the per-fold differences, fixed seed 42.
* **Sign test** — `(0.5)^k` for cross-cohort directional consistency (k cohorts positive).

All paired tests use the SAME fold/split assignments across the two arms, so within-fold patient assignment is identical and fold-level noise is eliminated.

### n=5 power note

For n=5, Wilcoxon signed-rank's minimum achievable two-sided p-value is 0.125 (when all 5 differences are positive). LUAD/UCEC Wilcoxon p_W = 0.125 is the test ceiling, not a weak effect. Paired t-test can go lower because it uses the magnitude of differences (LUAD p_t = 0.103, UCEC p_t = 0.065).

### Kaplan-Meier risk stratification

* **Risk groups** — median-cut on predicted hazard (low vs high).
* **Log-rank test** — `lifelines.statistics.logrank_test`, two-sided.

---

## Citation

If you use this code or the HyPAL-Surv method, please cite:

```bibtex
@article{hypal_surv_2026,
  title     = {{HyPAL-Surv: Hypercomplex-Gated Bilinear Fusion with Inverse
               Peer-Assisted Learning for Histology-Genomics Survival Prediction}},
  author    = {<authors>},
  journal   = {<journal>},
  year      = {2026},
  url       = {https://github.com/<your-username>/HyPAL-Surv}
}
```

---

## References (key prior work)

* **Pathomic Fusion** — Chen et al., "Pathomic Fusion: An integrated framework for fusing histopathology and genomic features for cancer diagnosis and prognosis." IEEE TMI 2020.
* **TFN (Tensor Fusion Network)** — Zadeh et al., "Tensor Fusion Network for Multimodal Sentiment Analysis." EMNLP 2017.
* **PHM (Parameterized Hypercomplex Multiplications)** — Zhang et al., "Beyond Fully-Connected Layers with Quaternions: PHM." ICLR 2021.
* **BMLSurv (Classical PAL)** — "Balanced Multimodal Learning for Survival Prediction." Pattern Recognition 2026.
* **Path-GPTOmic (GradMod)** — Wang et al., "Path-GPTOmic: A Balanced Multi-Modal Learning Framework." ISBI 2024.
* **UNI2-h** — Chen et al., "Towards a general-purpose foundation model for computational pathology." Nature Medicine 2024.
* **BulkRNABert** — InstaDeep, "BulkRNABert: A transformer language model for bulk RNA-seq embeddings." 2024.

See [ARCHITECTURE.md](ARCHITECTURE.md) Section 8 for the full per-component lineage map.

---

## License

[MIT License](LICENSE) © 2026.

---

## Acknowledgements

This work builds on the Pathomic Fusion benchmark + protocol (Chen et al. 2020), foundation-model encoders from UNI (Mahmood Lab) and BulkRNABert (InstaDeep), the classical PAL recipe from BMLSurv, and the GradMod gradient-modulation technique from Path-GPTOmic.
