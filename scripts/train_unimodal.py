"""
Stage 2 sample-level unimodal baselines for GBMLGG.

WHAT
  Same SELU+AlphaDropout matched MLP head as the patient-level Stage 2
  (in_dim -> 512 -> 256 -> 1, SELU activations, AlphaDropout(0.1)),
  trained per-ROI on our 592-patient / 1159-ROI working cohort.

  This is the apples-to-apples 'within-harness' anchor for sample-level
  comparisons: UNI2-h alone, BulkRNABert alone, identical head + identical
  training + identical 15-fold MCCV + identical 5 seeds. Only the input
  differs.

  Stage tags:
     sl_2_uni2h_alone_gbmlgg_5seed        --input uni2h
     sl_2_bulkrnabert_alone_gbmlgg_5seed  --input bulkrnabert

WHY THIS IS SAMPLE-LEVEL
  - Each ROI is one training example (multi-ROI patients up-weight by ROI count).
  - For BulkRNABert: patient's 256-d vector is broadcast to each of that
    patient's ROIs (the PF/PGPTomic gene-side convention).
  - Per-fold c-index is computed over ROI predictions (NOT patient-aggregated),
    matching PF Section IV.B.3 and Path-GPTOmic Section 3.1.

HEAD ARCHITECTURE (verbatim port of patient-level Stage 2 head)
  Linear(in_dim, 512) -> SELU -> AlphaDropout(0.1)
  Linear(512,    256) -> SELU -> AlphaDropout(0.1)
  Linear(256,      1)                                    -- Cox log-hazard
  Init: kaiming_normal(nonlinearity='linear') (Lecun-Normal-equivalent for SELU);
        bias = 0

TRAINING (verbatim from patient-level Stage 2 to keep configs comparable)
  Optimizer: Adam, lr=1e-4, weight_decay=0
  Epochs   : 200, full-batch
  Loss     : Cox NLL (harness/eval.py, stable logcumsumexp form)
  Eval     : Harrell c-index over per-ROI predictions (harness/eval.py)
  Splits   : 15-fold MCCV (harness/roi_splits.py reads PF's pnas_splits.csv
             restricted to the 592-patient working cohort, expanded per fold
             to ROI-level Train/Test lists)
  Seeds    : 5  (0..4); per-fold c-index = mean over seeds;
             final = mean +/- std over the 15 fold means.

INPUT SHAPES PER ROI
  --input uni2h         : (1536,)   per-ROI UNI2-h ViT-H/14 embedding
                                    (loaded from all_st_uni2h/<roi>.pt)
  --input bulkrnabert   : (256,)    patient's BulkRNABert embedding broadcast
                                    (one vector per patient, repeated for each
                                    of that patient's ROIs in this batch)

USAGE
  python -m stage2_unimodal_sample.run_unimodal_sample --input uni2h        [--smoke]
  python -m stage2_unimodal_sample.run_unimodal_sample --input bulkrnabert  [--smoke]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from harness.roi_splits import (
    iter_roi_folds, load_working_cohort, load_full_image_cohort, load_pf501_cohort,
    EXPECTED_WORKING_ROIS, EXPECTED_FULL_ROIS, EXPECTED_PF501_ROIS,
)
from harness.roi_loader import (
    load_bulkrnabert_patient_dict,
    load_patient_labels,
    assemble_image_only,
    assemble_gene_only,
)
# KIRC loaders (sample-level KIRC support).
# IMPORTANT: KIRC uses event = censored (NOT 1 - censored like GBMLGG). The
# convention is handled INSIDE load_kirc_patient_labels(), so the assembler
# functions assemble_image_only / assemble_gene_only work unchanged.
from harness.roi_splits_kirc import (
    iter_kirc_roi_folds, load_kirc_cohort, EXPECTED_KIRC_ROIS,
)
from harness.roi_loader_kirc import (
    load_kirc_bulkrnabert_patient_dict, load_kirc_patient_labels,
)
# LUAD loaders (sample-level LUAD support).
# IMPORTANT: LUAD uses event = 1 - censorship (SAME as GBMLGG, OPPOSITE of KIRC).
# Handled inside load_luad_patient_labels().
# LUAD has 1 slide per patient (sample-level = patient-level mathematically).
from harness.roi_splits_luad import (
    iter_luad_roi_folds, iter_luad_5split_folds,
    load_luad_cohort, EXPECTED_LUAD_SLIDES,
)
from harness.roi_loader_luad import (
    load_luad_bulkrnabert_patient_dict, load_luad_patient_labels,
)
# UCEC loaders (sample-level UCEC support).
# IMPORTANT: UCEC uses event = 1 - censorship (SAME as GBMLGG/LUAD, OPPOSITE of KIRC).
# Handled inside load_ucec_patient_labels().
# UCEC has 1 slide per patient (sample-level = patient-level mathematically).
# Working cohort: 478 patients (PF 480 minus 2 patients without BulkRNABert).
# Low event rate (15.7%): only the 5-split protocol is supported.
from harness.roi_splits_ucec import (
    iter_ucec_5split_folds, load_ucec_cohort, EXPECTED_UCEC_SLIDES,
)
from harness.roi_loader_ucec import (
    load_ucec_bulkrnabert_patient_dict, load_ucec_patient_labels,
)
# BRCA loaders (sample-level BRCA support).
# IMPORTANT: BRCA uses event = 1 - censorship (SAME as GBMLGG/LUAD/UCEC, OPPOSITE of KIRC).
# Convention verified against cBioPortal authoritative data. Handled inside
# load_brca_patient_labels().
# BRCA has 1 slide per patient (sample-level = patient-level mathematically).
# Working cohort: 950 patients (PF 957 minus 7 without UNI2-h .pt files).
# Low event rate (13.7%): only the 5-split protocol is supported.
from harness.roi_splits_brca import (
    iter_brca_5split_folds, load_brca_cohort, EXPECTED_BRCA_SLIDES,
)
from harness.roi_loader_brca import (
    load_brca_bulkrnabert_patient_dict, load_brca_patient_labels,
)
from harness.eval import cox_partial_likelihood_loss, cindex_lifeline
from harness.log import log_run


# ============================================================================
# Hyperparameters -- match patient-level Stage 2 verbatim so cross-protocol
# comparison is clean.
# ============================================================================
HIDDEN_DIMS    = (512, 256)
ALPHA_DROP     = 0.1
LR             = 1e-4
WEIGHT_DECAY   = 0.0
EPOCHS         = 200
N_SEEDS        = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Matched SELU+AlphaDropout head (patient-level Stage 2 port)
# ============================================================================
class SELUMLP(nn.Module):
    """in_dim -> 512 -> 256 -> 1, SELU + AlphaDropout(0.1).
    Last linear has no SELU/dropout (Cox log-hazard output).
    SELU + AlphaDropout requires LeCun-Normal init -- approximated by
    kaiming_normal_(weight, nonlinearity='linear') which is equivalent.
    """
    def __init__(self, in_dim: int,
                 hidden: tuple[int, ...] = HIDDEN_DIMS,
                 alpha: float = ALPHA_DROP):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            lin = nn.Linear(prev, h)
            nn.init.kaiming_normal_(lin.weight, nonlinearity='linear')
            nn.init.zeros_(lin.bias)
            layers += [lin, nn.SELU(inplace=True), nn.AlphaDropout(alpha)]
            prev = h
        out = nn.Linear(prev, 1)
        nn.init.kaiming_normal_(out.weight, nonlinearity='linear')
        nn.init.zeros_(out.bias)
        layers += [out]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ============================================================================
# One (fold, seed) fit
# ============================================================================
def fit_eval_one(
    X_train: np.ndarray, e_train: np.ndarray, t_train: np.ndarray,
    X_test:  np.ndarray, e_test:  np.ndarray, t_test:  np.ndarray,
    seed: int,
) -> float:
    """Train SELUMLP, return test c-index. ROI-level inputs throughout."""
    # 1) Standardize on train, apply to test (per-fold to prevent leakage)
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train).astype(np.float32)
    Xte = sc.transform(X_test).astype(np.float32)

    # 2) Seed
    torch.manual_seed(seed); np.random.seed(seed)
    Xtr_t = torch.from_numpy(Xtr).float().to(DEVICE)
    Xte_t = torch.from_numpy(Xte).float().to(DEVICE)
    etr_t = torch.from_numpy(e_train.astype(np.float32)).to(DEVICE)
    ttr_t = torch.from_numpy(t_train.astype(np.float32)).to(DEVICE)

    # 3) Model + optimizer
    model = SELUMLP(in_dim=Xtr.shape[1]).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 4) Full-batch training
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        theta = model(Xtr_t)
        loss  = cox_partial_likelihood_loss(theta, ttr_t, etr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    # 5) Eval
    model.eval()
    with torch.no_grad():
        hazards = model(Xte_t).cpu().numpy()
    return cindex_lifeline(hazards, e_test, t_test)


# ============================================================================
# Sweep: 15 folds x 5 seeds for one --input choice
# ============================================================================
def run_sweep(input_kind: str, seeds: list[int], smoke: bool = False,
              cohort_size: int = 592, dataset: str = "gbmlgg",
              protocol: str = "15fold") -> None:
    assert input_kind in {"uni2h", "bulkrnabert"}, input_kind
    assert dataset in {"gbmlgg", "kirc", "luad", "ucec", "brca"}, dataset
    assert protocol in {"15fold", "5split"}, protocol
    # 5-split is only supported for LUAD/UCEC/BRCA (BulkRNABert paper protocol).
    if protocol == "5split" and dataset not in {"luad", "ucec", "brca"}:
        raise ValueError(
            f"--protocol 5split currently only supported for LUAD, UCEC, BRCA "
            f"(BulkRNABert paper protocol). Got dataset={dataset}."
        )
    # UCEC/BRCA: 5-split is the ONLY supported protocol (low event rate, no 15-fold iter built).
    if dataset == "ucec" and protocol != "5split":
        raise ValueError(
            f"UCEC only supports --protocol 5split (low event rate; 15-fold "
            f"would give ~5 events/test fold, too noisy). Got protocol={protocol}."
        )
    if dataset == "brca" and protocol != "5split":
        raise ValueError(
            f"BRCA only supports --protocol 5split (event rate 13.7%; 15-fold "
            f"would give ~9 events/test fold, too noisy). Got protocol={protocol}."
        )
    # KIRC has exactly one cohort (417 patients); ignore --cohort if dataset=kirc.
    if dataset == "kirc":
        cohort_size = 417
    elif dataset == "luad":
        cohort_size = 450  # LUAD has exactly one cohort (450 patients, 1 slide each)
    elif dataset == "ucec":
        cohort_size = 478  # UCEC working cohort (PF 480 - 2 without BulkRNABert), 1 slide/patient
    elif dataset == "brca":
        cohort_size = 950  # BRCA working cohort (PF 957 - 7 without UNI2-h), 1 slide/patient
    else:
        assert cohort_size in {592, 769, 501}, cohort_size
        # Guard: 769-patient cohort only makes sense for image-only (no gene needed)
        if input_kind == "bulkrnabert" and cohort_size == 769:
            raise ValueError(
                "--cohort 769 is invalid for --input bulkrnabert. BulkRNABert "
                "embeddings exist only for 592 of PF's 769 patients (the other "
                "177 have no RNA-seq in TCGA). Use --cohort 592 with bulkrnabert."
            )
    t0 = time.time()
    print(f"=== Stage 2 sample-level: {input_kind} alone (dataset={dataset}, cohort={cohort_size}) ===")
    print(f"  device : {DEVICE}")
    print(f"  head   : Linear({{1536 or 256}}, 512) -> SELU -> AlphaDropout({ALPHA_DROP})")
    print(f"           Linear(512, 256) -> SELU -> AlphaDropout({ALPHA_DROP}) -> Linear(256, 1)")
    print(f"  optim  : Adam lr={LR}, wd={WEIGHT_DECAY}, full-batch, {EPOCHS} epochs")
    print(f"  seeds  : {seeds}")

    # Pre-load gene dict + labels once (used by all folds)
    # Dataset dispatch: KIRC uses its own loaders (with event=censored convention)
    # GBMLGG uses the canonical loaders.
    print("\nLoading shared dicts ...")
    if dataset == "kirc":
        labels    = load_kirc_patient_labels()
        gene_dict = None
        if input_kind == "bulkrnabert":
            gene_dict = load_kirc_bulkrnabert_patient_dict()
            print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
        print(f"  labels   : {len(labels)} patients (KIRC: event=censored convention)")
        cohort        = load_kirc_cohort()
        expected_rois = EXPECTED_KIRC_ROIS        # 11340
        print(f"  cohort: {len(cohort)} patients (expected 417), {expected_rois} ROIs")
    elif dataset == "luad":
        labels    = load_luad_patient_labels()
        gene_dict = None
        if input_kind == "bulkrnabert":
            gene_dict = load_luad_bulkrnabert_patient_dict()
            print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
        print(f"  labels   : {len(labels)} patients (LUAD: event = 1 - censorship)")
        cohort        = load_luad_cohort()
        expected_rois = EXPECTED_LUAD_SLIDES      # 450
        print(f"  cohort: {len(cohort)} patients (expected 450), {expected_rois} slides")
    elif dataset == "ucec":
        labels    = load_ucec_patient_labels()
        gene_dict = None
        if input_kind == "bulkrnabert":
            gene_dict = load_ucec_bulkrnabert_patient_dict()
            print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
        print(f"  labels   : {len(labels)} patients (UCEC: event = 1 - censorship)")
        cohort        = load_ucec_cohort()
        expected_rois = EXPECTED_UCEC_SLIDES      # 478
        print(f"  cohort: {len(cohort)} patients (expected 478), {expected_rois} slides")
    elif dataset == "brca":
        labels    = load_brca_patient_labels()
        gene_dict = None
        if input_kind == "bulkrnabert":
            gene_dict = load_brca_bulkrnabert_patient_dict()
            print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
        print(f"  labels   : {len(labels)} patients (BRCA: event = 1 - censorship)")
        cohort        = load_brca_cohort()
        expected_rois = EXPECTED_BRCA_SLIDES      # 950
        print(f"  cohort: {len(cohort)} patients (expected 950), {expected_rois} slides")
    else:
        labels    = load_patient_labels()
        gene_dict = None
        if input_kind == "bulkrnabert":
            gene_dict = load_bulkrnabert_patient_dict()
            print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
        print(f"  labels   : {len(labels)} patients")

        # Pick the right GBMLGG cohort.
        if cohort_size == 592:
            cohort        = load_working_cohort()
            expected_rois = EXPECTED_WORKING_ROIS    # 1159
        elif cohort_size == 769:
            cohort        = load_full_image_cohort()
            expected_rois = EXPECTED_FULL_ROIS       # 1505
        else:  # 501  (PF-exact RNA-seq subset intersected with our BulkRNABert)
            cohort        = load_pf501_cohort()
            expected_rois = EXPECTED_PF501_ROIS       # 997
        print(f"  cohort: {len(cohort)} patients (expected {cohort_size}), {expected_rois} ROIs")

    # Choose the right per-fold iterator for the dataset
    # 5-split protocol: use the BulkRNABert paper's 5 stratified 80/20 splits
    # 15-fold protocol: use the dataset's standard 15-fold CV (default)
    def _fold_iter():
        if dataset == "brca":
            # BRCA is 5-split-only (asserted upstream).
            return iter_brca_5split_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "ucec":
            # UCEC is 5-split-only (asserted upstream).
            return iter_ucec_5split_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "luad" and protocol == "5split":
            return iter_luad_5split_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "kirc":
            return iter_kirc_roi_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "luad":
            return iter_luad_roi_folds(cohort=cohort, expected_rois=expected_rois)
        return iter_roi_folds(cohort=cohort, expected_rois=expected_rois)

    # Print which protocol we're using for transparency
    if protocol == "5split":
        if dataset == "brca":
            split_size_text = "760 train + 190 test per split"
        elif dataset == "ucec":
            split_size_text = "382 train + 96 test per split"
        else:  # luad
            split_size_text = "360 train + 90 test per split"
        print(f"\nPROTOCOL: 5×80/20 BulkRNABert paper splits "
              f"(stratified by event, patient-disjoint, {split_size_text})")
    else:
        print(f"\nPROTOCOL: standard 15-fold CV")

    # 15-fold sweep
    if smoke:
        print("\nSMOKE: fold 1 only, seed 0 only")
        fold_iter = _fold_iter()
        fold_idx, train_rois, test_rois = next(fold_iter)
        if input_kind == "uni2h":
            X_train, e_train, t_train, _ = assemble_image_only(train_rois, labels)
            X_test,  e_test,  t_test,  _ = assemble_image_only(test_rois,  labels)
        else:
            X_train, e_train, t_train, _ = assemble_gene_only(train_rois, gene_dict, labels)
            X_test,  e_test,  t_test,  _ = assemble_gene_only(test_rois,  gene_dict, labels)
        print(f"  fold {fold_idx} train: X={X_train.shape}  events={int(e_train.sum())}/{len(e_train)}")
        print(f"  fold {fold_idx} test : X={X_test.shape}  events={int(e_test.sum())}/{len(e_test)}")
        ts = time.time()
        c = fit_eval_one(X_train, e_train, t_train, X_test, e_test, t_test, seed=0)
        print(f"  fold {fold_idx} c-index (seed 0): {c:.4f}  ({time.time()-ts:.1f}s)")
        print(f"SMOKE OK total {time.time()-t0:.1f}s")
        return

    if protocol == "5split":
        # 5×80/20 BulkRNABert paper protocol:
        # 5 splits, each is one (training, evaluation) measurement.
        # The split-derived seed (split_idx - 1) is used as both the
        # SPLIT randomization seed AND the model init seed -- matching the
        # paper's "5 different seeds" convention.
        print(f"\nFULL: 5×80/20 splits = 5 fits total (BulkRNABert paper protocol)")
        all_c = np.zeros(5, dtype=np.float64)
        for si, (split_idx, train_rois, test_rois) in enumerate(_fold_iter()):
            ts = time.time()
            seed = split_idx - 1   # split-derived model init seed
            if input_kind == "uni2h":
                X_train, e_train, t_train, _ = assemble_image_only(train_rois, labels)
                X_test,  e_test,  t_test,  _ = assemble_image_only(test_rois,  labels)
            else:
                X_train, e_train, t_train, _ = assemble_gene_only(train_rois, gene_dict, labels)
                X_test,  e_test,  t_test,  _ = assemble_gene_only(test_rois,  gene_dict, labels)
            c = fit_eval_one(X_train, e_train, t_train, X_test, e_test, t_test, seed=seed)
            all_c[si] = c
            print(f"  split {split_idx}: c={c:.4f}  "
                  f"({len(train_rois)} train, {len(test_rois)} test, "
                  f"time={time.time()-ts:.0f}s)")

        # Per-split values double as "per_fold_means" for downstream logging
        per_fold_means = all_c.copy()
        grand_mean = float(all_c.mean())
        grand_std  = float(all_c.std())

    else:
        # Standard 15-fold × 5 seeds protocol (default for GBMLGG/KIRC/LUAD legacy)
        print(f"\nFULL: 15 folds x {len(seeds)} seeds = {15*len(seeds)} fits")
        all_c = np.zeros((len(seeds), 15), dtype=np.float64)
        for si, seed in enumerate(seeds):
            ts = time.time()
            for fi, (fold_idx, train_rois, test_rois) in enumerate(_fold_iter()):
                if input_kind == "uni2h":
                    X_train, e_train, t_train, _ = assemble_image_only(train_rois, labels)
                    X_test,  e_test,  t_test,  _ = assemble_image_only(test_rois,  labels)
                else:
                    X_train, e_train, t_train, _ = assemble_gene_only(train_rois, gene_dict, labels)
                    X_test,  e_test,  t_test,  _ = assemble_gene_only(test_rois,  gene_dict, labels)
                c = fit_eval_one(X_train, e_train, t_train, X_test, e_test, t_test, seed=seed)
                all_c[si, fi] = c
            print(f"  seed {seed}: mean = {all_c[si].mean():.4f}  "
                  f"(min={all_c[si].min():.4f}, max={all_c[si].max():.4f}, "
                  f"time={time.time()-ts:.0f}s)")

        per_fold_means = all_c.mean(axis=0)
        grand_mean = float(per_fold_means.mean())
        grand_std  = float(per_fold_means.std())
    print(f"\n=== {input_kind} alone (sample-level): 5-seed averaged ===")
    print(f"  GRAND mean +/- fold std: {grand_mean:.4f} +/- {grand_std:.4f}")
    print(f"  per-fold means          : {per_fold_means.round(4).tolist()}")
    if dataset == "kirc":
        if input_kind == "uni2h":
            print(f"  comparators (KIRC):")
            print(f"    PF Histology CNN (paper, KIRC)        :  0.671 +/- 0.023")
        else:
            print(f"  comparators (KIRC):")
            print(f"    PF Genomic SNN (paper, KIRC)          :  0.684 +/- 0.025")
            print(f"    PF trimodal headline (KIRC paper)     :  0.720 +/- 0.028")
    elif dataset == "luad":
        print(f"  comparators (LUAD):")
        print(f"    BulkRNABert paper Table 3 (LUAD)      :  0.648 +/- 0.057")
        print(f"    our patient-level BulkRNABert (LUAD)  :  0.6275 +/- 0.109")
        print(f"    our patient-level UNI2-h     (LUAD)   :  0.5694 +/- 0.093")
    elif dataset == "ucec":
        print(f"  comparators (UCEC):")
        print(f"    BulkRNABert paper Table 3 (UCEC)      :  0.703 +/- 0.040  (their strongest cohort)")
    elif dataset == "brca":
        print(f"  comparators (BRCA):")
        print(f"    MCAT paper Table 1   (BRCA)           :  0.580 +/- 0.069")
        print(f"    MOTCat paper Table 1 (BRCA)           :  0.673 +/- 0.006")
        print(f"    CustOmics 2023 (BRCA)                 :  ~0.65    (Benkirane)")
    else:
        if input_kind == "uni2h":
            print(f"  comparators (GBMLGG):")
            print(f"    PF Histology CNN (paper, sample-level):  0.792 +/- 0.014")
            print(f"    our patient-level UNI2-h alone        :  0.7466 +/- 0.0195")
        else:
            print(f"  comparators (GBMLGG):")
            print(f"    PF Genomic SNN (paper, sample-level)  :  0.808 +/- 0.014")
            print(f"    our patient-level BulkRNABert alone   :  0.8124 +/- 0.0285")

    # Log to results_table.csv
    # Cohort suffix appears in stage_tag for non-default GBMLGG cohorts to
    # keep backward compatibility. For KIRC, the suffix is empty (only one
    # cohort) and the dataset suffix changes from gbmlgg -> kirc.
    if dataset == "kirc":
        cohort_suffix = ""
        dataset_suffix = "kirc"
    elif dataset == "luad":
        cohort_suffix = ""
        # Protocol suffix: distinguish 15-fold (default, _5seed tag) from 5-split.
        dataset_suffix = "luad_5split" if protocol == "5split" else "luad"
    elif dataset == "ucec":
        cohort_suffix = ""
        # UCEC is 5-split-only.
        dataset_suffix = "ucec_5split"
    elif dataset == "brca":
        cohort_suffix = ""
        # BRCA is 5-split-only.
        dataset_suffix = "brca_5split"
    else:
        if cohort_size == 769:
            cohort_suffix = "_full769"
        elif cohort_size == 501:
            cohort_suffix = "_pf501"
        else:
            cohort_suffix = ""
        dataset_suffix = "gbmlgg"
    if input_kind == "uni2h":
        stage_tag = f"sl_2_uni2h_alone{cohort_suffix}_{dataset_suffix}_5seed"
        image_enc = "UNI2-h-frozen-1536"
        gene_enc  = "none"
        mode_str  = "alone_image"
        in_dim    = 1536
    else:
        stage_tag = f"sl_2_bulkrnabert_alone{cohort_suffix}_{dataset_suffix}_5seed"
        image_enc = "none"
        gene_enc  = "BulkRNABert-frozen-256"
        mode_str  = "alone_gene"
        in_dim    = 256
    n_params = sum(p.numel() for p in SELUMLP(in_dim=in_dim).parameters()
                   if p.requires_grad)
    if dataset == "kirc":
        cohort_text = ("417-patient/11340-ROI PF KIRC cohort (matches PF Section IV.A "
                       "for TCGA-KIRC; 100% intersection with our UNI2-h ∩ BulkRNABert; "
                       "event = censored per KIRC convention).")
        cohort_field = "KIRC"
        splits_text  = "15-fold MCCV from PF's KIRC_st_1.pkl (patient-level partition)"
    elif dataset == "luad":
        cohort_text = ("450-patient/450-slide PF LUAD cohort (100% intersection with our "
                       "UNI2-h ∩ BulkRNABert; 1 slide per patient -> sample-level = "
                       "patient-level mathematically; event = 1 - censorship per LUAD "
                       "convention, SAME as GBMLGG).")
        cohort_field = "LUAD"
        if protocol == "5split":
            splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper "
                           "protocol; 360 train + 90 test per split, stratified by "
                           "event, patient-disjoint, seeds 0..4)")
        else:
            splits_text = "15-fold CV from master_splits.csv + splits_0..14.csv"
    elif dataset == "ucec":
        cohort_text = ("478-patient/478-slide UCEC working cohort (PF 480 minus 2 patients "
                       "with no BulkRNABert: TCGA-AP-A0LQ, TCGA-EY-A1GJ; 1 slide per "
                       "patient -> sample-level = patient-level mathematically; "
                       "event = 1 - censorship per UCEC convention, SAME as GBMLGG/LUAD; "
                       "low event rate 15.7%).")
        cohort_field = "UCEC"
        splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper protocol; "
                       "382 train + 96 test per split, stratified by event, "
                       "patient-disjoint, seeds 0..4)")
    elif dataset == "brca":
        cohort_text = ("950-patient/950-slide BRCA working cohort (PF 957 minus 7 patients "
                       "with no UNI2-h .pt files: TCGA-OL-A5RU/A5RV/A5RW/A5RX/A5RY/A5RZ/A5S0; "
                       "1 slide per patient -> sample-level = patient-level mathematically; "
                       "event = 1 - censorship per BRCA convention (verified vs cBioPortal), "
                       "SAME as GBMLGG/LUAD/UCEC; LOWEST event rate in our sweep 13.7%. "
                       "UNI2-h features are patch-level (n_patches, 1536) mean-pooled to "
                       "slide-level (1536,) at load time.")
        cohort_field = "BRCA"
        splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper protocol; "
                       "760 train + 190 test per split, stratified by event, "
                       "patient-disjoint, seeds 0..4)")
    elif cohort_size == 592:
        cohort_text = "592-patient/1159-ROI working cohort (PF cohort ∩ BulkRNABert)."
        cohort_field = "GBMLGG"
        splits_text  = "15-fold MCCV from pnas_splits.csv"
    elif cohort_size == 769:
        cohort_text = ("FULL 769-patient/1505-ROI PF cohort (image-only; matches PF "
                       "Histology CNN cohort -- NO BulkRNABert restriction).")
        cohort_field = "GBMLGG"
        splits_text  = "15-fold MCCV from pnas_splits.csv"
    else:  # 501
        cohort_text = ("501-patient/997-ROI PF-EXACT cohort (PF's 502-patient RNA-seq "
                       "subset INTERSECTED with our BulkRNABert NPZ -- 1 PF patient "
                       "(TCGA-06-0221) has no BulkRNABert embedding). Direct apples-"
                       "to-apples with PF Genomic SNN paper number 0.808.")
        cohort_field = "GBMLGG"
        splits_text  = "15-fold MCCV from pnas_splits.csv"
    log_run(
        stage=stage_tag,
        image_encoder=image_enc, gene_encoder=gene_enc,
        fusion="none", mode=mode_str, cohort=cohort_field,
        n_params=n_params, flops=0,
        c_index_mean=grand_mean, c_index_std=grand_std,
        per_fold_c_indices=per_fold_means.tolist(),
        seed=-1,
        notes=(
            f"Sample-level Stage 2 unimodal -- input={input_kind}, dataset={dataset}. "
            f"{cohort_text} "
            f"{splits_text} expanded patient->ROI. "
            f"5 seeds {seeds} averaged. ROI-level c-index over predictions "
            f"with patient-broadcast labels (PF/PGPTomic convention). "
            f"SELU+AlphaDropout(0.1) MLP({in_dim}->512->256->1), Cox NLL, "
            f"Adam lr={LR} wd={WEIGHT_DECAY}, {EPOCHS} epochs full-batch."
        ),
    )
    print(f"\n  logged stage={stage_tag}")
    print(f"  total wall time: {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", choices=["uni2h", "bulkrnabert"], required=True,
                    help="which unimodal input to evaluate at sample-level")
    ap.add_argument("--dataset", choices=["gbmlgg", "kirc", "luad", "ucec", "brca"], default="gbmlgg",
                    help="which dataset to run on. gbmlgg uses --cohort arg; "
                         "kirc uses its 417-patient cohort, luad its 450-patient cohort, "
                         "ucec its 478-patient cohort, brca its 950-patient cohort. "
                         "All non-gbmlgg cohorts ignore --cohort. "
                         "KIRC uses event = censored convention. "
                         "LUAD/UCEC/GBMLGG/BRCA use event = 1 - censored. "
                         "UCEC and BRCA are 5-split-only (low event rate). "
                         "All handled inside the per-dataset loaders.")
    ap.add_argument("--cohort", type=int, choices=[592, 769, 501], default=592,
                    help="GBMLGG cohort size (ignored when --dataset=kirc): "
                         "592 = PF ∩ BulkRNABert (default); "
                         "769 = full PF cohort (image-only, matches PF Histology CNN). "
                         "769 is invalid for --input bulkrnabert. "
                         "501 = PF-EXACT RNA-seq cohort ∩ BulkRNABert.")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)),
                    help="seeds to average over (default 0..4). Ignored under --protocol 5split.")
    ap.add_argument("--protocol", choices=["15fold", "5split"], default="15fold",
                    help="CV protocol. 15fold (default): standard k-fold CV used for "
                         "GBMLGG/KIRC/LUAD legacy. 5split: 5 × stratified 80/20 random "
                         "splits as used in BulkRNABert paper Table 3 (LUAD and UCEC). "
                         "Each split uses split-derived seed for BOTH split and model init "
                         "-- matches paper's '5 different seeds' convention. "
                         "UCEC ONLY supports 5split.")
    ap.add_argument("--smoke", action="store_true",
                    help="fold 1 + seed 0 only, no logging")
    args = ap.parse_args()
    run_sweep(args.input, args.seeds, smoke=args.smoke,
              cohort_size=args.cohort, dataset=args.dataset, protocol=args.protocol)


if __name__ == "__main__":
    main()
