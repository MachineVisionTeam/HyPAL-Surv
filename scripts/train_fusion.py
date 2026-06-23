"""
Stage 4 sample-level fusion sweep for GBMLGG.

WHAT
   UNI2-h (1536-d, per-ROI) x BulkRNABert (256-d, patient broadcast)
   -> ImageAdapter / BulkRNABertAdapter -> fusion op -> Cox head -> log-hazard

   Identical to the patient-level Stage 4 architecture; only iteration is per-ROI
   (one training example per ROI) instead of per-patient. Multi-ROI patients
   contribute multiple examples, with the same patient gene vector + (T, e)
   broadcast across them. This is the PF/Path-GPTOmic convention.

USAGE
   python -m stage4_fusion_sample.run_stage4_sample --fusion {concat|kronecker|lmf|phm}  [--smoke]

ARCHITECTURE
   ImageAdapter         : Linear(1536, 256) -> SELU -> AlphaDropout(0.1)        -> z_image
   BulkRNABertAdapter   : Linear( 256, 256) -> SELU -> AlphaDropout(0.1)        -> z_gene
   Fusion (one of)      : build_fusion(name, dim_a=256, dim_b=256, fused_dim=256,
                            n=4 for PHM)                                          -> fused (256,)
   Cox head             : Linear(256, 1)                                         -> hazard (1,)

TRAINING -- identical hyperparameters to patient-level Stage 4
   Optim   : Adam lr=1e-4, WD=0
   Epochs  : 200, full-batch
   Splits  : 15-fold MCCV, pnas_splits expanded patient->ROI (harness.roi_splits)
   Seeds   : 5 (0..4); per-fold c-index = mean over seeds;
             final = mean +/- std over the 15 fold means
   Loss    : Cox NLL (harness.eval.cox_partial_likelihood_loss)
   Eval    : c-Index over per-ROI predictions (harness.eval.cindex_lifeline)

STAGE TAGS LOGGED
   sl_4_concat_bulkrnabert_gbmlgg_5seed
   sl_4_kronecker_bulkrnabert_gbmlgg_5seed
   sl_4_lmf_bulkrnabert_gbmlgg_5seed
   sl_4_phm_bulkrnabert_gbmlgg_5seed   (Stage 5 reference cell)
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
    iter_roi_folds, load_working_cohort, load_pf501_cohort,
    EXPECTED_WORKING_ROIS, EXPECTED_PF501_ROIS,
)
from harness.roi_loader import (
    load_bulkrnabert_patient_dict,
    load_patient_labels,
    assemble_roi_batch,
)
# KIRC loaders (sample-level KIRC support).
# IMPORTANT: KIRC uses event = censored (NOT 1 - censored like GBMLGG). The
# convention is handled INSIDE load_kirc_patient_labels(), so the assembler
# (assemble_kirc_roi_batch) works the same as assemble_roi_batch.
from harness.roi_splits_kirc import (
    iter_kirc_roi_folds, load_kirc_cohort, EXPECTED_KIRC_ROIS,
)
from harness.roi_loader_kirc import (
    load_kirc_bulkrnabert_patient_dict, load_kirc_patient_labels,
    assemble_kirc_roi_batch,
)
# LUAD loaders (sample-level). event = 1 - censorship (SAME as GBMLGG).
from harness.roi_splits_luad import (
    iter_luad_roi_folds, iter_luad_5split_folds,
    load_luad_cohort, EXPECTED_LUAD_SLIDES,
)
from harness.roi_loader_luad import (
    load_luad_bulkrnabert_patient_dict, load_luad_patient_labels,
    assemble_luad_roi_batch,
)
# UCEC loaders (sample-level). event = 1 - censorship (SAME as GBMLGG/LUAD).
# 478-patient working cohort, 1 slide/patient, 5-split only.
from harness.roi_splits_ucec import (
    iter_ucec_5split_folds, load_ucec_cohort, EXPECTED_UCEC_SLIDES,
)
from harness.roi_loader_ucec import (
    load_ucec_bulkrnabert_patient_dict, load_ucec_patient_labels,
    assemble_ucec_roi_batch,
)
# BRCA loaders (sample-level). event = 1 - censorship (SAME as GBMLGG/LUAD/UCEC).
# 950-patient working cohort, 1 slide/patient, 5-split only. UNI2-h features
# are patch-level mean-pooled to slide-level inside the BRCA assembler.
from harness.roi_splits_brca import (
    iter_brca_5split_folds, load_brca_cohort, EXPECTED_BRCA_SLIDES,
)
from harness.roi_loader_brca import (
    load_brca_bulkrnabert_patient_dict, load_brca_patient_labels,
    assemble_brca_roi_batch,
)
from harness.eval import cox_partial_likelihood_loss, cindex_lifeline
from harness.log import log_run
from stage4_fusion_sample.fusion_ops import build_fusion


# ============================================================================
# Hyperparameters -- match patient-level Stage 4 verbatim
# ============================================================================
IMAGE_DIM      = 1536
BULKRNA_DIM    = 256
FUSION_DIM     = 256          # both branches project to 256-d
ALPHA_DROP     = 0.1
PHM_N          = 4
BKRON_BOTTLE   = 64    # bottleneck dim for BottleneckKronFusion / PHMKronHybridFusion
LR             = 1e-4
WEIGHT_DECAY   = 0.0
EPOCHS         = 200
N_SEEDS        = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


def _selu_linear(in_dim: int, out_dim: int) -> nn.Module:
    """Linear -> SELU -> AlphaDropout(0.1).  LeCun-Normal init via
    kaiming_normal(nonlinearity='linear'). Matches patient-level Stage 4."""
    lin = nn.Linear(in_dim, out_dim)
    nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
    nn.init.zeros_(lin.bias)
    return nn.Sequential(lin, nn.SELU(inplace=True), nn.AlphaDropout(ALPHA_DROP))


class ImageAdapter(nn.Module):
    """UNI2-h 1536 -> 256 (z_image)."""
    def __init__(self) -> None:
        super().__init__()
        self.net = _selu_linear(IMAGE_DIM, FUSION_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BulkRNABertAdapter(nn.Module):
    """BulkRNABert 256 -> 256 (z_gene). Symmetric to ImageAdapter."""
    def __init__(self) -> None:
        super().__init__()
        self.net = _selu_linear(BULKRNA_DIM, FUSION_DIM)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModel(nn.Module):
    """Sample-level Stage 4 model: ROI image + patient-broadcast gene
    -> fusion op -> Cox log-hazard.
    """
    def __init__(self, fusion_name: str) -> None:
        super().__init__()
        self.fusion_name   = fusion_name
        self.image_adapter = ImageAdapter()
        self.gene_encoder  = BulkRNABertAdapter()
        # Per-op kwargs:
        #   concat / kronecker / lmf : no special args
        #   phm                      : n=PHM_N (4)
        #   bkron                    : bottle=BKRON_BOTTLE (64)
        #   phm_kron                 : n=PHM_N + bottle=BKRON_BOTTLE
        if fusion_name == "phm":
            kw = {"n": PHM_N}
        elif fusion_name == "bkron":
            kw = {"bottle": BKRON_BOTTLE}
        elif fusion_name == "phm_kron":
            kw = {"n": PHM_N, "bottle": BKRON_BOTTLE}
        elif fusion_name == "abkron":
            kw = {"bottle": BKRON_BOTTLE}
        elif fusion_name == "slim_cpkf":
            kw = {"bottle": BKRON_BOTTLE, "n": PHM_N}
        else:
            kw = {}
        self.fusion = build_fusion(
            fusion_name,
            dim_a=FUSION_DIM, dim_b=FUSION_DIM, fused_dim=FUSION_DIM,
            **kw,
        )
        self.cox_head = nn.Linear(FUSION_DIM, 1)
        nn.init.kaiming_normal_(self.cox_head.weight, nonlinearity="linear")
        nn.init.zeros_(self.cox_head.bias)

    def forward(self, x_image: torch.Tensor, x_gene: torch.Tensor) -> torch.Tensor:
        z_image = self.image_adapter(x_image)              # (B, 256)
        z_gene  = self.gene_encoder(x_gene)                # (B, 256)
        fused   = self.fusion(z_image, z_gene)             # (B, 256)
        return self.cox_head(fused).squeeze(-1)            # (B,)


# ============================================================================
# Per-(fold, seed) fit
# ============================================================================
def fit_eval_one(
    fusion_name: str,
    X_img_train: np.ndarray, X_gene_train: np.ndarray,
    e_train:     np.ndarray, t_train:     np.ndarray,
    X_img_test:  np.ndarray, X_gene_test: np.ndarray,
    e_test:      np.ndarray, t_test:      np.ndarray,
    seed: int,
) -> float:
    """One sample-level Stage 4 fit. Returns test c-index over ROI predictions."""
    # 1) Z-score image + gene independently on train, apply to test
    sc_img  = StandardScaler()
    sc_gene = StandardScaler()
    Xi_tr = sc_img .fit_transform(X_img_train ).astype(np.float32)
    Xi_te = sc_img .transform    (X_img_test  ).astype(np.float32)
    Xg_tr = sc_gene.fit_transform(X_gene_train).astype(np.float32)
    Xg_te = sc_gene.transform    (X_gene_test ).astype(np.float32)

    # 2) Seed
    torch.manual_seed(seed); np.random.seed(seed)
    Xi_tr_t = torch.from_numpy(Xi_tr).float().to(DEVICE)
    Xi_te_t = torch.from_numpy(Xi_te).float().to(DEVICE)
    Xg_tr_t = torch.from_numpy(Xg_tr).float().to(DEVICE)
    Xg_te_t = torch.from_numpy(Xg_te).float().to(DEVICE)
    etr_t   = torch.from_numpy(e_train.astype(np.float32)).to(DEVICE)
    ttr_t   = torch.from_numpy(t_train.astype(np.float32)).to(DEVICE)

    # 3) Model
    model = FusionModel(fusion_name).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # 4) Full-batch training
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        theta = model(Xi_tr_t, Xg_tr_t)
        loss  = cox_partial_likelihood_loss(theta, ttr_t, etr_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

    # 5) Eval
    model.eval()
    with torch.no_grad():
        hazards = model(Xi_te_t, Xg_te_t).cpu().numpy()
    return cindex_lifeline(hazards, e_test, t_test)


# ============================================================================
# Sweep
# ============================================================================
def run_sweep(fusion_name: str, seeds: list[int], smoke: bool = False,
              cohort_size: int = 592, dataset: str = "gbmlgg",
              protocol: str = "15fold") -> None:
    assert fusion_name in {"concat", "kronecker", "lmf", "phm", "bkron", "phm_kron", "abkron", "slim_cpkf"}, fusion_name
    assert dataset in {"gbmlgg", "kirc", "luad", "ucec", "brca"}, dataset
    assert protocol in {"15fold", "5split"}, protocol
    # 5-split is only supported for LUAD/UCEC/BRCA (BulkRNABert paper protocol).
    if protocol == "5split" and dataset not in {"luad", "ucec", "brca"}:
        raise ValueError(
            f"--protocol 5split currently only supported for LUAD, UCEC, BRCA. "
            f"Got dataset={dataset}."
        )
    # UCEC/BRCA are 5-split-only (low event rate).
    if dataset == "ucec" and protocol != "5split":
        raise ValueError(
            f"UCEC only supports --protocol 5split. Got protocol={protocol}."
        )
    if dataset == "brca" and protocol != "5split":
        raise ValueError(
            f"BRCA only supports --protocol 5split. Got protocol={protocol}."
        )
    if dataset == "kirc":
        cohort_size = 417  # KIRC has exactly one cohort; ignore --cohort
    elif dataset == "luad":
        cohort_size = 450  # LUAD has exactly one cohort (450 patients, 1 slide each)
    elif dataset == "ucec":
        cohort_size = 478  # UCEC working cohort (PF 480 minus 2 without BulkRNABert)
    elif dataset == "brca":
        cohort_size = 950  # BRCA working cohort (PF 957 minus 7 without UNI2-h)
    else:
        assert cohort_size in {592, 501}, cohort_size
    t0 = time.time()
    print(f"=== Stage 4 sample-level: {fusion_name} (dataset={dataset}, UNI2-h x BulkRNABert) ===")
    print(f"  device       : {DEVICE}")
    print(f"  branches     : image (1536->256), gene (256->256) -- both SELU+AlphaDropout({ALPHA_DROP})")
    if fusion_name == "phm":
        print(f"  fusion op    : PHM (n={PHM_N})  256+256 -> 256")
    elif fusion_name == "slim_cpkf":
        print(f"  fusion op    : HGBF (PHM-modulated BKron in bottleneck, "
              f"bottle={BKRON_BOTTLE}, n={PHM_N})  256+256 -> 256")
    else:
        print(f"  fusion op    : {fusion_name}  256+256 -> 256")
    print(f"  cohort       : {cohort_size} patients")
    print(f"  optim        : Adam lr={LR}, wd={WEIGHT_DECAY}, full-batch, {EPOCHS} epochs")
    print(f"  seeds        : {seeds}")

    # Pre-load dicts once.
    # Dataset dispatch:
    #   KIRC  -> KIRC loaders (event = censored convention, 417 cohort)
    #   GBMLGG -> standard loaders (event = 1 - censored, 592/501 cohorts)
    print("\nLoading shared dicts ...")
    if dataset == "kirc":
        gene_dict = load_kirc_bulkrnabert_patient_dict()
        labels    = load_kirc_patient_labels()
        cohort        = load_kirc_cohort()
        expected_rois = EXPECTED_KIRC_ROIS    # 11340
        assembler     = assemble_kirc_roi_batch
        print(f"  gene_dict : {len(gene_dict)} patients, dim=256 (KIRC BulkRNABert)")
        print(f"  labels    : {len(labels)} patients (KIRC: event=censored)")
        print(f"  cohort    : {len(cohort)} patients (expected 417), {expected_rois} ROIs")
    elif dataset == "luad":
        gene_dict = load_luad_bulkrnabert_patient_dict()
        labels    = load_luad_patient_labels()
        cohort        = load_luad_cohort()
        expected_rois = EXPECTED_LUAD_SLIDES  # 450
        assembler     = assemble_luad_roi_batch
        print(f"  gene_dict : {len(gene_dict)} patients, dim=256 (LUAD BulkRNABert)")
        print(f"  labels    : {len(labels)} patients (LUAD: event = 1 - censorship)")
        print(f"  cohort    : {len(cohort)} patients (expected 450), {expected_rois} slides")
    elif dataset == "ucec":
        gene_dict = load_ucec_bulkrnabert_patient_dict()
        labels    = load_ucec_patient_labels()
        cohort        = load_ucec_cohort()
        expected_rois = EXPECTED_UCEC_SLIDES  # 478
        assembler     = assemble_ucec_roi_batch
        print(f"  gene_dict : {len(gene_dict)} patients, dim=256 (UCEC BulkRNABert)")
        print(f"  labels    : {len(labels)} patients (UCEC: event = 1 - censorship)")
        print(f"  cohort    : {len(cohort)} patients (expected 478), {expected_rois} slides")
    elif dataset == "brca":
        gene_dict = load_brca_bulkrnabert_patient_dict()
        labels    = load_brca_patient_labels()
        cohort        = load_brca_cohort()
        expected_rois = EXPECTED_BRCA_SLIDES  # 950
        assembler     = assemble_brca_roi_batch
        print(f"  gene_dict : {len(gene_dict)} patients, dim=256 (BRCA BulkRNABert)")
        print(f"  labels    : {len(labels)} patients (BRCA: event = 1 - censorship)")
        print(f"  cohort    : {len(cohort)} patients (expected 950), {expected_rois} slides")
    else:
        gene_dict = load_bulkrnabert_patient_dict()
        labels    = load_patient_labels()
        if cohort_size == 592:
            cohort        = load_working_cohort()
            expected_rois = EXPECTED_WORKING_ROIS    # 1159
        else:  # 501 -- PF-exact intersection
            cohort        = load_pf501_cohort()
            expected_rois = EXPECTED_PF501_ROIS      # 997
        assembler     = assemble_roi_batch
        print(f"  gene_dict     : {len(gene_dict)} patients, dim=256")
        print(f"  labels        : {len(labels)} patients")
        print(f"  cohort        : {len(cohort)} patients (expected {cohort_size}), "
              f"{expected_rois} ROIs expected")

    # Choose per-fold iterator based on dataset and protocol
    def _fold_iter():
        if dataset == "brca":
            return iter_brca_5split_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "ucec":
            return iter_ucec_5split_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "luad" and protocol == "5split":
            return iter_luad_5split_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "kirc":
            return iter_kirc_roi_folds(cohort=cohort, expected_rois=expected_rois)
        if dataset == "luad":
            return iter_luad_roi_folds(cohort=cohort, expected_rois=expected_rois)
        return iter_roi_folds(cohort=cohort, expected_rois=expected_rois)

    # Print protocol for transparency
    if protocol == "5split":
        if dataset == "brca":
            split_size_text = "760 train + 190 test per split"
        elif dataset == "ucec":
            split_size_text = "382 train + 96 test per split"
        else:  # luad
            split_size_text = "360 train + 90 test per split"
        print(f"\nPROTOCOL: 5×80/20 BulkRNABert paper splits "
              f"({split_size_text}, stratified by event)")
    else:
        print(f"\nPROTOCOL: standard 15-fold CV")

    if smoke:
        print("\nSMOKE: fold 1 only, seed 0 only")
        fold_iter = _fold_iter()
        fold_idx, train_rois, test_rois = next(fold_iter)
        Xi_tr, Xg_tr, e_tr, t_tr, _ = assembler(train_rois, gene_dict, labels, expected_gene_dim=256)
        Xi_te, Xg_te, e_te, t_te, _ = assembler(test_rois,  gene_dict, labels, expected_gene_dim=256)
        print(f"  fold {fold_idx} train: Xi={Xi_tr.shape}, Xg={Xg_tr.shape}, events={int(e_tr.sum())}/{len(e_tr)}")
        print(f"  fold {fold_idx} test : Xi={Xi_te.shape}, Xg={Xg_te.shape}, events={int(e_te.sum())}/{len(e_te)}")
        ts = time.time()
        c = fit_eval_one(fusion_name,
                         Xi_tr, Xg_tr, e_tr, t_tr,
                         Xi_te, Xg_te, e_te, t_te, seed=0)
        n_params = sum(p.numel() for p in FusionModel(fusion_name).parameters() if p.requires_grad)
        print(f"  fold {fold_idx} c-index (seed 0): {c:.4f}   ({time.time()-ts:.1f}s)")
        print(f"  model n_params: {n_params:,}")
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
            seed = split_idx - 1  # split-derived model init seed
            Xi_tr, Xg_tr, e_tr, t_tr, _ = assembler(train_rois, gene_dict, labels, expected_gene_dim=256)
            Xi_te, Xg_te, e_te, t_te, _ = assembler(test_rois,  gene_dict, labels, expected_gene_dim=256)
            c = fit_eval_one(fusion_name,
                             Xi_tr, Xg_tr, e_tr, t_tr,
                             Xi_te, Xg_te, e_te, t_te, seed=seed)
            all_c[si] = c
            print(f"  split {split_idx}: c={c:.4f}  "
                  f"({len(train_rois)} train, {len(test_rois)} test, "
                  f"time={time.time()-ts:.0f}s)")

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
                Xi_tr, Xg_tr, e_tr, t_tr, _ = assembler(train_rois, gene_dict, labels, expected_gene_dim=256)
                Xi_te, Xg_te, e_te, t_te, _ = assembler(test_rois,  gene_dict, labels, expected_gene_dim=256)
                c = fit_eval_one(fusion_name,
                                 Xi_tr, Xg_tr, e_tr, t_tr,
                                 Xi_te, Xg_te, e_te, t_te, seed=seed)
                all_c[si, fi] = c
            print(f"  seed {seed}: mean = {all_c[si].mean():.4f}  "
                  f"(min={all_c[si].min():.4f}, max={all_c[si].max():.4f}, "
                  f"time={time.time()-ts:.0f}s)")

        per_fold_means = all_c.mean(axis=0)
        grand_mean = float(per_fold_means.mean())
        grand_std  = float(per_fold_means.std())
    print(f"\n=== {fusion_name} (sample-level UNI2-h x BulkRNABert, dataset={dataset}): 5-seed averaged ===")
    print(f"  GRAND mean +/- fold std: {grand_mean:.4f} +/- {grand_std:.4f}")
    print(f"  per-fold means          : {per_fold_means.round(4).tolist()}")
    print(f"  comparators (sample-level):")
    if dataset == "kirc":
        print(f"    KIRC UNI2-h alone (this harness)         : 0.6824 +/- 0.0262")
        print(f"    KIRC BulkRNABert alone (this harness)    : 0.6678 +/- 0.0368")
        print(f"    PF KIRC trimodal (our replication)       : 0.7184 +/- 0.0513  <-- main anchor")
        print(f"    PF KIRC pathomic (our replication)       : 0.7049 +/- 0.0568")
    elif dataset == "luad":
        print(f"    BulkRNABert paper Table 3 (LUAD)         : 0.648 +/- 0.057")
        print(f"    patient-level BulkRNABert (LUAD)         : 0.6275 +/- 0.109")
        print(f"    patient-level UNI2-h     (LUAD)          : 0.5694 +/- 0.093")
        print(f"    patient-level Kronecker fusion (LUAD)    : 0.6055 +/- 0.100  <-- best patient-lvl fusion")
    elif dataset == "ucec":
        print(f"    BulkRNABert paper Table 3 (UCEC)         : 0.703 +/- 0.040  <-- their strongest cohort")
    elif dataset == "brca":
        print(f"    MCAT paper Table 1   (BRCA)              : 0.580 +/- 0.069")
        print(f"    MOTCat paper Table 1 (BRCA)              : 0.673 +/- 0.006")
        print(f"    CustOmics 2023 (BRCA)                    : ~0.65")
    else:
        print(f"    GBMLGG uni_anchor (BulkRNABert alone)    : 0.8139 +/- 0.0421")
        print(f"    PF GBMLGG TRIMODAL (our replication)     : 0.8174 +/- 0.0717")
        print(f"    Path-GPTOmic (paper-reported)            : 0.848  +/- 0.014")

    n_params = sum(p.numel() for p in FusionModel(fusion_name).parameters() if p.requires_grad)
    # Stage tag: switch dataset suffix (gbmlgg <-> kirc) and add cohort suffix for non-default GBMLGG.
    if dataset == "kirc":
        cohort_suffix = ""
        dataset_suffix = "kirc"
        cohort_field = "KIRC"
        cohort_text = ("417-patient/11340-ROI PF KIRC cohort (matches PF Section IV.A "
                       "for TCGA-KIRC; 100% intersection with our UNI2-h ∩ BulkRNABert; "
                       "event = censored per KIRC convention).")
        splits_text = "15-fold MCCV from PF's KIRC_st_1.pkl (patient-level partition)"
    elif dataset == "luad":
        cohort_suffix = ""
        dataset_suffix = "luad_5split" if protocol == "5split" else "luad"
        cohort_field = "LUAD"
        cohort_text = ("450-patient/450-slide PF LUAD cohort (1 slide/patient; "
                       "100% intersection with UNI2-h ∩ BulkRNABert; "
                       "event = 1 - censorship, SAME as GBMLGG).")
        if protocol == "5split":
            splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper "
                           "Table 3 protocol; 360 train + 90 test per split, "
                           "stratified by event, patient-disjoint, seeds 0..4)")
        else:
            splits_text = "15-fold CV from master_splits.csv + splits_0..14.csv"
    elif dataset == "ucec":
        cohort_suffix = ""
        dataset_suffix = "ucec_5split"
        cohort_field = "UCEC"
        cohort_text = ("478-patient/478-slide UCEC working cohort (PF 480 minus "
                       "TCGA-AP-A0LQ, TCGA-EY-A1GJ; 1 slide/patient; "
                       "event = 1 - censorship per UCEC convention, SAME as "
                       "GBMLGG/LUAD; low event rate 15.7%).")
        splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper "
                       "Table 3 protocol; 382 train + 96 test per split, "
                       "stratified by event, patient-disjoint, seeds 0..4)")
    elif dataset == "brca":
        cohort_suffix = ""
        dataset_suffix = "brca_5split"
        cohort_field = "BRCA"
        cohort_text = ("950-patient/950-slide BRCA working cohort (PF 957 minus 7 "
                       "TCGA-OL-A5R*/A5S0 without UNI2-h .pt files; 1 slide/patient; "
                       "event = 1 - censorship per BRCA convention, verified vs "
                       "cBioPortal; SAME as GBMLGG/LUAD/UCEC; LOWEST event rate "
                       "in our 5-cohort sweep at 13.7%. UNI2-h features mean-pooled "
                       "from patch-level (n_patches, 1536) to slide-level (1536,).")
        splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper "
                       "Table 3 protocol; 760 train + 190 test per split, "
                       "stratified by event, patient-disjoint, seeds 0..4)")
    else:
        cohort_suffix = "_pf501" if cohort_size == 501 else ""
        dataset_suffix = "gbmlgg"
        cohort_field = "GBMLGG"
        cohort_text = ("501-patient/997-ROI PF-EXACT cohort (PF's 502 RNA-seq subset "
                       "intersected with our BulkRNABert NPZ minus TCGA-06-0221). "
                       "Direct apples-to-apples with PF trimodal paper 0.826."
                       if cohort_size == 501
                       else "592-patient/1159-ROI working cohort (PF cohort ∩ BulkRNABert).")
        splits_text = "15-fold MCCV from pnas_splits.csv"
    stage_tag = f"sl_4_{fusion_name}_bulkrnabert{cohort_suffix}_{dataset_suffix}_5seed"
    log_run(
        stage=stage_tag,
        image_encoder="UNI2-h-frozen-1536",
        gene_encoder ="BulkRNABert-frozen-256",
        fusion=fusion_name, mode="fused", cohort=cohort_field,
        n_params=n_params, flops=0,
        c_index_mean=grand_mean, c_index_std=grand_std,
        per_fold_c_indices=per_fold_means.tolist(),
        seed=-1,
        notes=(
            f"Sample-level Stage 4 -- {fusion_name} fusion of "
            f"UNI2-h (1536, per-ROI) x BulkRNABert (256, patient broadcast), dataset={dataset}. "
            f"{cohort_text} {splits_text} "
            f"expanded patient->ROI. 5 seeds {seeds} averaged. ROI-level "
            f"Cox NLL + c-index over predictions with broadcast labels. "
            f"ImageAdapter+GeneAdapter use SELU+AlphaDropout({ALPHA_DROP}); "
            f"Adam lr={LR}, wd={WEIGHT_DECAY}, {EPOCHS} epochs full-batch."
        ),
    )
    print(f"\n  logged stage={stage_tag}")
    print(f"  total wall time: {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fusion",
        choices=["concat", "kronecker", "lmf", "phm", "bkron", "phm_kron", "abkron", "slim_cpkf"],
        required=True,
        help="bkron = bottleneck full bilinear (cheap Kronecker). "
             "phm_kron = gated PHM (linear) + BKron (bilinear) hybrid. "
             "abkron = BKron + gene-anchor skip (architectural PAL-spirit). "
             "slim_cpkf = NEW: PHM-modulated BKron in the bottleneck "
             "(content-aware compositional hybrid, ~1.09M params)."
    )
    ap.add_argument("--dataset", choices=["gbmlgg", "kirc", "luad", "ucec", "brca"], default="gbmlgg",
                    help="which dataset to run on. gbmlgg uses --cohort arg; "
                         "kirc uses its 417-patient cohort, luad its 450-patient, "
                         "ucec its 478-patient, brca its 950-patient cohort. "
                         "All non-gbmlgg cohorts ignore --cohort. "
                         "KIRC uses event = censored; LUAD/UCEC/GBMLGG/BRCA use event = 1 - censored. "
                         "UCEC and BRCA are 5-split-only. "
                         "All handled inside the per-dataset loaders.")
    ap.add_argument("--cohort", type=int, choices=[592, 501], default=592,
                    help="GBMLGG cohort size (ignored when --dataset=kirc): "
                         "592 = our default (PF ∩ BulkRNABert); "
                         "501 = PF-EXACT RNA-seq cohort intersection "
                         "(direct match to PF trimodal paper number).")
    ap.add_argument("--seeds",  type=int, nargs="+", default=list(range(N_SEEDS)),
                    help="seeds to average over (default 0..4). Ignored under --protocol 5split.")
    ap.add_argument("--protocol", choices=["15fold", "5split"], default="15fold",
                    help="CV protocol. 15fold (default): standard k-fold CV. "
                         "5split: 5 × stratified 80/20 random splits "
                         "(BulkRNABert paper Table 3 protocol, LUAD/UCEC/BRCA). "
                         "UCEC and BRCA ONLY support 5split.")
    ap.add_argument("--smoke",  action="store_true")
    args = ap.parse_args()
    run_sweep(args.fusion, args.seeds, smoke=args.smoke,
              cohort_size=args.cohort, dataset=args.dataset, protocol=args.protocol)


if __name__ == "__main__":
    main()
