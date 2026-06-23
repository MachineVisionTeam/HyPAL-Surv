"""
Stage 5 sample-level MoBalD on (UNI2-h x BulkRNABert) with configurable fusion.

WHAT IS THIS
   Classical MoBalD as defined in our patient-level project:
     - PAL  (Peer-Assistance Learning): gene-teacher pairwise ranking distillation
     - GradMod (Path-GPTOmic): per-branch gradient scaling based on aux Cox losses
   Applied at SAMPLE (ROI) level on the 592-patient / 1159-ROI working cohort,
   with the fusion operator selectable from the FUSION_REGISTRY.

   This file is the SAMPLE-LEVEL PORT of:
     pathgptomic_bulkrnabert_patient_level/stage5_fpal/run_stage5_distill.py
   Same losses, same hyperparameters, same training algorithm. The ONLY
   differences are:
     (a) per-ROI iteration via harness.roi_loader.assemble_roi_batch instead
         of per-patient feature arrays
     (b) fusion is configurable via --fusion {phm, bkron, ...} (default bkron)

USAGE
   # PAL only, no GradMod  (the original PAL paper convention)
   python -m stage5_mobald_sample.run_stage5_mobald --teacher gene --no_gradmod --fusion bkron
   # PAL + GradMod = MoBalD  (the headline)
   python -m stage5_mobald_sample.run_stage5_mobald --teacher gene             --fusion bkron
   # GradMod alone (no distillation; for ablation)
   python -m stage5_mobald_sample.run_stage5_mobald --teacher gene --no_pal     --fusion bkron

STAGE TAGS
   sl_5_pal_<fusion>_bulk_gbmlgg_5seed              (teacher=gene, no_gradmod)
   sl_5_gradmod_<fusion>_bulk_gbmlgg_5seed          (teacher=gene, no_pal, gradmod on)
   sl_5_mobald_<fusion>_bulk_gbmlgg_5seed           (teacher=gene, both on)  <- HEADLINE
   sl_5_fpal_<fusion>_bulk_gbmlgg_5seed             (teacher=fused, both on; for ablation)
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ))

from harness.roi_splits        import (
    iter_roi_folds, load_working_cohort, load_pf501_cohort,
    EXPECTED_WORKING_ROIS, EXPECTED_PF501_ROIS,
)
from harness.roi_loader        import (
    load_bulkrnabert_patient_dict, load_patient_labels, assemble_roi_batch,
)
# KIRC support. KIRC uses event = censored convention; that's handled inside
# load_kirc_patient_labels(), so the assembler (assemble_kirc_roi_batch) has
# the same call shape as assemble_roi_batch.
from harness.roi_splits_kirc   import (
    iter_kirc_roi_folds, load_kirc_cohort, EXPECTED_KIRC_ROIS,
)
from harness.roi_loader_kirc   import (
    load_kirc_bulkrnabert_patient_dict, load_kirc_patient_labels,
    assemble_kirc_roi_batch,
)
# LUAD loaders (sample-level). event = 1 - censorship (SAME as GBMLGG).
from harness.roi_splits_luad   import (
    iter_luad_roi_folds, iter_luad_5split_folds,
    load_luad_cohort, EXPECTED_LUAD_SLIDES,
)
from harness.roi_loader_luad   import (
    load_luad_bulkrnabert_patient_dict, load_luad_patient_labels,
    assemble_luad_roi_batch,
)
# UCEC loaders (sample-level). event = 1 - censorship (SAME as GBMLGG/LUAD).
# 478-patient working cohort, 1 slide/patient, 5-split only.
from harness.roi_splits_ucec   import (
    iter_ucec_5split_folds, load_ucec_cohort, EXPECTED_UCEC_SLIDES,
)
from harness.roi_loader_ucec   import (
    load_ucec_bulkrnabert_patient_dict, load_ucec_patient_labels,
    assemble_ucec_roi_batch,
)
# BRCA loaders (sample-level). event = 1 - censorship (SAME as GBMLGG/LUAD/UCEC).
# 950-patient working cohort, 1 slide/patient, 5-split only.
from harness.roi_splits_brca   import (
    iter_brca_5split_folds, load_brca_cohort, EXPECTED_BRCA_SLIDES,
)
from harness.roi_loader_brca   import (
    load_brca_bulkrnabert_patient_dict, load_brca_patient_labels,
    assemble_brca_roi_batch,
)
from harness.eval              import cox_partial_likelihood_loss, cindex_lifeline
from harness.log               import log_run
from stage4_fusion_sample.fusion_ops import build_fusion


# ============================================================================
# Hyperparameters -- verbatim from patient-level stage5_fpal/run_stage5_distill.py
# ============================================================================
IMAGE_DIM      = 1536
BULKRNA_DIM    = 256
FUSION_DIM     = 256
ALPHA_DROP     = 0.1
LR             = 1e-4
WEIGHT_DECAY   = 0.0
EPOCHS         = 200
GRAD_CLIP      = 1.0
PHM_N          = 4
BKRON_BOTTLE   = 64
GRADMOD_EPS    = 1e-6
LAMBDA_AUX     = 0.5   # weight on aux Cox losses (so aux heads train)
LAMBDA_DIST    = 0.3   # weight on PAL distillation losses
N_SEEDS        = 5
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# PAL ranking distillation -- verbatim from patient-level
# ============================================================================
def ranking_distillation_loss(student: torch.Tensor,
                              teacher: torch.Tensor,
                              time:    torch.Tensor,
                              event:   torch.Tensor) -> torch.Tensor:
    """MSE on pairwise hazard differences over uncensored-first-death pairs.

    For every pair (i, j) with event[i] == 1 AND time[i] < time[j]:
        s_diff[i,j] = student[i] - student[j]
        t_diff[i,j] = teacher[i] - teacher[j]     # teacher is detached upstream
        contribution = (s_diff - t_diff) ** 2
    Loss = sum of valid contributions / n_valid_pairs.
    """
    s_diff = student.unsqueeze(1) - student.unsqueeze(0)   # (B, B)
    t_diff = teacher.unsqueeze(1) - teacher.unsqueeze(0)   # (B, B)
    event_mask = (event.unsqueeze(1) > 0.5)                 # (B, 1) -> (B, B)
    time_mask  = (time.unsqueeze(1) < time.unsqueeze(0))    # (B, B)
    valid_mask = event_mask & time_mask                      # (B, B) bool
    n_valid = valid_mask.sum().float()
    sq_diff = (s_diff - t_diff) ** 2
    return (sq_diff * valid_mask.float()).sum() / n_valid.clamp(min=1.0)


# ============================================================================
# Adapter sub-modules -- same SELU/AlphaDropout as Stage 4 sample-level
# ============================================================================
def _selu_linear(in_dim: int, out_dim: int) -> nn.Module:
    lin = nn.Linear(in_dim, out_dim)
    nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
    nn.init.zeros_(lin.bias)
    return nn.Sequential(lin, nn.SELU(inplace=True), nn.AlphaDropout(ALPHA_DROP))


class ImageAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _selu_linear(IMAGE_DIM, FUSION_DIM)
    def forward(self, x): return self.net(x)


class BulkRNABertAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = _selu_linear(BULKRNA_DIM, FUSION_DIM)
    def forward(self, x): return self.net(x)


# ============================================================================
# FusionDistillModel -- like Stage 4 FusionModel but with two auxiliary heads
# ============================================================================
class FusionDistillModel(nn.Module):
    """Configurable fusion + main cox_head + two auxiliary heads (one per branch).

    forward(x_image, x_gene) -> (hazard_main, aux_g, aux_p)  each shape (B,)
       hazard_main : cox_head(fused)        -- main hazard from fused features
       aux_g       : aux_gene_head(z_gene)  -- gene-only hazard (used as PAL teacher
                                                 OR as branch-loss numerator for GradMod)
       aux_p       : aux_image_head(z_img)  -- image-only hazard (symmetric)
    """
    def __init__(self, fusion_name: str = "bkron"):
        super().__init__()
        self.fusion_name   = fusion_name
        self.image_adapter = ImageAdapter()
        self.gene_encoder  = BulkRNABertAdapter()

        # Per-fusion kwargs (must match Stage 4 conventions)
        if fusion_name == "phm":
            kw = {"n": PHM_N}
        elif fusion_name == "bkron":
            kw = {"bottle": BKRON_BOTTLE}
        elif fusion_name == "phm_kron":
            kw = {"n": PHM_N, "bottle": BKRON_BOTTLE}
        elif fusion_name == "abkron":
            kw = {"bottle": BKRON_BOTTLE}
        elif fusion_name == "slim_cpkf":
            kw = {"n": PHM_N, "bottle": BKRON_BOTTLE}
        else:
            kw = {}
        self.fusion = build_fusion(
            fusion_name,
            dim_a=FUSION_DIM, dim_b=FUSION_DIM, fused_dim=FUSION_DIM,
            **kw,
        )

        # Main + auxiliary heads
        self.cox_head       = nn.Linear(FUSION_DIM, 1)
        self.aux_image_head = nn.Linear(FUSION_DIM, 1)
        self.aux_gene_head  = nn.Linear(FUSION_DIM, 1)
        for lin in (self.cox_head, self.aux_image_head, self.aux_gene_head):
            nn.init.kaiming_normal_(lin.weight, nonlinearity="linear")
            nn.init.zeros_(lin.bias)

    def forward(self, x_image, x_gene):
        z_image = self.image_adapter(x_image)                  # (B, 256)
        z_gene  = self.gene_encoder(x_gene)                    # (B, 256)
        fused   = self.fusion(z_image, z_gene)                  # (B, 256)
        hazard_main = self.cox_head(fused).squeeze(-1)          # (B,)
        aux_p = self.aux_image_head(z_image).squeeze(-1)        # (B,)  image-only
        aux_g = self.aux_gene_head(z_gene).squeeze(-1)          # (B,)  gene-only
        return hazard_main, aux_g, aux_p


def split_param_ids(model: FusionDistillModel):
    """Returns (gene_ids, image_ids) sets of id(p) for each branch.
    Fusion and main cox_head are unmodulated 'glue' params (no scaling).
    """
    gene_ids  = {id(p) for p in model.gene_encoder.parameters()}
    gene_ids |= {id(p) for p in model.aux_gene_head.parameters()}
    image_ids  = {id(p) for p in model.image_adapter.parameters()}
    image_ids |= {id(p) for p in model.aux_image_head.parameters()}
    return gene_ids, image_ids


# ============================================================================
# Train one (fold, seed) at SAMPLE (ROI) level
# ============================================================================
def train_one_fold(fusion_name: str, teacher_mode: str, seed: int,
                   X_img_train: np.ndarray, X_gene_train: np.ndarray,
                   e_train: np.ndarray, t_train: np.ndarray,
                   X_img_test: np.ndarray, X_gene_test: np.ndarray,
                   e_test: np.ndarray, t_test: np.ndarray,
                   no_gradmod: bool = False, no_pal: bool = False,
                   target: str = "aux_p"):
    """Sample-level MoBalD training on one fold.

    Returns (c_index, hazards_test_np) so callers can optionally persist
    the per-ROI predictions for downstream visualizations (KM curves etc.).
    """
    # 1) Z-score per branch on train; apply same scaler to test
    sc_img  = StandardScaler(); sc_gene = StandardScaler()
    Xi_tr = sc_img .fit_transform(X_img_train ).astype(np.float32)
    Xi_te = sc_img .transform    (X_img_test  ).astype(np.float32)
    Xg_tr = sc_gene.fit_transform(X_gene_train).astype(np.float32)
    Xg_te = sc_gene.transform    (X_gene_test ).astype(np.float32)

    torch.manual_seed(seed); np.random.seed(seed)
    Xi_tr_t = torch.from_numpy(Xi_tr).float().to(DEVICE)
    Xi_te_t = torch.from_numpy(Xi_te).float().to(DEVICE)
    Xg_tr_t = torch.from_numpy(Xg_tr).float().to(DEVICE)
    Xg_te_t = torch.from_numpy(Xg_te).float().to(DEVICE)
    e_tr_t  = torch.from_numpy(e_train.astype(np.float32)).to(DEVICE)
    t_tr_t  = torch.from_numpy(t_train.astype(np.float32)).to(DEVICE)

    model = FusionDistillModel(fusion_name=fusion_name).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    gene_ids, image_ids = split_param_ids(model)

    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        hazard_main, aux_g, aux_p = model(Xi_tr_t, Xg_tr_t)

        # 1) Main + auxiliary Cox losses
        L_main  = cox_partial_likelihood_loss(hazard_main, t_tr_t, e_tr_t)
        L_aux_g = cox_partial_likelihood_loss(aux_g,       t_tr_t, e_tr_t)
        L_aux_p = cox_partial_likelihood_loss(aux_p,       t_tr_t, e_tr_t)

        # 2) PAL distillation (teacher detached). --no_pal zeros these out.
        #    Three possible distillation targets:
        #      L_dist_p     -> distill INTO aux_p (image-only head) [CLASSICAL PAL]
        #      L_dist_g     -> distill INTO aux_g (gene-only head)
        #      L_dist_main  -> distill INTO hazard_main (fused output) [INVERSE PAL]
        L_dist_p    = torch.zeros((), device=DEVICE, dtype=L_main.dtype)
        L_dist_g    = torch.zeros((), device=DEVICE, dtype=L_main.dtype)
        L_dist_main = torch.zeros((), device=DEVICE, dtype=L_main.dtype)

        if not no_pal:
            if teacher_mode == "gene":
                teacher = aux_g.detach()
                if target == "aux_p":      # CLASSICAL PAL (default, BMLSurv style)
                    L_dist_p = ranking_distillation_loss(aux_p, teacher, t_tr_t, e_tr_t)
                elif target == "main":     # INVERSE PAL (deep-research recipe D)
                    L_dist_main = ranking_distillation_loss(hazard_main, teacher, t_tr_t, e_tr_t)
                else:
                    raise ValueError(f"target must be 'aux_p' or 'main', got {target!r}")
            elif teacher_mode == "fused":
                teacher = hazard_main.detach()
                L_dist_p = ranking_distillation_loss(aux_p, teacher, t_tr_t, e_tr_t)
                L_dist_g = ranking_distillation_loss(aux_g, teacher, t_tr_t, e_tr_t)
            else:
                raise ValueError(f"teacher_mode must be 'gene' or 'fused', got {teacher_mode!r}")

        # 3) Total loss (all three L_dist_* terms summed; only one is non-zero per mode)
        L_total = (L_main
                   + LAMBDA_AUX  * (L_aux_g + L_aux_p)
                   + LAMBDA_DIST * (L_dist_g + L_dist_p + L_dist_main))
        L_total.backward()

        # 4) GradMod: shrink the gradient of the DOMINANT branch (the one with
        #    LOWER aux Cox loss). --no_gradmod skips this.
        with torch.no_grad():
            ag = float(L_aux_g.item()); ap = float(L_aux_p.item())
            rho_G = ap / max(ag, GRADMOD_EPS)   # >1 when gene dominates
            rho_P = ag / max(ap, GRADMOD_EPS)   # >1 when image dominates
            if no_gradmod:
                mod_G = 1.0; mod_P = 1.0
            else:
                mod_G = min(1.0 - math.tanh(rho_G - 1.0), 1.0) if rho_G > 1.0 else 1.0
                mod_P = min(1.0 - math.tanh(rho_P - 1.0), 1.0) if rho_P > 1.0 else 1.0
                mod_G = max(0.0, min(1.0, mod_G))
                mod_P = max(0.0, min(1.0, mod_P))

        # 5) Per-branch gradient scaling
        for p in model.parameters():
            if p.grad is None: continue
            pid = id(p)
            if pid in gene_ids:
                p.grad.mul_(mod_G)
            elif pid in image_ids:
                p.grad.mul_(mod_P)
            # else: fusion + main cox_head -- unmodulated 'glue' params

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)
        opt.step()

    # Eval: main fused hazard on test
    model.eval()
    with torch.no_grad():
        hazards_main, _, _ = model(Xi_te_t, Xg_te_t)
    hazards_np = hazards_main.cpu().numpy()
    c = cindex_lifeline(hazards_np, e_test, t_test)
    return c, hazards_np


# ============================================================================
# Sweep
# ============================================================================
def run_sweep(fusion_name: str, teacher_mode: str, seeds: list[int],
              no_gradmod: bool, no_pal: bool, smoke: bool = False,
              cohort_size: int = 592, target: str = "aux_p",
              dataset: str = "gbmlgg", protocol: str = "15fold",
              save_predictions: str | None = None) -> None:
    assert dataset in {"gbmlgg", "kirc", "luad", "ucec", "brca"}, dataset
    assert protocol in {"15fold", "5split"}, protocol
    if protocol == "5split" and dataset not in {"luad", "ucec", "brca"}:
        raise ValueError(
            f"--protocol 5split currently only supported for LUAD, UCEC, BRCA. "
            f"Got dataset={dataset}."
        )
    if dataset == "ucec" and protocol != "5split":
        raise ValueError(
            f"UCEC only supports --protocol 5split. Got protocol={protocol}."
        )
    if dataset == "brca" and protocol != "5split":
        raise ValueError(
            f"BRCA only supports --protocol 5split. Got protocol={protocol}."
        )
    if dataset == "kirc":
        cohort_size = 417  # KIRC has exactly one cohort
    elif dataset == "luad":
        cohort_size = 450  # LUAD has exactly one cohort (450 patients × 1 slide)
    elif dataset == "ucec":
        cohort_size = 478  # UCEC working cohort (PF 480 minus 2 without BulkRNABert)
    elif dataset == "brca":
        cohort_size = 950  # BRCA working cohort (PF 957 minus 7 without UNI2-h)
    if no_pal and no_gradmod:
        raise SystemExit("--no_pal AND --no_gradmod together means no MoBalD at all; "
                         "use Stage 4 (run_stage4_sample.py) instead.")
    # Decide a clean tag for this combination
    if no_pal:
        ablation = "gradmod"                        # PAL off, GradMod on
    elif no_gradmod:
        if target == "main":
            ablation = "invpal"                     # Inverse PAL alone (gene -> fused main output)
        else:
            ablation = "pal"                        # Classical PAL alone (gene -> aux_p)
    elif teacher_mode == "fused":
        ablation = "fpal"                           # F-PAL + GradMod
    else:
        if target == "main":
            ablation = "invmobald"                  # Inverse PAL + GradMod
        else:
            ablation = "mobald"                     # Classical PAL + GradMod (HEADLINE)

    t0 = time.time()
    print(f"=== Stage 5 sample-level: {ablation} on {fusion_name} x BulkRNABert ===")
    print(f"  device      : {DEVICE}")
    print(f"  fusion      : {fusion_name}")
    print(f"  teacher     : {teacher_mode}     target: {target}")
    print(f"  no_gradmod  : {no_gradmod}      no_pal: {no_pal}")
    print(f"  optim       : Adam lr={LR}, wd={WEIGHT_DECAY}, full-batch, {EPOCHS} epochs")
    print(f"  loss        : Cox + {LAMBDA_AUX}*(Cox_aux_g + Cox_aux_p) "
          f"+ {LAMBDA_DIST}*(L_dist_g + L_dist_p)")
    print(f"  seeds       : {seeds}")

    # Pre-load shared dicts.
    # Dataset dispatch:
    #   KIRC  -> KIRC loaders (event = censored convention, 417 cohort, 1260 ROIs)
    #   GBMLGG -> standard loaders (event = 1 - censored, 592/501 cohorts)
    print("\nLoading shared dicts ...")
    if dataset == "kirc":
        gene_dict = load_kirc_bulkrnabert_patient_dict()
        labels    = load_kirc_patient_labels()
        cohort        = load_kirc_cohort()
        expected_rois = EXPECTED_KIRC_ROIS    # 1260
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
        else:  # 501 PF-exact
            cohort        = load_pf501_cohort()
            expected_rois = EXPECTED_PF501_ROIS      # 997
        assembler     = assemble_roi_batch
        print(f"  gene_dict     : {len(gene_dict)} patients, dim=256")
        print(f"  labels        : {len(labels)} patients")
        print(f"  cohort        : {len(cohort)} patients (expected {cohort_size}), "
              f"{expected_rois} ROIs expected")

    # Per-fold iterator dispatcher (dataset + protocol aware)
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

    # Helper to persist per-test-ROI predictions for downstream KM plotting
    def _save_predictions(test_rois, hazards_np, e_te, t_te, fold_idx, seed):
        if save_predictions is None:
            return
        import os
        os.makedirs(save_predictions, exist_ok=True)
        out_path = os.path.join(
            save_predictions,
            f"{dataset}_fold{fold_idx}_seed{seed}.csv",
        )
        import pandas as pd
        pd.DataFrame({
            "patient_id": [r.patient_id   for r in test_rois],
            "roi_id"    : [r.roi_basename for r in test_rois],
            "hazard"    : hazards_np,
            "time"      : t_te,
            "event"     : e_te,
            "fold"      : fold_idx,
            "seed"      : seed,
        }).to_csv(out_path, index=False)

    if smoke:
        print("\nSMOKE: fold 1 only, seed 0 only")
        fold_iter = _fold_iter()
        fold_idx, train_rois, test_rois = next(fold_iter)
        Xi_tr, Xg_tr, e_tr, t_tr, _ = assembler(train_rois, gene_dict, labels,
                                                  expected_gene_dim=256)
        Xi_te, Xg_te, e_te, t_te, _ = assembler(test_rois,  gene_dict, labels,
                                                  expected_gene_dim=256)
        print(f"  fold {fold_idx} train: Xi={Xi_tr.shape}, Xg={Xg_tr.shape}, events={int(e_tr.sum())}/{len(e_tr)}")
        print(f"  fold {fold_idx} test : Xi={Xi_te.shape}, Xg={Xg_te.shape}, events={int(e_te.sum())}/{len(e_te)}")
        ts = time.time()
        c, hz = train_one_fold(fusion_name, teacher_mode, 0,
                               Xi_tr, Xg_tr, e_tr, t_tr,
                               Xi_te, Xg_te, e_te, t_te,
                               no_gradmod=no_gradmod, no_pal=no_pal, target=target)
        _save_predictions(test_rois, hz, e_te, t_te, fold_idx, 0)
        n_params = sum(p.numel() for p in FusionDistillModel(fusion_name).parameters()
                       if p.requires_grad)
        print(f"  fold {fold_idx} c-index (seed 0): {c:.4f}  ({time.time()-ts:.1f}s)")
        print(f"  model n_params: {n_params:,}")
        if save_predictions is not None:
            print(f"  predictions saved -> {save_predictions}")
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
            Xi_tr, Xg_tr, e_tr, t_tr, _ = assembler(train_rois, gene_dict, labels,
                                                      expected_gene_dim=256)
            Xi_te, Xg_te, e_te, t_te, _ = assembler(test_rois,  gene_dict, labels,
                                                      expected_gene_dim=256)
            c, hz = train_one_fold(fusion_name, teacher_mode, seed,
                                   Xi_tr, Xg_tr, e_tr, t_tr,
                                   Xi_te, Xg_te, e_te, t_te,
                                   no_gradmod=no_gradmod, no_pal=no_pal, target=target)
            _save_predictions(test_rois, hz, e_te, t_te, split_idx, seed)
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
                Xi_tr, Xg_tr, e_tr, t_tr, _ = assembler(train_rois, gene_dict, labels,
                                                          expected_gene_dim=256)
                Xi_te, Xg_te, e_te, t_te, _ = assembler(test_rois,  gene_dict, labels,
                                                          expected_gene_dim=256)
                c, hz = train_one_fold(fusion_name, teacher_mode, seed,
                                       Xi_tr, Xg_tr, e_tr, t_tr,
                                       Xi_te, Xg_te, e_te, t_te,
                                       no_gradmod=no_gradmod, no_pal=no_pal, target=target)
                _save_predictions(test_rois, hz, e_te, t_te, fold_idx, seed)
                all_c[si, fi] = c
            print(f"  seed {seed}: mean = {all_c[si].mean():.4f}  "
                  f"(min={all_c[si].min():.4f}, max={all_c[si].max():.4f}, "
                  f"time={time.time()-ts:.0f}s)")

        per_fold_means = all_c.mean(axis=0)
        grand_mean = float(per_fold_means.mean())
        grand_std  = float(per_fold_means.std())
    print(f"\n=== {ablation} on {fusion_name} x BulkRNABert (dataset={dataset}): 5-seed averaged ===")
    print(f"  GRAND mean +/- fold std: {grand_mean:.4f} +/- {grand_std:.4f}")
    print(f"  per-fold means          : {per_fold_means.round(4).tolist()}")
    print(f"  comparators:")
    if dataset == "kirc":
        print(f"    KIRC Stage 4 BKron baseline (this harness) : 0.7126 +/- 0.0432")
        print(f"    KIRC Stage 4 HGBF baseline            : 0.7041 +/- 0.0458")
        print(f"    KIRC BulkRNABert alone                     : 0.6794 +/- 0.0351")
        print(f"    KIRC UNI2-h alone                          : 0.6578 +/- 0.0363")
        print(f"    PF KIRC trimodal (our replication)         : 0.7184 +/- 0.0513  <-- main anchor")
        print(f"    PF KIRC pathomic (our replication)         : 0.7049 +/- 0.0568")
    elif dataset == "luad":
        print(f"    BulkRNABert paper Table 3 (LUAD)           : 0.648 +/- 0.057")
        print(f"    patient-level BulkRNABert (LUAD)           : 0.6275 +/- 0.109")
        print(f"    patient-level UNI2-h alone (LUAD)          : 0.5694 +/- 0.093")
        print(f"    patient-level Kronecker fusion (LUAD)      : 0.6055 +/- 0.100")
    elif dataset == "ucec":
        print(f"    BulkRNABert paper Table 3 (UCEC)           : 0.703 +/- 0.040  <-- their strongest cohort")
    elif dataset == "brca":
        print(f"    MCAT paper Table 1   (BRCA)                : 0.580 +/- 0.069")
        print(f"    MOTCat paper Table 1 (BRCA)                : 0.673 +/- 0.006")
        print(f"    CustOmics 2023 (BRCA)                      : ~0.65")
    else:
        print(f"    Stage 4 baseline (BKron x bulk)    : 0.8035 +/- 0.0462")
        print(f"    Stage 4 baseline (PHM x bulk)      : 0.7893 +/- 0.0382")
        print(f"    uni_anchor (BulkRNABert alone)     : 0.8139 +/- 0.0421")
        print(f"    PF trimodal (our replication)      : 0.8174 +/- 0.0717")
        print(f"    Path-GPTOmic headline (paper)      : 0.848  +/- 0.014")

    # Log: dataset-aware tag + cohort field
    if dataset == "kirc":
        cohort_suffix = ""
        dataset_suffix = "kirc"
        cohort_field = "KIRC"
        cohort_text = ("417-patient/1260-ROI PF KIRC PF-EXACT cohort (sample-level "
                       "matches PF Table II; event = censored convention).")
        splits_text = "15-fold MCCV from PF KIRC_st_1.pkl"
    elif dataset == "luad":
        cohort_suffix = ""
        dataset_suffix = "luad_5split" if protocol == "5split" else "luad"
        cohort_field = "LUAD"
        cohort_text = ("450-patient/450-slide PF LUAD cohort (1 slide/patient; "
                       "100% intersection with UNI2-h + BulkRNABert; "
                       "event = 1 - censorship per LUAD convention, same as GBMLGG).")
        if protocol == "5split":
            splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper "
                           "Table 3 protocol; 360 train + 90 test, stratified by event, "
                           "patient-disjoint, seeds 0..4)")
        else:
            splits_text = "15-fold CV from master_splits.csv"
    elif dataset == "ucec":
        cohort_suffix = ""
        dataset_suffix = "ucec_5split"
        cohort_field = "UCEC"
        cohort_text = ("478-patient/478-slide UCEC working cohort (PF 480 minus "
                       "TCGA-AP-A0LQ, TCGA-EY-A1GJ; 1 slide/patient; "
                       "event = 1 - censorship, SAME as GBMLGG/LUAD; "
                       "low event rate 15.7%).")
        splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper "
                       "Table 3 protocol; 382 train + 96 test, stratified by event, "
                       "patient-disjoint, seeds 0..4)")
    elif dataset == "brca":
        cohort_suffix = ""
        dataset_suffix = "brca_5split"
        cohort_field = "BRCA"
        cohort_text = ("950-patient/950-slide BRCA working cohort (PF 957 minus 7 "
                       "TCGA-OL-A5R*/A5S0 without UNI2-h .pt; 1 slide/patient; "
                       "event = 1 - censorship per BRCA convention, verified vs "
                       "cBioPortal; SAME as GBMLGG/LUAD/UCEC; LOWEST event rate "
                       "in sweep at 13.7%; UNI2-h features patch-pooled.")
        splits_text = ("5 × stratified 80/20 random splits (BulkRNABert paper "
                       "Table 3 protocol; 760 train + 190 test, stratified by event, "
                       "patient-disjoint, seeds 0..4)")
    else:
        cohort_suffix = "_pf501" if cohort_size == 501 else ""
        dataset_suffix = "gbmlgg"
        cohort_field = "GBMLGG"
        cohort_text = ("501-patient/997-ROI PF-EXACT cohort (intersection with PF's 502)"
                       if cohort_size == 501
                       else "592-patient/1159-ROI working cohort")
        splits_text = "15-fold MCCV from pnas_splits.csv"
    stage_tag = f"sl_5_{ablation}_{fusion_name}_bulk{cohort_suffix}_{dataset_suffix}_5seed"
    n_params = sum(p.numel() for p in FusionDistillModel(fusion_name).parameters()
                   if p.requires_grad)
    log_run(
        stage=stage_tag,
        image_encoder="UNI2-h-frozen-1536",
        gene_encoder ="BulkRNABert-frozen-256",
        fusion=fusion_name,
        mode=ablation,
        cohort=cohort_field,
        n_params=n_params, flops=0,
        c_index_mean=grand_mean, c_index_std=grand_std,
        per_fold_c_indices=per_fold_means.tolist(),
        seed=-1,
        notes=(
            f"Sample-level Stage 5 {ablation} on {fusion_name} x BulkRNABert, dataset={dataset}. "
            f"{cohort_text} {splits_text}. "
            f"5 seeds {seeds} averaged. Cox NLL + Aux Cox losses (LAMBDA_AUX={LAMBDA_AUX}) "
            f"+ PAL distillation (LAMBDA_DIST={LAMBDA_DIST}; rank-pair MSE, teacher={teacher_mode}, target={target}). "
            f"GradMod {'OFF' if no_gradmod else 'ON'} (tanh-modulated per-branch grad scaling). "
            f"PAL {'OFF' if no_pal else 'ON'}. Adam lr={LR} wd={WEIGHT_DECAY}, "
            f"{EPOCHS} epochs full-batch."
        ),
    )
    print(f"\n  logged stage={stage_tag}")
    print(f"  total wall time: {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion", choices=["phm", "bkron", "phm_kron", "abkron", "concat", "kronecker", "lmf", "slim_cpkf"],
                    default="bkron", help="fusion operator (default bkron)")
    ap.add_argument("--teacher", choices=["gene", "fused"], default="gene",
                    help="PAL teacher: 'gene' (gene aux head is teacher) "
                         "or 'fused' (F-PAL ablation, fused output is teacher)")
    ap.add_argument("--target", choices=["aux_p", "main"], default="aux_p",
                    help="PAL distillation target when teacher='gene': "
                         "'aux_p' = CLASSICAL PAL (BMLSurv: gene -> aux image head); "
                         "'main'  = INVERSE PAL (deep-research recipe D: gene -> fused output)")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(N_SEEDS)))
    ap.add_argument("--no_gradmod", action="store_true",
                    help="disable GradMod (PAL alone if --no_pal not set)")
    ap.add_argument("--no_pal", action="store_true",
                    help="disable PAL distillation (GradMod alone if --no_gradmod not set)")
    ap.add_argument("--smoke", action="store_true",
                    help="fold 1 + seed 0 only, no logging")
    ap.add_argument("--cohort", type=int, choices=[592, 501], default=592,
                    help="GBMLGG cohort size (ignored when --dataset=kirc): "
                         "592 = default (PF ∩ BulkRNABert); "
                         "501 = PF-EXACT RNA-seq cohort intersection")
    ap.add_argument("--dataset", choices=["gbmlgg", "kirc", "luad", "ucec", "brca"], default="gbmlgg",
                    help="which dataset to run on. gbmlgg uses --cohort; "
                         "kirc uses its 417-patient PF-EXACT 1260-ROI cohort; "
                         "luad its 450-patient 450-slide; "
                         "ucec its 478-patient 478-slide; "
                         "brca its 950-patient 950-slide cohort. "
                         "KIRC: event = censored. LUAD/UCEC/GBMLGG/BRCA: event = 1 - censored. "
                         "UCEC and BRCA are 5-split-only. "
                         "All handled in per-dataset loaders.")
    ap.add_argument("--protocol", choices=["15fold", "5split"], default="15fold",
                    help="CV protocol. 15fold (default): standard k-fold CV. "
                         "5split: 5 × stratified 80/20 random splits "
                         "(BulkRNABert paper Table 3 protocol, LUAD/UCEC/BRCA). "
                         "UCEC and BRCA ONLY support 5split.")
    ap.add_argument("--save_predictions", type=str, default=None,
                    help="If set, directory to dump per-test-ROI predictions as "
                         "<dataset>_fold{fold}_seed{seed}.csv (one row per ROI "
                         "with columns patient_id, roi_id, hazard, time, event). "
                         "Used downstream by stage6_visualization for KM plots.")
    args = ap.parse_args()
    run_sweep(args.fusion, args.teacher, args.seeds,
              no_gradmod=args.no_gradmod, no_pal=args.no_pal, smoke=args.smoke,
              cohort_size=args.cohort, target=args.target, dataset=args.dataset,
              protocol=args.protocol, save_predictions=args.save_predictions)


if __name__ == "__main__":
    main()
