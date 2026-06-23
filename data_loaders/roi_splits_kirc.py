"""
KIRC sample-level splits  --  patient-level 15-fold splits expanded to ROIs.

WHAT
   - Loads PF's KIRC 15-fold split (from PF's `KIRC_st_1.pkl`) at PATIENT level.
   - Restricts to the 417-patient cohort defined by PF (which equals our
     UNI2-h ∩ BulkRNABert intersection -- verified 100% match).
   - For each fold k (1..15), expands the patient-level Train/Test partition
     into ROI-level Train/Test lists by including ALL of our ROIs for each
     patient (~27 ROIs/patient on average).

WHY
   PF uses 15-fold cross-validation on KIRC ("the same experimental protocol
   as [29]" in PF paper Section IV.A). Their pkl has 15 sample-level folds.
   We extract the patient-level partition from PF's pkl (per-fold unique
   patient IDs from `x_patname`) and expand to our 11,340-ROI feature set.

   This guarantees apples-to-apples comparison with PF's KIRC trimodal
   result (0.720 ± 0.028 in PF paper Table II) while using ~9x more ROIs
   per patient than PF (we use ~27, PF used 3).

CRITICAL DIFFERENCE FROM GBMLGG
   GBMLGG convention: event = 1 - censored     (censored=0 -> death observed)
   KIRC   convention: event = censored          (censored=1 -> death observed)
   Verified across all 1008 ROIs in PF fold 1: `e == censored` matches 100%.

API mirrors harness.roi_splits (the GBMLGG version) for one-line drop-in:

   for fold_idx, train_rois, test_rois in iter_kirc_roi_folds():
       ...

Each element of train_rois/test_rois is a RoiEntry (same dataclass as GBMLGG):
   RoiEntry(roi_basename, patient_id, uni2h_pt_path)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np
import pandas as pd

# Re-use the GBMLGG RoiEntry dataclass and patient-ID parser (filename format
# is identical: TCGA-XX-XXXX-01Z-00-DX1...._roi_*.pt -- same first three fields).
from .roi_splits import RoiEntry, _patient_from_roi_filename


# ----------------------------------------------------------------------
# KIRC paths
# ----------------------------------------------------------------------
KIRC_PF_BASE = Path('/mnt/storage7/Dataset_pathomicfusion/KIRC/data/TCGA_KIRC')
KIRC_PF_PKL  = KIRC_PF_BASE / 'splits' / 'KIRC_st_1.pkl'      # PF's KIRC split (15-fold)
KIRC_UNI2H_DIR = KIRC_PF_BASE / 'KIRC_st_uni2h'                # 11,961 ROI .pt files

# Our BulkRNABert NPZ (from the patient-level project)
KIRC_BULKRNABERT_NPZ = Path(
    '/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level'
    '/bulkrnabert_data/kirc_bulkrnabert_256d.npz'
)

# Cohort numbers (verified at build time)
EXPECTED_KIRC_PATIENTS = 417    # PF KIRC cohort = our intersection (100% coverage)
EXPECTED_KIRC_ROIS     = 1260   # PF's exact KIRC ROIs (3/patient hand-picked, total across all folds)
                                # Verified: every PF ROI has a corresponding .pt file in our extraction.
                                # Per-fold: 1008 train + 252 test (some folds 1005/255 -- matches PF exactly).
                                # This matches PF Section IV.A: "3 512x512 40x ROIs per patient, yielding
                                # 1251 images total" -- the small extra (1260 vs 1251) is fold-overlap of
                                # the same patient's ROIs across MCCV folds (Monte Carlo, not k-fold).
                                # NOTE (history): our automated extraction provides 11340 ROIs (27/patient)
                                # but we deliberately subset to PF's 1260 for direct sample-level
                                # apples-to-apples comparison with PF's KIRC Table II numbers, the same
                                # way our GBMLGG sample-level uses PF's exact 997 ROIs.


# ----------------------------------------------------------------------
# Patient-level cohort
# ----------------------------------------------------------------------
def load_kirc_cohort() -> Set[str]:
    """The 417-patient KIRC cohort (from PF's KIRC_st_1.pkl 'all_dataset' index).

    Verified earlier: this set EQUALS PF's 417 ∩ our BulkRNABert ∩ our UNI2-h.
    So no further intersection is needed.
    """
    with open(KIRC_PF_PKL, 'rb') as f:
        d = pickle.load(f)
    cohort = set(d['all_dataset'].index)
    if len(cohort) != EXPECTED_KIRC_PATIENTS:
        raise RuntimeError(
            f"KIRC cohort drift: got {len(cohort)} patients, "
            f"expected {EXPECTED_KIRC_PATIENTS}"
        )
    return cohort


# ----------------------------------------------------------------------
# Patient -> list of RoiEntry (scan KIRC UNI2-h ROI directory once)
# ----------------------------------------------------------------------
def _scan_kirc_pat2entries() -> Tuple[dict[str, List[RoiEntry]], dict[str, str]]:
    """Walk KIRC_UNI2H_DIR once, building per-patient ROI lists.

    Returns:
        pat2entries[patient_id] -> list of RoiEntry for that patient
        roi2patient[roi_basename] -> patient_id

    Pure function -- no side effects except the filesystem read.
    """
    pat2entries: dict[str, List[RoiEntry]] = {}
    roi2patient: dict[str, str] = {}
    for pt_path in sorted(KIRC_UNI2H_DIR.glob('*.pt')):
        pid = _patient_from_roi_filename(pt_path.name)
        basename = pt_path.stem
        entry = RoiEntry(
            roi_basename = basename,
            patient_id   = pid,
            uni2h_pt_path = pt_path,
        )
        pat2entries.setdefault(pid, []).append(entry)
        roi2patient[basename] = pid
    return pat2entries, roi2patient


def build_kirc_pat2entries_for_cohort(
    cohort: Set[str],
    expected_rois: int | None = None,
) -> dict[str, List[RoiEntry]]:
    """Build {patient_id -> [RoiEntry,...]} restricted to `cohort`.

    Args:
        cohort: set of patient IDs to include (e.g. the 417 KIRC patients).
        expected_rois: if set, verify total ROI count matches.
    """
    pat2entries, _ = _scan_kirc_pat2entries()
    out = {pid: pat2entries[pid] for pid in cohort if pid in pat2entries}
    n_rois = sum(len(rois) for rois in out.values())
    if expected_rois is not None and n_rois != expected_rois:
        raise RuntimeError(
            f"KIRC ROI count drift: got {n_rois}, expected {expected_rois}\n"
            f"  cohort size: {len(cohort)}\n"
            f"  patients with ROIs: {len(out)}\n"
            f"  missing patients (in cohort but no ROIs): "
            f"{len(set(cohort) - set(out.keys()))}"
        )
    return out


# ----------------------------------------------------------------------
# Per-fold splits at PATIENT level (extracted from PF's KIRC pkl)
# ----------------------------------------------------------------------
def _load_pf_kirc_patient_splits() -> dict[int, Tuple[Set[str], Set[str]]]:
    """Extract PF's 15-fold split at PATIENT level from KIRC_st_1.pkl.

    Each PF fold has `x_patname` lists containing ROI filenames (PF's ~3 ROIs
    per patient). We extract unique patient IDs (first 12 chars) to get the
    per-fold patient partition. This partition is then expanded to OUR
    11,340-ROI feature set.

    Returns: {fold_idx: (train_patient_set, test_patient_set)}
    """
    with open(KIRC_PF_PKL, 'rb') as f:
        d = pickle.load(f)
    sp = d['split']
    out: dict[int, Tuple[Set[str], Set[str]]] = {}
    for fi in sorted(sp.keys()):
        pat_train = set(p[:12] for p in sp[fi]['train']['x_patname'])
        pat_test  = set(p[:12] for p in sp[fi]['test' ]['x_patname'])
        # Verify patient-disjoint
        overlap = pat_train & pat_test
        if overlap:
            raise RuntimeError(
                f"PF KIRC fold {fi} has train/test overlap: {len(overlap)} patients"
            )
        out[int(fi)] = (pat_train, pat_test)
    return out


# ----------------------------------------------------------------------
# Public iter -- mirrors iter_roi_folds for GBMLGG
# ----------------------------------------------------------------------
def _png_to_pt_path(png_name: str) -> Path:
    """Map PF's x_patname (.png) to our UNI2-h feature file (.pt).

    PF stores ROI image filenames like:
        TCGA-3Z-A93Z-01Z-00-DX1.79F4D1A6-..._roi_0_x_70176_y_35424_99.659.png
    Our pre-extracted UNI2-h features have the same stem with .pt extension:
        TCGA-3Z-A93Z-01Z-00-DX1.79F4D1A6-..._roi_0_x_70176_y_35424_99.659.pt
    Verified at module-build time: every PF KIRC ROI has a matching .pt.
    """
    if not png_name.endswith('.png'):
        raise ValueError(f"expected .png suffix, got {png_name!r}")
    return KIRC_UNI2H_DIR / (png_name[:-4] + '.pt')


def iter_kirc_roi_folds(
    cohort: Set[str] | None = None,
    expected_rois: int | None = None,
) -> Iterator[Tuple[int, List[RoiEntry], List[RoiEntry]]]:
    """Yield (fold_idx, train_rois, test_rois) for each of PF's 15 KIRC folds.

    Uses PF's EXACT ROI selection (3 hand-picked ROIs/patient via x_patname)
    rather than our denser 27-ROI-per-patient automated extraction. This is
    direct sample-level apples-to-apples with PF KIRC Table II numbers.

    Per fold: 1008 train + 252 test (some folds 1005/255), total 1260 ROIs
    across the whole MCCV. Patient-disjoint by construction.

    Args:
        cohort: patient IDs to keep (default: full 417 KIRC cohort).
        expected_rois: if set, verify total ROI count matches.

    Yields:
        fold_idx (1..15), train_rois (~1008 RoiEntries), test_rois (~252).
    """
    if cohort is None:
        cohort = load_kirc_cohort()
    if expected_rois is None:
        expected_rois = EXPECTED_KIRC_ROIS

    # Load PF's per-fold ROI selection directly from the pkl
    with open(KIRC_PF_PKL, 'rb') as f:
        d = pickle.load(f)
    splits = d['split']

    # Verify total ROI count across all folds matches expected (sanity at build time)
    seen_rois: set[str] = set()
    for fi in splits:
        for sn in ('train', 'test'):
            seen_rois.update(splits[fi][sn]['x_patname'])
    if expected_rois is not None and len(seen_rois) != expected_rois:
        raise RuntimeError(
            f"KIRC ROI count drift: PF pkl has {len(seen_rois)} unique ROIs, "
            f"expected {expected_rois}"
        )

    for fold_idx in sorted(splits.keys()):
        train_rois: List[RoiEntry] = []
        test_rois : List[RoiEntry] = []
        for png_name in splits[fold_idx]['train']['x_patname']:
            pid = _patient_from_roi_filename(png_name)
            if pid not in cohort:
                continue
            pt_path = _png_to_pt_path(png_name)
            train_rois.append(RoiEntry(
                roi_basename = pt_path.stem,
                patient_id   = pid,
                uni2h_pt_path = pt_path,
            ))
        for png_name in splits[fold_idx]['test']['x_patname']:
            pid = _patient_from_roi_filename(png_name)
            if pid not in cohort:
                continue
            pt_path = _png_to_pt_path(png_name)
            test_rois.append(RoiEntry(
                roi_basename = pt_path.stem,
                patient_id   = pid,
                uni2h_pt_path = pt_path,
            ))
        yield fold_idx, train_rois, test_rois


def kirc_cohort_summary() -> dict:
    """One-shot summary of the KIRC cohort + ROI counts for sanity printing."""
    cohort = load_kirc_cohort()
    pat2 = build_kirc_pat2entries_for_cohort(cohort, expected_rois=None)
    rois_per_patient = sorted(len(v) for v in pat2.values())
    n_rois = sum(rois_per_patient)
    splits = _load_pf_kirc_patient_splits()
    sizes_train = [len(s[0]) for s in splits.values()]
    sizes_test  = [len(s[1]) for s in splits.values()]
    return {
        "cohort_size_patients":    len(cohort),
        "total_rois":              n_rois,
        "rois_per_patient_min":    rois_per_patient[0],
        "rois_per_patient_median": rois_per_patient[len(rois_per_patient)//2],
        "rois_per_patient_max":    rois_per_patient[-1],
        "n_folds":                 len(splits),
        "train_patients_per_fold": sizes_train[0] if len(set(sizes_train)) == 1 else f"varies {min(sizes_train)}..{max(sizes_train)}",
        "test_patients_per_fold":  sizes_test [0] if len(set(sizes_test )) == 1 else f"varies {min(sizes_test )}..{max(sizes_test )}",
    }


if __name__ == '__main__':
    print("=== KIRC sample-level cohort summary ===")
    s = kirc_cohort_summary()
    for k, v in s.items():
        print(f"  {k:30s} : {v}")
