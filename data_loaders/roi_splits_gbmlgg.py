"""
GBMLGG sample-level splits -- patient-level 15-fold Monte Carlo CV expanded to ROIs.

WHAT
  - Loads PF's pnas_splits.csv (the canonical Mobadersany 2018 splits that PF and
    Path-GPTOmic both use for 15-fold MCCV on GBMLGG).
  - Restricts to the 592-patient working cohort (= PF ∩ BulkRNABert).
  - For each fold k (1..15), expands the patient-level Train/Test partition into
    ROI-level Train/Test lists by including ALL ROIs of each patient.

WHY
  - PF (2020) and Path-GPTOmic (2024) both train and evaluate at ROI granularity
    on these exact splits. Matching their split definitions guarantees apples-to-
    apples comparison of fold means.
  - Patient-disjoint guarantee: all of a patient's ROIs go to Train OR Test of
    any given fold (because we partition at the patient level FIRST, then expand).

API mirrors the patient-level project's harness.folds.iter_folds so swapping is
a one-line import change.

   for fold_idx, train_rois, test_rois in iter_roi_folds():
       ...

Each element of `train_rois` / `test_rois` is a `RoiEntry`:
   RoiEntry(roi_basename, patient_id, uni2h_pt_path)

The loader (roi_loader.py) takes a list of RoiEntry plus the gene NPZ + labels
CSV and yields per-ROI tensors.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# Paths to PF's canonical data (mounted, not in this project)
PF_BASE = Path('/mnt/storage7/Dataset_pathomicfusion/GBMLGG/data/TCGA_GBMLGG')
PNAS_SPLITS_CSV = PF_BASE / 'pnas_splits.csv'
UNI2H_ROI_DIR   = PF_BASE / 'all_st_uni2h'
LABELS_CSV      = PF_BASE / 'all_dataset.csv'

# BulkRNABert NPZ from the patient-level project (we don't duplicate it here)
BULKRNABERT_NPZ = Path(
    '/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level'
    '/bulkrnabert_data/gbmlgg_bulkrnabert_256d.npz'
)

# Locked cohort numbers (must match COHORT_AND_PROTOCOL.txt; verified at import time)
EXPECTED_WORKING_PATIENTS = 592
EXPECTED_WORKING_ROIS     = 1159

# Full PF GBMLGG cohort (image-only — no BulkRNABert filter applied)
# Matches PF paper Section IV.A: 769 patients / 1505 ROIs.
# Used ONLY for image-alone rows (UNI2-h alone) to directly mirror PF's
# Histology CNN cohort (PF uses 769 for image-only, drops to 502 for any
# RNA-seq-using row).
EXPECTED_FULL_PATIENTS = 769
EXPECTED_FULL_ROIS     = 1505

# PF-exact RNA-seq cohort (the 502 patients in PF's gbmlgg15cv_all_st_1_0_0_rnaseq.pkl)
# We can only evaluate on the intersection with our BulkRNABert NPZ -- 1 PF patient
# (TCGA-06-0221) has no BulkRNABert embedding in our pipeline -> 501 patients / 997 ROIs.
# Used to answer "what would our methods score on PF's exact gene-using cohort?"
PF_RNASEQ_PKL = Path(
    '/mnt/storage7/Dataset_pathomicfusion/GBMLGG/data/TCGA_GBMLGG/splits/'
    'gbmlgg15cv_all_st_1_0_0_rnaseq.pkl'
)
EXPECTED_PF501_PATIENTS = 501
EXPECTED_PF501_ROIS     = 997


@dataclass(frozen=True)
class RoiEntry:
    """One ROI-level training/test example identifier.

    Carries everything the loader needs to assemble a sample-level row at
    runtime: the patient-level gene lookup goes via `patient_id`; the image
    embedding loads via `uni2h_pt_path`; survival labels broadcast from patient
    via `patient_id` and a separate labels-frame in the loader.
    """
    roi_basename: str   # e.g. "TCGA-06-0166-01Z-00-DX1.f0c10e84-..._1"   (no .pt extension)
    patient_id  : str   # e.g. "TCGA-06-0166"
    uni2h_pt_path: Path # absolute path to the corresponding .pt file


def _patient_from_roi_filename(fn: str) -> str:
    """Parse TCGA-XX-XXXX from a UNI2-h ROI filename.
    Names look like 'TCGA-06-0166-01Z-00-DX1.f0c10e84-...-_1.pt' -- the patient
    barcode is the first three hyphen-separated fields before the first '.'.
    """
    base = fn.split('.')[0]
    parts = base.split('-')
    if len(parts) < 3:
        raise ValueError(f"unparseable patient id in filename: {fn}")
    pid = '-'.join(parts[:3])
    if not pid.startswith('TCGA-') or len(pid) < 12:
        raise ValueError(f"suspicious patient id '{pid}' from {fn}")
    return pid


def _scan_pat2roi_entries() -> Tuple[dict[str, List[RoiEntry]], dict[str, str]]:
    """Walk UNI2H_ROI_DIR once and build:
        pat2entries[patient_id] -> list of RoiEntry for that patient
        roi2patient[roi_basename] -> patient_id
    Idempotent / pure side-effect free apart from the filesystem walk."""
    pat2entries: dict[str, List[RoiEntry]] = {}
    roi2patient: dict[str, str] = {}
    for pt_path in sorted(UNI2H_ROI_DIR.glob('*.pt')):
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


def load_working_cohort() -> Set[str]:
    """Return the locked 592-patient working cohort: patients with BOTH a UNI2-h
    .pt file AND a BulkRNABert embedding AND a PF label AND a pnas_splits row.

    Verified against the COHORT_AND_PROTOCOL.txt counts; raises on mismatch so
    that drift is caught at the harness level rather than downstream.
    """
    pat2entries, _ = _scan_pat2roi_entries()
    pf_patients = set(pat2entries.keys())

    brb = np.load(BULKRNABERT_NPZ, allow_pickle=False)
    brb_patients = set(str(p) for p in brb['patient_ids'])

    labels = pd.read_csv(LABELS_CSV).set_index('TCGA ID')
    lab_patients = set(labels.index)

    splits = pd.read_csv(PNAS_SPLITS_CSV, index_col=0)
    split_patients = set(splits.index)

    working = pf_patients & brb_patients & lab_patients & split_patients
    n = len(working)
    if n != EXPECTED_WORKING_PATIENTS:
        raise RuntimeError(
            f"Working cohort drift: got {n} patients, expected "
            f"{EXPECTED_WORKING_PATIENTS}. Check COHORT_AND_PROTOCOL.txt.\n"
            f"  PF patients         : {len(pf_patients)}\n"
            f"  BulkRNABert patients: {len(brb_patients)}\n"
            f"  Labels patients     : {len(lab_patients)}\n"
            f"  Splits patients     : {len(split_patients)}\n"
        )
    return working


def load_pf501_cohort() -> Set[str]:
    """PF-exact 502-patient RNA-seq cohort INTERSECT our 592-patient BulkRNABert
    cohort. Returns 501 patients (1 PF patient -- TCGA-06-0221 -- has no
    BulkRNABert embedding in our pipeline).

    Used to ask: "What would BulkRNABert score on PF's exact gene-using cohort?"
    Direct apples-to-apples with PF Genomic SNN paper number 0.808 (which uses
    the 502-patient subset).

    Source of PF's 502: data_pd.index of gbmlgg15cv_all_st_1_0_0_rnaseq.pkl
    (the PF pkl variant whose data dataframe is restricted to 502 patients
    with complete 320-d genomic vectors).
    """
    import pickle
    with open(PF_RNASEQ_PKL, 'rb') as f:
        d = pickle.load(f)
    pf_502_patients = set(d['data_pd'].index)
    # Intersect with our 592 BulkRNABert cohort
    working_592 = load_working_cohort()
    cohort_501 = working_592 & pf_502_patients
    if len(cohort_501) != EXPECTED_PF501_PATIENTS:
        raise RuntimeError(
            f"PF-501 intersection drift: got {len(cohort_501)} patients, "
            f"expected {EXPECTED_PF501_PATIENTS}.\n"
            f"  PF 502  : {len(pf_502_patients)} patients\n"
            f"  Our 592 : {len(working_592)} patients"
        )
    return cohort_501


def load_full_image_cohort() -> Set[str]:
    """Full PF GBMLGG cohort for IMAGE-ONLY rows: patients with a UNI2-h .pt
    file AND a PF label AND a pnas_splits row. Does NOT intersect with
    BulkRNABert (which would drop to 592).

    Matches PF paper Section IV.A: 769 patients / 1505 ROIs.
    Use this cohort ONLY when the model doesn't consume gene features.
    Raises on drift from 769 so the file is the single source of truth.
    """
    pat2entries, _ = _scan_pat2roi_entries()
    pf_patients = set(pat2entries.keys())

    labels = pd.read_csv(LABELS_CSV).set_index('TCGA ID')
    lab_patients = set(labels.index)

    splits = pd.read_csv(PNAS_SPLITS_CSV, index_col=0)
    split_patients = set(splits.index)

    working = pf_patients & lab_patients & split_patients
    n = len(working)
    if n != EXPECTED_FULL_PATIENTS:
        raise RuntimeError(
            f"Full image cohort drift: got {n} patients, expected "
            f"{EXPECTED_FULL_PATIENTS}.\n"
            f"  PF patients         : {len(pf_patients)}\n"
            f"  Labels patients     : {len(lab_patients)}\n"
            f"  Splits patients     : {len(split_patients)}\n"
        )
    return working


def build_pat2entries_for_cohort(
    cohort: Set[str], expected_rois: int | None = None,
) -> dict[str, List[RoiEntry]]:
    """Restrict pat2entries to a given cohort. Verifies ROI count if
    `expected_rois` is given; defaults to EXPECTED_WORKING_ROIS (1159) for
    backward compatibility.
    """
    if expected_rois is None:
        expected_rois = EXPECTED_WORKING_ROIS
    pat2entries, _ = _scan_pat2roi_entries()
    out = {p: pat2entries[p] for p in cohort}
    n_roi = sum(len(v) for v in out.values())
    if n_roi != expected_rois:
        raise RuntimeError(
            f"ROI-count drift: got {n_roi} ROIs across {len(cohort)} patients, "
            f"expected {expected_rois}."
        )
    return out


def iter_roi_folds(
    cohort: Set[str] | None = None,
    seeds: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
    expected_rois: int | None = None,
) -> Iterator[Tuple[int, List[RoiEntry], List[RoiEntry]]]:
    """Generator yielding (fold_idx, train_rois, test_rois) for each fold in
    pnas_splits.csv intersected with the working cohort.

    Args:
        cohort: optional override of the 592-patient working set (used for
            testing / sanity-check variants). Default = load_working_cohort().
        seeds:  fold indices to iterate (1..15). Default = all 15.

    Yields:
        fold_idx     : int in {1..15} matching the "Randomization - {k}" column
        train_rois   : list[RoiEntry], all ROIs of patients flagged Train this fold
        test_rois    : list[RoiEntry], all ROIs of patients flagged Test this fold
    """
    if cohort is None:
        cohort = load_working_cohort()
    pat2entries = build_pat2entries_for_cohort(cohort, expected_rois=expected_rois)
    splits = pd.read_csv(PNAS_SPLITS_CSV, index_col=0)

    for k in seeds:
        col = f'Randomization - {k}'
        if col not in splits.columns:
            raise KeyError(f"pnas_splits.csv has no column '{col}' (fold {k})")
        assignments = splits[col]
        train_pids = [p for p in cohort if assignments.get(p, None) == 'Train']
        test_pids  = [p for p in cohort if assignments.get(p, None) == 'Test']
        # patient-disjoint expansion to ROI level
        train_rois: List[RoiEntry] = []
        for p in train_pids:
            train_rois.extend(pat2entries[p])
        test_rois: List[RoiEntry] = []
        for p in test_pids:
            test_rois.extend(pat2entries[p])
        yield k, train_rois, test_rois


def cohort_summary() -> dict:
    """One-shot audit -- prints the locked numbers and returns them as a dict
    so downstream code can log them next to results."""
    cohort = load_working_cohort()
    pat2entries = build_pat2entries_for_cohort(cohort)
    labels = pd.read_csv(LABELS_CSV).set_index('TCGA ID')
    events = (1 - labels.loc[sorted(cohort), 'censored']).astype(int)

    summary = {
        'n_patients'   : len(cohort),
        'n_rois'       : sum(len(v) for v in pat2entries.values()),
        'n_events'     : int(events.sum()),
        'n_censored'   : int((events == 0).sum()),
        'event_rate'   : float(events.mean()),
        'mean_rois_per_patient' : float(np.mean([len(v) for v in pat2entries.values()])),
        'max_rois_per_patient'  : int(np.max([len(v) for v in pat2entries.values()])),
    }
    return summary


if __name__ == '__main__':
    # Quick standalone smoke (does not load any tensors)
    summary = cohort_summary()
    print("=== Working cohort summary ===")
    for k, v in summary.items():
        print(f"  {k:30s} = {v}")

    print("\n=== Per-fold ROI counts ===")
    for fold_idx, tr, te in iter_roi_folds():
        # ratio of train_event_rate vs test_event_rate (rough)
        tr_pids = sorted({e.patient_id for e in tr})
        te_pids = sorted({e.patient_id for e in te})
        print(f"  fold {fold_idx:2d}: train n_patients={len(tr_pids):3d} "
              f"n_rois={len(tr):4d}  |  test n_patients={len(te_pids):3d} "
              f"n_rois={len(te):3d}")
