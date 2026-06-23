"""
UCEC sample-level splits  --  5×80/20 BulkRNABert paper protocol.

WHAT
   - Loads PF's TCGA-UCEC master_splits.csv (480 patients × 1 slide/patient)
   - Restricts to the 478-patient WORKING COHORT (PF cohort ∩ our BulkRNABert NPZ)
   - Two patients in PF cohort are missing from our BulkRNABert NPZ:
        TCGA-AP-A0LQ, TCGA-EY-A1GJ
     -> these are dropped from the working cohort (0.4% loss)
   - 5 stratified random 80/20 splits matching BulkRNABert paper protocol

DATA STRUCTURE
   master_splits.csv columns:
       case_id, slide_id, site, is_female, age, survival_months, censorship, fold

   slide_id format: 'TCGA-2E-A9G8-01Z-00-DX1.<UUID>.svs'
   Our .pt files:   'TCGA-2E-A9G8-01Z-00-DX1.<UUID>.pt' (slide_id minus .svs)

   The UCEC_uni2h_slide/ directory contains TWO formats:
     1. Plain (1536,) tensor .pt files            <-- WHAT WE USE
     2. Dict files named '..._patches.pt' with    <-- NOT USED (patch-level)
        (N, 1536) features + coords + metadata
   The slide_id mapping naturally points to (1).

CONVENTION (CRITICAL)
   UCEC survival: event = 1 - censorship   (SAME as GBMLGG/LUAD, OPPOSITE of KIRC)
       censorship=0 -> DEATH observed (event = 1)
       censorship=1 -> CENSORED at last follow-up (event = 0)

   Verified empirically (data audit):
       censorship=0 patients (n=75)  : mean OS = 28.8 months  (died earlier)
       censorship=1 patients (n=405) : mean OS = 40.8 months  (still alive)
       Event rate: 15.6% (75/480) -- LOW, typical for UCEC (good outcomes)

WHY 5-SPLIT INSTEAD OF 15-FOLD
   UCEC has very low event rate (15.6%). At 15-fold the test set is 32 patients
   → ~5 events per fold (too noisy for stable c-index). At 5-split the test
   set is 96 patients → ~15 events per split (much better statistical power).

   Also: matches BulkRNABert paper Table 3 protocol (0.703 ± 0.040 on UCEC,
   BulkRNABert's STRONGEST cohort), enabling direct apples-to-apples
   comparison with their reported number.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np
import pandas as pd

# Re-use the GBMLGG RoiEntry dataclass and patient-ID parser
from .roi_splits import RoiEntry, _patient_from_roi_filename


# ----------------------------------------------------------------------
# UCEC paths
# ----------------------------------------------------------------------
UCEC_PF_BASE     = Path('/mnt/storage7/Dataset_pathomicfusion/UCEC/data/TCGA_UCEC')
UCEC_UNI2H_DIR   = UCEC_PF_BASE / 'UCEC_uni2h_slide'
UCEC_SPLITS_DIR  = UCEC_PF_BASE / 'splits' / '15foldcv'
UCEC_MASTER_CSV  = UCEC_SPLITS_DIR / 'master_splits.csv'

# BulkRNABert NPZ for UCEC (from the patient-level project)
UCEC_BULKRNABERT_NPZ = Path(
    '/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level'
    '/bulkrnabert_data/ucec_bulkrnabert_256d.npz'
)

# Verified cohort numbers (verified at module-build time)
EXPECTED_UCEC_PATIENTS = 478       # PF 480 ∩ our BulkRNABert NPZ
EXPECTED_UCEC_SLIDES   = 478       # 1 slide/patient
EXPECTED_UCEC_PF_FULL  = 480       # PF master_splits full cohort (informational)

# Patients in PF cohort but MISSING from our BulkRNABert NPZ.
# Verified at data-audit time; dropping these is the only reason we have 478 not 480.
UCEC_MISSING_BULKRNABERT = frozenset({"TCGA-AP-A0LQ", "TCGA-EY-A1GJ"})


# ----------------------------------------------------------------------
# Patient-level cohort
# ----------------------------------------------------------------------
def load_ucec_cohort() -> Set[str]:
    """The 478-patient UCEC working cohort.

    = (PF master_splits 480 patients) ∩ (our BulkRNABert NPZ 545 patients)
    The 2 dropped patients (TCGA-AP-A0LQ, TCGA-EY-A1GJ) are missing from our
    BulkRNABert NPZ.
    """
    master = pd.read_csv(UCEC_MASTER_CSV)
    pf_480 = set(master['case_id'])
    if len(pf_480) != EXPECTED_UCEC_PF_FULL:
        raise RuntimeError(
            f"UCEC master_splits patient drift: got {len(pf_480)}, "
            f"expected {EXPECTED_UCEC_PF_FULL}"
        )
    # Intersect with BulkRNABert NPZ
    z = np.load(UCEC_BULKRNABERT_NPZ, allow_pickle=False)
    bulk_patients = set(str(p) for p in z['patient_ids'])
    cohort = pf_480 & bulk_patients
    if len(cohort) != EXPECTED_UCEC_PATIENTS:
        raise RuntimeError(
            f"UCEC cohort drift: got {len(cohort)} patients, "
            f"expected {EXPECTED_UCEC_PATIENTS}"
        )
    # Sanity: confirm the 2 missing patients are the expected ones
    actually_missing = pf_480 - bulk_patients
    if actually_missing != UCEC_MISSING_BULKRNABERT:
        raise RuntimeError(
            f"UCEC missing-from-BulkRNABert drift: expected "
            f"{sorted(UCEC_MISSING_BULKRNABERT)}, got {sorted(actually_missing)}"
        )
    return cohort


# ----------------------------------------------------------------------
# Patient -> list of RoiEntry helper
# ----------------------------------------------------------------------
def _slide_to_entry(slide_id_no_ext: str, patient_id: str) -> RoiEntry:
    """Build a RoiEntry from a slide_id stem (without .svs/.pt extension)."""
    pt_path = UCEC_UNI2H_DIR / f"{slide_id_no_ext}.pt"
    return RoiEntry(
        roi_basename = slide_id_no_ext,
        patient_id   = patient_id,
        uni2h_pt_path = pt_path,
    )


def build_ucec_pat2entries() -> dict[str, List[RoiEntry]]:
    """Build {patient_id -> [RoiEntry(slide stem, pid, .pt path)]} from
    master_splits.csv. Includes ALL 480 patients here -- iter functions
    apply cohort filtering downstream.

    For UCEC every patient has exactly 1 slide, so each list has length 1.
    The list-of-RoiEntry shape is kept for API consistency with KIRC/GBMLGG.
    """
    master = pd.read_csv(UCEC_MASTER_CSV)
    out: dict[str, List[RoiEntry]] = {}
    for _, row in master.iterrows():
        pid = row['case_id']
        slide_id_no_ext = row['slide_id'].replace('.svs', '')
        out.setdefault(pid, []).append(_slide_to_entry(slide_id_no_ext, pid))
    return out


# ----------------------------------------------------------------------
# 5×80/20 BulkRNABert paper protocol (PRIMARY iter for UCEC)
# ----------------------------------------------------------------------
def iter_ucec_5split_folds(
    cohort: Set[str] | None = None,
    expected_rois: int | None = None,
    split_seeds: List[int] = (0, 1, 2, 3, 4),
) -> Iterator[Tuple[int, List[RoiEntry], List[RoiEntry]]]:
    """Yield 5 stratified random 80/20 splits on the 478 UCEC cohort.

    Matches BulkRNABert paper Table 3 protocol (Lasry et al. 2024 Section 3.2):
       "For each task, the dataset is split into 80% train and 20% test.
        This is done with 5 different seeds, and the reported score corresponds
        to the mean score across these 5 seeds. The splits are stratified to
        ensure equal representation of each class in the train and test splits.
        Multiple RNA-seq samples might be available for a given patient; thus,
        we ensure that samples from the same patient are not mixed in the
        train and test splits."

    For UCEC specifically:
       - 478 patients × 0.20 = ~96 test patients per split
       - ~382 train patients per split
       - Patient-disjoint by construction
       - Stratified by EVENT (event = 1 - censorship)
       - Expected per-split event rate: ~15.6% (low, but stratified)

    Returns:
       Iterator yielding (split_idx, train_rois, test_rois) where split_idx is
       1-based (1..5) for consistency with KIRC/GBMLGG conventions.

    Multi-sample-per-patient caveat from the paper doesn't apply: UCEC has
    1 slide per patient.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    if cohort is None:
        cohort = load_ucec_cohort()
    if expected_rois is None:
        expected_rois = EXPECTED_UCEC_SLIDES

    # Build cohort-filtered patient → ROI mapping
    pat2entries_all = build_ucec_pat2entries()
    pat2entries = {pid: pat2entries_all[pid] for pid in cohort
                   if pid in pat2entries_all}

    # Sanity: every cohort patient has entries
    missing = cohort - set(pat2entries.keys())
    if missing:
        raise RuntimeError(
            f"UCEC iter: {len(missing)} cohort patients missing entries "
            f"in master_splits.csv: {sorted(missing)[:3]}"
        )

    # Total ROI count check
    total = sum(len(v) for v in pat2entries.values())
    if expected_rois is not None and total != expected_rois:
        raise RuntimeError(
            f"UCEC ROI count drift: got {total}, expected {expected_rois}"
        )

    # Build stratification arrays from master_splits, RESTRICTED to cohort
    master = pd.read_csv(UCEC_MASTER_CSV)
    master = master[master['case_id'].isin(cohort)].sort_values('case_id').reset_index(drop=True)
    if len(master) != len(cohort):
        raise RuntimeError(
            f"UCEC master/cohort mismatch: master has {len(master)} cohort entries, "
            f"cohort has {len(cohort)}"
        )
    pids   = master['case_id'].values
    strata = (1 - master['censorship'].astype(int)).values   # event labels (1=death, 0=censored)

    # 5 stratified random splits, one per seed
    for split_idx, seed in enumerate(split_seeds, start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, test_idx = next(sss.split(pids, strata))
        train_pids = set(pids[train_idx])
        test_pids  = set(pids[test_idx])

        # Patient-disjoint sanity (StratifiedShuffleSplit guarantees this, but verify)
        overlap = train_pids & test_pids
        if overlap:
            raise RuntimeError(
                f"UCEC 5-split seed {seed}: train/test overlap of {len(overlap)} patients"
            )

        # Expand patients to ROIs (slides; 1 per patient on UCEC)
        train_rois: List[RoiEntry] = []
        test_rois : List[RoiEntry] = []
        for pid in train_pids:
            train_rois.extend(pat2entries[pid])
        for pid in test_pids:
            test_rois.extend(pat2entries[pid])

        yield split_idx, train_rois, test_rois


# ----------------------------------------------------------------------
# Summaries for sanity-printing
# ----------------------------------------------------------------------
def ucec_5split_summary() -> list[dict]:
    """Per-split summary (size, event rate) for the 5-split protocol."""
    cohort = load_ucec_cohort()
    labels = pd.read_csv(UCEC_MASTER_CSV).set_index('case_id')
    out = []
    for split_idx, tr, te in iter_ucec_5split_folds(cohort=cohort):
        tr_pats = sorted({r.patient_id for r in tr})
        te_pats = sorted({r.patient_id for r in te})
        tr_events = int((1 - labels.loc[tr_pats, 'censorship']).sum())
        te_events = int((1 - labels.loc[te_pats, 'censorship']).sum())
        out.append({
            "split":                split_idx,
            "train_n_patients":     len(tr_pats),
            "train_n_events":       tr_events,
            "train_event_rate_pct": round(100 * tr_events / max(len(tr_pats), 1), 1),
            "test_n_patients":      len(te_pats),
            "test_n_events":        te_events,
            "test_event_rate_pct":  round(100 * te_events / max(len(te_pats), 1), 1),
        })
    return out


def ucec_cohort_summary() -> dict:
    """One-shot summary for sanity printing."""
    cohort = load_ucec_cohort()
    pat2 = build_ucec_pat2entries()
    cohort_entries = {p: pat2[p] for p in cohort if p in pat2}
    rois_per_patient = sorted(len(v) for v in cohort_entries.values())
    n_rois = sum(rois_per_patient)
    return {
        "cohort_size_patients":     len(cohort),
        "expected_pf_full_cohort":  EXPECTED_UCEC_PF_FULL,
        "missing_from_bulkrnabert": sorted(UCEC_MISSING_BULKRNABERT),
        "total_rois":               n_rois,
        "rois_per_patient_min":     rois_per_patient[0] if rois_per_patient else 0,
        "rois_per_patient_median":  rois_per_patient[len(rois_per_patient)//2] if rois_per_patient else 0,
        "rois_per_patient_max":     rois_per_patient[-1] if rois_per_patient else 0,
    }


if __name__ == '__main__':
    print("=== UCEC sample-level cohort summary ===")
    s = ucec_cohort_summary()
    for k, v in s.items():
        print(f"  {k:30s} : {v}")
    print("\n=== UCEC 5-split sanity (per-split sizes + event rates) ===")
    print(f"  {'split':>5}  {'tr_n':>5}  {'te_n':>5}  {'tr_ev':>6}  {'te_ev':>6}  {'tr_rate':>8}  {'te_rate':>8}")
    print(f"  {'-'*5}  {'-'*5}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*8}  {'-'*8}")
    for row in ucec_5split_summary():
        print(f"  {row['split']:>5}  {row['train_n_patients']:>5}  "
              f"{row['test_n_patients']:>5}  {row['train_n_events']:>6}  "
              f"{row['test_n_events']:>6}  {row['train_event_rate_pct']:>7}%  "
              f"{row['test_event_rate_pct']:>7}%")
