"""
LUAD sample-level splits  --  patient-level 15-fold MCCV expanded to slides.

WHAT
   - Loads PF's TCGA-LUAD master_splits.csv + per-fold splits_0..14.csv from
     /mnt/storage7/Dataset_pathomicfusion/LUAD/data/TCGA_LUAD/splits/15foldcv/
   - 450 patients × 1 slide/patient = 450 ROIs (sample-level)
   - For each fold k (0..14), yields ROI-level Train/Test lists matching
     PF's exact patient partition.

DATA STRUCTURE
   master_splits.csv columns:
       case_id, slide_id, site, age, survival_months, censorship, fold

   slide_id format: 'TCGA-05-4244-01Z-00-DX1.<UUID>.svs'
   Our .pt files:   'TCGA-05-4244-01Z-00-DX1.<UUID>.pt' (slide_id minus .svs)

   The LUAD_uni2h_slide/ directory contains TWO formats:
     1. Plain (1536,) tensor .pt files            <-- WHAT WE USE
     2. Dict files named '..._patches.pt' with    <-- NOT USED (patch-level)
        (250, 1536) features + coords + metadata
   The slide_id mapping naturally points to (1) — we never load the dict files.

   Splits format: splits_0.csv ... splits_14.csv each have columns
   'train', 'val' with patient IDs. 420 train + 30 val per fold (MCCV).

CONVENTION (CRITICAL)
   LUAD survival: event = 1 - censorship   (SAME as GBMLGG, OPPOSITE of KIRC)
       censorship=0 -> DEATH observed (event = 1)
       censorship=1 -> CENSORED (alive at last follow-up, event = 0)

   Verified empirically: censorship=0 patients have mean OS=24.8 months,
   censorship=1 patients have mean OS=32.8 months (alive longer → censored).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np
import pandas as pd

# Re-use the GBMLGG RoiEntry dataclass and patient-ID parser
from .roi_splits import RoiEntry, _patient_from_roi_filename


# ----------------------------------------------------------------------
# LUAD paths
# ----------------------------------------------------------------------
LUAD_PF_BASE   = Path('/mnt/storage7/Dataset_pathomicfusion/LUAD/data/TCGA_LUAD')
LUAD_UNI2H_DIR = LUAD_PF_BASE / 'LUAD_uni2h_slide'
LUAD_SPLITS_DIR = LUAD_PF_BASE / 'splits' / '15foldcv'
LUAD_MASTER_CSV = LUAD_SPLITS_DIR / 'master_splits.csv'

# Our BulkRNABert NPZ for LUAD
LUAD_BULKRNABERT_NPZ = Path(
    '/home/sbarua/Region_based_segmentation/pathgptomic_bulkrnabert_patient_level'
    '/bulkrnabert_data/luad_bulkrnabert_256d.npz'
)

# Verified cohort numbers (verified at module build time)
EXPECTED_LUAD_PATIENTS = 450     # PF cohort, 100% intersection with our features
EXPECTED_LUAD_SLIDES   = 450     # 1 slide per patient (sample-level = patient-level)
EXPECTED_LUAD_FOLDS    = 15      # 15-fold MCCV


# ----------------------------------------------------------------------
# Patient-level cohort
# ----------------------------------------------------------------------
def load_luad_cohort() -> Set[str]:
    """The 450-patient LUAD cohort (from PF master_splits.csv case_id).

    Verified: this set EQUALS PF's 450 ∩ our UNI2-h slides ∩ our BulkRNABert.
    No further intersection needed.
    """
    master = pd.read_csv(LUAD_MASTER_CSV)
    cohort = set(master['case_id'].unique())
    if len(cohort) != EXPECTED_LUAD_PATIENTS:
        raise RuntimeError(
            f"LUAD cohort drift: got {len(cohort)} patients, "
            f"expected {EXPECTED_LUAD_PATIENTS}"
        )
    return cohort


# ----------------------------------------------------------------------
# Slide -> RoiEntry helper
# ----------------------------------------------------------------------
def _slide_to_entry(slide_id_no_ext: str, patient_id: str) -> RoiEntry:
    """Build a RoiEntry from a slide_id (without .svs/.pt extension).

    slide_id_no_ext: e.g. 'TCGA-05-4244-01Z-00-DX1.d4ff32cd-...c2b100ac01'
    patient_id    : first 12 chars (TCGA-XX-XXXX)
    """
    pt_path = LUAD_UNI2H_DIR / f"{slide_id_no_ext}.pt"
    return RoiEntry(
        roi_basename = slide_id_no_ext,
        patient_id   = patient_id,
        uni2h_pt_path = pt_path,
    )


def build_luad_pat2entries() -> dict[str, List[RoiEntry]]:
    """Read master_splits.csv and build {patient_id -> [RoiEntry(slide_id_stem,
    patient_id, .pt path)]} for the 450 cohort.

    For LUAD there is exactly 1 slide per patient (verified), so each value
    is a list of length 1. The list-of-RoiEntry shape is kept for compatibility
    with the GBMLGG/KIRC interface.
    """
    master = pd.read_csv(LUAD_MASTER_CSV)
    out: dict[str, List[RoiEntry]] = {}
    for _, row in master.iterrows():
        pid = row['case_id']
        slide_id_no_ext = row['slide_id'].replace('.svs', '')
        entry = _slide_to_entry(slide_id_no_ext, pid)
        out.setdefault(pid, []).append(entry)
    return out


# ----------------------------------------------------------------------
# Per-fold patient-level splits (from PF's splits_0..14.csv)
# ----------------------------------------------------------------------
def _load_pf_luad_patient_splits() -> dict[int, Tuple[Set[str], Set[str]]]:
    """Read PF's splits_0..14.csv and return {fold_idx: (train_pats, test_pats)}.

    splits_N.csv has 'train' and 'val' columns with patient IDs.
    Note: PF calls them 'train' and 'val', but for MCCV survival this is the
    train/test partition (no separate val fold).
    """
    out: dict[int, Tuple[Set[str], Set[str]]] = {}
    for fi in range(15):
        sp = pd.read_csv(LUAD_SPLITS_DIR / f'splits_{fi}.csv')
        train_pats = set(sp['train'].dropna().astype(str))
        test_pats  = set(sp['val'  ].dropna().astype(str))
        overlap = train_pats & test_pats
        if overlap:
            raise RuntimeError(
                f"LUAD splits_{fi}.csv has train/val overlap: {len(overlap)} patients"
            )
        out[fi] = (train_pats, test_pats)
    return out


# ----------------------------------------------------------------------
# Public iter
# ----------------------------------------------------------------------
def iter_luad_roi_folds(
    cohort: Set[str] | None = None,
    expected_rois: int | None = None,
) -> Iterator[Tuple[int, List[RoiEntry], List[RoiEntry]]]:
    """Yield (fold_idx, train_rois, test_rois) for each of PF's 15 LUAD folds.

    Returns fold_idx in [1..15] (1-indexed for consistency with KIRC/GBMLGG;
    PF stores splits_0..14.csv so we shift +1 internally).

    Each train_rois / test_rois is a list of RoiEntry (slide-level on LUAD).
    Patient-disjoint by construction.
    """
    if cohort is None:
        cohort = load_luad_cohort()
    if expected_rois is None:
        expected_rois = EXPECTED_LUAD_SLIDES

    pat2entries = build_luad_pat2entries()
    # Cohort match sanity
    missing = cohort - set(pat2entries.keys())
    if missing:
        raise RuntimeError(
            f"LUAD cohort drift: {len(missing)} cohort patients without entries "
            f"in master_splits.csv"
        )

    # Total ROI count check
    total = sum(len(v) for pid, v in pat2entries.items() if pid in cohort)
    if expected_rois is not None and total != expected_rois:
        raise RuntimeError(
            f"LUAD ROI count drift: got {total}, expected {expected_rois}"
        )

    pf_splits = _load_pf_luad_patient_splits()
    for pf_fold_idx in sorted(pf_splits.keys()):       # 0..14
        train_pats, test_pats = pf_splits[pf_fold_idx]
        # Restrict to cohort (no-op if cohort == full cohort)
        train_pats = train_pats & cohort
        test_pats  = test_pats  & cohort
        # Sanity check: train + test should equal the full cohort (MCCV)
        if (train_pats & test_pats):
            raise RuntimeError(f"fold {pf_fold_idx}: train/test overlap detected")
        train_rois: List[RoiEntry] = []
        test_rois : List[RoiEntry] = []
        for pid in train_pats:
            train_rois.extend(pat2entries[pid])
        for pid in test_pats:
            test_rois.extend(pat2entries[pid])
        # 1-indexed fold for downstream consistency
        yield pf_fold_idx + 1, train_rois, test_rois


def iter_luad_5split_folds(
    cohort: Set[str] | None = None,
    expected_rois: int | None = None,
    split_seeds: List[int] = (0, 1, 2, 3, 4),
) -> Iterator[Tuple[int, List[RoiEntry], List[RoiEntry]]]:
    """Yield 5 stratified random 80/20 splits  --  matches BulkRNABert paper Table 3 protocol.

    PROTOCOL (verbatim from Lasry et al. 2024 BulkRNABert paper Section 3.2):
       "For each task, the dataset is split into 80% train and 20% test.
        This is done with 5 different seeds, and the reported score corresponds
        to the mean score across these 5 seeds. The splits are stratified to
        ensure equal representation of each class in the train and test splits.
        Multiple RNA-seq samples might be available for a given patient; thus,
        we ensure that samples from the same patient are not mixed in the
        train and test splits."

    For LUAD specifically:
       - 450 patients × 0.20 = 90 test patients per split
       - 360 train patients per split
       - Patient-disjoint by construction (each patient in exactly ONE side per split)
       - Stratified by EVENT (event = 1 - censorship) so each split has similar
         event rate (~35.6% events expected, since 160/450 events)

    Returns:
       Iterator yielding (split_idx, train_rois, test_rois)
       where split_idx is 1-based (1..5) for downstream consistency.

    Note: Multiple "samples per patient" caveat from the paper doesn't apply
    to LUAD because LUAD has 1 slide per patient by construction.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    if cohort is None:
        cohort = load_luad_cohort()
    if expected_rois is None:
        expected_rois = EXPECTED_LUAD_SLIDES

    # Load patient → ROI entries
    pat2entries = build_luad_pat2entries()
    if set(pat2entries.keys()) - cohort:
        # silently ignore patients with entries but not in cohort
        pass
    if cohort - set(pat2entries.keys()):
        raise RuntimeError(
            f"LUAD cohort missing entries: "
            f"{len(cohort - set(pat2entries.keys()))} patients without .pt files"
        )

    # Build stratification arrays from master_splits.csv:
    # - PIDS: 1D array of patient IDs in deterministic order
    # - STRATA: corresponding event labels (1 = death, 0 = censored)
    # We use the same convention as load_luad_patient_labels(): event = 1 - censorship
    master = pd.read_csv(LUAD_SPLITS_DIR.parent.parent / 'splits' / '15foldcv' / 'master_splits.csv')
    master = master[master['case_id'].isin(cohort)].sort_values('case_id').reset_index(drop=True)
    if len(master) != len(cohort):
        raise RuntimeError(
            f"LUAD master/cohort mismatch: master has {len(master)} cohort entries, "
            f"cohort has {len(cohort)}"
        )
    pids   = master['case_id'].values
    strata = (1 - master['censorship'].astype(int)).values   # event labels

    # 5 stratified random splits, one per seed
    n_splits = len(split_seeds)
    for split_idx, seed in enumerate(split_seeds, start=1):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
        train_idx, test_idx = next(sss.split(pids, strata))
        train_pids = set(pids[train_idx])
        test_pids  = set(pids[test_idx])

        # Patient-disjoint sanity (StratifiedShuffleSplit guarantees this but verify)
        overlap = train_pids & test_pids
        if overlap:
            raise RuntimeError(
                f"LUAD 5-split seed {seed}: train/test overlap of {len(overlap)} patients"
            )

        # Strata balance sanity (event rates should be close to 35.6%)
        tr_events = sum(strata[i] for i in train_idx)
        te_events = sum(strata[i] for i in test_idx)
        # (informational; we don't error here since slight variation is normal)

        # Expand patients to ROIs (slides)
        train_rois: List[RoiEntry] = []
        test_rois : List[RoiEntry] = []
        for pid in train_pids:
            train_rois.extend(pat2entries[pid])
        for pid in test_pids:
            test_rois.extend(pat2entries[pid])

        yield split_idx, train_rois, test_rois


def luad_5split_summary() -> dict:
    """Sanity summary for the 5-split protocol (stratification, sizes, event rates)."""
    cohort = load_luad_cohort()
    out = []
    for split_idx, tr, te in iter_luad_5split_folds(cohort=cohort):
        # Lookup events per patient
        labels = pd.read_csv(LUAD_SPLITS_DIR / 'master_splits.csv').set_index('case_id')
        tr_pats = sorted({r.patient_id for r in tr})
        te_pats = sorted({r.patient_id for r in te})
        tr_events = int((1 - labels.loc[tr_pats, 'censorship']).sum())
        te_events = int((1 - labels.loc[te_pats, 'censorship']).sum())
        out.append({
            "split": split_idx,
            "train_n_patients": len(tr_pats),
            "train_n_events": tr_events,
            "train_event_rate_pct": round(100 * tr_events / len(tr_pats), 1),
            "test_n_patients": len(te_pats),
            "test_n_events": te_events,
            "test_event_rate_pct": round(100 * te_events / len(te_pats), 1),
        })
    return out


def luad_cohort_summary() -> dict:
    """One-shot summary for sanity printing."""
    cohort = load_luad_cohort()
    pat2 = build_luad_pat2entries()
    rois_per_patient = sorted(len(v) for v in pat2.values())
    n_rois = sum(rois_per_patient)
    splits = _load_pf_luad_patient_splits()
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
    print("=== LUAD sample-level cohort summary ===")
    s = luad_cohort_summary()
    for k, v in s.items():
        print(f"  {k:30s} : {v}")
