"""Cohort splits and ROI/slide loaders for the 4 TCGA cohorts used in the
HyPAL-Surv evaluation:

    TCGA-GBMLGG  (PF-501, 15-fold MCCV)        roi_splits_gbmlgg.py + roi_loader.py
    TCGA-KIRC    (PF-417, 15-fold MCCV)        roi_splits_kirc.py   + roi_loader_kirc.py
    TCGA-LUAD    (450 pt, 5x80/20 splits)      roi_splits_luad.py   + roi_loader_luad.py
    TCGA-UCEC    (478 pt, 5x80/20 splits)      roi_splits_ucec.py   + roi_loader_ucec.py

Each cohort module exposes:
    get_splits(seed: int) -> List[Dict]    fold-level (train, val, test) patient IDs
    assemble_roi_batch(...)                 batches of (image, gene, T, event)
"""
