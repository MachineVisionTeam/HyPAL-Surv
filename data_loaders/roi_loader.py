"""
Sample-level data assembly for GBMLGG.

For each ROI (one row in train/test set), we need a 4-tuple:
   x_image (1536,)  -- UNI2-h embedding of this ROI (loaded from .pt)
   x_gene  (D,)     -- PATIENT's gene vector, broadcast across all of that
                       patient's ROIs. D = 256 for BulkRNABert, 19062 for MLP-19K.
   time    (float)  -- patient's survival months
   event   (int)    -- patient's death event (= 1 - censored)

The broadcast pattern (one gene vector per patient, repeated for each of that
patient's ROIs) is exactly what PF (2020) and Path-GPTOmic (2024) do at
sample-level. See PF paper Section IV.A: "320 genomic features... for each
patient."

The loader returns numpy arrays (no torch tensors) so downstream code can
either feed to torch directly (full-batch training is fine -- ~930 ROIs per
fold) or wrap in a DataLoader if needed for bigger cohorts later.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .roi_splits import (
    BULKRNABERT_NPZ, LABELS_CSV, RoiEntry,
)


# ---------------------------------------------------------------------------
# Gene-side loaders -- each returns a {patient_id -> 1-d numpy array} dict.
# ---------------------------------------------------------------------------
def load_bulkrnabert_patient_dict() -> dict[str, np.ndarray]:
    """800-patient NPZ -> {patient_id: (256,) float32}.
    Matches the patient-level project's load_bulkrnabert_at_patient_level.
    Each entry is broadcast at runtime to each of that patient's ROIs.
    """
    z = np.load(BULKRNABERT_NPZ, allow_pickle=False)
    pids = z['patient_ids']
    emb  = z['embeddings']     # (800, 256)
    out: dict[str, np.ndarray] = {}
    for i, p in enumerate(pids):
        out[str(p)] = emb[i].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Labels loader
# ---------------------------------------------------------------------------
def load_patient_labels() -> dict[str, Tuple[int, float]]:
    """Returns {patient_id: (event, time)}.
       event = 1 - censored       (PF convention: censored=1 means alive)
       time  = Survival months    (PF's column name)

    UNITS NOTE: despite the "Survival months" column name, the values are
    actually in DAYS (max 6423 = 17.6 years across the cohort, median 523).
    PF's own train/test pkl uses identical values; PF/Path-GPTOmic and our
    patient-level project all consume them as-is. Cox regression is scale-
    invariant so this does not affect c-index. We do NOT convert.
    """
    df = pd.read_csv(LABELS_CSV).set_index('TCGA ID')
    if 'censored' not in df.columns or 'Survival months' not in df.columns:
        raise KeyError(
            f"PF labels file missing required columns. Got: {list(df.columns)[:8]}"
        )
    events = (1 - df['censored']).astype(int)
    times  = df['Survival months'].astype(float)
    return {pid: (int(events.loc[pid]), float(times.loc[pid])) for pid in df.index}


# ---------------------------------------------------------------------------
# Per-ROI assembler -- the central function
# ---------------------------------------------------------------------------
def assemble_roi_batch(
    roi_entries: Sequence[RoiEntry],
    gene_dict: dict[str, np.ndarray],
    labels: dict[str, Tuple[int, float]],
    expected_gene_dim: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assemble a per-ROI batch from a list of RoiEntry objects.

    Args:
        roi_entries: list of RoiEntry (from iter_roi_folds).
        gene_dict:   {patient_id -> 1-d gene vector}. Looked up by patient_id.
        labels:      {patient_id -> (event, time)}.
        expected_gene_dim: if set, raise if the resolved gene dim differs.

    Returns:
        X_img : (N, 1536) float32  -- per-ROI UNI2-h embedding
        X_gene: (N, D)    float32  -- patient's gene vector broadcast to each ROI
        events: (N,)      float32  -- broadcast patient event
        times : (N,)      float32  -- broadcast patient survival months
        roi_ids: list[str] of length N -- the .stem of each .pt path (for joining)
    """
    n = len(roi_entries)
    if n == 0:
        raise ValueError("assemble_roi_batch called with empty roi_entries")

    # Probe gene dim from the first ROI's patient (any patient -- they should all
    # have the same dim by construction since the gene_dict is uniform).
    first_pid = roi_entries[0].patient_id
    if first_pid not in gene_dict:
        raise KeyError(f"patient {first_pid} missing from gene_dict")
    gene_dim = gene_dict[first_pid].shape[0]
    if expected_gene_dim is not None and gene_dim != expected_gene_dim:
        raise ValueError(
            f"gene_dim mismatch: got {gene_dim}, expected {expected_gene_dim}"
        )

    X_img  = np.zeros((n, 1536),     dtype=np.float32)
    X_gene = np.zeros((n, gene_dim), dtype=np.float32)
    events = np.zeros(n,             dtype=np.float32)
    times  = np.zeros(n,             dtype=np.float32)
    roi_ids: list[str] = []

    for i, entry in enumerate(roi_entries):
        # Image: load the .pt -- it's a single (1536,) tensor per ROI in our pipeline
        ten = torch.load(entry.uni2h_pt_path, map_location='cpu', weights_only=False)
        if isinstance(ten, dict):
            # safety net: if a future variant stores a dict, take 'features' mean
            ten = ten['features'].mean(dim=0)
        if hasattr(ten, 'numpy'):
            arr = ten.float().numpy()
        else:
            arr = np.asarray(ten, dtype=np.float32)
        # BRCA-compat: SurvPath_features/BRCA_uni2h/*.pt are patch-level
        # (n_patches, 1536) -- mean-pool to slide-level (1536,) here.
        # No-op for 1D slide-level tensors used by LUAD/UCEC/GBMLGG/KIRC.
        if len(arr.shape) == 2 and arr.shape[1] == 1536:
            arr = arr.mean(axis=0)
        if arr.shape != (1536,):
            raise ValueError(
                f"UNI2-h tensor for {entry.roi_basename} has shape {arr.shape}, "
                f"expected (1536,)"
            )
        X_img[i] = arr

        # Gene: broadcast patient vector
        pid = entry.patient_id
        if pid not in gene_dict:
            raise KeyError(f"patient {pid} missing from gene_dict")
        X_gene[i] = gene_dict[pid]

        # Labels: broadcast patient (event, time)
        if pid not in labels:
            raise KeyError(f"patient {pid} missing from labels")
        e, t = labels[pid]
        events[i] = float(e)
        times[i]  = float(t)

        roi_ids.append(entry.roi_basename)

    return X_img, X_gene, events, times, roi_ids


# ---------------------------------------------------------------------------
# Convenience: load image-only or gene-only batches (used by Stage 2 unimodal)
# ---------------------------------------------------------------------------
def assemble_image_only(
    roi_entries: Sequence[RoiEntry],
    labels: dict[str, Tuple[int, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """For sl_2_uni2h_alone -- skip the gene vector entirely."""
    n = len(roi_entries)
    X_img  = np.zeros((n, 1536), dtype=np.float32)
    events = np.zeros(n,         dtype=np.float32)
    times  = np.zeros(n,         dtype=np.float32)
    roi_ids: list[str] = []
    for i, entry in enumerate(roi_entries):
        ten = torch.load(entry.uni2h_pt_path, map_location='cpu', weights_only=False)
        if isinstance(ten, dict):
            ten = ten['features'].mean(dim=0)
        if hasattr(ten, 'numpy'):
            arr = ten.float().numpy()
        else:
            arr = np.asarray(ten, dtype=np.float32)
        # BRCA-compat: patch-level (n_patches, 1536) -> slide-level (1536,)
        # No-op for 1D slide-level tensors used by other cohorts.
        if len(arr.shape) == 2 and arr.shape[1] == 1536:
            arr = arr.mean(axis=0)
        if arr.shape != (1536,):
            raise ValueError(
                f"UNI2-h tensor for {entry.roi_basename} has shape {arr.shape}, "
                f"expected (1536,)"
            )
        X_img[i] = arr
        e, t = labels[entry.patient_id]
        events[i] = float(e); times[i] = float(t)
        roi_ids.append(entry.roi_basename)
    return X_img, events, times, roi_ids


def assemble_gene_only(
    roi_entries: Sequence[RoiEntry],
    gene_dict: dict[str, np.ndarray],
    labels: dict[str, Tuple[int, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """For sl_2_bulkrnabert_alone -- per-ROI gene broadcast (so multi-ROI
    patients up-weight, matching PF's ROI-level convention)."""
    n = len(roi_entries)
    first_pid = roi_entries[0].patient_id
    gene_dim = gene_dict[first_pid].shape[0]
    X_gene = np.zeros((n, gene_dim), dtype=np.float32)
    events = np.zeros(n, dtype=np.float32)
    times  = np.zeros(n, dtype=np.float32)
    roi_ids: list[str] = []
    for i, entry in enumerate(roi_entries):
        pid = entry.patient_id
        X_gene[i] = gene_dict[pid]
        e, t = labels[pid]
        events[i] = float(e); times[i] = float(t)
        roi_ids.append(entry.roi_basename)
    return X_gene, events, times, roi_ids


if __name__ == '__main__':
    # Smoke: load fold 1, assemble one full bimodal batch (train side), verify shapes
    import time
    from .roi_splits import iter_roi_folds

    print("Loading gene dict + labels ...")
    gene_dict = load_bulkrnabert_patient_dict()
    labels    = load_patient_labels()
    print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
    print(f"  labels   : {len(labels)} patients")

    print("\nLoading fold 1 ROIs ...")
    fold_iter = iter_roi_folds()
    fold_idx, train_rois, test_rois = next(fold_iter)
    print(f"  fold {fold_idx}: {len(train_rois)} train ROIs, {len(test_rois)} test ROIs")

    print("\nAssembling fold 1 train batch (this loads ~930 UNI2-h .pt files)...")
    t0 = time.time()
    X_img, X_gene, events, times, roi_ids = assemble_roi_batch(
        train_rois, gene_dict, labels, expected_gene_dim=256
    )
    print(f"  X_img : {X_img.shape}  dtype={X_img.dtype}  "
          f"range=[{X_img.min():.3f}, {X_img.max():.3f}]")
    print(f"  X_gene: {X_gene.shape}  dtype={X_gene.dtype}  "
          f"range=[{X_gene.min():.3f}, {X_gene.max():.3f}]")
    print(f"  events: {int(events.sum())} dead / {int((events==0).sum())} censored")
    print(f"  times : range=[{times.min():.1f}, {times.max():.1f}] months")
    print(f"  roi_ids[0] = {roi_ids[0]}")
    print(f"  load time : {time.time()-t0:.1f}s")

    # Verify broadcast: same patient should have identical X_gene rows
    pid_first = train_rois[0].patient_id
    same_pid_rows = [i for i, e in enumerate(train_rois) if e.patient_id == pid_first]
    if len(same_pid_rows) > 1:
        i0, i1 = same_pid_rows[0], same_pid_rows[1]
        same = np.allclose(X_gene[i0], X_gene[i1])
        print(f"\n  Broadcast check: patient {pid_first} has {len(same_pid_rows)} ROIs, "
              f"gene vectors identical = {same}")
