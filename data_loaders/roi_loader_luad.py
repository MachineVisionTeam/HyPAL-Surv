"""
LUAD sample-level data assembly.

Mirrors the API of `harness.roi_loader` (GBMLGG) and `harness.roi_loader_kirc`
(KIRC) so the runners work with a one-line import change. Two key differences
from KIRC:

  (1) BulkRNABert NPZ path  : luad_bulkrnabert_256d.npz (517 patients × 256-d)
  (2) Event convention      : event = 1 - censorship  (SAME as GBMLGG, OPPOSITE of KIRC)

For each slide we assemble the same 4-tuple as KIRC:
   x_image (1536,)  -- UNI2-h SLIDE embedding (already aggregated)
   x_gene  (256,)   -- patient's BulkRNABert vector
   time    (float)  -- patient's survival_months
   event   (int)    -- patient's death event (= 1 - censorship)

CRITICAL CONVENTION NOTE
   LUAD master_splits.csv censorship convention (VERIFIED by mean-OS test):
       censorship=0  -> DEATH observed at survival_months (event = 1)
       censorship=1  -> CENSORED / alive at survival_months (event = 0)

   So event = 1 - censorship. This is the SAME convention as GBMLGG but
   the OPPOSITE of KIRC. Mixing them up would invert Cox loss and yield
   c-index ~0.30 (worse than random).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .roi_splits      import RoiEntry
from .roi_splits_luad import (
    LUAD_BULKRNABERT_NPZ, LUAD_MASTER_CSV, LUAD_UNI2H_DIR,
)


# ---------------------------------------------------------------------------
# Gene-side loader
# ---------------------------------------------------------------------------
def load_luad_bulkrnabert_patient_dict() -> dict[str, np.ndarray]:
    """LUAD BulkRNABert NPZ -> {patient_id: (256,) float32}.

    Source: `luad_bulkrnabert_256d.npz` (517 patients × 256-d).
    Verified: all 450 PF LUAD cohort patients have entries here.
    """
    z = np.load(LUAD_BULKRNABERT_NPZ, allow_pickle=False)
    pids = z['patient_ids']
    emb  = z['embeddings']            # (517, 256)
    if emb.shape[1] != 256:
        raise ValueError(f"LUAD BulkRNABert dim mismatch: got {emb.shape[1]}, expected 256")
    out: dict[str, np.ndarray] = {}
    for i, p in enumerate(pids):
        out[str(p)] = emb[i].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Labels loader -- LUAD-SPECIFIC EVENT CONVENTION
# ---------------------------------------------------------------------------
def load_luad_patient_labels() -> dict[str, Tuple[int, float]]:
    """Returns {patient_id: (event, time)} for the 450 LUAD cohort.

    *** CRITICAL CONVENTION ***

    LUAD master_splits.csv stores `censorship`:
       censorship=0 -> DEATH observed (event = 1)
       censorship=1 -> CENSORED (alive) (event = 0)

    So event = 1 - censorship   (SAME as GBMLGG, OPPOSITE of KIRC).

    Verified empirically (in roi_splits_luad.py module docstring):
       censorship=0 patients mean OS = 24.8 months  (died earlier)
       censorship=1 patients mean OS = 32.8 months  (still alive)
    """
    master = pd.read_csv(LUAD_MASTER_CSV)
    required = {'case_id', 'survival_months', 'censorship'}
    if not required.issubset(master.columns):
        raise KeyError(
            f"LUAD master_splits.csv missing required columns. "
            f"Got: {list(master.columns)}"
        )
    # event = 1 - censorship (DEATH if censorship == 0)
    out: dict[str, Tuple[int, float]] = {}
    for _, row in master.iterrows():
        pid = row['case_id']
        event = int(1 - row['censorship'])
        time  = float(row['survival_months'])
        out[pid] = (event, time)
    return out


# ---------------------------------------------------------------------------
# Per-slide assembler
# ---------------------------------------------------------------------------
def assemble_luad_roi_batch(
    roi_entries: Sequence[RoiEntry],
    gene_dict: dict[str, np.ndarray],
    labels: dict[str, Tuple[int, float]],
    expected_gene_dim: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assemble a per-slide (= per-ROI) bimodal batch.

    Returns:
        X_img : (N, 1536) float32 -- per-slide UNI2-h embedding
        X_gene: (N, 256)  float32 -- patient gene vector broadcast to each slide
        events: (N,)      float32 -- broadcast patient event (LUAD: 1 - censorship)
        times : (N,)      float32 -- broadcast patient survival_months
        roi_ids: list[str] of length N -- slide_id stems
    """
    n = len(roi_entries)
    if n == 0:
        raise ValueError("assemble_luad_roi_batch called with empty roi_entries")

    first_pid = roi_entries[0].patient_id
    if first_pid not in gene_dict:
        raise KeyError(f"patient {first_pid} missing from LUAD gene_dict")
    gene_dim = gene_dict[first_pid].shape[0]
    if expected_gene_dim is not None and gene_dim != expected_gene_dim:
        raise ValueError(
            f"LUAD gene_dim mismatch: got {gene_dim}, expected {expected_gene_dim}"
        )

    X_img  = np.zeros((n, 1536),     dtype=np.float32)
    X_gene = np.zeros((n, gene_dim), dtype=np.float32)
    events = np.zeros(n,             dtype=np.float32)
    times  = np.zeros(n,             dtype=np.float32)
    roi_ids: list[str] = []

    for i, entry in enumerate(roi_entries):
        # Image: load the .pt file -- LUAD slides are (1536,) tensors
        # (the _patches.pt files in the same dir are dict format, NEVER used here)
        ten = torch.load(entry.uni2h_pt_path, map_location='cpu', weights_only=False)
        if isinstance(ten, dict):
            # Defensive: should never happen because slide_id mapping points
            # to non-patches .pt files. If it does, average patches.
            ten = ten['features'].mean(dim=0)
        if hasattr(ten, 'numpy'):
            arr = ten.float().numpy()
        else:
            arr = np.asarray(ten, dtype=np.float32)
        if arr.shape != (1536,):
            raise ValueError(
                f"LUAD UNI2-h tensor for {entry.roi_basename} has shape {arr.shape}, "
                f"expected (1536,)"
            )
        X_img[i] = arr

        pid = entry.patient_id
        if pid not in gene_dict:
            raise KeyError(f"patient {pid} missing from LUAD gene_dict")
        X_gene[i] = gene_dict[pid]

        if pid not in labels:
            raise KeyError(f"patient {pid} missing from LUAD labels")
        e, t = labels[pid]
        events[i] = float(e)
        times[i]  = float(t)

        roi_ids.append(entry.roi_basename)

    return X_img, X_gene, events, times, roi_ids


if __name__ == '__main__':
    """Smoke: load fold 1, verify shapes / distributions / no NaN."""
    import time
    from .roi_splits_luad import iter_luad_roi_folds, load_luad_cohort

    print("Loading shared LUAD dicts ...")
    gene_dict = load_luad_bulkrnabert_patient_dict()
    labels    = load_luad_patient_labels()
    cohort    = load_luad_cohort()
    print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
    print(f"  labels   : {len(labels)} patients")
    print(f"  cohort   : {len(cohort)} patients")

    # All 450 cohort patients must have gene + labels
    missing_gene = cohort - set(gene_dict.keys())
    missing_lab  = cohort - set(labels.keys())
    print(f"  cohort missing gene  : {len(missing_gene)}")
    print(f"  cohort missing label : {len(missing_lab)}")
    assert len(missing_gene) == 0, f"LUAD cohort has {len(missing_gene)} patients without gene"
    assert len(missing_lab) == 0,  f"LUAD cohort has {len(missing_lab)} patients without label"

    # Label distribution sanity (event = 1 - censorship)
    n_events = sum(1 for pid in cohort if labels[pid][0] == 1)
    n_censored = sum(1 for pid in cohort if labels[pid][0] == 0)
    print(f"\n  Label distribution (LUAD: event = 1 - censorship):")
    print(f"    events (deaths)  : {n_events}  ({100*n_events/len(cohort):.1f}%)")
    print(f"    censored (alive) : {n_censored}  ({100*n_censored/len(cohort):.1f}%)")
    print(f"    expected events  : 160 (PF master_splits.censorship == 0)")
    assert n_events == 160, f"expected 160 events, got {n_events}"

    # Load fold 1
    print("\nLoading fold 1 (PF fold 0) ROIs ...")
    fi, train_rois, test_rois = next(iter_luad_roi_folds())
    print(f"  fold {fi}: {len(train_rois)} train slides, {len(test_rois)} test slides")

    # Assemble train batch
    print("\nAssembling fold 1 train batch ...")
    t0 = time.time()
    X_img, X_gene, events, times, roi_ids = assemble_luad_roi_batch(
        train_rois, gene_dict, labels, expected_gene_dim=256
    )
    print(f"  X_img : {X_img.shape}  dtype={X_img.dtype}  "
          f"range=[{X_img.min():.3f}, {X_img.max():.3f}]")
    print(f"  X_gene: {X_gene.shape}  dtype={X_gene.dtype}  "
          f"range=[{X_gene.min():.3f}, {X_gene.max():.3f}]")
    print(f"  events: {int(events.sum())} dead / {int((events==0).sum())} censored")
    print(f"  times : range=[{times.min():.1f}, {times.max():.1f}] months")
    print(f"  load time : {time.time()-t0:.1f}s")

    print(f"\n  NaN check:")
    print(f"    X_img  NaN : {int(np.isnan(X_img).sum())}")
    print(f"    X_gene NaN : {int(np.isnan(X_gene).sum())}")
    print(f"    events NaN : {int(np.isnan(events).sum())}")
    print(f"    times  NaN : {int(np.isnan(times).sum())}")
    assert not np.isnan(X_img).any(),  "X_img has NaN"
    assert not np.isnan(X_gene).any(), "X_gene has NaN"

    print("\nSMOKE OK")
