"""
KIRC sample-level data assembly.

Mirrors the API of `harness.roi_loader` (the GBMLGG version) so the runners
work for KIRC with a one-line import change. The two important differences:

  (1) BulkRNABert NPZ path:  kirc_bulkrnabert_256d.npz   (533 patients × 256-d)
  (2) Event convention:      event = censored  (KIRC), NOT 1 - censored (GBMLGG)

For each ROI we assemble the same 4-tuple as GBMLGG:
   x_image (1536,)  -- UNI2-h embedding of this ROI (loaded from .pt)
   x_gene  (256,)   -- patient's BulkRNABert vector, broadcast to all of that
                       patient's ROIs
   time    (float)  -- patient's OS_month
   event   (int)    -- patient's death event (= censored, NOT 1-censored)
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .roi_splits     import RoiEntry
from .roi_splits_kirc import KIRC_BULKRNABERT_NPZ, KIRC_PF_PKL


# ---------------------------------------------------------------------------
# Gene-side loader
# ---------------------------------------------------------------------------
def load_kirc_bulkrnabert_patient_dict() -> dict[str, np.ndarray]:
    """KIRC BulkRNABert NPZ -> {patient_id: (256,) float32}.

    Source: `kirc_bulkrnabert_256d.npz` (533 patients × 256-d).
    All 417 PF KIRC patients have entries here (verified earlier).
    """
    z = np.load(KIRC_BULKRNABERT_NPZ, allow_pickle=False)
    pids = z['patient_ids']
    emb  = z['embeddings']            # (533, 256)
    if emb.shape[1] != 256:
        raise ValueError(f"KIRC BulkRNABert dim mismatch: got {emb.shape[1]}, expected 256")
    out: dict[str, np.ndarray] = {}
    for i, p in enumerate(pids):
        out[str(p)] = emb[i].astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Labels loader -- KIRC-SPECIFIC EVENT CONVENTION
# ---------------------------------------------------------------------------
def load_kirc_patient_labels() -> dict[str, Tuple[int, float]]:
    """Returns {patient_id: (event, time)} for the 417 KIRC cohort.

    *** CRITICAL CONVENTION DIFFERENCE FROM GBMLGG ***

    KIRC convention (verified across all 1008 ROIs in PF fold 1):
       event = censored         (censored=1 -> death observed)
       time  = OS_month         (in months; range [0, 149] for KIRC)

    Whereas GBMLGG uses:
       event = 1 - censored     (censored=0 -> death observed)

    PF's KIRC pkl stores the same `e` array as `censored` -- they are
    identical to floating-point precision in every fold. Using the wrong
    convention (1 - censored) would produce c-index ~0.30 (worse than random).
    """
    with open(KIRC_PF_PKL, 'rb') as f:
        d = pickle.load(f)
    ad = d['all_dataset']
    if 'censored' not in ad.columns or 'OS_month' not in ad.columns:
        raise KeyError(
            f"PF KIRC pkl 'all_dataset' missing required columns. "
            f"Got: {list(ad.columns)[:8]}"
        )
    # KIRC: event = censored (NOT 1 - censored)
    events = ad['censored'].astype(int)
    times  = ad['OS_month'].astype(float)
    return {pid: (int(events.loc[pid]), float(times.loc[pid])) for pid in ad.index}


# ---------------------------------------------------------------------------
# Per-ROI assembler -- the central function (mirrors GBMLGG signature)
# ---------------------------------------------------------------------------
def assemble_kirc_roi_batch(
    roi_entries: Sequence[RoiEntry],
    gene_dict: dict[str, np.ndarray],
    labels: dict[str, Tuple[int, float]],
    expected_gene_dim: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Assemble a per-ROI bimodal batch from a list of RoiEntry objects.

    Returns:
        X_img : (N, 1536) float32 -- per-ROI UNI2-h embedding
        X_gene: (N, 256)  float32 -- patient gene vector broadcast to each ROI
        events: (N,)      float32 -- broadcast patient event (KIRC: censored)
        times : (N,)      float32 -- broadcast patient OS_month
        roi_ids: list[str] of length N -- the .stem of each .pt path
    """
    n = len(roi_entries)
    if n == 0:
        raise ValueError("assemble_kirc_roi_batch called with empty roi_entries")

    # Probe gene dim from the first ROI's patient
    first_pid = roi_entries[0].patient_id
    if first_pid not in gene_dict:
        raise KeyError(f"patient {first_pid} missing from KIRC gene_dict")
    gene_dim = gene_dict[first_pid].shape[0]
    if expected_gene_dim is not None and gene_dim != expected_gene_dim:
        raise ValueError(
            f"KIRC gene_dim mismatch: got {gene_dim}, expected {expected_gene_dim}"
        )

    X_img  = np.zeros((n, 1536),     dtype=np.float32)
    X_gene = np.zeros((n, gene_dim), dtype=np.float32)
    events = np.zeros(n,             dtype=np.float32)
    times  = np.zeros(n,             dtype=np.float32)
    roi_ids: list[str] = []

    for i, entry in enumerate(roi_entries):
        # Image: load the .pt -- single (1536,) tensor per ROI
        ten = torch.load(entry.uni2h_pt_path, map_location='cpu', weights_only=False)
        if isinstance(ten, dict):
            ten = ten['features'].mean(dim=0)
        if hasattr(ten, 'numpy'):
            arr = ten.float().numpy()
        else:
            arr = np.asarray(ten, dtype=np.float32)
        if arr.shape != (1536,):
            raise ValueError(
                f"KIRC UNI2-h tensor for {entry.roi_basename} has shape {arr.shape}, "
                f"expected (1536,)"
            )
        X_img[i] = arr

        # Gene: broadcast patient vector
        pid = entry.patient_id
        if pid not in gene_dict:
            raise KeyError(f"patient {pid} missing from KIRC gene_dict")
        X_gene[i] = gene_dict[pid]

        # Labels: broadcast patient (event, time)
        if pid not in labels:
            raise KeyError(f"patient {pid} missing from KIRC labels")
        e, t = labels[pid]
        events[i] = float(e)
        times[i]  = float(t)

        roi_ids.append(entry.roi_basename)

    return X_img, X_gene, events, times, roi_ids


if __name__ == '__main__':
    """Smoke: load fold 1, assemble train batch, verify shapes + distributions."""
    import time
    from .roi_splits_kirc import iter_kirc_roi_folds, load_kirc_cohort

    print("Loading shared KIRC dicts ...")
    gene_dict = load_kirc_bulkrnabert_patient_dict()
    labels    = load_kirc_patient_labels()
    cohort    = load_kirc_cohort()
    print(f"  gene_dict: {len(gene_dict)} patients, dim={next(iter(gene_dict.values())).shape[0]}")
    print(f"  labels   : {len(labels)} patients")
    print(f"  cohort   : {len(cohort)} patients")

    # All 417 cohort patients must have gene + labels
    missing_gene = cohort - set(gene_dict.keys())
    missing_lab  = cohort - set(labels.keys())
    print(f"  cohort patients missing gene : {len(missing_gene)}")
    print(f"  cohort patients missing label: {len(missing_lab)}")
    assert len(missing_gene) == 0, f"KIRC cohort has {len(missing_gene)} patients without gene"
    assert len(missing_lab) == 0,  f"KIRC cohort has {len(missing_lab)} patients without label"

    # Label distribution sanity
    n_events = sum(1 for pid in cohort if labels[pid][0] == 1)
    n_censored = sum(1 for pid in cohort if labels[pid][0] == 0)
    print(f"\n  Label distribution (KIRC convention: event = censored):")
    print(f"    events (deaths)   : {n_events}  ({100*n_events/len(cohort):.1f}%)")
    print(f"    censored (alive)  : {n_censored}  ({100*n_censored/len(cohort):.1f}%)")
    print(f"    expected events   : 135 (~32.4%)")

    # Load fold 1
    print("\nLoading fold 1 ROIs ...")
    fi, train_rois, test_rois = next(iter_kirc_roi_folds())
    print(f"  fold {fi}: {len(train_rois)} train ROIs, {len(test_rois)} test ROIs")

    # Assemble train batch -- THIS IS THE BIG IO STEP
    print("\nAssembling fold 1 train batch ...")
    t0 = time.time()
    X_img, X_gene, events, times, roi_ids = assemble_kirc_roi_batch(
        train_rois, gene_dict, labels, expected_gene_dim=256
    )
    print(f"  X_img : {X_img.shape}  dtype={X_img.dtype}  "
          f"range=[{X_img.min():.3f}, {X_img.max():.3f}]")
    print(f"  X_gene: {X_gene.shape}  dtype={X_gene.dtype}  "
          f"range=[{X_gene.min():.3f}, {X_gene.max():.3f}]")
    print(f"  events: {int(events.sum())} dead / {int((events==0).sum())} censored")
    print(f"  times : range=[{times.min():.1f}, {times.max():.1f}] months")
    print(f"  load time : {time.time()-t0:.1f}s")

    # Verify broadcast
    pid_first = train_rois[0].patient_id
    same_pid_rows = [i for i, e in enumerate(train_rois) if e.patient_id == pid_first]
    if len(same_pid_rows) > 1:
        i0, i1 = same_pid_rows[0], same_pid_rows[1]
        same_gene  = np.allclose(X_gene[i0], X_gene[i1])
        same_event = events[i0] == events[i1]
        same_time  = times [i0] == times [i1]
        print(f"\n  Broadcast check: patient {pid_first} has {len(same_pid_rows)} ROIs")
        print(f"    gene identical : {same_gene}")
        print(f"    event identical: {same_event}")
        print(f"    time identical : {same_time}")
        assert same_gene and same_event and same_time, "broadcast invariant violated"

    # Check NO NaN in any column
    print(f"\n  NaN check:")
    print(f"    X_img  NaN : {int(np.isnan(X_img).sum())}")
    print(f"    X_gene NaN : {int(np.isnan(X_gene).sum())}")
    print(f"    events NaN : {int(np.isnan(events).sum())}")
    print(f"    times  NaN : {int(np.isnan(times).sum())}")
    assert not np.isnan(X_img).any(),  "X_img has NaN"
    assert not np.isnan(X_gene).any(), "X_gene has NaN"
    assert not np.isnan(events).any(), "events has NaN"
    assert not np.isnan(times).any(),  "times has NaN"

    print("\nSMOKE OK")
