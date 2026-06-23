"""
Aggregate per-fold-per-seed HyPAL-Surv predictions into one CSV per cohort,
keyed on patient_id. Output is the input to the KM plotting script.

INPUT  (produced by run_stage5_mobald.py --save_predictions <dir>):
   results/predictions/<dataset>/<dataset>_fold{fold}_seed{seed}.csv
   columns: patient_id, roi_id, hazard, time, event, fold, seed

OUTPUT:
   results/predictions/<dataset>_per_patient.csv
   columns: patient_id, mean_hazard, std_hazard, n_observations, time, event

AGGREGATION RULES
   For each cohort:
     1) Concatenate all per-fold-per-seed CSVs.
     2) For ROI-level cohorts (GBMLGG, KIRC) average hazards across all rows
        of the same patient_id (each patient appears in exactly 1 test fold per
        seed, and may have multiple ROIs per fold). This collapses across both
        ROIs and seeds. n_observations records the number of (roi, seed) pairs.
     3) For slide-level cohorts (LUAD, UCEC, 1 slide/patient) the same logic
        works: each patient appears in some splits exactly once per split they
        land in, n_observations = number of splits the patient was test in.
     4) Sanity: every patient in the cohort must have at least 1 observation,
        and (time, event) values must agree across all that patient's rows
        (broadcast labels are constant per patient by construction).

USAGE
   python -m stage6_visualization.aggregate_predictions
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
PRED_DIR = PROJ / "results" / "predictions"


def aggregate_one(dataset: str) -> pd.DataFrame:
    src = PRED_DIR / dataset
    files = sorted(src.glob(f"{dataset}_fold*_seed*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No per-fold prediction CSVs found under {src} -- did the "
            f"stage5 refit run with --save_predictions {src}?"
        )
    print(f"\n[{dataset}] aggregating {len(files)} per-fold CSVs")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    print(f"  total rows           : {len(df):,}")
    print(f"  unique patients      : {df['patient_id'].nunique():,}")
    print(f"  unique (fold, seed)  : {df.groupby(['fold','seed']).ngroups:,}")

    # Sanity: (time, event) must agree within a patient_id
    bad = df.groupby('patient_id').agg(
        n_time=('time', 'nunique'),
        n_event=('event', 'nunique'),
    )
    inconsistent = bad[(bad.n_time != 1) | (bad.n_event != 1)]
    if len(inconsistent) > 0:
        raise RuntimeError(
            f"[{dataset}] {len(inconsistent)} patients have inconsistent "
            f"(time, event) across observations:\n{inconsistent.head()}"
        )

    # Mean / std hazard per patient
    out = df.groupby('patient_id').agg(
        mean_hazard=('hazard', 'mean'),
        std_hazard=('hazard', 'std'),
        n_observations=('hazard', 'count'),
        time=('time', 'first'),
        event=('event', 'first'),
    ).reset_index()

    # std_hazard is NaN if n_observations == 1; fill with 0 for those rows
    out['std_hazard'] = out['std_hazard'].fillna(0.0)

    print(f"  output rows          : {len(out):,}")
    print(f"  n_observations range : "
          f"{int(out.n_observations.min())} .. {int(out.n_observations.max())} "
          f"(median {int(out.n_observations.median())})")
    print(f"  events / total       : "
          f"{int(out.event.sum())} / {len(out)} ({100*out.event.mean():.1f}%)")
    print(f"  mean_hazard range    : "
          f"[{out.mean_hazard.min():.3f}, {out.mean_hazard.max():.3f}]")
    print(f"  time range           : "
          f"[{out.time.min():.1f}, {out.time.max():.1f}]")

    return out


def main() -> None:
    for dataset in ("gbmlgg", "kirc", "luad", "ucec"):
        try:
            out_df = aggregate_one(dataset)
        except FileNotFoundError as e:
            print(f"[{dataset}] SKIP: {e}")
            continue
        out_path = PRED_DIR / f"{dataset}_per_patient.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
