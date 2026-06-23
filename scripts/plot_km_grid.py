"""
4-panel Kaplan-Meier grid for HyPAL-Surv on TCGA-GBMLGG/KIRC/LUAD/UCEC.

INPUT  (produced by aggregate_predictions.py):
   results/predictions/<dataset>_per_patient.csv
   columns: patient_id, mean_hazard, std_hazard, n_observations, time, event

OUTPUT:
   results/figures/fig_km_grid.png
   results/figures/fig_km_grid.pdf

LAYOUT (MOTCat Fig 4 template)
   2 x 2 grid, one panel per cohort, identical visual conventions:
     - Green curve = low risk (hazard below median)
     - Red curve   = high risk (hazard above median)
     - Shaded 95% confidence bands per stratum
     - Log-rank p-value annotated in the panel
     - X-axis label: time in months
     - Y-axis label: survival probability
     - Title: TCGA-<dataset> + (n_patients, n_events)

TIME UNIT CONVENTION
   GBMLGG and KIRC use PF's .pkl convention storing survival time in DAYS.
   LUAD and UCEC use master_splits.csv storing survival_months in MONTHS.
   We convert all times to MONTHS for display so the 4 panels share a unit.

USAGE
   python -m stage6_visualization.plot_km_grid
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

PROJ      = Path(__file__).resolve().parent.parent
PRED_DIR  = PROJ / "results" / "predictions"
FIG_DIR   = PROJ / "results" / "figures"

# Per-cohort metadata: title + time unit
COHORT_META = {
    "gbmlgg": {
        "title":     "TCGA-GBMLGG",
        "time_unit": "days",     # PF pkl convention (verified: max ~6423d ~17.6y)
    },
    "kirc": {
        "title":     "TCGA-KIRC",
        "time_unit": "months",   # PF KIRC_st_1.pkl stores months (verified: max ~149mo ~12.5y)
    },
    "luad": {
        "title":     "TCGA-LUAD",
        "time_unit": "months",   # master_splits convention (verified: max ~238mo)
    },
    "ucec": {
        "title":     "TCGA-UCEC",
        "time_unit": "months",   # master_splits convention (verified: max ~225mo)
    },
}

# Visual style (MOTCat Fig 4)
COLOR_LOW   = "#2E7D32"   # dark green
COLOR_HIGH  = "#C62828"   # dark red
ALPHA_BAND  = 0.18


def to_months(time_arr: np.ndarray, unit: str) -> np.ndarray:
    """Convert raw time array to months."""
    if unit == "months":
        return time_arr.astype(np.float64)
    if unit == "days":
        return time_arr.astype(np.float64) / (365.25 / 12.0)  # days -> months
    raise ValueError(f"unknown time unit {unit!r}")


def plot_one_panel(ax: plt.Axes, dataset: str) -> dict:
    src = PRED_DIR / f"{dataset}_per_patient.csv"
    df = pd.read_csv(src)
    meta = COHORT_META[dataset]
    times = to_months(df["time"].values, meta["time_unit"])
    events = df["event"].values.astype(int)
    haz    = df["mean_hazard"].values

    # Median split on hazard
    cut = np.median(haz)
    is_high = haz >= cut

    times_low,  events_low  = times[~is_high], events[~is_high]
    times_high, events_high = times[ is_high], events[ is_high]

    # Log-rank
    lr = logrank_test(times_low, times_high, events_low, events_high)
    p_value = lr.p_value

    # KM fits with 95% CI bands (default)
    kmf_low  = KaplanMeierFitter(label=f"Low risk (n={len(times_low)})")
    kmf_high = KaplanMeierFitter(label=f"High risk (n={len(times_high)})")
    kmf_low .fit(times_low,  event_observed=events_low)
    kmf_high.fit(times_high, event_observed=events_high)

    kmf_low .plot_survival_function(ax=ax, color=COLOR_LOW,
                                    ci_show=True, ci_alpha=ALPHA_BAND)
    kmf_high.plot_survival_function(ax=ax, color=COLOR_HIGH,
                                    ci_show=True, ci_alpha=ALPHA_BAND)

    # Title and labels
    n_pat   = len(df)
    n_event = int(events.sum())
    ax.set_title(f"{meta['title']}  (n={n_pat}, events={n_event})",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Time (months)", fontsize=10)
    ax.set_ylabel("Survival probability", fontsize=10)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.25)

    # P-value annotation (MOTCat style)
    if p_value < 1e-4:
        p_str = f"log-rank p = {p_value:.2e}"
    else:
        p_str = f"log-rank p = {p_value:.4f}"
    ax.text(0.04, 0.05, p_str, transform=ax.transAxes,
            fontsize=10, fontweight="bold",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5",
                      boxstyle="round,pad=0.3"))

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    return {
        "dataset":  dataset,
        "n":        n_pat,
        "n_events": n_event,
        "p_value":  float(p_value),
        "median_hazard": float(cut),
        "n_low":   int((~is_high).sum()),
        "n_high":  int(( is_high).sum()),
        "n_low_events":  int(events_low.sum()),
        "n_high_events": int(events_high.sum()),
    }


def plot_single_cohort(dataset: str) -> dict | None:
    """One standalone KM figure for the given cohort -- larger, cleaner,
    intended for per-cohort use in the manuscript or slides."""
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        s = plot_one_panel(ax, dataset)
    except FileNotFoundError as e:
        print(f"[{dataset}] SKIP: {e}")
        plt.close(fig)
        return None

    # No suptitle; keep the per-panel title which labels the cohort + n.
    plt.tight_layout()
    out_png = FIG_DIR / f"fig_km_{dataset}.png"
    out_pdf = FIG_DIR / f"fig_km_{dataset}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED : {out_png}")
    print(f"SAVED : {out_pdf}")
    return s


def plot_grid(datasets=("gbmlgg", "kirc", "luad", "ucec")) -> list[dict]:
    """2x2 grid combining all 4 cohorts (overview figure)."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    layout = dict(zip(datasets, axes.flat))
    summaries = []
    for dataset, ax in layout.items():
        try:
            s = plot_one_panel(ax, dataset)
            summaries.append(s)
        except FileNotFoundError as e:
            ax.text(0.5, 0.5, f"{dataset} predictions\nnot found",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(dataset)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = FIG_DIR / "fig_km_grid.png"
    out_pdf = FIG_DIR / "fig_km_grid.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"\nSAVED : {out_png}")
    print(f"SAVED : {out_pdf}")
    return summaries


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1) ONE STANDALONE FIGURE PER COHORT
    print("=== PER-COHORT KM FIGURES ===")
    per_cohort = []
    for ds in ("gbmlgg", "kirc", "luad", "ucec"):
        s = plot_single_cohort(ds)
        if s is not None:
            per_cohort.append(s)

    # 2) ONE COMBINED 2x2 GRID (overview)
    print("\n=== 2x2 OVERVIEW GRID ===")
    summaries = plot_grid()

    # 3) Console summary table
    print("\nPER-COHORT SUMMARY")
    print(f"  {'cohort':<8} {'n_pat':>6} {'events':>7} "
          f"{'n_low':>6} {'n_low_e':>8} {'n_high':>7} {'n_high_e':>9} "
          f"{'p_value':>12}")
    for s in summaries:
        print(f"  {s['dataset']:<8} {s['n']:>6d} {s['n_events']:>7d} "
              f"{s['n_low']:>6d} {s['n_low_events']:>8d} "
              f"{s['n_high']:>7d} {s['n_high_events']:>9d} "
              f"{s['p_value']:>12.2e}")


if __name__ == "__main__":
    main()
