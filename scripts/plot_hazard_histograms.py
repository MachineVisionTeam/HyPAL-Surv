"""
Per-cohort hazard-distribution histograms for HyPAL-Surv predictions.

PURPOSE
   Show that the model's PREDICTED HAZARDS actually separate short- and
   long-surviving patients along the prediction axis, BEFORE we apply any
   thresholding. This complements the KM curves (which apply a median
   split) by showing the underlying score distribution directly.

   Template: Pathomic Fusion Fig 3A (glioma) and Fig 4 (CCRCC).
     - Red histogram   = short-surviving patients (died before cutoff)
     - Blue histogram  = long-surviving patients  (alive at cutoff)
     - Density-normalized overlay so the two groups are visually comparable
     - X-axis: z-scored hazard (per cohort) so the visual scale matches
       across cohorts and the cohort-mean is 0 by construction.

CLASSIFICATION RULES (the honest part)
   For each patient with (time, event) and per-cohort cutoff T_c:
     SHORT survivor  : event == 1  AND  time <  T_c   (death observed early)
     LONG survivor   : time >= T_c                    (known alive at T_c)
     CENSORED-EARLY  : event == 0  AND  time <  T_c   -> EXCLUDED
                                                         from the figure
                                                         (we don't know
                                                         if they lived past
                                                         the cutoff)

   This matches Pathomic Fusion Fig 3A / Fig 4 exactly. We do NOT count a
   patient who was censored at month 24 (cutoff 60 months) as a "short
   survivor" -- they might still be alive.

PER-COHORT CUTOFF CONVENTIONS (clinically motivated)
   GBMLGG  : 5-year   (60 mo)   -- PF Fig 3A glioma convention
   KIRC    : 3.5-year (42 mo)   -- PF Fig 4 CCRCC convention
   LUAD    : 3-year   (36 mo)   -- LUAD 5-yr survival is ~22%; 3-yr is
                                   the more clinically informative split
   UCEC    : 5-year   (60 mo)   -- UCEC 5-yr survival is ~80%; this is
                                   the standard reference horizon

OUTPUT
   results/figures/fig_hazard_hist_<cohort>.{png,pdf}    -- 4 per-cohort
   results/figures/fig_hazard_hist_grid.{png,pdf}        -- 2x2 overview

USAGE
   python -m stage6_visualization.plot_hazard_histograms
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reuse the per-cohort metadata + time-unit converter from plot_km_grid
# so we have ONE place where time-unit conventions live.
from stage6_visualization.plot_km_grid import COHORT_META, to_months

PROJ      = Path(__file__).resolve().parent.parent
PRED_DIR  = PROJ / "results" / "predictions"
FIG_DIR   = PROJ / "results" / "figures"

# Clinically motivated cutoffs (months). See module docstring for rationale.
CUTOFF_MONTHS: dict[str, float] = {
    "gbmlgg": 60.0,
    "kirc"  : 42.0,
    "luad"  : 36.0,
    "ucec"  : 60.0,
}

# Pathomic Fusion Fig 3A / Fig 4 visual conventions
COLOR_SHORT  = "#C62828"   # red  -- short survivors (died early)
COLOR_LONG   = "#1565C0"   # blue -- long survivors (alive at cutoff)
ALPHA_FILL   = 0.45
N_BINS       = 30


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def classify_patients(
    times_months: np.ndarray,
    events: np.ndarray,
    cutoff_months: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split patients into SHORT, LONG, and EXCLUDED boolean masks.

    SHORT     : event == 1 AND time <  cutoff   (observed death before cutoff)
    LONG      : time >= cutoff                  (known alive at cutoff)
    EXCLUDED  : event == 0 AND time <  cutoff   (censored before cutoff;
                                                  outcome uncertain)
    """
    times_months = np.asarray(times_months, dtype=np.float64)
    events       = np.asarray(events,       dtype=np.int64)
    short_mask   = (events == 1) & (times_months <  cutoff_months)
    long_mask    =                (times_months >= cutoff_months)
    excluded     = (events == 0) & (times_months <  cutoff_months)
    # Sanity: every patient is in exactly one of the 3 sets.
    n_assigned = int(short_mask.sum() + long_mask.sum() + excluded.sum())
    if n_assigned != len(times_months):
        raise RuntimeError(
            f"classify_patients lost rows: {n_assigned} vs {len(times_months)}"
        )
    return short_mask, long_mask, excluded


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-std effect size between two 1-D arrays.

    d = (mean(a) - mean(b)) / s_pooled,  with s_pooled = sqrt((var_a + var_b)/2)
    Positive d => group a has higher mean than group b (in our convention,
                  positive d means short survivors have higher hazard, the
                  desired direction).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    s_pooled = np.sqrt(0.5 * (a.var(ddof=1) + b.var(ddof=1)))
    if s_pooled <= 0:
        return float("nan")
    return float((a.mean() - b.mean()) / s_pooled)


def load_cohort(dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load per-patient (z_hazard, time_in_months, event) for one cohort.

    Z-scores the cohort's hazards using its own population statistics so the
    plot's x-axis is cohort-mean-centered and unit-variance.
    """
    src = PRED_DIR / f"{dataset}_per_patient.csv"
    df  = pd.read_csv(src)
    meta = COHORT_META[dataset]
    times_months = to_months(df["time"].values, meta["time_unit"])
    events       = df["event"].values.astype(int)
    raw_haz      = df["mean_hazard"].values.astype(np.float64)
    # Per-cohort z-score (avoid divide-by-zero if variance is degenerate)
    mu  = raw_haz.mean()
    sig = raw_haz.std(ddof=0)
    if sig <= 0:
        raise RuntimeError(
            f"[{dataset}] hazard std is zero -- predictions are constant?"
        )
    z_haz = (raw_haz - mu) / sig
    return z_haz, times_months, events


def plot_one_panel(ax: plt.Axes, dataset: str) -> dict:
    z_haz, times_months, events = load_cohort(dataset)
    cutoff = CUTOFF_MONTHS[dataset]
    short_mask, long_mask, excluded_mask = classify_patients(
        times_months, events, cutoff,
    )

    n_short    = int(short_mask.sum())
    n_long     = int(long_mask.sum())
    n_excluded = int(excluded_mask.sum())

    # If a group is empty we still want a useful panel. Guard the histogram
    # and the effect-size calculation.
    has_short = n_short >= 1
    has_long  = n_long  >= 1

    # Shared bin edges across both groups so the histograms align
    if has_short or has_long:
        z_all = np.concatenate([
            z_haz[short_mask] if has_short else np.empty(0),
            z_haz[long_mask]  if has_long  else np.empty(0),
        ])
        bins = np.linspace(z_all.min() - 0.05, z_all.max() + 0.05, N_BINS + 1)
    else:
        bins = np.linspace(-3, 3, N_BINS + 1)

    if has_long:
        ax.hist(z_haz[long_mask],  bins=bins, density=True,
                color=COLOR_LONG,  alpha=ALPHA_FILL,
                edgecolor=COLOR_LONG, linewidth=0.7,
                label=f"Long survivor (n={n_long})")
    if has_short:
        ax.hist(z_haz[short_mask], bins=bins, density=True,
                color=COLOR_SHORT, alpha=ALPHA_FILL,
                edgecolor=COLOR_SHORT, linewidth=0.7,
                label=f"Short survivor (n={n_short})")

    # Group means as vertical reference lines
    if has_long:
        mu_long = z_haz[long_mask].mean()
        ax.axvline(mu_long, color=COLOR_LONG,
                   linestyle="--", linewidth=1.5, alpha=0.85)
    if has_short:
        mu_short = z_haz[short_mask].mean()
        ax.axvline(mu_short, color=COLOR_SHORT,
                   linestyle="--", linewidth=1.5, alpha=0.85)

    # Effect size between groups (Cohen's d; positive = short > long, desired)
    d = (cohens_d(z_haz[short_mask], z_haz[long_mask])
         if (has_short and has_long) else float("nan"))

    # Cutoff annotation in years for readability
    cutoff_years = cutoff / 12.0
    meta = COHORT_META[dataset]
    title = (
        f"{meta['title']}  hazard distribution  "
        f"(cutoff = {cutoff_years:.1f} years)"
    )
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted hazard (z-scored per cohort)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.grid(alpha=0.25)

    # Stats annotation box
    info = (
        f"short / long / excluded:\n"
        f"    {n_short}  /  {n_long}  /  {n_excluded}\n"
        f"Cohen's d (short - long) = {d:.2f}"
        if not np.isnan(d)
        else f"short / long / excluded:\n    {n_short} / {n_long} / {n_excluded}"
    )
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=9, va="top",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="0.5",
                      boxstyle="round,pad=0.4"))

    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    return {
        "dataset":     dataset,
        "cutoff_mo":   cutoff,
        "n_short":     n_short,
        "n_long":      n_long,
        "n_excluded":  n_excluded,
        "mean_short":  float(z_haz[short_mask].mean()) if has_short else float("nan"),
        "mean_long":   float(z_haz[long_mask].mean())  if has_long  else float("nan"),
        "cohens_d":    d,
    }


def plot_single_cohort(dataset: str) -> dict | None:
    fig, ax = plt.subplots(figsize=(8, 6))
    try:
        s = plot_one_panel(ax, dataset)
    except FileNotFoundError as e:
        print(f"[{dataset}] SKIP: {e}")
        plt.close(fig)
        return None

    # No suptitle; keep the per-panel title which labels the cohort.
    plt.tight_layout()

    out_png = FIG_DIR / f"fig_hazard_hist_{dataset}.png"
    out_pdf = FIG_DIR / f"fig_hazard_hist_{dataset}.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"SAVED : {out_png}")
    print(f"SAVED : {out_pdf}")
    return s


def plot_grid(datasets=("gbmlgg", "kirc", "luad", "ucec")) -> list[dict]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
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
    out_png = FIG_DIR / "fig_hazard_hist_grid.png"
    out_pdf = FIG_DIR / "fig_hazard_hist_grid.pdf"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(out_pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"\nSAVED : {out_png}")
    print(f"SAVED : {out_pdf}")
    return summaries


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("=== PER-COHORT HAZARD HISTOGRAMS ===")
    per_cohort = []
    for ds in ("gbmlgg", "kirc", "luad", "ucec"):
        s = plot_single_cohort(ds)
        if s is not None:
            per_cohort.append(s)

    print("\n=== 2x2 OVERVIEW GRID ===")
    summaries = plot_grid()

    print("\nPER-COHORT SUMMARY")
    hdr = (f"  {'cohort':<8} {'cutoff_mo':>10} "
           f"{'n_short':>8} {'n_long':>7} {'n_excl':>7} "
           f"{'mean_short':>11} {'mean_long':>10} {'Cohens_d':>9}")
    print(hdr)
    for s in summaries:
        ms = ("n/a" if np.isnan(s["mean_short"]) else f"{s['mean_short']:.2f}")
        ml = ("n/a" if np.isnan(s["mean_long" ]) else f"{s['mean_long' ]:.2f}")
        cd = ("n/a" if np.isnan(s["cohens_d"  ]) else f"{s['cohens_d'  ]:.2f}")
        print(f"  {s['dataset']:<8} {s['cutoff_mo']:>10.0f} "
              f"{s['n_short']:>8d} {s['n_long']:>7d} {s['n_excluded']:>7d} "
              f"{ms:>11} {ml:>10} {cd:>9}")


if __name__ == "__main__":
    main()
