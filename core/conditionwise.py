"""
core/conditionwise.py

Per-condition analysis: aggregate per-file CSVs, QC/outlier removal,
diffusion histogram and bar-graph plots, HMM fitting, state survival
curves, bleaching analysis, and movie generation.

Public API
----------
run_conditionwise(condition, settings)
    Run the full condition-level analysis pipeline for one condition.
"""

from __future__ import annotations

import os
import re
import pickle

import pandas as pd

import core  # noqa: F401
from core.utils import aggregate_csv
from core import plots


def run_conditionwise(condition: str, settings: dict) -> None:
    """Run all condition-level analyses for *condition*.

    Steps performed (in order):
    1. Aggregate posterior, MLE, and rollingMLE CSVs for this condition.
    2. QC / outlier removal using KS-distance on posterior distributions.
    3. Generate per-condition histogram + stacked bar-graph PDF.
    4. Plot the empirical state-survival curve from the naive
       (threshold-only) population assignment, before HMM refinement.
    5. Fit a Gaussian HMM to the rolling-window MLE diffusion rates.
    6. Plot the empirical state-survival curve again, now from the
       HMM-refined population assignment, for comparison against step 4.
    7. Save a plotting PKL for use by :func:`~core.aggregate.run_aggregate`.
    8. Measure bleaching curves and plot them.
    9. Generate overlay MP4 movies for a random sample of trajectories.

    Parameters
    ----------
    condition:
        Condition label string (matches the prefix of per-file CSV names).
    settings:
        Fully-resolved workflow settings dict.
    """
    print(f"\n{'='*60}")
    print(f"Condition-wise analysis: {condition}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Aggregate per-file CSVs
    # ------------------------------------------------------------------
    posterior_df = aggregate_csv(settings["io"]["post_directory"], f"{condition}*.csv")
    posterior_df["condition"] = [
        re.sub(r"_[0-9]*_traj\.csv$", "", f) for f in posterior_df["file"]
    ]

    MLE_df = aggregate_csv(settings["io"]["MLE_directory"], f"{condition}*.csv")
    MLE_df["file_basename"] = [os.path.basename(f) for f in MLE_df["source_file"]]
    MLE_df["condition"] = [
        re.sub(r"_[0-9]*_traj\.csv$", "", f) for f in MLE_df["file_basename"]
    ]

    rollingMLE = aggregate_csv(
        settings["io"]["rolling_window_directory"], f"{condition}*.csv"
    )
    rollingMLE["file_basename"] = [
        re.sub(r"::[0-9]*$", ".csv", traj) for traj in rollingMLE["ur_trajectory"]
    ]
    rollingMLE["condition"] = [
        re.sub(r"_[0-9]*_traj\.csv$", "", traj) for traj in rollingMLE["file_basename"]
    ]

    # ------------------------------------------------------------------
    # 2. QC / outlier removal
    # ------------------------------------------------------------------
    outliers = plots.find_outlier_filenames(
        posterior_df, mult_on_sd=settings["plot"]["mult_on_sd"]
    )

    print(f"\nAllowed mult_on_sd: {settings['plot']['mult_on_sd']}")
    print("Removing outliers:")
    for ol in outliers:
        print(f"    {ol}")
    print()

    posterior_df = posterior_df[~posterior_df["file"].isin(outliers)]
    MLE_df = MLE_df[~MLE_df["file_basename"].isin(outliers)]
    rollingMLE = rollingMLE[~rollingMLE["file_basename"].isin(outliers)]

    # ------------------------------------------------------------------
    # 3. Generate histogram and bar-graph plots
    # ------------------------------------------------------------------
    posterior_plot_df = plots.generate_posterior_plotting_df(posterior_df)
    plots.generate_plots(
        MLE_df, posterior_plot_df, posterior_df, settings, showPlot=False
    )

    # ------------------------------------------------------------------
    # 4. Plot survival curves from the naive (threshold-only) population
    #    assignment, BEFORE any HMM state refinement — a baseline to
    #    compare against the HMM-refined curves plotted after step 5.
    # ------------------------------------------------------------------
    rollingMLE = plots.compute_naive_pop(rollingMLE, settings, min_obs=7)
    plots.plot_survival_HMM(
        rollingMLE, settings, condition, showPlot=False, min_run=0,
        pop_column="naive_pop",
    )

    # ------------------------------------------------------------------
    # 5. Fit HMM
    # ------------------------------------------------------------------
    rollingMLE = plots.fitHMM(rollingMLE, settings, condition, min_obs=7)

    # Handle slow-SPT data where rolling-window analysis spans >1 s frames:
    # assign all detections to population 0 to avoid HMM artefacts.
    if settings["saspt"]["frame_interval"] > 1:
        rollingMLE["posterior_pop"] = 0
        rollingMLE["naive_pop"] = 0

    # ------------------------------------------------------------------
    # 6. Plot empirical state survival from the HMM-refined population
    #    assignment, for comparison against the naive curves from step 4.
    # ------------------------------------------------------------------
    plots.plot_survival_HMM(
        rollingMLE, settings, condition, showPlot=False, min_run=0,
        pop_column="posterior_pop",
    )

    print("Finished HMM")

    # ------------------------------------------------------------------
    # 7. Save plotting PKL
    # ------------------------------------------------------------------
    plotting_pkl_dir = os.path.join(settings["io"]["plot_directory"], "plotting_pkls")
    os.makedirs(plotting_pkl_dir, exist_ok=True)

    plotting_pkl = os.path.join(plotting_pkl_dir, f"{condition}.pkl")
    with open(plotting_pkl, "wb") as fh:
        pickle.dump(
            {
                "MLE_df": MLE_df,
                "posterior_plot_df": posterior_plot_df,
                "rollingMLE": rollingMLE,
            },
            fh,
        )

    # ------------------------------------------------------------------
    # 8. Bleaching curves
    # ------------------------------------------------------------------
    bc = plots.aggregate_bleaching_data(rollingMLE, settings, n_particles=250)
    plots.plot_bleaching_curves(settings, bc)

    # ------------------------------------------------------------------
    # 9. Generate movies
    # ------------------------------------------------------------------
    plots.generate_movies(rollingMLE, settings)

    print(f"\n  Condition-wise analysis complete: {condition}")
