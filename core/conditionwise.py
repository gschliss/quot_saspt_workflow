"""
core/conditionwise.py

Per-condition analysis. Branches on ``settings["io"]["mode"]``:

- slowSPT: just a Kaplan-Meier trajectory survival curve straight from this
  condition's raw ``_traj.csv`` files (no SASPT/rolling-window output exists
  to aggregate in slowSPT -- see core/filewise.py).
- fastSPT: the full pipeline -- aggregate per-file CSVs, QC/outlier removal,
  diffusion histogram and bar-graph plots, HMM fitting, a per-HMM-state
  Kaplan-Meier dwell-time survival curve, bleaching analysis, and movie
  generation.

Public API
----------
run_conditionwise(condition, settings)
    Run the mode-appropriate condition-level analysis for one condition.
"""

from __future__ import annotations

import os
import re
import glob
import pickle

import pandas as pd

import core  # noqa: F401
from core.utils import aggregate_csv
from core import plots


def run_conditionwise(condition: str, settings: dict) -> None:
    """Run condition-level analysis for *condition*, per ``settings["io"]["mode"]``.

    Parameters
    ----------
    condition:
        Condition label string (matches the prefix of per-file CSV names).
    settings:
        Fully-resolved workflow settings dict.
    """
    print(f"\n{'='*60}")
    print(f"Condition-wise analysis: {condition} (mode={settings['io']['mode']})")
    print(f"{'='*60}")

    if settings["io"]["mode"] == "slowSPT":
        _run_conditionwise_slowSPT(condition, settings)
    else:
        _run_conditionwise_fastSPT(condition, settings)

    print(f"\n  Condition-wise analysis complete: {condition}")


def _run_conditionwise_slowSPT(condition: str, settings: dict) -> None:
    """slowSPT: Kaplan-Meier survival curve straight from raw _traj.csv, done.

    Writes a minimal marker PKL (no MLE/posterior data exists in slowSPT) to
    ``plotting_pkls/<condition>.pkl`` purely so the Snakemake DAG has the
    output file it expects; :func:`~core.aggregate.run_aggregate` skips
    conditions whose PKL looks like this.

    If ``settings['survival']['background_condition']`` is set (and this
    isn't that condition), also writes a background-corrected companion
    curve -- see :func:`~core.plots.plot_survival_background_corrected`.
    """
    traj_csvs = sorted(
        glob.glob(os.path.join(settings["io"]["traj_directory"], f"{condition}*_traj.csv"))
    )
    plots.plot_survival_kaplan_meier(traj_csvs, settings, condition, showPlot=False)

    background_condition = settings["survival"]["background_condition"]
    if background_condition is not None and condition != background_condition:
        background_traj_csvs = sorted(
            glob.glob(os.path.join(settings["io"]["traj_directory"], f"{background_condition}*_traj.csv"))
        )
        if background_traj_csvs:
            plots.plot_survival_background_corrected(
                traj_csvs, background_traj_csvs, settings, condition, background_condition,
                showPlot=False,
            )
        else:
            print(
                f"  [survival corrected] background condition {background_condition!r} has "
                f"no _traj.csv files; skipping correction for {condition}"
            )

    plotting_pkl_dir = os.path.join(settings["io"]["plot_directory"], "plotting_pkls")
    os.makedirs(plotting_pkl_dir, exist_ok=True)
    plotting_pkl = os.path.join(plotting_pkl_dir, f"{condition}.pkl")
    with open(plotting_pkl, "wb") as fh:
        pickle.dump({"mode": "slowSPT"}, fh)


def _run_conditionwise_fastSPT(condition: str, settings: dict) -> None:
    """fastSPT: full pipeline (SASPT aggregation, HMM, movies, plotting).

    Steps performed (in order):
    1. Aggregate posterior, MLE, and rollingMLE CSVs for this condition.
    2. QC / outlier removal using KS-distance on posterior distributions.
    3. Generate per-condition histogram + stacked bar-graph PDF.
    4. Fit a Gaussian HMM to the rolling-window MLE diffusion rates
       (writes ``HMM_output/<condition>_HMM.txt``).
    5. Plot a Kaplan-Meier survival curve per HMM state (dwell time in each
       diffusive population before transitioning out), one subplot per
       population.
    6. Save a plotting PKL for use by :func:`~core.aggregate.run_aggregate`.
    7. Measure bleaching curves and plot them.
    8. Generate overlay MP4 movies for a random sample of trajectories.
    """
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
    # 4. Fit HMM (state assignment written to HMM_output/<condition>_HMM.txt,
    #    and posterior_pop labels feed the per-state survival curve in step 5)
    # ------------------------------------------------------------------
    rollingMLE = plots.fitHMM(rollingMLE, settings, condition, min_obs=7)
    print("Finished HMM")

    # ------------------------------------------------------------------
    # 5. Plot a Kaplan-Meier survival curve per HMM-assigned diffusive state.
    # ------------------------------------------------------------------
    plots.plot_survival_by_hmm_state(rollingMLE, settings, condition, showPlot=False)

    # ------------------------------------------------------------------
    # 6. Save plotting PKL
    # ------------------------------------------------------------------
    plotting_pkl_dir = os.path.join(settings["io"]["plot_directory"], "plotting_pkls")
    os.makedirs(plotting_pkl_dir, exist_ok=True)

    plotting_pkl = os.path.join(plotting_pkl_dir, f"{condition}.pkl")
    with open(plotting_pkl, "wb") as fh:
        pickle.dump(
            {
                "mode": "fastSPT",
                "MLE_df": MLE_df,
                "posterior_plot_df": posterior_plot_df,
                "rollingMLE": rollingMLE,
            },
            fh,
        )

    # ------------------------------------------------------------------
    # 7. Bleaching curves
    # ------------------------------------------------------------------
    bc = plots.aggregate_bleaching_data(rollingMLE, settings, n_particles=250)
    plots.plot_bleaching_curves(settings, bc)

    # ------------------------------------------------------------------
    # 8. Generate movies
    # ------------------------------------------------------------------
    plots.generate_movies(rollingMLE, settings)
