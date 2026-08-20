"""
core/aggregate.py

Cross-condition comparison: load per-condition plotting PKLs and produce
combined PDFs that overlay all conditions on the same axes.

Public API
----------
run_aggregate(settings)
    Called once after all condition-wise jobs have finished.
"""

from __future__ import annotations

import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import core  # noqa: F401
from core.plots import (
    _render_population_bar,
    plot_survival_kaplan_meier,
    plot_survival_kaplan_meier_overlay,
    plot_survival_kaplan_meier_overlay_corrected,
)
from core.utils import extract_metadata, group_files_by_metadata


def run_aggregate(settings: dict) -> None:
    """Produce cross-condition comparison plots.

    Loads the per-condition ``.pkl`` files written by
    :func:`~core.conditionwise.run_conditionwise` and generates combined PDFs:

    - ``<plot_directory>/aggregate_posterior.pdf`` — overlaid posterior
      diffusion-rate histograms, one curve per condition.
    - ``<plot_directory>/aggregate_MLE_bar.pdf`` — stacked bar chart
      comparing diffusion-population fractions across all conditions.

    Parameters
    ----------
    settings:
        Fully-resolved workflow settings dict.
    """
    plotting_pkl_dir = os.path.join(settings["io"]["plot_directory"], "plotting_pkls")
    pkl_files = sorted(glob.glob(os.path.join(plotting_pkl_dir, "*.pkl")))

    if not pkl_files:
        print("run_aggregate: no plotting PKL files found — skipping.")
        return

    # ------------------------------------------------------------------
    # Load all per-condition data
    # ------------------------------------------------------------------
    all_posterior_plot: dict[str, pd.DataFrame] = {}
    all_MLE: dict[str, pd.DataFrame] = {}
    skipped_slow: list[str] = []

    for pkl_path in pkl_files:
        condition = os.path.splitext(os.path.basename(pkl_path))[0]
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        # slowSPT conditions write a minimal marker PKL (no SASPT/MLE data
        # exists in slowSPT -- see core/conditionwise.py) purely to satisfy
        # the Snakemake DAG; they have nothing to contribute to these
        # SASPT-derived cross-condition plots, so skip them here instead of
        # crashing on the missing keys.
        if data.get("mode") == "slowSPT" or "posterior_plot_df" not in data:
            skipped_slow.append(condition)
            continue
        all_posterior_plot[condition] = data["posterior_plot_df"]
        all_MLE[condition] = data["MLE_df"]

    if skipped_slow:
        print(f"run_aggregate: skipping {len(skipped_slow)} slowSPT condition(s) — {skipped_slow}")

    conditions = list(all_posterior_plot.keys())
    if not conditions:
        print("run_aggregate: no fastSPT conditions with SASPT data found — skipping.")
        return
    print(f"run_aggregate: found {len(conditions)} conditions — {conditions}")

    # ------------------------------------------------------------------
    # Overlaid posterior histogram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))

    for condition, ppdf in all_posterior_plot.items():
        ax.plot(
            np.log10(ppdf["D"]),
            50 * ppdf["density"],
            linewidth=1.5,
            label=condition,
        )

    tick_positions = [-2, -1, 0, 1, 2]
    tick_labels = [
        f"{10**v:.2f}" if 10**v < 1 else f"{10**v:.0f}" for v in tick_positions
    ]
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlabel("Diffusion coefficient (μm²/s)")
    ax.set_ylabel("Probability density")
    ax.set_title("Aggregate posterior distributions")
    ax.set_ylim(settings["plot"]["ylim"][0], settings["plot"]["ylim"][1])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize="small", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(settings["io"]["plot_directory"], "aggregate_posterior.pdf")
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ------------------------------------------------------------------
    # Aggregate MLE stacked bar chart
    # ------------------------------------------------------------------
    bin_labels = settings["plot"]["barGraphLabels"]
    bar_data = pd.DataFrame(index=bin_labels)

    for condition, mle_df in all_MLE.items():
        counts, _ = np.histogram(mle_df["MLE_D"], bins=settings["plot"]["barGraphBreaks"])
        bar_data[condition] = counts

    bar_data = bar_data.T
    bar_data = bar_data.div(bar_data.sum(axis=1), axis=0)

    # Width scales with number of conditions so each bar is ~0.9 in wide
    fig_width = max(4, len(conditions) * 0.9 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    _render_population_bar(ax, bar_data)
    ax.set_title("Aggregate diffusion-population fractions")
    plt.tight_layout()

    out_path = os.path.join(settings["io"]["plot_directory"], "aggregate_MLE_bar.pdf")
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    print("\nrun_aggregate complete.")


def run_aggregate_survival_by_exposure(
    settings: dict, min_length: int = 2, max_frame: int = 30
) -> None:
    """Kaplan-Meier trajectory survival curve pooling ALL conditions that share
    the same ``exp=`` exposure-time tag embedded in their filenames, straight
    from raw ``_traj.csv`` files.

    Conditions differ by sender/receiver plasmid but several can share the
    same exposure time; this groups those together (one plot per distinct
    ``exp=`` value) so the effect of exposure time on trajectory survival
    can be compared directly, independent of the per-condition curve from
    :func:`~core.conditionwise.run_conditionwise`.

    Parameters
    ----------
    settings:
        Fully-resolved workflow settings dict.
    min_length, max_frame:
        Passed through to :func:`~core.plots.plot_survival_kaplan_meier`.
    """
    grouped_nd2 = group_files_by_metadata(settings["io"]["data_directory"])
    if not grouped_nd2:
        print("run_aggregate_survival_by_exposure: no conditions found — skipping.")
        return

    by_exposure: dict[str, list[str]] = {}
    for condition, nd2_files in grouped_nd2.items():
        exp = extract_metadata(condition, ["exp"]).get("exp")
        if exp is None:
            print(f"  [survival-by-exposure] no 'exp=' tag in condition {condition!r}; skipping")
            continue
        traj_csvs = [
            os.path.join(settings["io"]["traj_directory"], os.path.splitext(f)[0] + "_traj.csv")
            for f in nd2_files
        ]
        by_exposure.setdefault(exp, []).extend(f for f in traj_csvs if os.path.exists(f))

    print(f"run_aggregate_survival_by_exposure: exposure groups found — {sorted(by_exposure)}")

    for exp, traj_csvs in sorted(by_exposure.items()):
        plot_survival_kaplan_meier(
            traj_csvs, settings, f"exp={exp}",
            min_length=min_length, max_frame=max_frame, showPlot=False,
        )

    print("\nrun_aggregate_survival_by_exposure complete.")


def run_aggregate_survival_by_interval(
    settings: dict, min_length: int = 2, max_frame: int = 30
) -> None:
    """One Kaplan-Meier survival plot per distinct ``int=`` (imaging
    interval) tag, with every condition sharing that interval broken out as
    its own overlaid trace on one common set of axes -- lets
    condition-to-condition differences be compared directly within a fixed
    imaging interval (as opposed to :func:`run_aggregate_survival_by_exposure`,
    which pools everything sharing an ``exp=`` tag into a single curve).

    Skipped entirely if no condition's filename carries an ``int=`` tag --
    not every dataset encodes one.

    Parameters
    ----------
    settings:
        Fully-resolved workflow settings dict.
    min_length, max_frame:
        Passed through to :func:`~core.plots.plot_survival_kaplan_meier_overlay`.
    """
    grouped_nd2 = group_files_by_metadata(settings["io"]["data_directory"])
    if not grouped_nd2:
        print("run_aggregate_survival_by_interval: no conditions found — skipping.")
        return

    # conditions_by_interval[interval][condition] = [traj_csv, ...]
    conditions_by_interval: dict[str, dict[str, list[str]]] = {}
    for condition, nd2_files in grouped_nd2.items():
        interval = extract_metadata(condition, ["int"]).get("int")
        if interval is None:
            continue
        traj_csvs = [
            os.path.join(settings["io"]["traj_directory"], os.path.splitext(f)[0] + "_traj.csv")
            for f in nd2_files
        ]
        traj_csvs = [f for f in traj_csvs if os.path.exists(f)]
        if traj_csvs:
            conditions_by_interval.setdefault(interval, {})[condition] = traj_csvs

    if not conditions_by_interval:
        print("run_aggregate_survival_by_interval: no 'int=' tag found on any condition — skipping.")
        return

    print(f"run_aggregate_survival_by_interval: interval groups found — {sorted(conditions_by_interval)}")

    # If configured, resolve the background condition's own trajectories once
    # up front -- same background estimate is reused for every interval group,
    # since it's the background condition's raw contamination rate, not
    # anything specific to a given interval.
    background_condition = settings["survival"]["background_condition"]
    background_traj_csvs = None
    if background_condition is not None:
        background_traj_csvs = sorted(
            glob.glob(os.path.join(settings["io"]["traj_directory"], f"{background_condition}*_traj.csv"))
        )
        if not background_traj_csvs:
            print(
                f"run_aggregate_survival_by_interval: background condition "
                f"{background_condition!r} has no _traj.csv files -- skipping corrected overlays."
            )
            background_traj_csvs = None

    for interval, condition_traj_csvs in sorted(conditions_by_interval.items()):
        plot_survival_kaplan_meier_overlay(
            condition_traj_csvs, settings, f"int={interval}",
            min_length=min_length, max_frame=max_frame,
        )
        if background_traj_csvs:
            plot_survival_kaplan_meier_overlay_corrected(
                condition_traj_csvs, background_traj_csvs, background_condition, settings,
                f"int={interval}", min_length=min_length, max_frame=max_frame,
            )

    print("\nrun_aggregate_survival_by_interval complete.")
