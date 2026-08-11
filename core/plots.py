"""
core/plots.py

All plotting and visualisation functions, moved from plotFunctions.py.

Includes:
- Outlier detection (find_outlier_filenames, find_outlier_positions)
- Posterior/MLE histogram plots (generate_posterior_plotting_df, generate_plots)
- Trajectory survival curves (compute_trajectory_durations, kaplan_meier,
  plot_survival_kaplan_meier) -- Kaplan-Meier estimate straight from raw
  _traj.csv files, right-censoring trajectories still alive at the last
  observed frame instead of dropping them
- Hidden Markov Model fitting (fitHMM) -- state assignment only; no longer
  feeds a survival plot, just the HMM_output/<condition>_HMM.txt summary
- Detections-per-trajectory QC histogram (plot_detections_histogram)
- Movie / annotation helpers (isolate_traj, annotate_trajectory_building_tail,
  save_overlay_movie, save_overlay_tiff)
- Bleaching-curve analysis (bleaching_curve, normalize_signal_by_baseline,
  aggregate_bleaching_data, plot_bleaching_curves)
- Movie generation (generate_movies, generate_fullfield_overlay_movie)
- Trajectory scoring (trajectory_score_df, compute_runs_df)

Note: The TrackMate/ImageJ block that existed in the original file has been
intentionally excluded as it was already commented out.
"""

from __future__ import annotations

import os
import re
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import cm
from PIL import Image, ImageDraw
import imageio
import pims
from tifffile import imwrite
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "DejaVu Sans"


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def find_outlier_filenames(posterior_df: pd.DataFrame, mult_on_sd: float = 2) -> list[str]:
    """Identify filenames whose posterior distribution is a statistical outlier.

    Uses KS-distance from the median posterior across files; files whose
    KS distance exceeds ``median + mult_on_sd * std`` are flagged.

    Parameters
    ----------
    posterior_df:
        DataFrame with columns ``file`` and ``density``.
    mult_on_sd:
        Multiplier on the standard deviation to set the outlier threshold.

    Returns
    -------
    List of outlier filenames (values from the ``file`` column).
    """
    grouped_density = [
        group["density"].to_numpy() for _, group in posterior_df.groupby("file")
    ]
    group_names = posterior_df["file"].drop_duplicates().tolist()
    outliers = find_outlier_positions(grouped_density, group_names, mult_on_sd=mult_on_sd)
    return outliers


def find_outlier_positions(
    grouped_density: list[np.ndarray],
    group_names: list[str],
    mult_on_sd: float = 2,
) -> list[str]:
    """Core statistical work for :func:`find_outlier_filenames`.

    For each distribution in *grouped_density*, compute the KS distance from
    the median CDF across all distributions and flag those beyond the threshold.
    """
    pdfs = np.array(grouped_density)
    cdfs = np.cumsum(pdfs, axis=1)
    cdfs = cdfs / cdfs[:, -1][:, np.newaxis]
    median_cdf = np.median(cdfs, axis=0)

    def ks_statistic(empirical, reference):
        return np.max(np.abs(empirical - reference))

    ks_stats = [ks_statistic(cdf, median_cdf) for cdf in cdfs]
    threshold = np.median(ks_stats) + mult_on_sd * np.std(ks_stats)
    outlier_files = [group_names[i] for i, stat in enumerate(ks_stats) if stat > threshold]
    return outlier_files


# ---------------------------------------------------------------------------
# Posterior / MLE histogram plots
# ---------------------------------------------------------------------------

def generate_posterior_plotting_df(posterior_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the condition-level mean posterior density from per-file posteriors.

    Averages the posterior density across all files, normalises so densities
    sum to 1, and records the condition name.

    Parameters
    ----------
    posterior_df:
        DataFrame with columns ``D``, ``density``, and ``file``.

    Returns
    -------
    DataFrame with columns ``D``, ``density``, and ``condition``.
    """
    posterior_plot_df = posterior_df.copy()
    posterior_plot_df = (
        posterior_plot_df.groupby("D")["density"].mean().reset_index()
    )
    posterior_plot_df["density"] = (
        posterior_plot_df["density"] / posterior_plot_df["density"].sum()
    )
    posterior_plot_df["condition"] = np.unique(
        [re.sub(r"_[0-9]*_traj\.csv", "", f) for f in posterior_df["file"]]
    )[0]
    return posterior_plot_df


def _render_population_bar(ax: plt.Axes, bar_data: pd.DataFrame) -> None:
    """Render a normalised stacked population bar chart on *ax*.

    Parameters
    ----------
    ax:
        Axes to draw on.
    bar_data:
        DataFrame with rows = conditions, columns = population labels,
        values = fractions (rows must already sum to 1).
    """
    n_pops = len(bar_data.columns)
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_pops, 1)))
    bar_data.plot(
        kind="bar", stacked=True, ax=ax,
        edgecolor="black", color=colors, width=0.6, legend=False,
    )
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(
        title="Population", bbox_to_anchor=(1.05, 1),
        loc="upper left", fontsize="small",
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def generate_plots(
    MLE_df: pd.DataFrame,
    posterior_plot_df: pd.DataFrame,
    all_posteriors: pd.DataFrame,
    settings: dict,
    showPlot: bool = True,
    conditions: list[str] | None = None,
) -> None:
    """Generate histogram + bar-graph PDFs for one or more conditions.

    The histogram shows the log10(MLE_D) distribution with per-file grey
    curves and a condition-mean black curve.  The stacked bar chart shows
    the fraction of molecules in each diffusion population.

    Saved to ``settings['io']['plot_directory']/<condition>_hist.pdf``.
    """
    if conditions is None:
        conditions = []

    if len(conditions) == 0:
        conditions = MLE_df["condition"].unique()

    bin_labels = settings["plot"]["barGraphLabels"]
    barGraphs = pd.DataFrame(index=bin_labels)

    for condition in conditions:
        sm_df = posterior_plot_df[posterior_plot_df["condition"] == condition]
        sm_MLE = MLE_df[MLE_df["condition"] == condition]
        sm_all_posterior = all_posteriors[all_posteriors["condition"] == condition]

        normalized_df = sm_all_posterior.copy()
        normalized_df["density_norm"] = normalized_df.groupby("file")["density"].transform(
            lambda x: x / x.sum()
        )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(9, 3),
            gridspec_kw={"width_ratios": [8, 1]},
        )
        ax_hist, ax_bar = axes

        ax_hist.hist(
            np.log10(sm_MLE["MLE_D"]),
            bins=np.linspace(-2, 2, settings["plot"]["nbins"]),
            density=True,
            color="#D28F8F",
        )

        for f, group in normalized_df.groupby("file"):
            ax_hist.plot(
                np.log10(group["D"]),
                50 * group["density_norm"],
                linewidth=1.5,
                color="darkgray",
                alpha=0.4,
            )

        ax_hist.plot(
            np.log10(sm_df["D"]), 50 * sm_df["density"], linewidth=1.5, color="black"
        )

        tick_positions = [-2, -1, 0, 1, 2]
        tick_labels = [
            f"{10**v:.2f}" if 10**v < 1 else f"{10**v:.0f}" for v in tick_positions
        ]
        ax_hist.set_xticks(tick_positions, tick_labels)

        minor_ticks = []
        for base in [0.01, 0.1, 1, 10]:
            for i in range(2, 10):
                tick = base * i
                if tick <= 100:
                    minor_ticks.append(tick)
        minor_ticks = np.log10(minor_ticks)
        ax_hist.set_xticks(minor_ticks, minor=True)
        ax_hist.tick_params(axis="x", which="minor", length=4, color="gray")

        ax_hist.set_ylim(settings["plot"]["ylim"][0], settings["plot"]["ylim"][1])
        ax_hist.set_title(
            f"{condition} ; n_jumps = {settings['saspt']['splitsize']} ;"
            f" {len(sm_MLE['MLE_D'])} particles",
            pad=20,
        )
        ax_hist.set_xlabel("Value")
        ax_hist.set_ylabel("Probability Density")
        for spine in ["top", "right"]:
            ax_hist.spines[spine].set_visible(False)

        counts, _ = np.histogram(sm_MLE["MLE_D"], bins=settings["plot"]["barGraphBreaks"])
        barGraphs[condition] = counts

        bar_row = barGraphs.T.copy()
        bar_row = bar_row.div(bar_row.sum(axis=1), axis=0)
        _render_population_bar(ax_bar, bar_row)
        ax_bar.set_xticklabels([])
        ax_bar.set_xlabel("")

        plt.tight_layout()
        fig.savefig(
            f"{settings['io']['plot_directory']}/{condition}_hist.pdf",
            dpi=300,
            format="pdf",
            bbox_inches="tight",
        )

        if showPlot:
            plt.show()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Survival curves
# ---------------------------------------------------------------------------

def compute_trajectory_durations(traj_csv: str) -> pd.DataFrame:
    """One row per trajectory in *traj_csv*: duration (frames elapsed since
    first detection), number of detections, and whether it's right-censored
    (its last detection is on the last frame observed anywhere in this file).

    Ported from the standalone ``playspace/survival_curve.py`` prototype.
    """
    df = pd.read_csv(traj_csv, comment="#")
    max_frame = df["frame"].max()

    per_traj = df.groupby("trajectory")["frame"].agg(
        first_frame="min", last_frame="max", n_detections="count"
    )
    per_traj["duration"] = per_traj["last_frame"] - per_traj["first_frame"]
    per_traj["censored"] = per_traj["last_frame"] == max_frame
    per_traj["source_file"] = os.path.basename(traj_csv)
    return per_traj.reset_index()


def kaplan_meier(durations: np.ndarray, censored: np.ndarray) -> pd.DataFrame:
    """Kaplan-Meier survival estimate with a Greenwood's-formula 95% CI.

    Parameters
    ----------
    durations:
        Observed time-to-event (or time-to-censoring) for each trajectory.
    censored:
        True where the corresponding duration is right-censored (the
        trajectory was merely last SEEN alive at that time, not observed to
        end).

    Returns
    -------
    DataFrame with columns: time, at_risk, n_events, survival, ci_low, ci_high.
    Starts with a ``time=0, survival=1`` row so the curve/plot anchors at
    the origin.
    """
    n = len(durations)
    event_times = np.unique(durations[~censored])

    rows = [{"time": 0, "at_risk": n, "n_events": 0, "survival": 1.0, "ci_low": 1.0, "ci_high": 1.0}]
    survival = 1.0
    greenwood_sum = 0.0  # running sum of d / (n_at_risk * (n_at_risk - d))

    for t in event_times:
        n_at_risk = int(np.sum(durations >= t))
        n_events = int(np.sum((durations == t) & (~censored)))
        if n_at_risk == 0 or n_events == 0:
            continue

        survival *= 1 - n_events / n_at_risk
        if n_at_risk > n_events:
            greenwood_sum += n_events / (n_at_risk * (n_at_risk - n_events))

        se = survival * np.sqrt(greenwood_sum)
        rows.append({
            "time": t,
            "at_risk": n_at_risk,
            "n_events": n_events,
            "survival": survival,
            "ci_low": max(0.0, survival - 1.96 * se),
            "ci_high": min(1.0, survival + 1.96 * se),
        })

    return pd.DataFrame(rows)


def plot_survival_kaplan_meier(
    traj_csvs: list[str],
    settings: dict,
    label: str,
    min_length: int = 2,
    max_frame: int = 30,
    showPlot: bool = False,
) -> None:
    """Kaplan-Meier trajectory survival curve straight from raw ``_traj.csv``
    files -- one curve pooling every trajectory across all of *traj_csvs*.

    Per trajectory, "survival time" is frames elapsed between its first and
    last detection (the csv has no real timestamp). Trajectories whose last
    detection lands on the last frame observed anywhere in their own source
    file are right-censored (the movie just stopped recording, so we don't
    know whether they really ended there) rather than dropped -- a censored
    trajectory stays in the risk set up to its own last frame but isn't
    counted as a "death" there, which is more information-preserving than
    discarding it outright. See Greenwood's formula for the 95% CI.

    Trajectories with fewer than *min_length* detections are dropped before
    fitting (default 2 -- excludes single-frame blips, which would otherwise
    dominate the curve's initial drop). *max_frame* only limits the plotted
    x-range; the KM fit itself always uses every remaining trajectory's full
    duration, since truncating the input would incorrectly shrink the risk
    set at frames <= max_frame.

    Caveat: censoring is detected from the last frame with ANY detection in
    a given file, not that movie's true frame count (not available in the
    csv) -- if a movie's last few frames have zero detections, a handful of
    censored trajectories could be misclassified as real deaths.

    Saved to ``settings['io']['plot_directory']/survival_curves/<label>_survival.pdf``.
    """
    per_traj = pd.concat([compute_trajectory_durations(f) for f in traj_csvs], ignore_index=True)

    n_dropped = int((per_traj["n_detections"] < min_length).sum())
    per_traj = per_traj[per_traj["n_detections"] >= min_length]

    n_total = len(per_traj)
    if n_total == 0:
        print(f"  [KM survival] no trajectories with >= {min_length} detections for {label}; skipping")
        return
    n_censored = int(per_traj["censored"].sum())
    print(
        f"  [KM survival] {label}: {n_total} trajectories (dropped {n_dropped} "
        f"with < {min_length} detections), {n_censored} right-censored"
    )

    km_df = kaplan_meier(per_traj["duration"].to_numpy(), per_traj["censored"].to_numpy())

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.fill_between(km_df["time"], km_df["ci_low"], km_df["ci_high"], step="post", alpha=0.25)
    ax.step(km_df["time"], km_df["survival"], where="post", color="black")

    ax.set_xlabel("Frames elapsed since first detection")
    ax.set_ylabel("Survival probability (Kaplan-Meier)")
    ax.set_title(label)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_frame)
    ax.text(
        0.98, 0.95, f"n={n_total}, censored={n_censored}",
        transform=ax.transAxes, ha="right", va="top", fontsize="small",
    )
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    output_dirname = os.path.join(settings["io"]["plot_directory"], "survival_curves")
    os.makedirs(output_dirname, exist_ok=True)
    output_filename = os.path.join(output_dirname, f"{label}_survival.pdf")
    fig.savefig(output_filename, dpi=300, format="pdf", bbox_inches="tight")
    print(f"  Saved {os.path.basename(output_filename)}")

    if showPlot:
        plt.show()
    plt.close(fig)


def plot_survival_kaplan_meier_overlay(
    grouped_traj_csvs: dict[str, list[str]],
    settings: dict,
    plot_label: str,
    min_length: int = 2,
    max_frame: int = 30,
) -> None:
    """Kaplan-Meier survival curves for several groups overlaid on one shared
    set of axes -- one step-curve per group, e.g. one trace per condition
    within a fixed imaging interval, rather than each getting its own
    separate plot.

    Same fitting/censoring logic as :func:`plot_survival_kaplan_meier`
    (see its docstring), just without the per-group confidence-interval
    shading, which would overlap illegibly with more than one group on the
    same axes.

    Parameters
    ----------
    grouped_traj_csvs:
        Maps a group's display label (e.g. a condition string) to the list
        of ``_traj.csv`` paths for that group -- not pooled with any other
        group's.
    plot_label:
        Used for the plot title and output filename.

    Saved to
    ``settings['io']['plot_directory']/survival_curves/<plot_label>_survival_overlay.pdf``.
    Skipped entirely if no group has any qualifying trajectory.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    any_plotted = False

    for group_label, traj_csvs in sorted(grouped_traj_csvs.items()):
        per_traj = pd.concat([compute_trajectory_durations(f) for f in traj_csvs], ignore_index=True)
        n_dropped = int((per_traj["n_detections"] < min_length).sum())
        per_traj = per_traj[per_traj["n_detections"] >= min_length]

        n_total = len(per_traj)
        if n_total == 0:
            print(f"  [KM survival overlay] no trajectories with >= {min_length} detections for {group_label}; skipping")
            continue
        n_censored = int(per_traj["censored"].sum())
        print(
            f"  [KM survival overlay] {group_label}: {n_total} trajectories (dropped {n_dropped} "
            f"with < {min_length} detections), {n_censored} right-censored"
        )

        km_df = kaplan_meier(per_traj["duration"].to_numpy(), per_traj["censored"].to_numpy())
        ax.step(km_df["time"], km_df["survival"], where="post", label=f"{group_label} (n={n_total})")
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        print(f"  [KM survival overlay] no group had qualifying trajectories for {plot_label}; skipping")
        return

    ax.set_xlabel("Frames elapsed since first detection")
    ax.set_ylabel("Survival probability (Kaplan-Meier)")
    ax.set_title(plot_label)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, max_frame)
    ax.legend(fontsize="small", loc="best")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    output_dirname = os.path.join(settings["io"]["plot_directory"], "survival_curves")
    os.makedirs(output_dirname, exist_ok=True)
    output_filename = os.path.join(output_dirname, f"{plot_label}_survival_overlay.pdf")
    fig.savefig(output_filename, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {os.path.basename(output_filename)}")


def compute_hmm_state_dwell_runs(
    rollingMLE: pd.DataFrame,
    state_col: str = "posterior_pop",
) -> pd.DataFrame:
    """Break each trajectory's HMM-labeled detections into contiguous dwell runs.

    A run continues while consecutive rows (sorted by frame, within one
    ``ur_trajectory``) share the same *state_col* value and their frames are
    exactly 1 apart; it ends on a state change, a frame gap (e.g. a linking
    blink), or a NaN/unassigned label (the HMM couldn't classify that
    detection, or the trajectory was too short to be fit at all -- see
    :func:`fitHMM`'s ``min_obs``). A run ending because the trajectory
    simply has no further rows is right-censored -- we didn't observe it
    actually leave that state, unlike a run that ends because the next
    detection is genuinely labeled with a different state.

    Parameters
    ----------
    rollingMLE:
        Rolling-window MLE DataFrame after :func:`fitHMM`, i.e. with a
        populated *state_col* (``posterior_pop`` by default).
    state_col:
        Column holding the per-detection state label.

    Returns
    -------
    DataFrame with columns: ur_trajectory, state, duration, n_detections,
    censored. One row per dwell run -- a trajectory that revisits a state
    contributes one row per distinct visit, not one row total.
    """
    rows: list[dict] = []
    for traj_id, group in rollingMLE.groupby("ur_trajectory"):
        group = group.sort_values("frame")
        frames = group["frame"].to_numpy()
        states = group[state_col].to_numpy()
        n = len(group)

        run_start = 0
        for i in range(1, n + 1):
            continues_run = (
                i < n
                and not pd.isna(states[i])
                and not pd.isna(states[run_start])
                and states[i] == states[run_start]
                and frames[i] == frames[i - 1] + 1
            )
            if continues_run:
                continue

            if not pd.isna(states[run_start]):
                rows.append({
                    "ur_trajectory": traj_id,
                    "state": int(states[run_start]),
                    "duration": int(frames[i - 1] - frames[run_start]),
                    "n_detections": i - run_start,
                    "censored": i == n,
                })
            run_start = i

    return pd.DataFrame(
        rows, columns=["ur_trajectory", "state", "duration", "n_detections", "censored"]
    )


def plot_survival_by_hmm_state(
    rollingMLE: pd.DataFrame,
    settings: dict,
    condition: str,
    min_length: int = 2,
    max_frame: int = 30,
    showPlot: bool = False,
) -> None:
    """Kaplan-Meier survival curve per HMM-assigned diffusive state.

    Unlike :func:`plot_survival_kaplan_meier` (which pools raw, unlabeled
    trajectory durations straight from ``_traj.csv``), this measures how
    long a trajectory *dwells* in each of :func:`fitHMM`'s ``posterior_pop``
    states before transitioning out -- one Kaplan-Meier curve per state, laid
    out as one subplot per population (state 0 = slowest-diffusing / most
    confined, per ``settings['plot']['barGraphLabels']`` order). See
    :func:`compute_hmm_state_dwell_runs` for the run/censoring definition.

    Only meaningful for fastSPT conditions, where *rollingMLE* has already
    been through :func:`fitHMM`; slowSPT never runs SASPT/HMM at all and
    keeps the raw-trajectory :func:`plot_survival_kaplan_meier` curve instead.

    Saved to ``settings['io']['plot_directory']/survival_curves/
    <condition>_survival_by_state.pdf``.
    """
    nPopulations = len(settings["plot"]["barGraphLabels"])
    state_labels = settings["plot"]["barGraphLabels"]

    runs = compute_hmm_state_dwell_runs(rollingMLE)
    if len(runs):
        runs = runs[runs["n_detections"] >= min_length]

    fig, axes = plt.subplots(1, nPopulations, figsize=(4 * nPopulations, 5), squeeze=False)
    axes = axes[0]

    for state in range(nPopulations):
        ax = axes[state]
        label = state_labels[state] if state < len(state_labels) else f"state {state}"
        state_runs = runs[runs["state"] == state] if len(runs) else runs

        if len(state_runs) == 0:
            ax.text(0.5, 0.5, "no dwell runs", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{label} (state {state})")
            ax.set_xlim(0, max_frame)
            ax.set_ylim(0, 1.05)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            print(
                f"  [KM survival by state] {condition}: no dwell runs with "
                f">= {min_length} detections for state {state} ({label})"
            )
            continue

        km_df = kaplan_meier(
            state_runs["duration"].to_numpy(), state_runs["censored"].to_numpy()
        )
        n_total = len(state_runs)
        n_censored = int(state_runs["censored"].sum())

        ax.fill_between(km_df["time"], km_df["ci_low"], km_df["ci_high"], step="post", alpha=0.25)
        ax.step(km_df["time"], km_df["survival"], where="post", color="black")
        ax.set_xlabel("Frames elapsed since entering state")
        if state == 0:
            ax.set_ylabel("Survival probability (Kaplan-Meier)")
        ax.set_title(f"{label} (state {state})")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, max_frame)
        ax.text(
            0.98, 0.95, f"n={n_total}, censored={n_censored}",
            transform=ax.transAxes, ha="right", va="top", fontsize="small",
        )
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

        print(
            f"  [KM survival by state] {condition} state {state} ({label}): "
            f"{n_total} dwell runs, {n_censored} right-censored"
        )

    fig.suptitle(condition)
    plt.tight_layout()

    output_dirname = os.path.join(settings["io"]["plot_directory"], "survival_curves")
    os.makedirs(output_dirname, exist_ok=True)
    output_filename = os.path.join(output_dirname, f"{condition}_survival_by_state.pdf")
    fig.savefig(output_filename, dpi=300, format="pdf", bbox_inches="tight")
    print(f"  Saved {os.path.basename(output_filename)}")

    if showPlot:
        plt.show()
    plt.close(fig)


def plot_detections_histogram(
    rolling_MLE: pd.DataFrame,
    settings: dict,
    cond: str,
    y_max: float = 0.25,
    showPlot: bool = False,
) -> None:
    """Plot a density histogram of detections (raw localizations) per trajectory.

    Real SPT datasets are dominated by very short trajectories, so a handful
    of low bins can be 10-100x taller than the rest of the distribution.
    The y-axis is plotted as density (fraction of trajectories per bin, not
    raw count) and capped at a fixed *y_max* rather than scaled per-condition,
    so histograms from different conditions are visually comparable at a
    glance. Any bar taller than *y_max* is colored red and its true density
    listed in a corner box, so the clipping is visible rather than silent.

    Saved to ``settings['io']['plot_directory']/survival_curves/
    <cond>_detections_histogram.pdf``.
    """
    counts = (
        rolling_MLE.dropna(subset=["ur_trajectory"]).groupby("ur_trajectory").size()
    )
    if len(counts) == 0:
        print(f"  [detections histogram] no trajectories for {cond}; skipping")
        return

    max_count = int(counts.max())
    bins = np.arange(1, max_count + 2) - 0.5

    fig, ax = plt.subplots(figsize=(8, 5))
    bin_heights, _, patches = ax.hist(
        counts, bins=bins, density=True, color="#4C72B0", edgecolor="black"
    )

    for height, patch in zip(bin_heights, patches):
        if height > y_max:
            patch.set_color("#B44C4C")

    # Adjacent clipped bins (nearly always the smallest detection counts,
    # e.g. 1-4) sit right next to each other in x, so per-bar text labels
    # collide into an unreadable stack. List them in one corner box instead.
    clipped = [
        (int(round(patch.get_x() + patch.get_width() / 2)), height)
        for height, patch in zip(bin_heights, patches)
        if height > y_max
    ]
    if clipped:
        max_lines = 8
        lines = [f"{x}: {h:.2f}" for x, h in clipped[:max_lines]]
        if len(clipped) > max_lines:
            lines.append(f"... +{len(clipped) - max_lines} more")
        text = "clipped bins\n(detections: density)\n" + "\n".join(lines)
        ax.text(
            0.98, 0.98, text, transform=ax.transAxes, ha="right", va="top",
            fontsize="x-small",
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85),
        )

    ax.set_ylim(0, y_max)
    ax.set_xlabel("Detections per trajectory")
    ax.set_ylabel("Density")
    ax.set_title(f"{cond} — detections per trajectory (n={len(counts)})")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    output_dirname = os.path.join(settings["io"]["plot_directory"], "survival_curves")
    os.makedirs(output_dirname, exist_ok=True)
    output_filename = os.path.join(output_dirname, f"{cond}_detections_histogram.pdf")
    fig.savefig(output_filename, dpi=300, format="pdf", bbox_inches="tight")
    print(f"  Saved {os.path.basename(output_filename)}")

    if showPlot:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# HMM fitting
# ---------------------------------------------------------------------------

def _prepare_rolling_mle_for_hmm(
    rollingMLE: pd.DataFrame, settings: dict, min_obs: int = 7
):
    """Shared prep for fitHMM/compute_naive_pop.

    Filters to trajectories long enough to include (``min_obs``), builds the
    ``log10(MLE_D)`` observation matrix the HMM fits, and computes the
    threshold-only ``naive_pop`` label (independent of any HMM fit) by
    binning against ``settings['plot']['barGraphBreaks']``.

    Returns
    -------
    (dynamic_MLE_df, filtered_df, X, data, zones, log_zones, nPopulations)
    ``dynamic_MLE_df`` is a copy of *rollingMLE* with a populated
    ``naive_pop`` column.
    """
    nPopulations = len(settings["plot"]["barGraphLabels"])

    dynamic_MLE_df = rollingMLE.copy()
    dynamic_MLE_df["MLE_D"] = dynamic_MLE_df["MLE_D"].replace([np.inf, -np.inf], np.nan)

    valid_trajs = (
        dynamic_MLE_df.groupby("ur_trajectory")["frame"].transform("count") >= min_obs
    )
    filtered_df = dynamic_MLE_df[
        valid_trajs & dynamic_MLE_df["MLE_D"].notna()
    ].copy()

    grouped = filtered_df.sort_values(["ur_trajectory", "frame"]).groupby("ur_trajectory")
    data = [group["MLE_D"].values for _, group in grouped if len(group) >= min_obs]

    if data:
        X = np.concatenate(data).reshape(-1, 1)
        X = np.log10(X + 1e-10)
    else:
        X = np.empty((0, 1))

    eps = 1e-10
    boundaries = [b if b > 0 else eps for b in settings["plot"]["barGraphBreaks"]]
    zones = list(zip(boundaries[:-1], boundaries[1:]))
    log_zones = [(np.log10(a), np.log10(b)) for a, b in zones]

    naive_pop = np.full_like(X, fill_value=np.nan, dtype=float)
    for i, (low, high) in enumerate(zones):
        mask = (X >= np.log10(low)) & (X < np.log10(high))
        naive_pop[mask] = i

    dynamic_MLE_df["naive_pop"] = np.nan
    start_idx = 0
    grouped = filtered_df.sort_values(["ur_trajectory", "frame"]).groupby("ur_trajectory")
    for traj_id, group in grouped:
        traj_len = len(group)
        if traj_len <= min_obs:
            continue
        dynamic_MLE_df.loc[group.index, "naive_pop"] = naive_pop[
            start_idx : start_idx + traj_len
        ]
        start_idx += traj_len

    return dynamic_MLE_df, filtered_df, X, data, zones, log_zones, nPopulations


def fitHMM(
    rollingMLE: pd.DataFrame,
    settings: dict,
    condition: str,
    min_obs: int = 7,
) -> pd.DataFrame:
    """Fit a Gaussian HMM to the log10(MLE_D) time series.

    Each diffusion-coefficient population defined in
    ``settings['plot']['barGraphBreaks']`` is treated as one HMM state.
    HMM parameters are written to
    ``settings['io']['plot_directory']/HMM_output/<condition>_HMM.txt``.

    Parameters
    ----------
    rollingMLE:
        Rolling-window MLE DataFrame with columns ``ur_trajectory``,
        ``frame``, and ``MLE_D``.
    settings:
        Workflow settings dict.
    condition:
        Condition label used for the output filename.
    min_obs:
        Minimum number of observations required for a trajectory to be
        included in the HMM fit.

    Returns
    -------
    A copy of *rollingMLE* with added columns ``posterior_pop`` and
    ``naive_pop``.
    """
    np.set_printoptions(suppress=True, precision=4)

    dynamic_MLE_df, filtered_df, X, data, zones, log_zones, nPopulations = (
        _prepare_rolling_mle_for_hmm(rollingMLE, settings, min_obs=min_obs)
    )
    lengths = [len(traj) for traj in data]

    print("Running HMM")

    class BoundedMeanHMM(GaussianHMM):
        """GaussianHMM with optional per-state mean bounds (from Chad)."""

        def __init__(self, n_components=1, mean_bounds=None, **kwargs):
            super().__init__(n_components=n_components, **kwargs)
            self.mean_bounds = mean_bounds

        def _do_mstep(self, stats):
            super()._do_mstep(stats)
            if self.mean_bounds is not None:
                for i, (low, high) in enumerate(self.mean_bounds):
                    self.means_[i, :] = np.clip(self.means_[i, :], low, high)

    model = BoundedMeanHMM(
        n_components=nPopulations,
        covariance_type="full",
        mean_bounds=log_zones,
        n_iter=500,
        tol=1e-8,
        random_state=42,
    )

    # A GaussianHMM with several components needs enough independent,
    # non-degenerate observations to estimate a covariance per state. Short
    # trajectories tend to pile up at the diffusion-coefficient grid floor
    # (MLE_D clamped to its minimum), which can starve one or more states of
    # variance and drive hmmlearn's internal covariance estimate to NaN/inf
    # ("array must not contain infs or NaNs" from scipy's cholesky). Treat
    # that the same way run_saspt treats too-few-trajectories: skip with a
    # clear warning and fall back to the threshold-based naive_pop, rather
    # than crashing the whole condition (and, transitively, the whole batch).
    hmm_fit_ok = True
    try:
        model.fit(X)
        gamma = model.predict_proba(X)
        fractions_soft = gamma.mean(axis=0)
        posterior_pop = np.array(gamma.argmax(axis=1))
    except (ValueError, np.linalg.LinAlgError) as e:
        print(
            f"  WARNING: HMM fit failed for condition {condition} ({e}); "
            "likely too little/degenerate rolling-window data for "
            f"{nPopulations} states. Falling back to naive (threshold-only) "
            "population assignment; posterior_pop will be left as NaN and "
            "no HMM_output file will be written for this condition."
        )
        hmm_fit_ok = False
        posterior_pop = np.full(X.shape[0], np.nan)

    # naive_pop was already assigned by _prepare_rolling_mle_for_hmm; only
    # posterior_pop (the HMM-derived label) needs assigning here.
    start_idx = 0
    dynamic_MLE_df["posterior_pop"] = np.nan

    grouped = filtered_df.sort_values(["ur_trajectory", "frame"]).groupby("ur_trajectory")
    for traj_id, group in grouped:
        traj_len = len(group)
        if traj_len <= min_obs:
            continue
        dynamic_MLE_df.loc[group.index, "posterior_pop"] = posterior_pop[
            start_idx : start_idx + traj_len
        ]
        start_idx += traj_len

    if not hmm_fit_ok:
        return dynamic_MLE_df

    # Sort states by ascending mean and reorder model attributes accordingly
    means = model.means_.flatten()
    sort_order = np.argsort(means)
    model.startprob_ = fractions_soft[sort_order]
    model.transmat_ = model.transmat_[sort_order][:, sort_order]
    model.means_ = model.means_[sort_order]
    model.covars_ = model.covars_[sort_order]

    output_dirname = os.path.join(settings["io"]["plot_directory"], "HMM_output")
    os.makedirs(output_dirname, exist_ok=True)
    output_file = os.path.join(output_dirname, f"{condition}_HMM.txt")
    with open(output_file, "w") as fh:
        print(f"{condition}", file=fh)
        print("", file=fh)
        print("Transition matrix:", file=fh)
        print(model.transmat_, file=fh)
        print("", file=fh)
        print("Means of each state:", file=fh)
        print(10 ** (model.means_), file=fh)
        print("", file=fh)
        print("Occupancy of each state:", file=fh)
        print(model.startprob_, file=fh)
        print("", file=fh)
        print("State lifetimes", file=fh)
        transmat = model.transmat_
        lifetimes = 1 / (1 - np.diag(transmat)) * settings["saspt"]["frame_interval"]
        for i, tau in enumerate(lifetimes):
            print(f"State {i}: expected lifetime = {tau:.4f} seconds", file=fh)

    return dynamic_MLE_df


# ---------------------------------------------------------------------------
# Trajectory isolation and annotation for movies
# ---------------------------------------------------------------------------

def isolate_traj(
    all_traj: pd.DataFrame,
    ur_trajectory: str,
    half_size: int = 15,
    max_frame: int = 500,
) -> tuple[pd.DataFrame, dict]:
    """Extract a padded sub-table for one trajectory plus its spatial/temporal bounding box.

    Parameters
    ----------
    all_traj:
        Full rolling-MLE DataFrame with columns ``ur_trajectory``, ``x``,
        ``y``, ``frame``, ``condition``.
    ur_trajectory:
        Unique trajectory identifier to isolate.
    half_size:
        Half-width of the spatial crop (pixels).
    max_frame:
        Maximum allowed frame index.

    Returns
    -------
    ``(this_traj, bounding_box)`` where *bounding_box* is a dict with
    ``timeBox`` and ``spaceBox`` keys.
    """
    this_traj = all_traj[all_traj["ur_trajectory"] == ur_trajectory].copy()

    this_traj_center = (int(np.mean(this_traj["y"])), int(np.mean(this_traj["x"])))
    timeBox = (
        max(0, np.min(this_traj["frame"]) - 5),
        min(max_frame, np.max(this_traj["frame"]) + 5),
    )

    spaceBox = {
        "xmin": this_traj_center[1] - half_size,
        "xmax": this_traj_center[1] + half_size + 1,
        "ymin": this_traj_center[0] - half_size,
        "ymax": this_traj_center[0] + half_size + 1,
    }

    this_traj["og_frame"] = this_traj["frame"]
    this_traj["rel_frame"] = this_traj["frame"] - np.min(timeBox)

    first_frame = this_traj["frame"].min()
    last_frame = this_traj["frame"].max()
    new_frames = list(range(first_frame - 5, first_frame)) + list(
        range(last_frame + 1, last_frame + 6)
    )
    new_frames = [x for x in new_frames if x > 0 and x < max_frame]

    nan_rows = pd.DataFrame(np.nan, index=range(len(new_frames)), columns=this_traj.columns)
    traj_id = this_traj["ur_trajectory"].iloc[0]
    nan_rows["frame"] = new_frames
    nan_rows["og_frame"] = new_frames
    nan_rows["ur_trajectory"] = traj_id
    nan_rows["condition"] = this_traj["condition"].iloc[0]
    nan_rows["rel_frame"] = nan_rows["frame"] - np.min(timeBox)

    bounding_box = {"timeBox": timeBox, "spaceBox": spaceBox}
    this_traj = pd.concat([nan_rows, this_traj], ignore_index=True).sort_values(
        "frame", ignore_index=True
    )
    return this_traj, bounding_box


def annotate_trajectory_building_tail(
    box: np.ndarray,
    traj_df: pd.DataFrame,
    spaceBox: dict,
    upscale: int = 8,
    spot_radius: int = 8,
    lwd: int = 4,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Render an upscaled annotation layer colourised by MLE_D.

    Each frame shows the trajectory tail (all displacements up to that time)
    with line and spot colours mapped to log10(MLE_D) using the viridis
    colormap in the fixed range [-2, 2].

    Parameters
    ----------
    box:
        Cropped image stack, shape ``(T, Y, X)``.
    traj_df:
        Trajectory table for the cropped region with columns
        ``frame``, ``x``, ``y``, ``rel_frame``, ``ur_trajectory``, ``MLE_D``.
    spaceBox:
        Dict with keys ``xmin``, ``xmax``, ``ymin``, ``ymax``.
    upscale:
        Integer upscaling factor for sub-pixel rendering.
    spot_radius:
        Radius of the drawn spot circle in upscaled pixels.
    lwd:
        Line width for trajectory segments.

    Returns
    -------
    ``(upsized_input, annotated_frames)`` — lists of RGB arrays.
    """
    annotated_frames: list[np.ndarray] = []
    upsized_input: list[np.ndarray] = []

    traj_df = traj_df.sort_values(["ur_trajectory", "frame"])
    grouped = traj_df.groupby("ur_trajectory")

    trajectory_segments_by_frame: dict[int, list] = {t: [] for t in range(len(box))}

    cmap = cm.viridis
    vmin, vmax = -2, 2

    def mle_to_color(val):
        val = max(float(val), 1e-4)  # guard log10(0) and log10(negative)
        val_clipped = np.clip(np.log10(val), vmin, vmax)
        fraction = (val_clipped - vmin) / (vmax - vmin)
        return tuple(int(255 * c) for c in cmap(fraction)[:3])

    for _, group in grouped:
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            f_prev = group.loc[i - 1, "rel_frame"]
            f_curr = group.loc[i, "rel_frame"]
            if f_curr == f_prev + 1:
                x1 = (group.loc[i - 1, "x"] - spaceBox["xmin"]) * upscale
                y1 = (group.loc[i - 1, "y"] - spaceBox["ymin"]) * upscale
                x2 = (group.loc[i, "x"] - spaceBox["xmin"]) * upscale
                y2 = (group.loc[i, "y"] - spaceBox["ymin"]) * upscale
                val = group.loc[i, "MLE_D"]
                if not np.isnan(val):
                    color = mle_to_color(val)
                    for t in range(int(f_curr), len(box)):
                        trajectory_segments_by_frame[t].append(
                            ((x1, y1), (x2, y2), color)
                        )

    for t in range(len(box)):
        height, width = box[t].shape
        frame_arr = np.zeros((height, width, 3), dtype=np.uint8)
        img = Image.fromarray(frame_arr, "RGB")
        img_up = img.resize(
            (img.width * upscale, img.height * upscale), resample=Image.BICUBIC
        )
        draw = ImageDraw.Draw(img_up)

        # PIL's fromarray requires uint8; .nd2 data is typically uint16
        frame = box[t]
        if frame.dtype != np.uint8:
            fmin, fmax = float(frame.min()), float(frame.max())
            if fmax > fmin:
                frame = ((frame - fmin) / (fmax - fmin) * 255).astype(np.uint8)
            else:
                frame = np.zeros_like(frame, dtype=np.uint8)
        boximg = Image.fromarray(frame)
        boximg_up = boximg.resize(
            (boximg.width * upscale, boximg.height * upscale), resample=Image.BICUBIC
        )

        for (x1, y1), (x2, y2), color in trajectory_segments_by_frame[t]:
            if np.isnan([x1, x2, y1, y2]).any():
                continue
            draw.line([(x1, y1), (x2, y2)], fill=color, width=lwd)

        current_detections = traj_df[traj_df["rel_frame"] == t]
        for _, row in current_detections.iterrows():
            x = (row["x"] - spaceBox["xmin"]) * upscale
            y = (row["y"] - spaceBox["ymin"]) * upscale
            bbox = [x - spot_radius, y - spot_radius, x + spot_radius, y + spot_radius]
            if not np.isnan(row["MLE_D"]):
                ellipse_color = mle_to_color(row["MLE_D"])
                draw.ellipse(bbox, outline=ellipse_color, width=lwd)

        annotated_frames.append(np.array(img_up))
        upsized_input.append(np.array(boximg_up))

    return upsized_input, annotated_frames


def save_overlay_movie(
    box_resized,
    box_annotated,
    output_path: str = "overlay_output.mp4",
    fps: int = 20,
) -> None:
    """Composite grayscale frames with a colourised annotation layer and save as MP4.

    The grayscale channel is contrast-normalised globally; annotation pixels
    replace the RGB value at their location.
    """
    frames_rgb = []

    box_resized = np.array(box_resized, dtype=np.float32)
    if box_resized.size == 0:
        raise ValueError("no frames to write")
    box_resized = box_resized - np.min(box_resized)
    _max = float(np.max(box_resized))
    if _max > 0:
        box_resized = (box_resized / _max * 255).astype(np.uint8)
    else:
        box_resized = np.zeros_like(box_resized, dtype=np.uint8)
    box_resized = [255 - frame for frame in box_resized]

    box_resized = np.array(box_resized, dtype=np.float32)
    box_resized = box_resized - np.percentile(box_resized, 2)
    _p99 = float(np.percentile(box_resized, 99))
    if _p99 > 0:
        box_resized = box_resized / _p99 * 255
    box_resized = np.clip(box_resized, 0, 255).astype(np.uint8)

    for gray, blue in zip(box_resized, box_annotated):
        gray = np.squeeze(gray)
        blue = np.squeeze(blue)

        rgb = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
        rgb[..., 0] = gray
        rgb[..., 1] = gray
        rgb[..., 2] = gray

        mask = blue.sum(axis=2) > 0
        rgb[..., 0][mask] = blue[..., 0][mask]
        rgb[..., 1][mask] = blue[..., 1][mask]
        rgb[..., 2][mask] = blue[..., 2][mask]

        frames_rgb.append(rgb)

    # imageio v3 doesn't always expose libx264; use v2 writer which reliably
    # picks up the system ffmpeg.  Fall back to GIF if mp4 still fails.
    try:
        imageio.v2.mimsave(output_path, frames_rgb, fps=fps, codec="libx264")
    except Exception as e:
        gif_path = os.path.splitext(output_path)[0] + ".gif"
        try:
            imageio.v2.mimsave(gif_path, frames_rgb, fps=fps)
        except Exception as e2:
            raise RuntimeError(
                f"Failed to write movie as mp4 ({e}) and gif ({e2})"
            ) from e2


def save_overlay_tiff(
    box,
    box_anotations,
    output_path: str = "overlay_output.tif",
    fps: int = 20,
) -> None:
    """Save a two-channel TIFF: grayscale image and binary annotation mask."""
    box = box - np.min(box)
    box = (box / np.max(box) * 255).astype(np.uint8)

    box_gray = np.dot(box[..., :3], [0.2989, 0.5870, 0.1140])
    annotations_binary = [
        np.where(np.any(frame != 0, axis=2), 255, 0).astype(np.uint8)
        for frame in box_anotations
    ]

    combined_channels = np.stack(
        [np.stack(box_gray, axis=0), np.stack(annotations_binary, axis=0)], axis=-1
    )
    imwrite(
        output_path,
        combined_channels,
        photometric="minisblack",
        planarconfig="contig",
    )


# ---------------------------------------------------------------------------
# Bleaching analysis
# ---------------------------------------------------------------------------

def bleaching_curve(
    img: np.ndarray,
    traj_df: pd.DataFrame,
    signal_size: int = 3,
    background_size: int = 7,
    post_frames: int = 10,
) -> pd.DataFrame:
    """Measure signal and local background around each detected position.

    Parameters
    ----------
    img:
        Image stack of shape ``(T, Y, X)``.
    traj_df:
        Trajectory table with columns ``x``, ``y``, ``frame``,
        ``ur_trajectory``, ``condition``.
    signal_size:
        Side length of the signal square (pixels).
    background_size:
        Side length of the background annulus outer square (pixels).
    post_frames:
        Number of frames to continue measuring after the last detected frame.

    Returns
    -------
    DataFrame with added columns ``signal``, ``background``, ``rel_frame``.
    """
    half_signal = signal_size // 2
    half_background = background_size // 2
    results = []

    traj_valid = traj_df.dropna(subset=["x", "y", "frame"]).copy()

    if len(traj_valid) > 0:
        last_x = traj_valid["x"].iloc[-1]
        last_y = traj_valid["y"].iloc[-1]
        last_frame = int(traj_valid["frame"].iloc[-1])

        post_rows = []
        for f in range(last_frame + 1, min(last_frame + 1 + post_frames, img.shape[0])):
            post_rows.append(
                {
                    "x": last_x,
                    "y": last_y,
                    "frame": f,
                    "ur_trajectory": traj_valid["ur_trajectory"].iloc[-1],
                    "condition": traj_valid["condition"].iloc[-1],
                }
            )
        if post_rows:
            traj_valid = pd.concat(
                [traj_valid, pd.DataFrame(post_rows)], ignore_index=True
            )

    for _, row in traj_valid.iterrows():
        x = int(round(row["x"]))
        y = int(round(row["y"]))
        f = int(row["frame"])

        if np.isnan(x) or np.isnan(y) or f >= img.shape[0]:
            signal = np.nan
            background = np.nan
        else:
            frame_img = img[f]
            h, w = frame_img.shape

            y1s, y2s = np.clip([y - half_signal, y + half_signal + 1], 0, h)
            x1s, x2s = np.clip([x - half_signal, x + half_signal + 1], 0, w)
            y1b, y2b = np.clip([y - half_background, y + half_background + 1], 0, h)
            x1b, x2b = np.clip([x - half_background, x + half_background + 1], 0, w)

            signal_patch = frame_img[y1s:y2s, x1s:x2s]
            background_patch = frame_img[y1b:y2b, x1b:x2b]

            signal = np.mean(signal_patch)
            background = np.median(background_patch)

        result = row.to_dict()
        result["signal"] = signal
        result["background"] = background
        results.append(result)

    df_out = pd.DataFrame(results)

    if not df_out.empty:
        non_nan = df_out[df_out["I0"].notna()]
        last_frame = non_nan["frame"].max()
        df_out["rel_frame"] = df_out["frame"] - last_frame

    return df_out


def normalize_signal_by_baseline(df: pd.DataFrame, pre_window: int = 10) -> pd.DataFrame:
    """Normalise ``signal`` for each trajectory by its pre-bleach baseline.

    The baseline is the mean of ``corrected_signal`` over the *pre_window*
    frames immediately before ``rel_frame == 0``.
    """
    df = df.copy()

    def normalize(traj):
        mask = (traj["rel_frame"] < 0) & (traj["rel_frame"] >= -pre_window)
        baseline = traj.loc[mask, "corrected_signal"].mean()
        if pd.isna(baseline) or baseline == 0:
            traj["signal_norm"] = traj["corrected_signal"]
        else:
            traj["signal_norm"] = traj["corrected_signal"] / baseline
        return traj

    return df.groupby("ur_trajectory", group_keys=False).apply(normalize)


def aggregate_bleaching_data(
    rollingMLE: pd.DataFrame,
    settings: dict,
    n_particles: int = 200,
    min_obs: int = 7,
) -> pd.DataFrame:
    """Measure bleaching curves for a random sample of trajectories.

    Parameters
    ----------
    rollingMLE:
        Rolling-MLE DataFrame.
    settings:
        Workflow settings dict.
    n_particles:
        Maximum number of trajectories to sample.
    min_obs:
        Minimum number of observations for a trajectory to be eligible.

    Returns
    -------
    Concatenated DataFrame of per-trajectory bleaching measurements.
    """
    counts = rollingMLE["ur_trajectory"].value_counts()
    valid_trajectories = counts[counts >= min_obs].index
    rollingMLE_filtered = rollingMLE[
        rollingMLE["ur_trajectory"].isin(valid_trajectories)
    ].copy()

    bc_rows: list[pd.DataFrame] = []
    prev_open_image = ""
    img_stack = None

    unique_trajs = np.unique(rollingMLE_filtered["ur_trajectory"])
    n = min(n_particles, len(unique_trajs))
    sampled_ids = sorted(np.random.choice(unique_trajs, size=n, replace=False))
    sampled_trajs = rollingMLE_filtered[
        rollingMLE_filtered["ur_trajectory"].isin(sampled_ids)
    ].sort_values(by=["ur_trajectory", "frame"], ignore_index=True)

    for traj_id in sampled_ids:
        this_traj, boundingBox = isolate_traj(rollingMLE_filtered, traj_id)
        try:
            file_pattern = this_traj["ur_trajectory"].replace(
                r"_traj::\d+", ".nd2", regex=True
            ).iloc[0]
            image_file = glob.glob(
                os.path.join(settings["io"]["data_directory"], file_pattern)
            )[0]
            if image_file != prev_open_image:
                img_stack = np.array(pims.open(image_file))
                prev_open_image = image_file
            bc_rows.append(bleaching_curve(img_stack, this_traj))
        except Exception as exc:
            print(f"  [bleaching] skipping {traj_id}: {exc}")

    return pd.concat(bc_rows, ignore_index=True) if bc_rows else pd.DataFrame()


def plot_bleaching_curves(settings: dict, bc: pd.DataFrame) -> None:
    """Plot per-trajectory and median bleaching curves for each condition.

    Saves one PDF per condition to
    ``settings['io']['plot_directory']/bleaching_curves/<condition>.pdf``.
    """
    plot_xlim = (
        settings["saspt"]["frame_interval"] * -30 * 1000,
        settings["saspt"]["frame_interval"] * 10 * 1000,
    )

    output_dirname = os.path.join(settings["io"]["plot_directory"], "bleaching_curves")
    os.makedirs(output_dirname, exist_ok=True)

    bc["corrected_signal"] = bc["signal"] - bc["background"]
    bc_norm = normalize_signal_by_baseline(bc, pre_window=10)
    bc_norm["signal_smooth"] = (
        bc_norm.groupby("ur_trajectory")["signal_norm"].transform(
            lambda x: x.rolling(3, min_periods=1).mean()
        )
    )

    median_fluorescence = (
        bc_norm.groupby(["condition", "rel_frame"], as_index=False).agg(
            median_signal=("signal_norm", "median")
        )
    )

    for cond in bc_norm["condition"].unique():
        output_filename = os.path.join(output_dirname, f"{cond}.pdf")
        print(f"Writing to {output_filename}")

        df_cond = bc_norm[bc_norm["condition"] == cond].copy()
        df_median = median_fluorescence[median_fluorescence["condition"] == cond].copy()

        df_cond["rel_frame"] = (
            df_cond["rel_frame"] * settings["saspt"]["frame_interval"] * 1000
        )
        df_median["rel_frame"] = (
            df_median["rel_frame"] * settings["saspt"]["frame_interval"] * 1000
        )

        plt.figure(figsize=(8, 6))
        for traj_id, traj_data in df_cond.groupby("ur_trajectory"):
            plt.plot(
                traj_data["rel_frame"],
                traj_data["signal_smooth"],
                color="darkgray",
                alpha=0.25,
            )
        plt.plot(df_median["rel_frame"], df_median["median_signal"], color="black")

        plt.xlim(plot_xlim[0], plot_xlim[1])
        plt.ylim(-0.5, 3.5)
        plt.xlabel("Time rel bleaching (ms)")
        plt.ylabel("Signal")
        plt.title(f"Condition: {cond}")
        plt.tight_layout()
        plt.savefig(output_filename, dpi=300)
        plt.close()


# ---------------------------------------------------------------------------
# Movie generation
# ---------------------------------------------------------------------------

def generate_movies(
    rollingMLE: pd.DataFrame,
    settings: dict,
    min_obs: int = 8,
    overlay: bool = True,
) -> None:
    """Render overlay movies for a random sample of trajectories.

    Movies are saved to
    ``settings['io']['movie_directory']/<condition>/<ur_trajectory>.mp4``.

    Parameters
    ----------
    rollingMLE:
        Rolling-MLE DataFrame.
    settings:
        Workflow settings dict.
    min_obs:
        Minimum trajectory length required for inclusion.
    overlay:
        If True, composite the annotation layer onto the grayscale image.
    """
    counts = rollingMLE["ur_trajectory"].value_counts()
    valid_trajectories = counts[counts >= min_obs].index
    rollingMLE_filtered = rollingMLE[
        rollingMLE["ur_trajectory"].isin(valid_trajectories)
    ].copy()

    condition = (
        rollingMLE_filtered["ur_trajectory"]
        .str.replace(r"_\d+_traj::\d+", "", regex=True)
        .unique()[0]
    )

    unique_trajs = np.unique(rollingMLE_filtered["ur_trajectory"])
    n = min(100, len(unique_trajs))
    sampled_trajs = np.random.choice(unique_trajs, size=n, replace=False)

    output_dirname = os.path.join(settings["io"]["movie_directory"], condition)
    os.makedirs(output_dirname, exist_ok=True)
    prev_open_image = ""

    for traj_id in sampled_trajs:
        this_traj, boundingBox = isolate_traj(rollingMLE_filtered, traj_id)
        output_filename = os.path.join(
            output_dirname, f"{this_traj['ur_trajectory'].iloc[0]}.mp4"
        )

        try:
            file_pattern = this_traj["ur_trajectory"].replace(
                r"_traj::\d+", ".nd2", regex=True
            ).iloc[0]
            matches = glob.glob(
                os.path.join(settings["io"]["data_directory"], file_pattern)
            )
            if not matches:
                print(f"  [movie] skipping {traj_id}: no .nd2 file matched '{file_pattern}'")
                continue
            image_file = matches[0]

            if image_file != prev_open_image:
                img = pims.open(image_file)
                img_stack = np.array(img)
                prev_open_image = image_file

            # Clamp all bounding-box indices to the actual image dimensions.
            # isolate_traj uses max_frame=500 by default, which is wrong for
            # longer movies; clamping here corrects that without restructuring
            # the loop.
            T, H, W = img_stack.shape
            t0   = max(0, boundingBox["timeBox"][0])
            t1   = min(T, max(t0 + 1, boundingBox["timeBox"][1]))
            ymin = max(0, boundingBox["spaceBox"]["ymin"])
            ymax = min(H, boundingBox["spaceBox"]["ymax"])
            xmin = max(0, boundingBox["spaceBox"]["xmin"])
            xmax = min(W, boundingBox["spaceBox"]["xmax"])

            if t0 >= t1 or ymin >= ymax or xmin >= xmax:
                print(
                    f"  [movie] skipping {traj_id}: bounding box "
                    f"[t={t0}:{t1}, y={ymin}:{ymax}, x={xmin}:{xmax}] "
                    f"is empty for image shape ({T}, {H}, {W})"
                )
                continue

            clamped_spaceBox = {"xmin": xmin, "xmax": xmax, "ymin": ymin, "ymax": ymax}
            box = img_stack[t0:t1, ymin:ymax, xmin:xmax]

            box_resized, box_resized_annotations = annotate_trajectory_building_tail(
                box, this_traj, clamped_spaceBox, upscale=8, spot_radius=12, lwd=3
            )

            if overlay:
                save_overlay_movie(box_resized, box_resized_annotations, output_filename)
            else:
                box_resized_annotations = np.zeros_like(np.array(box_resized_annotations))
                save_overlay_movie(box_resized, box_resized_annotations, output_filename)

        except Exception as e:
            print(f"  [movie] skipping {traj_id}: {type(e).__name__}: {e}")


def generate_fullfield_overlay_movie(
    traj_csv: str,
    nd2_path: str,
    output_path: str,
    upscale: int = 1,
    spot_radius: int = 8,
    lwd: int = 2,
    fps: int = 10,
    color: tuple = (255, 60, 60),
) -> None:
    """Render one whole-field-of-view overlay movie straight from a raw
    ``_traj.csv``: every detection in every frame is circled, and detections
    linked into the same trajectory across consecutive frames are joined by
    a line (accumulating over time, like a growing tail).

    Unlike :func:`generate_movies` (which crops to a small window around a
    handful of *sampled* trajectories and colours by MLE_D), this shows
    every detection quot called for one file, at native frame scale, with a
    single fixed colour -- useful as an illustrative "what did tracking
    actually see" movie for one raw file, independent of any downstream
    SASPT/rolling-window analysis.

    Parameters
    ----------
    traj_csv:
        Path to the ``_traj.csv`` written by :func:`~core.filewise.run_tracking`
        (columns ``x``, ``y``, ``frame``, ``trajectory``).
    nd2_path:
        Path to the corresponding raw .nd2 movie.
    output_path:
        Where to write the ``.mp4``.
    upscale:
        Integer upscaling factor (1 = native resolution).
    spot_radius, lwd:
        Circle radius and line width, in upscaled pixels.
    fps:
        Playback frame rate of the output movie.
    color:
        RGB colour used for both circles and connecting lines.
    """
    df = pd.read_csv(traj_csv, comment="#").sort_values(["trajectory", "frame"])

    stack = np.array(pims.open(nd2_path))
    T, H, W = stack.shape

    segments_by_frame: dict[int, list] = {t: [] for t in range(T)}
    for _, group in df.groupby("trajectory"):
        group = group.reset_index(drop=True)
        for i in range(1, len(group)):
            f_prev, f_curr = int(group.loc[i - 1, "frame"]), int(group.loc[i, "frame"])
            if f_curr == f_prev + 1:
                pt1 = (group.loc[i - 1, "x"] * upscale, group.loc[i - 1, "y"] * upscale)
                pt2 = (group.loc[i, "x"] * upscale, group.loc[i, "y"] * upscale)
                for t in range(f_curr, T):
                    segments_by_frame[t].append((pt1, pt2))

    detections_by_frame = {
        t: g[["x", "y"]].values for t, g in df.groupby("frame")
    }

    annotated_frames: list[np.ndarray] = []
    for t in range(T):
        img_up = Image.fromarray(np.zeros((H * upscale, W * upscale, 3), dtype=np.uint8), "RGB")
        draw = ImageDraw.Draw(img_up)

        for pt1, pt2 in segments_by_frame[t]:
            draw.line([pt1, pt2], fill=color, width=lwd)

        for x, y in detections_by_frame.get(t, np.empty((0, 2))):
            x, y = x * upscale, y * upscale
            bbox = [x - spot_radius, y - spot_radius, x + spot_radius, y + spot_radius]
            draw.ellipse(bbox, outline=color, width=lwd)

        annotated_frames.append(np.array(img_up))

    if upscale != 1:
        stack = np.array([
            np.array(Image.fromarray(f).resize((W * upscale, H * upscale), resample=Image.BICUBIC))
            for f in stack
        ])

    save_overlay_movie(stack, annotated_frames, output_path, fps=fps)
    print(f"  Saved {output_path}")


# ---------------------------------------------------------------------------
# Trajectory scoring
# ---------------------------------------------------------------------------

def compute_runs_df(
    traj_df: pd.DataFrame,
    tol: float = 0.1,
    column: str = "MLE_D",
) -> list[tuple]:
    """Compute consecutive runs of roughly constant diffusion rate.

    Parameters
    ----------
    traj_df:
        Single-trajectory DataFrame.
    tol:
        Maximum absolute difference to consider two values the same state.
    column:
        Column name containing the diffusion-rate values.

    Returns
    -------
    List of ``(state_value, run_length)`` tuples.
    """
    traj = traj_df[column].values
    runs: list[tuple] = []
    if len(traj) == 0:
        return runs
    current_val = traj[0]
    run_len = 1
    for val in traj[1:]:
        if abs(val - current_val) <= tol:
            run_len += 1
        else:
            runs.append((current_val, run_len))
            current_val = val
            run_len = 1
    runs.append((current_val, run_len))
    return runs


def trajectory_score_df(
    traj_df: pd.DataFrame,
    run_tol: float = 0.1,
    min_run_len: int = 5,
    column: str = "MLE_D",
) -> tuple[int, int, float]:
    """Compute simple metrics for filtering visually interesting trajectories.

    Returns
    -------
    ``(length, num_long_runs, state_var)`` where *length* is the trajectory
    length in frames, *num_long_runs* is the number of runs exceeding
    *min_run_len*, and *state_var* is the standard deviation of MLE_D.
    """
    runs = compute_runs_df(traj_df, tol=run_tol, column=column)
    long_runs = [r for r in runs if r[1] >= min_run_len]
    num_long_runs = len(long_runs)
    state_var = traj_df[column].std()
    length = len(traj_df)
    return length, num_long_runs, state_var
