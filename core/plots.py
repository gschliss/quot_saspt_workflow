"""
core/plots.py

All plotting and visualisation functions, moved from plotFunctions.py.

Includes:
- Outlier detection (find_outlier_filenames, find_outlier_positions)
- Posterior/MLE histogram plots (generate_posterior_plotting_df, generate_plots)
- Trajectory survival curves (plot_survival_by_metadata, plot_survival_HMM)
- Hidden Markov Model fitting (fitHMM)
- Survival-curve helpers (run_lengths_to_survival_curve, get_runs)
- Movie / annotation helpers (isolate_traj, annotate_trajectory_building_tail,
  save_overlay_movie, save_overlay_tiff)
- Bleaching-curve analysis (bleaching_curve, normalize_signal_by_baseline,
  aggregate_bleaching_data, plot_bleaching_curves)
- Movie generation (generate_movies)
- Trajectory scoring (trajectory_score_df, compute_runs_df)

Note: The TrackMate/ImageJ block that existed in the original file has been
intentionally excluded as it was already commented out.
"""

from __future__ import annotations

import os
import re
import glob
import warnings
from collections import defaultdict

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

from core.utils import extract_metadata, wilson_ci

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

def plot_survival_by_metadata(
    settings: dict,
    metadata_fields: list[str] | None = None,
    subset: list[str] | None = None,
    min_detections: int = 2,
    xlim: float = 120,
    title: str = "",
) -> None:
    """Plot trajectory survival curves grouped by filename metadata.

    Metadata values are extracted from filenames using
    :func:`~core.utils.extract_metadata`.

    Parameters
    ----------
    settings:
        Workflow settings dict.
    metadata_fields:
        List of metadata keys to look for in filenames (e.g. ``['send', 'rec']``).
    subset:
        Optional list of substrings; only files whose name contains one of these
        substrings are included.
    min_detections:
        Minimum number of detections required for a trajectory to be counted.
    xlim:
        Upper x-axis limit (in the units of ``frame_interval``).
    title:
        Optional suffix appended to the output PDF filename.
    """
    if metadata_fields is None:
        metadata_fields = ["send", "rec"]
    if subset is None:
        subset = []

    metadata_groups: dict[str, dict] = {field: defaultdict(list) for field in metadata_fields}

    file_list = [
        os.path.join(settings["io"]["traj_directory"], fi)
        for fi in os.listdir(settings["io"]["traj_directory"])
        if fi.endswith("csv")
    ]
    file_pattern = [re.sub(r"_\d{3}_traj\.csv$", "", f) for f in file_list]
    file_pattern = list(dict.fromkeys(file_pattern))

    for f in file_pattern:
        metadata = extract_metadata(f, metadata_fields)
        for field in metadata_fields:
            if field in metadata:
                metadata_groups[field][metadata[field]].append(f)

    if subset:
        for field in metadata_groups:
            for key, files in metadata_groups[field].items():
                filtered_files: list[str] = []
                for sub in subset:
                    filtered_files.extend(f for f in files if sub in f)
                metadata_groups[field][key] = filtered_files

    for field in metadata_groups:
        empty_keys = [k for k, v in metadata_groups[field].items() if not v]
        for k in empty_keys:
            del metadata_groups[field][k]

    for field in metadata_groups:
        for value, patterns in metadata_groups[field].items():
            plt.figure(figsize=(8, 5))
            for p in patterns:
                matching_files = glob.glob(p + "*.csv")
                dat = pd.DataFrame()
                for f in matching_files:
                    newDat = pd.read_csv(f)
                    newDat["trajectory"] = f"{f}::" + newDat["trajectory"].astype(str)
                    dat = pd.concat([dat, newDat], ignore_index=True)
                df_relative = dat.copy()
                df_relative["frame_offset"] = df_relative["frame"] - df_relative.groupby(
                    "trajectory"
                )["frame"].transform("min")
                df_relative = df_relative[
                    df_relative["frame_offset"] >= (min_detections - 1)
                ]
                df_relative = df_relative.groupby("trajectory").filter(
                    lambda g: g["I0"].mean() > settings["quot"]["track"]["min_I0"]
                )

                survival_counts = (
                    df_relative.groupby("frame_offset")["trajectory"].nunique().sort_index()
                )
                total_trajectories = df_relative["trajectory"].nunique()

                results = []
                for frame_offset, n_alive in survival_counts.items():
                    frac = n_alive / total_trajectories
                    ci_low, ci_high = wilson_ci(n_alive, total_trajectories)
                    results.append((frame_offset, frac, ci_low, ci_high))

                survival_df = pd.DataFrame(
                    results, columns=["relative_frame", "survival_prob", "ci_low", "ci_high"]
                )
                survival_df["time"] = (
                    survival_df["relative_frame"] * settings["quot"]["track"]["frame_interval"]
                )

                label = os.path.basename(p)
                plt.fill_between(
                    survival_df["time"], survival_df["ci_low"], survival_df["ci_high"], alpha=0.3
                )
                plt.plot(survival_df["time"], survival_df["survival_prob"], label=label)

            plotname = (
                f"survival_by_{field}_{value}"
                if title == ""
                else f"survival_by_{field}_{value}_{title}"
            )
            plotfile = os.path.join(settings["io"]["plot_directory"], f"{plotname}.pdf")
            plt.xlabel("Time (s)")
            plt.ylabel("Survival probability")
            plt.title(f"Survival curves — {field} = {value}")
            plt.xlim(0, xlim)
            plt.legend(fontsize="small")
            plt.tight_layout()
            plt.savefig(plotfile, dpi=300, format="pdf", bbox_inches="tight")
            plt.close()
            print(f"  Saved {os.path.basename(plotfile)}")


def run_lengths_to_survival_curve(
    run_lengths, max_lag: int | None = None, min_run: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert a list of run lengths to a survival curve with Wilson 95 % CIs.

    Parameters
    ----------
    run_lengths:
        1-D array-like of integer run lengths.
    max_lag:
        Maximum Δt to compute (defaults to ``max(run_lengths)``).
    min_run:
        Exclude runs shorter than this value.

    Returns
    -------
    ``(survival_curve, ci_low, ci_high)`` — three 1-D numpy arrays indexed
    by Δt.  All three are empty if there are no valid run lengths.
    """
    run_lengths = np.array(run_lengths)
    run_lengths = run_lengths[run_lengths > min_run]
    if len(run_lengths) == 0:
        return np.array([]), np.array([]), np.array([])
    if max_lag is None:
        max_lag = int(run_lengths.max())
    total = len(run_lengths)
    survival_curve = np.zeros(max_lag)
    ci_low = np.zeros(max_lag)
    ci_high = np.zeros(max_lag)
    for dt in range(max_lag):
        n_alive = int(np.sum(run_lengths > dt))
        survival_curve[dt] = n_alive / total
        ci_low[dt], ci_high[dt] = wilson_ci(n_alive, total)
    return survival_curve, ci_low, ci_high


def get_runs(pops: np.ndarray, pop: int) -> np.ndarray:
    """Return lengths of consecutive runs of state *pop* in the *pops* array."""
    is_pop = (pops == pop).astype(int)
    if len(is_pop) == 0:
        return []
    padded = np.pad(is_pop, (1, 1), mode="constant")
    diff = np.diff(padded)
    run_starts = np.where(diff == 1)[0]
    run_ends = np.where(diff == -1)[0]
    return run_ends - run_starts


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


def compute_naive_pop(
    rollingMLE: pd.DataFrame, settings: dict, min_obs: int = 7
) -> pd.DataFrame:
    """Classify each rolling-window MLE_D observation by simple thresholding.

    Bins ``log10(MLE_D)`` against ``settings['plot']['barGraphBreaks']``,
    independent of any HMM fit. Intended to be plotted (via
    :func:`plot_survival_HMM` with ``pop_column="naive_pop"``) *before*
    :func:`fitHMM`'s state refinement, as a baseline for comparison.

    Returns
    -------
    A copy of *rollingMLE* with an added ``naive_pop`` column.
    """
    dynamic_MLE_df, *_ = _prepare_rolling_mle_for_hmm(rollingMLE, settings, min_obs=min_obs)
    return dynamic_MLE_df


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


def plot_survival_HMM(
    rolling_MLE: pd.DataFrame,
    settings: dict,
    cond: str,
    max_lag: int = 50,
    showPlot: bool = False,
    min_run: int = 0,
    pop_column: str = "posterior_pop",
) -> None:
    """Plot empirical state-dwell survival curves for all populations.

    One curve per population is drawn on the same axes, with 95 % Wilson CI
    shading.  Population labels come from
    ``settings['plot']['barGraphLabels']`` when available.

    Parameters
    ----------
    pop_column:
        Which per-observation population label to use: ``"naive_pop"``
        (threshold-only, independent of any HMM fit — see
        :func:`compute_naive_pop`) or ``"posterior_pop"`` (HMM-refined,
        from :func:`fitHMM`). Determines both the plot title and output
        filename.

    Saved to ``settings['io']['plot_directory']/survival_curves/
    <cond>_<naive|HMM>_survival.pdf``.
    """
    fit_label = {"naive_pop": "naive", "posterior_pop": "HMM"}.get(pop_column, pop_column)

    df = rolling_MLE.copy()
    df = df.dropna(subset=[pop_column])
    df[pop_column] = df[pop_column].astype(int)

    populations = sorted(df[pop_column].unique())
    pop_labels = settings["plot"].get("barGraphLabels", None)

    run_lengths_dict: dict[int, list] = {pop: [] for pop in populations}
    for traj_id, traj in df.groupby("ur_trajectory"):
        traj = traj.sort_values("frame")
        pops = traj[pop_column].values
        for pop in populations:
            run_lengths_dict[pop].extend(get_runs(pops, pop))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(populations), 1)))
    dt = settings["saspt"]["frame_interval"]

    # Compute all survival curves first (no frame cap) so we know the true
    # data extent before deciding axis units and limits.
    curves: dict[int, tuple] = {}
    for pop in populations:
        sc, ci_lo, ci_hi = run_lengths_to_survival_curve(
            run_lengths_dict[pop], max_lag=None, min_run=min_run
        )
        if len(sc) > 0:
            curves[pop] = (sc, ci_lo, ci_hi)

    # x-axis limits and units:
    #   - fast SPT (natural data extent < 100 ms): always show [0, 1000 ms]
    #   - otherwise: cap at 120 s or full data extent, whichever is shorter;
    #     use seconds when the displayed range >= 1 s, else milliseconds.
    t_max_s = max((len(sc) * dt for sc, _, _ in curves.values()), default=0.0)
    if dt < 0.1:                           # fast SPT (dt < 100 ms) — force 1000 ms window
        time_scale = 1000.0
        xlabel     = "Time (ms)"
        xlim_max   = 1000.0
    else:
        xlim_s = min(t_max_s, 120.0)
        if xlim_s >= 1.0:
            time_scale = 1.0
            xlabel     = "Time (s)"
            xlim_max   = xlim_s
        else:
            time_scale = 1000.0
            xlabel     = "Time (ms)"
            xlim_max   = xlim_s * 1000.0

    for pop, color in zip(populations, colors):
        if pop not in curves:
            continue
        sc, ci_lo, ci_hi = curves[pop]
        t = np.arange(len(sc)) * dt * time_scale
        # Clip to display range
        mask   = t <= xlim_max
        t      = t[mask]
        sc     = sc[mask]
        ci_lo  = ci_lo[mask]
        ci_hi  = ci_hi[mask]

        n_runs = len([r for r in run_lengths_dict[pop] if r > min_run])
        label_name = (
            pop_labels[pop]
            if pop_labels is not None and pop < len(pop_labels)
            else f"Pop {pop}"
        )
        label = f"{label_name}  (n={n_runs})"

        ax.fill_between(t, ci_lo, ci_hi, alpha=0.25, color=color)
        ax.plot(t, sc, color=color, label=label)

    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, xlim_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Survival probability")
    ax.set_title(f"{cond} — {fit_label} state dwell-time survival")
    ax.legend(fontsize="small")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()

    output_dirname = os.path.join(settings["io"]["plot_directory"], "survival_curves")
    os.makedirs(output_dirname, exist_ok=True)
    output_filename = os.path.join(output_dirname, f"{cond}_{fit_label}_survival.pdf")
    fig.savefig(output_filename, dpi=300, format="pdf", bbox_inches="tight")
    print(f"  Saved {os.path.basename(output_filename)}")

    if showPlot:
        plt.show()
    plt.close(fig)


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
