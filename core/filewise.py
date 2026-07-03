"""
core/filewise.py

Per-file processing: detection, localization, tracking (via quot), SASPT
state inference, and rolling-window SASPT.

Public API
----------
run_tracking(nd2_path, settings)
    Run quot detection+localization+tracking on one .nd2 file.
    Writes ``<traj_directory>/<basename>_traj.csv``.

run_saspt(traj_csv, settings)
    Run SASPT on one ``_traj.csv``.
    Writes ``<post_directory>/<basename>_posterior.csv`` and
    ``<MLE_directory>/<basename>_MLE.csv``.

run_rolling_windows(traj_csv, settings)
    Run rolling-window SASPT on one ``_traj.csv``.
    Writes ``<rolling_window_directory>/<basename>_rollingMLE.csv``.

run_filewise(nd2_path, settings)
    Convenience wrapper: calls all three functions in sequence.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pandas as pd

import core  # noqa: F401  — ensures fastQuot is on sys.path before quot import
from quot.core import track_file

try:
    from saspt import StateArray
    _SASPT_AVAILABLE = True
except ImportError:
    _SASPT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def run_tracking(nd2_path: str, settings: dict) -> str:
    """Run quot detection + localization + tracking on one .nd2 file.

    Parameters
    ----------
    nd2_path:
        Absolute path to the input .nd2 file.
    settings:
        Workflow settings dict.  The ``quot`` sub-dict is forwarded to
        ``quot.core.track_file``.

    Returns
    -------
    Absolute path to the written ``_traj.csv`` file.
    """
    input_basename = os.path.basename(nd2_path)
    quot_outcsv = os.path.join(
        settings["io"]["traj_directory"],
        re.sub(r"\.nd2$", "_traj.csv", input_basename),
    )

    print(f"\n  Tracking: {input_basename}")
    print(f"  Saving trajectories to: {quot_outcsv}")
    track_file(nd2_path, out_csv=quot_outcsv, **settings["quot"])
    return quot_outcsv


# ---------------------------------------------------------------------------
# SASPT
# ---------------------------------------------------------------------------

def run_saspt(traj_csv: str, settings: dict) -> None:
    """Run SASPT on one ``_traj.csv`` file.

    Writes:
    - ``<MLE_directory>/<basename>_MLE.csv``
    - ``<post_directory>/<basename>_posterior.csv``

    Parameters
    ----------
    traj_csv:
        Absolute path to the trajectory CSV produced by :func:`run_tracking`.
    settings:
        Workflow settings dict.
    """
    if not _SASPT_AVAILABLE:
        raise RuntimeError(
            "saspt is not installed. Cannot run SASPT. "
            "Install it with: pip install saspt"
        )

    print(f"  Sending posterior output to : {settings['io']['post_directory']}")
    print(f"  Sending MLE output to       : {settings['io']['MLE_directory']}")
    print(f"  Loading spots from          : {traj_csv}")

    spots = pd.read_csv(traj_csv)

    # Keep all trajectories at this step (no length filter yet)

    # Filter trajectories with low mean intensity
    spots = spots.groupby("trajectory").filter(
        lambda g: g["I0"].mean() > settings["quot"]["track"]["min_I0"]
    )

    # SASPT (like a Markov/HMM state array) needs at least one trajectory with
    # more than one linked detection to estimate any state transition. A very
    # short movie (or one where min_I0 filtering removes most detections) can
    # leave zero such trajectories; saspt itself doesn't raise in that case,
    # it just silently returns a degenerate all-zero result. Detect that here
    # and skip explicitly instead, so the output files clearly reflect
    # "nothing usable was found" rather than looking like a normal result.
    n_multiframe_trajs = int((spots.groupby("trajectory").size() > 1).sum())
    if n_multiframe_trajs == 0:
        print(
            f"  [SASPT] {os.path.basename(traj_csv)} has no multi-frame trajectories "
            "after min_I0 filtering (movie too short/sparse to fit a state array); "
            "skipping SASPT and writing empty MLE/posterior output."
        )
        basename = os.path.basename(traj_csv)
        pd.DataFrame(
            columns=["orig_trajectory", "track_length", "MLE_D", "source_file"]
        ).to_csv(
            os.path.join(settings["io"]["MLE_directory"], basename.replace("_traj", "_MLE")),
            index=False,
        )
        pd.DataFrame(
            {
                "D": settings["saspt"]["diff_coefs"],
                "density": np.zeros(len(settings["saspt"]["diff_coefs"])),
                "file": basename,
            }
        ).to_csv(
            os.path.join(settings["io"]["post_directory"], basename.replace("_traj", "_posterior")),
            index=False,
        )
        return

    print("  Running SASPT")
    SA = StateArray.from_detections(spots, **settings["saspt"])
    marginal_D = SA.posterior_assignment_probabilities.sum(axis=1)

    print("  Extracting posterior and formatting export")
    MLE_D_vals = settings["saspt"]["diff_coefs"][marginal_D.argmax(axis=0)]
    soft_D = marginal_D.sum(axis=1)
    softD_df = pd.DataFrame(
        {
            "D": settings["saspt"]["diff_coefs"],
            "density": soft_D,
            "file": os.path.basename(traj_csv),
        }
    )

    detections_df = SA.trajectories.detections
    MLE_D_df = pd.DataFrame(MLE_D_vals, columns=["MLE_D"])
    MLE_D_df["trajectory"] = range(MLE_D_df.shape[0])
    detections_df = pd.merge(detections_df, MLE_D_df, on="trajectory")

    useful_trajname = os.path.basename(os.path.dirname(traj_csv))  # quality string
    useful_trajname = f"{useful_trajname}::{os.path.basename(traj_csv)}"

    MLE_df = SA.trajectories.detections.drop_duplicates(subset="trajectory")
    MLE_df = pd.merge(MLE_df, MLE_D_df, on="trajectory")
    MLE_df = MLE_df[["orig_trajectory", "track_length", "MLE_D"]]
    MLE_df["source_file"] = traj_csv
    MLE_df["orig_trajectory"] = (
        useful_trajname + "::" + MLE_df["orig_trajectory"].astype(str)
    )

    print("  Exporting data")
    basename = os.path.basename(traj_csv)
    MLE_filename = basename.replace("_traj", "_MLE")
    MLE_df.to_csv(
        os.path.join(settings["io"]["MLE_directory"], MLE_filename), index=False
    )
    posterior_filename = basename.replace("_traj", "_posterior")
    softD_df.to_csv(
        os.path.join(settings["io"]["post_directory"], posterior_filename), index=False
    )


# ---------------------------------------------------------------------------
# Rolling-window SASPT
# ---------------------------------------------------------------------------

def run_rolling_windows(traj_csv: str, settings: dict) -> pd.DataFrame:
    """Run rolling-window SASPT on one ``_traj.csv`` file.

    Each detection is described by an instantaneous diffusion rate computed
    from a symmetric window centred on that frame.  The window size is
    ``settings['saspt']['splitsize']``.

    Writes ``<rolling_window_directory>/<basename>_rollingMLE.csv``.

    Parameters
    ----------
    traj_csv:
        Absolute path to the trajectory CSV.
    settings:
        Workflow settings dict.

    Returns
    -------
    The output DataFrame (also saved to disk).
    """
    if not _SASPT_AVAILABLE:
        raise RuntimeError(
            "saspt is not installed. Cannot run rolling-window analysis. "
            "Install it with: pip install saspt"
        )

    window_size: int = settings["saspt"]["splitsize"]
    if window_size % 2 == 1:  # odd window: symmetric
        half_window = window_size // 2
        left_window = half_window
        right_window = half_window
    else:  # even window: left-shifted
        left_window = window_size // 2 - 1
        right_window = window_size // 2

    print("  Successfully initialised half-windows")
    print(f"  left window  : {left_window}")
    print(f"  right window : {right_window}")

    basename = os.path.basename(traj_csv).replace(".csv", "")

    df = pd.read_csv(traj_csv)
    df["ur_trajectory"] = df["trajectory"].apply(lambda t: f"{basename}::{t}")

    # Keep all trajectories in rolling-windows analysis

    all_rows: list[pd.DataFrame] = []
    new_traj_id = 0

    for _, group in df.groupby("ur_trajectory"):
        group = group.sort_values("frame").reset_index(drop=True)
        for i in range(left_window, len(group) - right_window):
            window = group.iloc[i - left_window : i + right_window + 1].copy()
            window["focal_point"] = False
            window.iloc[left_window, window.columns.get_loc("focal_point")] = True
            window["trajectory"] = new_traj_id
            all_rows.append(window)
            new_traj_id += 1

    if not all_rows:
        print(
            f"  [rolling windows] no windows produced for {basename} "
            f"(all trajectories shorter than window_size={window_size}); skipping SASPT."
        )
        return pd.DataFrame(columns=df.columns.tolist() + ["focal_point", "MLE_D"])

    out_df = pd.concat(all_rows, ignore_index=True)

    SA = StateArray.from_detections(out_df, **settings["saspt"])
    marginal_D = SA.posterior_assignment_probabilities.sum(axis=1)

    print("  Extracting posterior and formatting export")
    MLE_D_vals = settings["saspt"]["diff_coefs"][marginal_D.argmax(axis=0)]

    detections_df = SA.trajectories.detections
    MLE_D_df = pd.DataFrame(MLE_D_vals, columns=["MLE_D"])
    MLE_D_df["trajectory"] = range(MLE_D_df.shape[0])
    detections_df = pd.merge(detections_df, MLE_D_df, on="trajectory")
    detections_df = detections_df[detections_df["focal_point"] == True]

    merge_key = ["y", "x", "I0", "frame", "ur_trajectory"]
    new_df = pd.merge(
        df,
        detections_df[merge_key + ["MLE_D"]],
        on=merge_key,
        how="left",
    )
    new_df = new_df[merge_key + ["MLE_D"]]

    output_file = basename.replace("_traj", "_rollingMLE") + ".csv"
    output_file = os.path.join(settings["io"]["rolling_window_directory"], output_file)
    new_df.to_csv(output_file, index=False)

    return new_df


# ---------------------------------------------------------------------------
# Combined per-file entry point
# ---------------------------------------------------------------------------

def run_filewise(nd2_path: str, settings: dict) -> None:
    """Run tracking + SASPT + rolling windows on one .nd2 file.

    This is the single entry point called for each file, either from
    :mod:`run_local` or from the Snakemake ``filewise`` rule.

    Parameters
    ----------
    nd2_path:
        Absolute path to the .nd2 file to process.
    settings:
        Fully-resolved workflow settings dict.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(nd2_path)}")
    print(f"{'='*60}")

    try:
        traj_csv = run_tracking(nd2_path, settings)
    except Exception as e:
        # e.g. a movie with zero readable frames raises inside quot's own
        # localize_file(); nothing downstream can run without a traj_csv.
        print(f"WARNING: tracking failed for {os.path.basename(nd2_path)}: {e}")
        print("  Skipping SASPT and rolling windows for this file.")
        return

    print("\n  Executing SASPT")
    try:
        run_saspt(traj_csv, settings)
    except Exception as e:
        print(f"WARNING: SASPT failed for {os.path.basename(traj_csv)}: {e}")

    print("\n  Analysing rolling windows")
    try:
        run_rolling_windows(traj_csv, settings)
    except Exception as e:
        print(f"WARNING: rolling-window analysis failed for {os.path.basename(traj_csv)}: {e}")

    print(f"\n  Done: {os.path.basename(nd2_path)}")
