"""
core/settings.py

All settings logic:
- Default settings definition (get_default_settings)
- Directory-layout expansion (update_default_settings_for_analysis)
- Image-metadata updates (update_settings_with_image_metadata)
- Convenience loader that wires all three steps together (load_settings)
- Utility helpers: deep_update, print_nested_dict
"""

from __future__ import annotations

import os
import glob
import yaml
import numpy as np
import pims

# Ensure fastQuot is on the path (harmless to call again if __init__ already ran)
import core  # noqa: F401  — triggers core/__init__.py sys.path insert


# ---------------------------------------------------------------------------
# Utility helpers (moved from trackingCodeFunctions.py)
# ---------------------------------------------------------------------------

def deep_update(d: dict, u: dict) -> None:
    """Recursively update *d* in-place with values from *u*.

    Nested dicts are merged; all other values are replaced.
    A null/empty YAML key (parsed as None) is silently skipped when the
    existing value is a dict — this prevents an accidental bare ``io:``
    in settings_override.yaml from wiping out the entire io section.
    """
    for k, v in u.items():
        if v is None and isinstance(d.get(k), dict):
            continue  # bare YAML key with no value — don't clobber the dict
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v


def print_nested_dict(d: dict, indent: int = 0, file=None) -> None:
    """Pretty-print a nested dict, optionally writing to a file path or object."""
    should_close = False
    if isinstance(file, str):
        file = open(file, "w")
        should_close = True

    for key, value in d.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key} ::", file=file)
            print_nested_dict(value, indent + 2, file=file)
        else:
            print(" " * indent + f"{key} :: {value}", file=file)

    if should_close:
        file.close()


# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

def get_default_settings() -> dict:
    """Return a fresh settings dict populated with sensible defaults."""
    from saspt import RBME  # imported here so saspt is not required at import time

    settings: dict = {}
    settings["io"] = {}
    settings["quot"] = {}
    settings["saspt"] = {}
    settings["plot"] = {}

    # -- I/O ------------------------------------------------------------------
    settings["io"]["home_directory"] = ""
    settings["io"]["data_directory"] = ""       # set at runtime by load_settings
    settings["io"]["code_directory"] = ""       # set at runtime
    settings["io"]["split_dimension"] = 0       # pixel dimension to split large images
    settings["io"]["forceQUOT"] = True
    settings["io"]["plotOnly"] = False

    # -- quot filter ----------------------------------------------------------
    settings["quot"]["filter"] = {}
    settings["quot"]["filter"]["start"] = 0
    settings["quot"]["filter"]["method"] = "identity"
    settings["quot"]["filter"]["chunk_size"] = 100

    # -- quot detect ----------------------------------------------------------
    settings["quot"]["detect"] = {}
    settings["quot"]["detect"]["method"] = "llr"
    settings["quot"]["detect"]["k"] = 2.0
    settings["quot"]["detect"]["w"] = 15
    settings["quot"]["detect"]["t"] = 20.00

    # -- quot localize --------------------------------------------------------
    settings["quot"]["localize"] = {}
    settings["quot"]["localize"]["method"] = "ls_int_gaussian"
    settings["quot"]["localize"]["window_size"] = 15
    settings["quot"]["localize"]["sigma"] = 2.5
    settings["quot"]["localize"]["ridge"] = 0.00001
    settings["quot"]["localize"]["max_iter"] = 50
    settings["quot"]["localize"]["damp"] = 1
    settings["quot"]["localize"]["camera_bg"] = 100

    # -- quot track -----------------------------------------------------------
    settings["quot"]["track"] = {}
    settings["quot"]["track"]["method"] = "euclidean"
    settings["quot"]["track"]["pixel_size_um"] = 0.11
    settings["quot"]["track"]["frame_interval"] = 0.006
    settings["quot"]["track"]["search_radius"] = 1.2
    settings["quot"]["track"]["max_blinks"] = 0
    settings["quot"]["track"]["min_I0"] = 100.0
    settings["quot"]["track"]["scale"] = 1.0

    # -- saspt ----------------------------------------------------------------
    settings["saspt"]["likelihood_type"] = RBME
    settings["saspt"]["pixel_size_um"] = 0.11
    settings["saspt"]["frame_interval"] = 0.006
    settings["saspt"]["focal_depth"] = 0.1
    settings["saspt"]["progress_bar"] = "False"
    settings["saspt"]["splitsize"] = 5
    settings["saspt"]["sample_size"] = 100000
    settings["saspt"]["num_workers"] = 4
    settings["saspt"]["diff_coefs"] = np.power(10, np.linspace(-2, 2, 125))
    settings["saspt"]["loc_errors"] = np.linspace(0.025, 0.028, 5)
    settings["saspt"]["start_frame"] = 0

    # -- plot -----------------------------------------------------------------
    settings["plot"]["xlim"] = (-2, 2)
    settings["plot"]["bleach_xlim"] = (-300, 50)
    settings["plot"]["ylim"] = [0, 5]
    settings["plot"]["nbins"] = 100
    settings["plot"]["tick_positions"] = (-2, -1, 0, 1, 2)
    settings["plot"]["mult_on_sd"] = 5.0
    settings["plot"]["barGraphBreaks"] = [-10000, 10000]
    settings["plot"]["barGraphLabels"] = ["None"]

    return settings


# ---------------------------------------------------------------------------
# Directory layout expansion
# ---------------------------------------------------------------------------

def update_default_settings_for_analysis(settings: dict, data_directory: str) -> dict:
    """Populate all derivative I/O paths in *settings* and create directories.

    Parameters
    ----------
    settings:
        Settings dict (mutated in-place and also returned).
    data_directory:
        Absolute path to the folder containing .nd2 files.  All analysis
        output is placed in a sibling directory next to it.
    """
    settings["io"]["data_directory"] = data_directory
    settings["io"]["home_directory"] = os.path.dirname(data_directory)

    # code_directory is always the workflow root (parent of core/)
    settings["io"]["code_directory"] = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    traj_filestring = (
        f"tracking_output_q="
        f"{str(float(settings['quot']['detect']['t'])).replace('.', 'p')}"
    )
    settings["io"]["analysis_directory"] = os.path.join(
        settings["io"]["home_directory"], traj_filestring
    )
    settings["io"]["traj_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "trajectories"
    )
    settings["io"]["post_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "posterior"
    )
    settings["io"]["MLE_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "MLE"
    )
    settings["io"]["plot_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "plots"
    )
    settings["io"]["split_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "split_directory"
    )
    settings["io"]["split_traj_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "split_trajectories"
    )
    settings["io"]["rolling_window_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "rolling_windows"
    )
    settings["io"]["movie_directory"] = os.path.join(
        settings["io"]["analysis_directory"], "movies"
    )

    settings["quot"]["filter"]["start"] = 0

    os.makedirs(settings["io"]["analysis_directory"], exist_ok=True)
    os.makedirs(settings["io"]["traj_directory"], exist_ok=True)
    os.makedirs(settings["io"]["post_directory"], exist_ok=True)
    os.makedirs(settings["io"]["MLE_directory"], exist_ok=True)
    os.makedirs(settings["io"]["plot_directory"], exist_ok=True)
    os.makedirs(settings["io"]["split_directory"], exist_ok=True)
    os.makedirs(settings["io"]["split_traj_directory"], exist_ok=True)
    os.makedirs(settings["io"]["rolling_window_directory"], exist_ok=True)
    os.makedirs(settings["io"]["movie_directory"], exist_ok=True)

    return settings


# ---------------------------------------------------------------------------
# Image-metadata update
# ---------------------------------------------------------------------------

def update_settings_with_image_metadata(settings: dict, cond: str | None = None) -> dict:
    """Read frame-rate and background level from a representative .nd2 file.

    Parameters
    ----------
    settings:
        Settings dict (mutated in-place and also returned).
    cond:
        If provided, glob for ``*{cond}*.nd2`` inside the data directory;
        otherwise use the first .nd2 file found.
    """
    if cond is None:
        arbitrary_image = glob.glob(f"{settings['io']['data_directory']}/*.nd2")[0]
    else:
        arbitrary_image = glob.glob(
            f"{settings['io']['data_directory']}/*{cond}*.nd2"
        )[0]

    img = pims.open(arbitrary_image)
    frame = img[0]
    bg_level = np.median(frame)
    dt = 1 / img.frame_rate

    settings["quot"]["localize"]["camera_bg"] = bg_level
    settings["quot"]["track"]["frame_interval"] = dt
    settings["saspt"]["frame_interval"] = dt

    return settings


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_settings(data_directory: str) -> dict:
    """Build a fully-resolved settings dict for *data_directory*.

    Steps:
    1. Start with ``get_default_settings()``.
    2. Look for ``settings_override.yaml`` inside *data_directory*; if found,
       merge it with ``deep_update``.
    3. Call ``update_default_settings_for_analysis(settings, data_directory)``
       to expand all derivative I/O paths and create output directories.
    4. Return the fully-populated settings dict.

    The ``settings_override.yaml`` file must live in the same directory as
    the .nd2 files.  It must not (and need not) specify ``io.data_filepath``
    or any other path — the data directory is always taken from the argument
    passed to this function.

    Parameters
    ----------
    data_directory:
        Absolute path to the folder that contains .nd2 files.
    """
    settings = get_default_settings()

    override_path = os.path.join(data_directory, "settings_override.yaml")
    if os.path.exists(override_path):
        with open(override_path) as fh:
            override_cfg = yaml.load(fh, Loader=yaml.FullLoader)
        deep_update(settings, override_cfg)
        print(f"Applied settings overrides from {override_path}")

    update_default_settings_for_analysis(settings, data_directory)

    return settings
