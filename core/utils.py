"""
core/utils.py

Shared utility functions used across the workflow:
- File discovery and grouping
- CSV aggregation
- Statistics helpers
- Metadata extraction from filenames
- Image tiling (for large-FOV SPT)
- SLURM job-monitoring helpers
- YAML serialisation helpers
"""

from __future__ import annotations

import os
import re
import copy
import subprocess
import glob as _glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import tifffile
from scipy.stats import norm

import core  # noqa: F401  — ensures fastQuot is on sys.path


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def group_files_by_metadata(directory: str, extension: str = ".nd2") -> dict[str, list[str]]:
    """Group .nd2 filenames by their condition metadata prefix.

    Files are expected to follow the naming convention
    ``<condition-string>_###.nd2`` where ``###`` is a zero-padded index.
    Files that do not match this pattern are silently ignored.

    Parameters
    ----------
    directory:
        Path to the folder containing the data files.
    extension:
        File extension to search for (default: ``.nd2``).

    Returns
    -------
    dict mapping condition string → sorted list of matching filenames
    (basenames only, not full paths).
    """
    files_by_metadata: dict[str, list[str]] = defaultdict(list)

    for filename in os.listdir(directory):
        if not filename.endswith(extension):
            continue
        match = re.match(r"(.*)_\d{3}" + re.escape(extension) + r"$", filename)
        if match:
            metadata_key = match.group(1)
            files_by_metadata[metadata_key].append(filename)

    return dict(files_by_metadata)


# ---------------------------------------------------------------------------
# Missing-file detection
# ---------------------------------------------------------------------------

def identify_missing_filewise(settings: dict, force: bool = False) -> dict[str, list[str]]:
    """Return a dict of condition → [.nd2 filenames] for files that still need processing.

    If *force* is True every .nd2 file in the data directory is returned,
    regardless of whether output files already exist.

    Uses ``settings['io']['data_directory']`` directly (not
    ``home_directory + data_filepath``) so callers only need to set
    ``data_directory`` once via :func:`~core.settings.load_settings`.
    """
    data_directory = settings["io"]["data_directory"]
    nd2_files_by_condition = group_files_by_metadata(data_directory)

    if force:
        return nd2_files_by_condition

    from core.settings import update_settings_with_image_metadata

    # Build the expected output paths for every input file. A slowSPT file
    # (see core/filewise.py) only ever produces a _traj.csv -- checking for
    # posterior/MLE/rollingMLE on one would always report it "missing" and
    # reprocess it on every invocation, since those are never written for a
    # slowSPT run. Mode is detected the same way core/filewise.py itself
    # does, on a throwaway settings copy so this peek doesn't disturb the
    # caller's settings dict (the real per-file update happens again, on the
    # actual settings dict, right before run_filewise).
    target_files: dict[str, list[str]] = {}
    for condition, file_list in nd2_files_by_condition.items():
        target_list: list[str] = []
        for f in file_list:
            base = os.path.splitext(f)[0]
            target_list.append(
                os.path.join(settings["io"]["traj_directory"], base + "_traj.csv")
            )

            peek_settings = copy.deepcopy(settings)
            update_settings_with_image_metadata(
                peek_settings, nd2_path=os.path.join(data_directory, f)
            )
            if peek_settings["io"]["mode"] == "fastSPT":
                target_list.append(
                    os.path.join(settings["io"]["post_directory"], base + "_posterior.csv")
                )
                target_list.append(
                    os.path.join(settings["io"]["MLE_directory"], base + "_MLE.csv")
                )
                target_list.append(
                    os.path.join(
                        settings["io"]["rolling_window_directory"], base + "_rollingMLE.csv"
                    )
                )
        target_files[condition] = target_list

    # Identify which expected outputs are absent
    missing_files: dict[str, list[str]] = {}
    for key, file_list in target_files.items():
        missing = [f for f in file_list if not os.path.exists(f)]
        if missing:
            missing_files[key] = missing

    if missing_files:
        print("\nMissing files.")
    else:
        print("\nAll files exist.\n")

    # Convert missing CSV paths back to the corresponding .nd2 basenames
    nd2_files_by_condition = {}
    for key, files in missing_files.items():
        nd2_set: set[str] = set()
        for f in files:
            base = os.path.basename(f)
            # Strip the analysis suffix and replace with .nd2
            base = re.sub(r"(_\d+)(?:_.*)?\.csv$", r"\1.nd2", base)
            nd2_set.add(base)
        nd2_files_by_condition[key] = sorted(nd2_set)

    print("\nRun file-wise operations for .nd2 files (grouped by condition):")
    for key, files in nd2_files_by_condition.items():
        print(f"\n{key}:")
        for f in files:
            print(f"\t{f}")

    return nd2_files_by_condition


# ---------------------------------------------------------------------------
# CSV aggregation
# ---------------------------------------------------------------------------

def aggregate_csv(directory_string: str, filter_string: str = "*.csv") -> pd.DataFrame:
    """Load and concatenate all CSVs matching *filter_string* in *directory_string*.

    Parameters
    ----------
    directory_string:
        Path to the directory to search.
    filter_string:
        Glob pattern relative to *directory_string* (default: ``*.csv``).

    Returns
    -------
    A single concatenated :class:`pandas.DataFrame`.
    """
    csv_dir = Path(directory_string)
    csv_files = sorted(csv_dir.glob(filter_string))
    df_list = [pd.read_csv(f, comment="#") for f in csv_files]
    return pd.concat(df_list, ignore_index=True)


# ---------------------------------------------------------------------------
# Metadata header (auditing)
# ---------------------------------------------------------------------------

def _format_metadata_value(value) -> str:
    """Render a settings value as a single-line string for the CSV header."""
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return "[]"
        return f"min={value.min():.4g}, max={value.max():.4g}, n={value.size}"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _flatten_dict(d: dict, prefix: str = "") -> list[tuple[str, object]]:
    items: list[tuple[str, object]] = []
    for key, value in d.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            items.extend(_flatten_dict(value, full_key))
        else:
            items.append((full_key, value))
    return items


def metadata_header_lines(settings: dict, source: str | None = None) -> list[str]:
    """Build human-readable audit lines describing the settings used to
    produce a given output file (pixel size, imaging interval, and every
    other ``quot``/``saspt`` parameter). Meant to be written as ``#``-prefixed
    comment lines at the top of a trajectory/posterior/MLE CSV -- for
    auditing only, never read back by the pipeline itself.
    """
    lines = ["quot_saspt_workflow run metadata (for auditing only)"]
    if source is not None:
        lines.append(f"generated_from: {source}")
    for section in ("quot", "saspt"):
        for key, value in _flatten_dict(settings[section]):
            lines.append(f"{section}.{key}: {_format_metadata_value(value)}")
    return lines


def write_csv_with_metadata_header(
    df: pd.DataFrame, path: str, settings: dict, source: str | None = None, index: bool = False
) -> None:
    """Write *df* to *path* as CSV, preceded by a ``#``-commented metadata header."""
    with open(path, "w", newline="") as fh:
        for line in metadata_header_lines(settings, source=source):
            fh.write(f"# {line}\n")
        df.to_csv(fh, index=index)


def prepend_metadata_header(path: str, settings: dict, source: str | None = None) -> None:
    """Prepend a ``#``-commented metadata header to an already-written CSV at *path*.

    Used for files written by code with no header hook (e.g. quot's own
    ``track_file``, which calls ``DataFrame.to_csv`` internally) -- the file
    is read back once it's fully written, then rewritten with the header on top.
    """
    with open(path, "r") as fh:
        original = fh.read()
    header = "".join(f"# {line}\n" for line in metadata_header_lines(settings, source=source))
    with open(path, "w") as fh:
        fh.write(header)
        fh.write(original)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Compute Wilson score confidence interval for a binomial proportion.

    Parameters
    ----------
    k:
        Number of successes (trajectories still alive at time *t*).
    n:
        Total number of trials.
    confidence:
        Desired confidence level (default: 0.95).

    Returns
    -------
    ``(lower, upper)`` bounds of the confidence interval.
    """
    if n == 0:
        return (0.0, 0.0)
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = k / n
    denominator = 1 + z**2 / n
    centre = phat + z**2 / (2 * n)
    margin = z * np.sqrt((phat * (1 - phat) + z**2 / (4 * n)) / n)
    lower = (centre - margin) / denominator
    upper = (centre + margin) / denominator
    return lower, upper


# ---------------------------------------------------------------------------
# Filename metadata extraction
# ---------------------------------------------------------------------------

def extract_metadata(file_path: str, keys: list[str]) -> dict[str, str]:
    """Extract ``key=value`` pairs embedded in a filename.

    Parameters
    ----------
    file_path:
        Path or basename to parse.
    keys:
        List of key names to search for.

    Returns
    -------
    Dict of ``{key: value}`` for every key that was found.
    """
    metadata: dict[str, str] = {}
    for key in keys:
        match = re.search(rf"{key}=([^_]+)", file_path)
        if match:
            metadata[key] = match.group(1)
    return metadata


# ---------------------------------------------------------------------------
# Type-conversion helpers (YAML serialisation)
# ---------------------------------------------------------------------------

def to_builtin(obj):
    """Recursively convert numpy scalar/array types to Python built-ins.

    Required before serialising a settings dict with :mod:`yaml` or
    :mod:`toml`, which do not know about numpy types.
    """
    if isinstance(obj, dict):
        return {k: to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin(i) for i in obj]
    elif isinstance(obj, (np.float64, np.float32, np.floating)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32, np.integer)):
        return int(obj)
    else:
        return obj


def save_dict_to_yaml(data: dict, filepath: str) -> None:
    """Serialise *data* to a YAML file at *filepath*."""
    with open(filepath, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False)


# ---------------------------------------------------------------------------
# Image tiling (for dense / large-FOV SPT)
# ---------------------------------------------------------------------------

def split_image_stack(
    img: np.ndarray,
    tile_size: int = 110,
    overlap: int = 10,
) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    """Split a 3-D image stack (T, Y, X) into overlapping spatial tiles.

    Parameters
    ----------
    img:
        Array of shape ``(T, Y, X)``.
    tile_size:
        Side length of each square tile in pixels.
    overlap:
        Overlap between adjacent tiles in pixels.

    Returns
    -------
    ``(tiles, positions)`` where *positions* is a list of ``(x_start, y_start)``
    tuples corresponding to each tile.
    """
    T, Y, X = img.shape
    step = tile_size - overlap
    tiles: list[np.ndarray] = []
    positions: list[tuple[int, int]] = []

    def _starts(dim: int) -> list[int]:
        starts = list(range(0, dim - tile_size + 1, step))
        if starts and starts[-1] + tile_size < dim:
            starts.append(starts[-1] + step)
        elif not starts:
            starts = [0]
        return starts

    for y_start in _starts(Y):
        for x_start in _starts(X):
            tile = img[:, y_start : y_start + tile_size, x_start : x_start + tile_size]
            tiles.append(tile)
            positions.append((x_start, y_start))

    return tiles, positions


def save_tiles(
    tiles: list[np.ndarray],
    positions: list[tuple[int, int]],
    out_dir: str,
    nd2_filename: str,
) -> list[str]:
    """Write each tile to a TIFF file in *out_dir*.

    Returns a list of saved basenames.
    """
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(nd2_filename))[0]
    saved_basenames: list[str] = []

    for tile, (x, y) in zip(tiles, positions):
        fname = f"{base_name}_x={x}_y={y}.tif"
        fpath = os.path.join(out_dir, fname)
        tifffile.imwrite(fpath, tile.astype(np.uint16))
        saved_basenames.append(fname)

    return saved_basenames


def extract_offsets(filename: str) -> tuple[int, int]:
    """Return the ``(x_offset, y_offset)`` encoded in *filename*.

    Expects a pattern like ``..._x=42_y=110...`` in the filename.
    Raises :class:`ValueError` if no match is found.
    """
    match = re.search(r"_x=(\d+)_y=(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Could not extract x/y offsets from filename: {filename}")


def update_csv_with_offsets(filename: str) -> pd.DataFrame:
    """Load a CSV and shift x/y columns by the offsets encoded in the filename.

    Columns ``x``, ``x_detect``, ``y``, ``y_detect`` are shifted if present.
    """
    x_offset, y_offset = extract_offsets(filename)
    df = pd.read_csv(filename)
    for col in ["x", "x_detect"]:
        if col in df.columns:
            df[col] += x_offset
    for col in ["y", "y_detect"]:
        if col in df.columns:
            df[col] += y_offset
    return df


# ---------------------------------------------------------------------------
# SLURM job-monitoring helpers
# ---------------------------------------------------------------------------

def is_job_running(job_id: int | str) -> bool:
    """Return True if SLURM job *job_id* is still in the queue."""
    result = subprocess.run(
        ["squeue", "--job", str(job_id)], capture_output=True, text=True
    )
    return str(job_id) in result.stdout


def count_jobs_running(job_ids) -> int:
    """Return the number of job IDs from *job_ids* that are still running."""
    job_ids = set(str(jid) for jid in job_ids)
    result = subprocess.run(
        ["squeue", "--noheader", "--format=%i"], capture_output=True, text=True
    )
    running_ids = set(result.stdout.strip().splitlines())
    return len(job_ids & running_ids)


def get_running_job_ids() -> set[str]:
    """Return the set of all job IDs currently in the SLURM queue."""
    result = subprocess.run(
        ["squeue", "--noheader", "--format=%i"], capture_output=True, text=True
    )
    return set(result.stdout.strip().splitlines())


def keys_with_all_jobs_finished(job_dict: dict) -> list:
    """Return keys from *job_dict* for which all associated job IDs have finished.

    Parameters
    ----------
    job_dict:
        Mapping of ``{key: [job_id, ...]}`` as returned by the SLURM submission
        loop in :mod:`analyze_directory`.
    """
    running = get_running_job_ids()
    finished_keys = []
    for key, job_ids in job_dict.items():
        job_ids_str = set(str(j) for j in job_ids)
        if job_ids_str.isdisjoint(running):
            finished_keys.append(key)
    return finished_keys
