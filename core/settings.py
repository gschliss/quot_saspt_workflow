"""
core/settings.py

All settings logic:
- Default settings definition (get_default_settings)
- Directory-layout expansion (update_default_settings_for_analysis)
- Image-metadata updates (update_settings_with_image_metadata)
- Loader that wires the above together given an explicit override file
  (load_settings)
- Command-line-driven override construction and freezing (
  build_override_from_args, resolve_and_freeze_override) -- the entry point
  for prepare_run.py and run_local.py. Every run's override file lives in
  that run's own analysis directory, never in data_directory, so unrelated
  concurrent runs never share a mutable settings file.
- Utility helpers: deep_update, settings_equal, print_nested_dict
"""

from __future__ import annotations

import os
import glob
import sys
import yaml
import numpy as np
import pims

# Ensure fastQuot is on the path (harmless to call again if __init__ already ran)
import core  # noqa: F401  — triggers core/__init__.py sys.path insert

# min_I0 default, calibrated against a 16-bit camera's dynamic range (0-65535).
# update_settings_with_image_metadata() rescales this proportionally when a
# movie's own pixel values indicate a lower bit depth (e.g. an 8-bit .nd2),
# but only if the user hasn't already overridden min_I0 in settings_override.yaml.
_DEFAULT_MIN_I0_16BIT = 100.0
_REFERENCE_BIT_DEPTH = 16


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


def settings_equal(a, b) -> bool:
    """Deep-compare two settings values, treating numpy arrays element-wise.

    Plain ``==``/``!=`` on a dict containing ndarray values (e.g.
    ``saspt.diff_coefs``) raises ``ValueError: truth value of an array
    ... is ambiguous``, so this recurses manually instead.

    Float arrays are compared with ``np.allclose`` rather than exact
    ``np.array_equal``: values like ``diff_coefs``/``loc_errors`` are
    recomputed via ``np.power``/``np.linspace`` in every fresh process, and
    on a heterogeneous cluster a settings.pkl frozen on one compute node's
    CPU can differ from a freshly-resolved array on a different node's CPU
    in the last bit or two -- exact equality produced real, reproducible
    (not merely transient) false-positive mismatches across nodes in
    practice (confirmed 2026-07-17: a job on sh04-14n21 disagreed with a
    settings.pkl frozen on an sh02 node, despite an unmodified override
    file). A tolerant comparison still catches any actual, intended
    settings change (e.g. a different search_radius), which differs by far
    more than floating-point noise.
    """
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return False
        if np.issubdtype(a.dtype, np.floating) and np.issubdtype(b.dtype, np.floating):
            return a.shape == b.shape and np.allclose(a, b, rtol=1e-9, atol=1e-12)
        return np.array_equal(a, b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(settings_equal(a[k], b[k]) for k in a)
    return a == b


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
    settings["survival"] = {}

    # -- I/O ------------------------------------------------------------------
    # slowSPT vs fastSPT is auto-detected from each file/condition's own
    # frame_interval (set in update_settings_with_image_metadata): a frame
    # interval above slow_spt_threshold_s means slowSPT (quot + a raw-
    # trajectory survival curve only, no SASPT/HMM/movies). force_mode
    # overrides that auto-detection when set via --force-fastSPT/
    # --force-slowSPT (see prepare_run.py / run_local.py) -- unlike
    # slow_spt_threshold_s, force_mode is not exposed as a generic --set key
    # so it can't be silently tweaked mid-sweep the way the threshold can.
    settings["io"]["slow_spt_threshold_s"] = 2.0
    settings["io"]["force_mode"] = None
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
    settings["quot"]["detect"]["k"] = 15.0
    settings["quot"]["detect"]["w"] = 15
    settings["quot"]["detect"]["t"] = 20.00

    # -- quot localize --------------------------------------------------------
    settings["quot"]["localize"] = {}
    settings["quot"]["localize"]["method"] = "ls_int_gaussian"
    settings["quot"]["localize"]["window_size"] = 11
    settings["quot"]["localize"]["sigma"] = 1.5
    settings["quot"]["localize"]["ridge"] = 0.0000001
    settings["quot"]["localize"]["max_iter"] = 100
    settings["quot"]["localize"]["damp"] = 1
    settings["quot"]["localize"]["camera_bg"] = 100

    # -- quot track -----------------------------------------------------------
    settings["quot"]["track"] = {}
    settings["quot"]["track"]["method"] = "euclidean"
    settings["quot"]["track"]["pixel_size_um"] = 0.11
    settings["quot"]["track"]["frame_interval"] = 0.006
    settings["quot"]["track"]["search_radius"] = 1.2
    settings["quot"]["track"]["max_blinks"] = 0
    settings["quot"]["track"]["min_I0"] = _DEFAULT_MIN_I0_16BIT
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

    # -- gpu ------------------------------------------------------------------
    # Set to False to force CPU execution even when CUDA is available.
    settings["gpu"] = {}
    settings["gpu"]["use_gpu"] = True

    # -- plot -----------------------------------------------------------------
    settings["plot"]["xlim"] = (-2, 2)
    settings["plot"]["bleach_xlim"] = (-300, 50)
    settings["plot"]["ylim"] = [0, 5]
    settings["plot"]["nbins"] = 100
    settings["plot"]["tick_positions"] = (-2, -1, 0, 1, 2)
    settings["plot"]["mult_on_sd"] = 5.0
    settings["plot"]["barGraphBreaks"] = [-10000, 10000]
    settings["plot"]["barGraphLabels"] = ["None"]

    # -- survival ---------------------------------------------------------
    # Name of a condition to treat as a background/no-signal control (e.g. an
    # untagged or non-recruiting construct) -- see plot_survival_background_
    # corrected in core/plots.py. None (default) disables background
    # correction entirely; existing survival plots are unaffected either way.
    settings["survival"]["background_condition"] = None

    return settings


# ---------------------------------------------------------------------------
# Directory layout expansion
# ---------------------------------------------------------------------------

def update_default_settings_for_analysis(
    settings: dict, data_directory: str, analysis_directory: str | None = None
) -> dict:
    """Populate all derivative I/O paths in *settings* and create directories.

    Parameters
    ----------
    settings:
        Settings dict (mutated in-place and also returned).
    data_directory:
        Absolute path to the folder containing .nd2 files.  All analysis
        output is placed in a sibling directory next to it (unless
        *analysis_directory* is given explicitly).
    analysis_directory:
        If given, used as-is instead of deriving a ``tracking_output_q=...``
        name from ``settings['quot']['detect']['t']``. Callers that already
        know which analysis directory they're operating on (e.g. the
        Snakefile, told explicitly via ``config['analysis_directory']``)
        should always pass this, so the directory a job actually writes to
        never depends on re-deriving it from settings resolved in that same
        call -- see :func:`resolve_and_freeze_override`.
    """
    settings["io"]["data_directory"] = data_directory
    settings["io"]["home_directory"] = os.path.dirname(data_directory)

    # code_directory is always the workflow root (parent of core/)
    settings["io"]["code_directory"] = os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )

    if analysis_directory is None:
        traj_filestring = (
            f"tracking_output_q="
            f"{str(float(settings['quot']['detect']['t'])).replace('.', 'p')}"
        )
        analysis_directory = os.path.join(
            settings["io"]["home_directory"], traj_filestring
        )
    settings["io"]["analysis_directory"] = analysis_directory
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

    # Only create the directories every mode actually writes into. slowSPT
    # never touches post_directory/MLE_directory/rolling_window_directory/
    # movie_directory, and split_directory/split_traj_directory are unused by
    # the pipeline entirely (split_image_stack/save_tiles in core/utils.py are
    # standalone helpers, never called from filewise/conditionwise/aggregate) --
    # each of those is created lazily, right before its first write, by the
    # function that actually writes into it.
    os.makedirs(settings["io"]["analysis_directory"], exist_ok=True)
    os.makedirs(settings["io"]["traj_directory"], exist_ok=True)
    os.makedirs(settings["io"]["plot_directory"], exist_ok=True)

    return settings


def validate_background_condition(settings: dict) -> None:
    """Guard against survival background-correction silently pointing at the
    wrong condition: ``settings['survival']['background_condition']``, if
    set, must contain ``'000'`` or ``'322'`` -- this project's naming
    convention for background/no-signal controls. Raises ``ValueError``
    otherwise. A ``None`` background_condition (the default, correction
    disabled) always passes.

    Called from :func:`load_settings` and :func:`resolve_and_freeze_override`
    so a typo'd condition name fails fast at settings-resolution time
    instead of silently producing a bogus corrected survival curve deep in
    a per-condition job.
    """
    background_condition = settings["survival"]["background_condition"]
    if background_condition is None:
        return
    if "000" not in background_condition and "322" not in background_condition:
        raise ValueError(
            f"survival.background_condition={background_condition!r} must contain "
            "'000' or '322' (background/control naming convention) -- refusing to "
            "use it as a background sample."
        )


# ---------------------------------------------------------------------------
# Image-metadata update
# ---------------------------------------------------------------------------

def _infer_bit_depth(img, first_frame, max_frames_to_sample: int = 10) -> int:
    """Infer the nominal camera bit depth from observed pixel values.

    nd2reader always reports frame dtype as float64 regardless of the
    camera's actual bit depth, and nd2 metadata does not expose a bit-depth
    field either — so the only reliable signal available is the observed
    intensity range across a handful of frames.
    """
    n_frames = len(img)
    observed_max = float(np.max(first_frame))
    for i in range(1, min(n_frames, max_frames_to_sample)):
        observed_max = max(observed_max, float(np.max(img[i])))

    if observed_max <= 255:
        return 8
    elif observed_max <= 4095:
        return 12
    return 16


def update_settings_with_image_metadata(
    settings: dict, cond: str | None = None, nd2_path: str | None = None
) -> dict:
    """Read pixel size, frame interval, and background level from an .nd2 file.

    Parameters
    ----------
    settings:
        Settings dict (mutated in-place and also returned).
    cond:
        If provided (and *nd2_path* is not), glob for ``*{cond}*.nd2`` inside
        the data directory and use the first openable match as a
        representative file (unreadable/corrupt files are skipped). If
        neither *cond* nor *nd2_path* is given, use the first openable .nd2
        file found in the data directory.
    nd2_path:
        If provided, read metadata directly from this exact file instead of
        an arbitrary representative file. This is what the filewise rule
        uses, so each file's own pixel size and frame interval are picked up
        even if they differ within a condition.
    """
    if nd2_path is not None:
        candidates = [nd2_path]
    elif cond is None:
        candidates = sorted(glob.glob(f"{settings['io']['data_directory']}/*.nd2"))
    else:
        candidates = sorted(
            glob.glob(f"{settings['io']['data_directory']}/*{cond}*.nd2")
        )

    img = None
    image_path = None
    skipped = []
    for candidate in candidates:
        try:
            img = pims.open(candidate)
            image_path = candidate
            break
        except Exception as e:
            skipped.append((candidate, e))

    if img is None:
        raise RuntimeError(
            f"Could not open any .nd2 file as a representative for metadata "
            f"(cond={cond!r}, nd2_path={nd2_path!r}). Tried: "
            + "; ".join(f"{os.path.basename(c)}: {e}" for c, e in skipped)
        )

    for candidate, e in skipped:
        print(
            f"WARNING: skipping unreadable representative-file candidate "
            f"{os.path.basename(candidate)} ({e}); trying the next match."
        )
    frame = img[0]
    bg_level = np.median(frame)
    dt = 1 / img.frame_rate

    fallback_pixel_size_um = settings["quot"]["track"]["pixel_size_um"]
    pixel_size_um = img.metadata.get("pixel_microns", None)
    if pixel_size_um is None:
        pixel_size_um = fallback_pixel_size_um
        print(
            f"WARNING: no 'pixel_microns' calibration found in {image_path}; "
            f"falling back to pixel_size_um={fallback_pixel_size_um}"
        )

    settings["quot"]["localize"]["camera_bg"] = bg_level
    settings["quot"]["track"]["frame_interval"] = dt
    settings["saspt"]["frame_interval"] = dt
    settings["quot"]["track"]["pixel_size_um"] = pixel_size_um
    settings["saspt"]["pixel_size_um"] = pixel_size_um

    # -- slowSPT vs fastSPT mode -------------------------------------------
    detected_mode = (
        "slowSPT" if dt > settings["io"]["slow_spt_threshold_s"] else "fastSPT"
    )
    settings["io"]["detected_mode"] = detected_mode
    force_mode = settings["io"].get("force_mode")
    if force_mode is not None:
        settings["io"]["mode"] = force_mode
        if force_mode != detected_mode:
            print(
                f"NOTE: {image_path} has frame_interval={dt:g}s, which would "
                f"auto-detect as {detected_mode!r}, but force_mode={force_mode!r} "
                "is set; using the forced mode."
            )
    else:
        settings["io"]["mode"] = detected_mode

    # -- bit-depth-aware min_I0 -------------------------------------------
    # min_I0 (spot-amplitude threshold) is calibrated by default against a
    # 16-bit camera's dynamic range. An 8-bit (or 12-bit) .nd2 tops out far
    # below that, so a flat 100.0 AU threshold can silently discard most or
    # all genuine detections. Rescale proportionally, unless the user has
    # explicitly set min_I0 in settings_override.yaml.
    bit_depth = _infer_bit_depth(img, frame)
    settings["io"]["detected_bit_depth"] = bit_depth

    min_I0_baseline = settings["io"].get("_min_I0_baseline", _DEFAULT_MIN_I0_16BIT)
    if bit_depth < _REFERENCE_BIT_DEPTH:
        scale = (2 ** bit_depth - 1) / (2 ** _REFERENCE_BIT_DEPTH - 1)
        rescaled_min_I0 = min_I0_baseline * scale
        settings["quot"]["track"]["min_I0"] = rescaled_min_I0
        print(
            f"NOTE: {image_path} looks like {bit_depth}-bit data (max observed "
            f"pixel value <= {2 ** bit_depth - 1}); rescaling min_I0 from "
            f"{min_I0_baseline:g} to {rescaled_min_I0:g} to match its dynamic "
            f"range. Set 'quot.track.min_I0' in settings_override.yaml to override."
        )
    else:
        settings["quot"]["track"]["min_I0"] = min_I0_baseline

    return settings


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_settings(
    data_directory: str, override_path: str, analysis_directory: str | None = None
) -> dict:
    """Build a fully-resolved settings dict for *data_directory*.

    Steps:
    1. Start with ``get_default_settings()``.
    2. If *override_path* exists, merge it in with ``deep_update``.
    3. Call ``update_default_settings_for_analysis(settings, data_directory,
       analysis_directory)`` to expand all derivative I/O paths and create
       output directories.
    4. Return the fully-populated settings dict.

    *override_path* is always explicit -- this function does NOT look inside
    *data_directory* for a ``settings_override.yaml`` on its own. A run's
    override file should live in that run's own analysis directory (see
    :func:`resolve_and_freeze_override`, which creates it there), not in
    *data_directory* -- a shared, mutable file next to the raw data is what
    let unrelated concurrent runs stomp on each other's settings by editing
    it out from under an already-dispatched sweep.

    Parameters
    ----------
    data_directory:
        Absolute path to the folder that contains .nd2 files.
    override_path:
        Absolute path to the YAML override file for this specific run.
        Read if it exists; not an error if it doesn't (a run with no
        explicit overrides is valid).
    analysis_directory:
        Passed through to :func:`update_default_settings_for_analysis`.
    """
    settings = get_default_settings()

    if os.path.exists(override_path):
        with open(override_path) as fh:
            override_cfg = yaml.load(fh, Loader=yaml.FullLoader)
        deep_update(settings, override_cfg or {})
        print(f"Applied settings overrides from {override_path}", file=sys.stderr)

    update_default_settings_for_analysis(settings, data_directory, analysis_directory)

    # Snapshot the user-configured (default-or-override) min_I0 *before* any
    # per-file bit-depth rescaling happens in update_settings_with_image_metadata.
    # Kept under "io" (rather than "quot"/"track") so it never gets spread as a
    # stray kwarg into quot's track_file()/track() calls, which pass
    # settings["quot"] / settings["quot"]["track"] straight through as **kwargs.
    settings["io"]["_min_I0_baseline"] = settings["quot"]["track"]["min_I0"]

    validate_background_condition(settings)

    return settings


# ---------------------------------------------------------------------------
# Command-line-driven override construction
# ---------------------------------------------------------------------------

def build_override_from_args(set_args: list[str]) -> dict:
    """Parse ``["quot.detect.t=10.0", "quot.track.search_radius=0.1"]`` into
    a nested override dict: ``{"quot": {"detect": {"t": 10.0}, "track":
    {"search_radius": 0.1}}}``.

    Each value is parsed with ``yaml.safe_load`` for the same scalar
    coercion a hand-written override YAML file would get (``"10.0"`` ->
    float, ``"true"`` -> bool, ``"[1,2,3]"`` -> list, anything else -> str).
    """
    override: dict = {}
    for item in set_args:
        if "=" not in item:
            raise ValueError(f"--set expects <dotted.key>=<value>, got: {item!r}")
        key_path, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)

        node = override
        parts = key_path.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return override


def resolve_and_freeze_override(data_directory: str, overrides: dict, force: bool = False) -> dict:
    """Resolve settings from *overrides* + code defaults, create this run's
    analysis directory, and persist *overrides* there as
    ``<analysis_directory>/settings_override.yaml`` -- the durable, per-run
    location every later reparse (see the Snakefile) reads from instead of
    anything inside *data_directory*.

    Only the explicit *overrides* are written, not the full resolved
    settings dict -- same convention as a hand-written override file, so a
    later code-default change is still picked up on --forceall re-resolution
    instead of being frozen in by accident.

    If *analysis_directory* already has a settings_override.yaml whose
    content differs from *overrides*, this raises instead of silently
    overwriting it -- caught in practice while testing a CLI wrapper
    (2026-08-12): an existing, fully-computed fastSPT directory's override
    file got silently clobbered by an unrelated test run, which would have
    corrupted that directory's provenance record (its actual tracked
    outputs were unaffected, since nothing was re-executed, but a later
    reader of settings_override.yaml would have been misled about what
    produced them). Pass *force=True* once you're sure every output already
    in that directory should be recomputed with the new settings.

    Parameters
    ----------
    data_directory:
        Absolute path to the folder that contains .nd2 files.
    overrides:
        Nested override dict, e.g. from :func:`build_override_from_args`.
    force:
        Overwrite an existing, differing settings_override.yaml anyway.

    Returns
    -------
    The fully-resolved settings dict (also used to create/locate all
    analysis-directory subdirectories via `update_default_settings_for_analysis`).
    """
    settings = get_default_settings()
    deep_update(settings, overrides)
    update_default_settings_for_analysis(settings, data_directory)
    settings["io"]["_min_I0_baseline"] = settings["quot"]["track"]["min_I0"]
    validate_background_condition(settings)

    override_path = os.path.join(settings["io"]["analysis_directory"], "settings_override.yaml")
    if os.path.exists(override_path) and not force:
        with open(override_path) as fh:
            existing_overrides = yaml.load(fh, Loader=yaml.FullLoader) or {}
        if existing_overrides != overrides:
            raise RuntimeError(
                f"{override_path} already exists with different overrides than requested "
                "-- refusing to overwrite.\n"
                f"Existing: {existing_overrides}\n"
                f"Requested: {overrides}\n"
                "Pass force=True (prepare_run.py --force) once you're sure every output "
                "already in that directory should be recomputed with the new settings."
            )

    with open(override_path, "w") as fh:
        yaml.dump(overrides, fh, default_flow_style=False, sort_keys=False)
    print(f"Wrote settings overrides to {override_path}", file=sys.stderr)

    return settings
