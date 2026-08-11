# Snakefile — HPC DAG runner for the quot/SASPT workflow
#
# Always invoked through prepare_run.py, which resolves settings from
# command-line overrides, creates this run's own analysis directory, and
# freezes the overrides there as <analysis_directory>/settings_override.yaml.
# `config` must carry BOTH `data_directory` (the .nd2 folder) and
# `analysis_directory` (this specific run's own output folder, already
# created by prepare_run.py) -- see its docstring for the full invocation.
#
#   ANALYSIS_DIR=$(python3 prepare_run.py /path/to/raw_data \
#       --set quot.detect.t=10.0 --set quot.track.search_radius=0.1 \
#       --write-controller)
#   sbatch "$ANALYSIS_DIR/run_snakemake_controller.slurm"
#
# Why analysis_directory must be explicit, not derived here: deriving it from
# settings[quot][detect][t] would require resolving settings first, which
# would require knowing where to read the override file from -- and reading
# it from a shared location next to the .nd2 files (the old design) is
# exactly what let one sweep's edits silently redirect another sweep's
# already-dispatched-but-not-yet-executed jobs (see incident notes in
# project memory, 2026-07-17). With analysis_directory passed in explicitly
# and forwarded to every recursive per-job reparse via `--config` (Snakemake
# always forwards `--config` to cluster job subprocesses -- see
# snakemake/executors/__init__.py's `general_args`), every reparse of this
# run reads the same fixed, run-private override file. No two runs ever
# share a mutable settings file, so there's nothing left for a differently-
# timed sweep to race against.
#
# Concurrency / --cluster reparse note (still true, just no longer a hazard):
# legacy `--cluster "sbatch ..."` mode does not serialize this module's
# top-level state into each dispatched job. Every individual SLURM job
# re-invokes `python -m snakemake --snakefile <this file> ...` as a fresh
# subprocess (see snakemake/executors/__init__.py's `format_job_exec`),
# which reparses this whole file from scratch -- so `load_settings(...)`
# below reruns at whatever moment that job actually executes, not at
# submission time. That's fine now: it re-reads the SAME run-private
# override file every time, not a shared one that could have moved on to a
# different sweep in the meantime.
#
# The settings.pkl freeze-and-guard below is still worth keeping on top of
# that: it catches someone hand-editing THIS run's own
# analysis_directory/settings_override.yaml while its jobs are still in
# flight, turning a silent parameter-mix into a loud, per-job failure unless
# --forceall is passed.
import copy
import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: ensure core/ (and therefore fastQuot) is importable from here.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(workflow.snakefile))

from core.settings import load_settings, settings_equal, update_settings_with_image_metadata
from core.utils import group_files_by_metadata


# ---------------------------------------------------------------------------
# Resolve settings and discover files at DAG-construction time.
# ---------------------------------------------------------------------------
import pickle as _pickle

_DATA_DIR      = config["data_directory"]
_ANALYSIS_DIR  = config["analysis_directory"]
_OVERRIDE_PATH = os.path.join(_ANALYSIS_DIR, "settings_override.yaml")
_SETTINGS  = load_settings(_DATA_DIR, override_path=_OVERRIDE_PATH, analysis_directory=_ANALYSIS_DIR)
_GROUPED   = group_files_by_metadata(_DATA_DIR)   # {condition: [filename, ...]}
_CONDITIONS = list(_GROUPED.keys())

# Freeze the settings this analysis directory was first resolved with, and
# guard every later reparse (including each individual cluster job's own
# reparse -- see the concurrency note above) against a since-changed
# settings_override.yaml silently taking effect partway through a run.
_PROVENANCE_PKL = os.path.join(_SETTINGS["io"]["analysis_directory"], "settings.pkl")
_FORCE_SETTINGS_REFRESH = bool(
    {"--forceall", "-F", "--forcerun", "-R"} & set(sys.argv)
)

if os.path.exists(_PROVENANCE_PKL):
    with open(_PROVENANCE_PKL, "rb") as _fh:
        _FROZEN_SETTINGS = _pickle.load(_fh)
    if not _FORCE_SETTINGS_REFRESH and not settings_equal(_FROZEN_SETTINGS, _SETTINGS):
        raise RuntimeError(
            f"{_PROVENANCE_PKL} (frozen the first time this analysis directory's "
            f"jobs were dispatched) no longer matches the current "
            f"{_OVERRIDE_PATH}. Proceeding would silently mix old and new "
            "parameters across this directory's jobs. Re-run with --forceall "
            "once every output should actually be recomputed with the new "
            f"settings, or restore {_OVERRIDE_PATH} to what it was when this "
            "directory's run started."
        )

if _FORCE_SETTINGS_REFRESH or not os.path.exists(_PROVENANCE_PKL):
    with open(_PROVENANCE_PKL, "wb") as _fh:
        _pickle.dump(_SETTINGS, _fh)


def _traj_csv(nd2_basename):
    """Return expected trajectory CSV path for a given .nd2 basename."""
    base = os.path.splitext(nd2_basename)[0]
    return os.path.join(_SETTINGS["io"]["traj_directory"], base + "_traj.csv")


def _condition_pkl(condition):
    """Return expected plotting PKL path for a given condition."""
    return os.path.join(
        _SETTINGS["io"]["plot_directory"], "plotting_pkls", f"{condition}.pkl"
    )


# All per-file trajectory CSVs (filewise outputs)
ALL_TRAJ_CSVS = [
    _traj_csv(f)
    for file_list in _GROUPED.values()
    for f in file_list
]

# All per-condition PKLs (conditionwise outputs)
ALL_COND_PKLS = [_condition_pkl(c) for c in _CONDITIONS]

# Final aggregate outputs
AGGREGATE_OUTPUTS = [
    os.path.join(_SETTINGS["io"]["plot_directory"], "aggregate_posterior.pdf"),
    os.path.join(_SETTINGS["io"]["plot_directory"], "aggregate_MLE_bar.pdf"),
]


# ---------------------------------------------------------------------------
# rule all — top-level target
# ---------------------------------------------------------------------------
rule all:
    input:
        AGGREGATE_OUTPUTS


# ---------------------------------------------------------------------------
# rule filewise — one job per .nd2 file
# ---------------------------------------------------------------------------
rule filewise:
    input:
        nd2 = lambda wc: os.path.join(_DATA_DIR, wc.base + ".nd2")
    output:
        # Output must be a plain wildcard pattern (Snakemake only allows
        # functions for `input`, not `output`) so it can be pattern-matched
        # against the explicit traj CSV paths requested by rule conditionwise.
        traj = os.path.join(_SETTINGS["io"]["traj_directory"], "{base}_traj.csv")
    params:
        # Deep-copied per job so this job's own settings snapshot (baked
        # into its serialized job file) is independent of every other job's.
        settings = lambda wc: copy.deepcopy(_SETTINGS)
    run:
        from core.settings import update_settings_with_image_metadata
        from core.filewise import run_filewise

        nd2_path = input.nd2
        settings = params.settings

        # Update pixel size / frame interval / background from this exact
        # file's own .nd2 metadata.
        update_settings_with_image_metadata(settings, nd2_path=nd2_path)

        run_filewise(nd2_path, settings)


# ---------------------------------------------------------------------------
# rule conditionwise — one job per condition, depends on all filewise outputs
# ---------------------------------------------------------------------------
rule conditionwise:
    input:
        trajs = lambda wc: [_traj_csv(f) for f in _GROUPED[wc.condition]]
    output:
        pkl = os.path.join(_SETTINGS["io"]["plot_directory"], "plotting_pkls", "{condition}.pkl")
    params:
        settings = lambda wc: copy.deepcopy(_SETTINGS)
    run:
        from core.settings import update_settings_with_image_metadata
        from core.conditionwise import run_conditionwise

        settings = params.settings
        update_settings_with_image_metadata(settings, cond=wildcards.condition)
        run_conditionwise(wildcards.condition, settings)


# ---------------------------------------------------------------------------
# rule aggregate — one job, depends on all conditionwise outputs
# ---------------------------------------------------------------------------
rule aggregate:
    input:
        pkls = ALL_COND_PKLS
    output:
        AGGREGATE_OUTPUTS
    params:
        settings = copy.deepcopy(_SETTINGS)
    run:
        from core.aggregate import run_aggregate, run_aggregate_survival_by_exposure

        settings = params.settings
        run_aggregate(settings)
        run_aggregate_survival_by_exposure(settings, min_length=2)
