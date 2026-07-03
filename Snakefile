# Snakefile — HPC DAG runner for the quot/SASPT workflow
#
# Local (sequential):
#   snakemake --cores 4 --config data_directory=/path/to/nd2s
#
# SLURM (parallel):
#   snakemake --executor slurm --jobs 100 --config data_directory=/path/to/nd2s
#
# Each rule uses a run: block that imports directly from core/ and calls the
# appropriate function — no intermediate shell scripts required.

import os
import sys

# ---------------------------------------------------------------------------
# Bootstrap: ensure core/ (and therefore fastQuot) is importable from here.
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(workflow.snakefile))

from core.settings import load_settings, update_settings_with_image_metadata
from core.utils import group_files_by_metadata


# ---------------------------------------------------------------------------
# Resolve settings and discover files at DAG-construction time.
# The settings PKL is written here so that individual rule run: blocks can
# reload it in their (possibly separate) worker processes.
# ---------------------------------------------------------------------------
import pickle as _pickle

_DATA_DIR  = config["data_directory"]
_SETTINGS  = load_settings(_DATA_DIR)
_GROUPED   = group_files_by_metadata(_DATA_DIR)   # {condition: [filename, ...]}
_CONDITIONS = list(_GROUPED.keys())

# Persist settings so rule run: blocks can load them by path
_PKL_PATH = os.path.join(_SETTINGS["io"]["analysis_directory"], "settings.pkl")
with open(_PKL_PATH, "wb") as _fh:
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
        nd2 = lambda wc: os.path.join(_DATA_DIR, wc.nd2_basename)
    output:
        traj = lambda wc: _traj_csv(wc.nd2_basename)
    run:
        import pickle
        from core.settings import update_settings_with_image_metadata
        from core.filewise import run_filewise

        nd2_basename = wildcards.nd2_basename
        nd2_path = os.path.join(_DATA_DIR, nd2_basename)

        # Load a fresh copy of settings (Snakemake jobs may run in separate
        # processes) and update pixel size / frame interval / background
        # from this exact file's own .nd2 metadata.
        with open(os.path.join(_SETTINGS["io"]["analysis_directory"], "settings.pkl"), "rb") as fh:
            settings = pickle.load(fh)

        update_settings_with_image_metadata(settings, nd2_path=nd2_path)

        run_filewise(nd2_path, settings)


# ---------------------------------------------------------------------------
# rule conditionwise — one job per condition, depends on all filewise outputs
# ---------------------------------------------------------------------------
rule conditionwise:
    input:
        trajs = lambda wc: [_traj_csv(f) for f in _GROUPED[wc.condition]]
    output:
        pkl = lambda wc: _condition_pkl(wc.condition)
    run:
        import pickle
        from core.settings import update_settings_with_image_metadata
        from core.conditionwise import run_conditionwise

        with open(os.path.join(_SETTINGS["io"]["analysis_directory"], "settings.pkl"), "rb") as fh:
            settings = pickle.load(fh)

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
    run:
        import pickle
        from core.aggregate import run_aggregate

        with open(os.path.join(_SETTINGS["io"]["analysis_directory"], "settings.pkl"), "rb") as fh:
            settings = pickle.load(fh)

        run_aggregate(settings)
