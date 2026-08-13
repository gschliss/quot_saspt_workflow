#!/usr/bin/env python3
"""prepare_run.py — resolve settings from command-line overrides and freeze
them into a fresh, run-specific analysis directory.

This is the entry point for any cluster (Snakefile) sweep. It replaces
hand-editing a shared ``settings_override.yaml`` inside the data directory --
that file is mutable and shared across every sweep pointed at the same data,
so editing it for sweep B while sweep A's jobs are still queued/running can
silently redirect one of A's not-yet-executed jobs into B's settings/output.
Each run's override now lives inside that run's own analysis directory
(``tracking_output_q=<t>/settings_override.yaml``), created by this script,
and nothing ever reads a settings_override.yaml from the data directory.

Usage
-----
    python3 prepare_run.py /path/to/raw_data --set quot.detect.t=10.0 \\
        --set quot.track.search_radius=0.1 [--write-controller]

Prints the resulting analysis_directory path on success (and nothing else),
so it's safe to capture directly:

    ANALYSIS_DIR=$(python3 prepare_run.py /path/to/raw_data \\
        --set quot.detect.t=10.0 --set quot.track.search_radius=0.1 \\
        --write-controller)
    sbatch "$ANALYSIS_DIR/run_snakemake_controller.slurm"

--write-controller also writes a ready-to-submit
``<analysis_directory>/run_snakemake_controller.slurm``, with
``--directory <analysis_directory>`` (so this sweep's Snakemake lock/metadata
never collides with a concurrent sweep's) and ``--latency-wait 60`` (NFS
output-visibility latency on $GROUP_HOME caused real job failures without
this) already set.
"""

from __future__ import annotations

import argparse
import os
import stat
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from core.settings import build_override_from_args, resolve_and_freeze_override

_CONTROLLER_TEMPLATE = """#!/bin/bash
#SBATCH -p normal
#SBATCH --time=04:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
#SBATCH -o {analysis_directory}/logs/controller_%j.log
#SBATCH -J saspt_controller_{tag}

set -euo pipefail

source "{software_directory}/load_saspt_modules.sh"
source "{software_directory}/saspt_env/bin/activate"

mkdir -p "{analysis_directory}/logs"

# --rerun-triggers omits Snakemake's default 'params' trigger: it fingerprints
# each job's params (including params.settings, the whole resolved settings
# dict) via an exact repr() string comparison against what was last
# recorded, with no tolerance for legitimate-but-cosmetic differences. Two
# concrete ways that's bitten this pipeline (see the longer note in
# Snakefile, next to the settings.pkl freeze-and-guard): a rule-body/params
# shape change alone makes every already-computed output look "changed"
# the moment new code runs against it (confirmed root cause of a full,
# unwanted reprocessing on 2026-08-10 against an already-complete analysis
# directory); repr() also embeds saspt.diff_coefs/loc_errors float64
# arrays, which can in principle round differently across sufficiently
# different node CPUs, the same category of cross-node divergence
# settings_equal()'s docstring documents from 2026-07-17. Either way,
# Snakemake's own 'params' trigger is redundant here: the Snakefile's
# settings.pkl freeze-and-guard already re-detects genuine settings drift,
# with the floating-point tolerance repr()-based comparison lacks.
snakemake --snakefile "{snakefile}" --directory "{analysis_directory}" \\
    --jobs 40 \\
    --latency-wait 60 \\
    --cluster "sbatch -p normal --time=01:00:00 --mem=16G --cpus-per-task=2 -o {analysis_directory}/logs/snakejob_%j.log" \\
    --keep-going \\
    --rerun-triggers mtime input code software-env \\
    --config data_directory="{data_directory}" analysis_directory="{analysis_directory}" \\
    "$@"
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_directory", help="Absolute path to the folder containing .nd2 files.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted settings key=value, repeatable (e.g. --set quot.detect.t=10.0).",
    )
    parser.add_argument(
        "--write-controller",
        action="store_true",
        default=False,
        help="Also write <analysis_directory>/run_snakemake_controller.slurm, ready to sbatch.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help=(
            "Overwrite an existing analysis directory's settings_override.yaml even if its "
            "content differs from what --set requested (default: refuse and print the diff)."
        ),
    )
    args = parser.parse_args()

    data_directory = os.path.abspath(args.data_directory)
    overrides = build_override_from_args(args.overrides)
    settings = resolve_and_freeze_override(data_directory, overrides, force=args.force)
    analysis_directory = settings["io"]["analysis_directory"]

    if args.write_controller:
        snakefile = os.path.join(_HERE, "Snakefile")
        tag = os.path.basename(analysis_directory.rstrip("/"))
        controller_path = os.path.join(analysis_directory, "run_snakemake_controller.slurm")
        with open(controller_path, "w") as fh:
            fh.write(
                _CONTROLLER_TEMPLATE.format(
                    analysis_directory=analysis_directory,
                    data_directory=data_directory,
                    snakefile=snakefile,
                    tag=tag,
                    software_directory=os.path.dirname(_HERE),
                )
            )
        os.chmod(controller_path, os.stat(controller_path).st_mode | stat.S_IEXEC)
        print(f"Wrote {controller_path}", file=sys.stderr)

    print(analysis_directory)


if __name__ == "__main__":
    main()
