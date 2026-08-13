#!/bin/bash
# run-spt-analysis -- prepare and submit a full SPT tracking/SASPT analysis
# on Sherlock, without needing an interactive Claude Code session.
#
# Usage:
#   run-spt-analysis [--threshold=T] [--linking=R] [--gaps=G] \
#       [--set key.path=value]... [--write-only] <date_directory>
#
# <date_directory> is the dated folder containing raw_data/ (e.g.
# 2026_08_07), absolute or relative.
#
# --threshold / --linking / --gaps map to quot.detect.t /
# quot.track.search_radius / quot.track.max_blinks -- the three knobs every
# sweep on this dataset has actually varied. Each defaults to this
# project's usual starting point (threshold=12, linking=1.1, gaps=0) if
# omitted. Everything else defaults to this project's established
# calibration (w=11, localize.window_size=7/max_iter=10/damp=1,
# track.method=conservative, min_I0=200.0, plot.bleach_xlim=[-125000,25000]);
# pass --set key=value (repeatable) to override any of those, or set
# anything not covered by the three named flags.
#
# --write-only prepares the analysis directory + controller script without
# submitting it -- e.g. to check settings_override.yaml first.
#
# If <date_directory>/tracking_output_q=<threshold>p.../settings_override.yaml
# already exists with different settings than requested, this refuses to
# overwrite it (protects an existing run's provenance/outputs from being
# silently clobbered by a mistyped flag) -- pass --force once you're sure
# every output already there should be recomputed with the new settings.
#
# Org policy: never run Python directly on the login node. The lightweight
# prepare_run.py step (no tracking/compute, just resolves settings and
# writes two small files) runs via `srun -p dev`; only the sbatch
# controller this then submits dispatches the real per-file/condition
# tracking jobs, on -p normal.

set -euo pipefail

SW_DIR="/oak/stanford/groups/gschliss/Gavin/software"
REPO_DIR="$SW_DIR/quot_saspt_workflow"

# Defaults applied unless overridden on the command line.
DEFAULT_THRESHOLD="12"
DEFAULT_LINKING="1.1"
DEFAULT_GAPS="0"

THRESHOLD="$DEFAULT_THRESHOLD"
LINKING="$DEFAULT_LINKING"
GAPS="$DEFAULT_GAPS"
EXTRA_SETS=()
WRITE_ONLY=0
FORCE=0
DATE_DIR=""

usage() {
    cat >&2 <<USAGE
Usage: run-spt-analysis [--threshold=T] [--linking=R] [--gaps=G] [--set key.path=value]... [--write-only] [--force] <date_directory>

  --threshold=T   quot.detect.t (quality threshold), default $DEFAULT_THRESHOLD
  --linking=R     quot.track.search_radius (linking distance, um), default $DEFAULT_LINKING
  --gaps=G        quot.track.max_blinks (gap-closing tolerance, frames), default $DEFAULT_GAPS
  --set K=V       extra dotted settings override, repeatable
  --write-only    prepare the analysis directory + controller script, don't submit it
  --force         overwrite an existing, differing settings_override.yaml instead of refusing
USAGE
    exit 1
}

for arg in "$@"; do
    case "$arg" in
        --threshold=*) THRESHOLD="${arg#*=}" ;;
        --linking=*)   LINKING="${arg#*=}" ;;
        --gaps=*)      GAPS="${arg#*=}" ;;
        --set=*)       EXTRA_SETS+=("${arg#*=}") ;;
        --write-only)  WRITE_ONLY=1 ;;
        --force)       FORCE=1 ;;
        -h|--help)     usage ;;
        -*)            echo "Unknown option: $arg" >&2; usage ;;
        *)             DATE_DIR="$arg" ;;
    esac
done

[[ -z "$DATE_DIR" ]] && usage

if [[ ! -d "$DATE_DIR" ]]; then
    echo "No such directory: $DATE_DIR" >&2
    exit 1
fi
DATA_DIR="$(cd "$DATE_DIR" && pwd)/raw_data"
if [[ ! -d "$DATA_DIR" ]]; then
    echo "No raw_data/ directory found under $DATE_DIR" >&2
    exit 1
fi

SET_ARGS=(
    --set "quot.detect.t=$THRESHOLD"
    --set "quot.track.search_radius=$LINKING"
    --set "quot.track.max_blinks=$GAPS"
    --set "quot.detect.w=11"
    --set "quot.localize.window_size=7"
    --set "quot.localize.max_iter=10"
    --set "quot.localize.damp=1"
    --set "quot.track.method=conservative"
    --set "quot.track.min_I0=200.0"
    --set "plot.bleach_xlim=[-125000,25000]"
)
for kv in "${EXTRA_SETS[@]:-}"; do
    [[ -n "$kv" ]] && SET_ARGS+=(--set "$kv")
done

PREPARE_ARGS=("$DATA_DIR" "${SET_ARGS[@]}" --write-controller)
[[ "$FORCE" -eq 1 ]] && PREPARE_ARGS+=(--force)

echo "Preparing analysis directory (threshold=$THRESHOLD, linking=$LINKING, gaps=$GAPS)..."
# stderr (prepare_run.py's own "Wrote ..." messages, and any refusal error
# with a settings diff) is left to print directly to the terminal -- only
# stdout (the final analysis_directory path) is captured below.
ANALYSIS_DIR=$(srun -p dev --time=00:05:00 --mem=4G --cpus-per-task=1 \
    "$REPO_DIR/prepare_run_env.sh" "${PREPARE_ARGS[@]}" | tail -1)

if [[ -z "$ANALYSIS_DIR" || ! -d "$ANALYSIS_DIR" ]]; then
    echo "prepare_run.py did not return a valid analysis directory -- see errors above." >&2
    exit 1
fi

echo "Prepared: $ANALYSIS_DIR"
echo "Settings: $ANALYSIS_DIR/settings_override.yaml"

if [[ "$WRITE_ONLY" -eq 1 ]]; then
    echo "Controller written, not submitted (--write-only)."
    echo "Submit later with: sbatch \"$ANALYSIS_DIR/run_snakemake_controller.slurm\""
else
    sbatch "$ANALYSIS_DIR/run_snakemake_controller.slurm"
fi
