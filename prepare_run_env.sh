#!/bin/bash
# prepare_run_env.sh -- sources the Lmod modules + venv prepare_run.py
# needs, then execs it with the given arguments.
#
# Exists so callers (e.g. run_spt_analysis.sh) can invoke prepare_run.py
# via `srun -p dev ... prepare_run_env.sh ...` with plain argument passing,
# instead of building a `bash -c "source ...; python3 ..."` string --
# nested shell-quoting around --set values like
# 'plot.bleach_xlim=[-125000,25000]' is a real footgun otherwise.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SW_DIR="$(dirname "$HERE")"

source "$SW_DIR/load_saspt_modules.sh"
source "$SW_DIR/saspt_env/bin/activate"

exec python3 "$HERE/prepare_run.py" "$@"
