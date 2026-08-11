"""
run_local.py — Sequential local runner (no SLURM required).

Usage
-----
    python run_local.py --input_directory /path/to/nd2s \\
        [--set quot.detect.t=10.0 --set quot.track.search_radius=0.1 ...] \\
        [--force] [--mode all|track|condition|aggregate]

Modes
-----
track       Run run_filewise on every .nd2 file in the input directory.
condition   Run run_conditionwise on every condition.
aggregate   Run run_aggregate once (requires per-condition PKLs to exist).
all         track → condition → aggregate  (default)

The --force flag re-processes all .nd2 files even if their outputs already
exist.

Settings come only from code defaults plus any --set overrides given here --
this never reads a settings_override.yaml from --input_directory. Overrides
are frozen into <analysis_directory>/settings_override.yaml for provenance
(see core.settings.resolve_and_freeze_override), where <analysis_directory>
is this run's own ``tracking_output_q=<t>/`` folder, not shared with any
other run.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

# ---------------------------------------------------------------------------
# Ensure this script can locate the core package regardless of cwd.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import core  # noqa: F401  — boots fastQuot sys.path insert
from core import _apply_gpu_settings, _report_gpu_status
from core.settings import (
    build_override_from_args,
    resolve_and_freeze_override,
    update_settings_with_image_metadata,
    print_nested_dict,
)
from core.utils import group_files_by_metadata, identify_missing_filewise
from core.filewise import run_filewise
from core.conditionwise import run_conditionwise
from core.aggregate import run_aggregate, run_aggregate_survival_by_exposure
from core.publish import publish_run_report


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential local runner for the quot/SASPT workflow."
    )
    parser.add_argument(
        "--input_directory",
        required=True,
        help="Absolute path to the folder containing .nd2 files.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted settings key=value, repeatable (e.g. --set quot.detect.t=10.0).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-processing of all files, ignoring existing outputs.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "track", "condition", "aggregate"],
        default="all",
        help=(
            "Which stage(s) to run.  "
            "'all' runs track → condition → aggregate (default)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    data_directory = os.path.abspath(args.input_directory)
    print(f"\nWorking from directory: {data_directory}")
    print(f"Mode: {args.mode}  |  Force: {args.force}\n")

    # ------------------------------------------------------------------
    # Load and persist settings
    # ------------------------------------------------------------------
    overrides = build_override_from_args(args.overrides)
    settings = resolve_and_freeze_override(data_directory, overrides)
    _apply_gpu_settings(settings)
    print(f"  {_report_gpu_status()}\n")

    pkl_path = os.path.join(settings["io"]["analysis_directory"], "settings.pkl")
    with open(pkl_path, "wb") as fh:
        pickle.dump(settings, fh)

    txt_path = os.path.join(settings["io"]["analysis_directory"], "settings.txt")
    print_nested_dict(settings, file=txt_path)
    print(f"Settings saved to: {txt_path}\n")

    # ------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------
    grouped_files = group_files_by_metadata(settings["io"]["data_directory"])

    # ------------------------------------------------------------------
    # TRACK stage
    # ------------------------------------------------------------------
    if args.mode in ("all", "track"):
        nd2_files = identify_missing_filewise(settings, force=args.force)

        for cond, file_list in nd2_files.items():
            print(f"\n--- Tracking condition: {cond} ({len(file_list)} files) ---")

            for filename in file_list:
                nd2_path = os.path.join(settings["io"]["data_directory"], filename)

                # Update pixel size, frame interval, and background from
                # this exact file's own .nd2 metadata.
                update_settings_with_image_metadata(settings, nd2_path=nd2_path)

                run_filewise(nd2_path, settings)

    # ------------------------------------------------------------------
    # CONDITION stage
    # ------------------------------------------------------------------
    if args.mode in ("all", "condition"):
        conditions = list(grouped_files.keys())
        print(f"\n--- Condition-wise analysis for: {conditions} ---")

        for cond in conditions:
            # Re-apply per-condition image metadata before condition analysis
            update_settings_with_image_metadata(settings, cond=cond)
            try:
                run_conditionwise(cond, settings)
            except Exception as e:
                print(f"WARNING: condition-wise analysis failed for {cond}: {e}")
                print("  Skipping to the next condition.")

    # ------------------------------------------------------------------
    # AGGREGATE stage
    # ------------------------------------------------------------------
    if args.mode in ("all", "aggregate"):
        print("\n--- Aggregate cross-condition plots ---")
        run_aggregate(settings)
        run_aggregate_survival_by_exposure(settings, min_length=2)
        publish_run_report(settings)

    print("\nDone.")


if __name__ == "__main__":
    main()
