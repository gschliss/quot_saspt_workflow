"""
core/aggregate.py

Cross-condition comparison: load per-condition plotting PKLs and produce
combined PDFs that overlay all conditions on the same axes.

Public API
----------
run_aggregate(settings)
    Called once after all condition-wise jobs have finished.
"""

from __future__ import annotations

import os
import glob
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import core  # noqa: F401
from core.plots import _render_population_bar


def run_aggregate(settings: dict) -> None:
    """Produce cross-condition comparison plots.

    Loads the per-condition ``.pkl`` files written by
    :func:`~core.conditionwise.run_conditionwise` and generates combined PDFs:

    - ``<plot_directory>/aggregate_posterior.pdf`` — overlaid posterior
      diffusion-rate histograms, one curve per condition.
    - ``<plot_directory>/aggregate_MLE_bar.pdf`` — stacked bar chart
      comparing diffusion-population fractions across all conditions.

    Parameters
    ----------
    settings:
        Fully-resolved workflow settings dict.
    """
    plotting_pkl_dir = os.path.join(settings["io"]["plot_directory"], "plotting_pkls")
    pkl_files = sorted(glob.glob(os.path.join(plotting_pkl_dir, "*.pkl")))

    if not pkl_files:
        print("run_aggregate: no plotting PKL files found — skipping.")
        return

    # ------------------------------------------------------------------
    # Load all per-condition data
    # ------------------------------------------------------------------
    all_posterior_plot: dict[str, pd.DataFrame] = {}
    all_MLE: dict[str, pd.DataFrame] = {}

    for pkl_path in pkl_files:
        condition = os.path.splitext(os.path.basename(pkl_path))[0]
        with open(pkl_path, "rb") as fh:
            data = pickle.load(fh)
        all_posterior_plot[condition] = data["posterior_plot_df"]
        all_MLE[condition] = data["MLE_df"]

    conditions = list(all_posterior_plot.keys())
    print(f"run_aggregate: found {len(conditions)} conditions — {conditions}")

    # ------------------------------------------------------------------
    # Overlaid posterior histogram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3))

    for condition, ppdf in all_posterior_plot.items():
        ax.plot(
            np.log10(ppdf["D"]),
            50 * ppdf["density"],
            linewidth=1.5,
            label=condition,
        )

    tick_positions = [-2, -1, 0, 1, 2]
    tick_labels = [
        f"{10**v:.2f}" if 10**v < 1 else f"{10**v:.0f}" for v in tick_positions
    ]
    ax.set_xticks(tick_positions, tick_labels)
    ax.set_xlabel("Diffusion coefficient (μm²/s)")
    ax.set_ylabel("Probability density")
    ax.set_title("Aggregate posterior distributions")
    ax.set_ylim(settings["plot"]["ylim"][0], settings["plot"]["ylim"][1])
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize="small", bbox_to_anchor=(1.01, 1), loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(settings["io"]["plot_directory"], "aggregate_posterior.pdf")
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    # ------------------------------------------------------------------
    # Aggregate MLE stacked bar chart
    # ------------------------------------------------------------------
    bin_labels = settings["plot"]["barGraphLabels"]
    bar_data = pd.DataFrame(index=bin_labels)

    for condition, mle_df in all_MLE.items():
        counts, _ = np.histogram(mle_df["MLE_D"], bins=settings["plot"]["barGraphBreaks"])
        bar_data[condition] = counts

    bar_data = bar_data.T
    bar_data = bar_data.div(bar_data.sum(axis=1), axis=0)

    # Width scales with number of conditions so each bar is ~0.9 in wide
    fig_width = max(4, len(conditions) * 0.9 + 2.5)
    fig, ax = plt.subplots(figsize=(fig_width, 4))
    _render_population_bar(ax, bar_data)
    ax.set_title("Aggregate diffusion-population fractions")
    plt.tight_layout()

    out_path = os.path.join(settings["io"]["plot_directory"], "aggregate_MLE_bar.pdf")
    fig.savefig(out_path, dpi=300, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")

    print("\nrun_aggregate complete.")
