# quot_saspt_workflow

Single-particle tracking (SPT) pipeline combining **quot** (detection, localization, linking) with **SASPT** (Bayesian diffusion-state inference). Designed for both local sequential runs and HPC cluster (SLURM) parallelism.

The repository bundles a performance-optimized fork of quot (`fastQuot/`) so no external quot installation is needed.

---

## Repository layout

```
quot_saspt_workflow/
├── fastQuot/               # Bundled quot fork (auto-added to sys.path by core/)
│   └── quot/               # The quot Python package
├── core/                   # Workflow library
│   ├── __init__.py         # sys.path bootstrap (finds bundled fastQuot)
│   ├── settings.py         # Settings loading / merging
│   ├── filewise.py         # Per-file: detect → localize → link → SASPT
│   ├── conditionwise.py    # Per-condition: aggregate, QC, HMM, plots
│   ├── aggregate.py        # Cross-condition comparison plots
│   ├── plots.py            # All plotting functions
│   └── utils.py            # CSV aggregation, file-grouping helpers
├── calling_scripts/        # Example settings_override.yaml and SLURM caller
├── run_local.py            # Sequential local runner (no SLURM needed)
├── Snakefile               # Snakemake DAG for HPC parallel execution
├── requirements.txt
└── README.md
```

---

## Installation

```bash
git clone https://github.com/gschliss/quot_saspt_workflow.git
cd quot_saspt_workflow
pip install -r requirements.txt
```

`fastQuot` is bundled — no separate quot install required.

---

## Quick start (local)

```bash
python run_local.py --input_directory /path/to/nd2_files/
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--input_directory` | *(required)* | Folder containing `.nd2` files |
| `--mode` | `all` | `track`, `condition`, `aggregate`, or `all` |
| `--force` | off | Re-run even if output files already exist |

### Run only tracking

```bash
python run_local.py --input_directory /path/to/nd2s --mode track
```

### Run only plotting (after tracking is done)

```bash
python run_local.py --input_directory /path/to/nd2s --mode condition
python run_local.py --input_directory /path/to/nd2s --mode aggregate
```

---

## Settings override

Drop a `settings_override.yaml` in the same folder as your `.nd2` files to override any default parameter. The file is optional — sensible defaults are used if it is absent.

**Example (`settings_override.yaml`):**

```yaml
quot:
  detect:
    t: 15.0          # detection threshold (lower = more spots)
    w: 11             # half-window size for LLR filter
  localize:
    window_size: 7
  track:
    method: conservative
    search_radius: 0.15   # μm
    min_I0: 200.0

plot:
  ylim: [0, 2.5]
  barGraphBreaks: [0, 0.05, 0.35, 2, 100]   # D boundaries (μm²/s)
  barGraphLabels: [Immobile, Confined, Membrane, Free]
  mult_on_sd: 1.0    # outlier-rejection threshold (KS-distance SDs)

saspt:
  focal_depth: 0.1
  splitsize: 5
```

You do **not** need to specify any file paths in the override file — the data directory is inferred from where the YAML lives.

---

## Output structure

All output is written to a sibling directory named `tracking_output_q=<threshold>/`:

```
tracking_output_q=15p01/
├── trajectories/           # Per-file _traj.csv (localization + linking output)
├── posterior/              # Per-file posterior diffusion-state CSVs
├── MLE/                    # Per-file MLE diffusion-coefficient CSVs
├── rolling_windows/        # Per-file rolling-window MLE CSVs
├── plots/
│   ├── plotting_pkls/      # Per-condition serialized data for aggregate step
│   ├── <condition>_posterior.pdf
│   ├── <condition>_bar.pdf
│   ├── <condition>_survival.pdf
│   ├── aggregate_posterior.pdf
│   └── aggregate_MLE_bar.pdf
├── movies/                 # Overlay MP4s of sampled trajectories
├── settings.pkl
└── settings.txt
```

---

## HPC / SLURM

See `Snakefile` and `calling_scripts/` for cluster submission. The DAG parallelizes the per-file tracking stage as a job array and runs condition-wise and aggregate steps after all file jobs complete.

---

## Dependencies

- [saspt](https://github.com/alecheckert/saspt) — Bayesian diffusion inference
- numpy, pandas, scipy, matplotlib
- pims, nd2reader — image I/O
- imageio, imageio-ffmpeg — movie writing
- hmmlearn — HMM state assignment
- PyYAML — settings files
- snakemake — HPC workflow (optional for local use)
