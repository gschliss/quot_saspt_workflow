# quot_saspt_workflow

Single-particle tracking (SPT) pipeline combining **quot** (detection, localization, linking) with **SASPT** (Bayesian diffusion-state inference). Designed for both local sequential runs and HPC cluster (SLURM) parallelism.

The repository bundles a performance-optimized fork of quot (`fastQuot/`) so no external quot installation is needed.

---

## Repository layout

```
quot_saspt_workflow/
├── fastQuot/               # Bundled quot fork (auto-added to sys.path by core/)
│   └── quot/               # The quot Python package (with GPU acceleration)
├── core/                   # Workflow library
│   ├── __init__.py         # sys.path bootstrap + GPU control helpers
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

### 1. Clone the repository

```bash
git clone https://github.com/gschliss/quot_saspt_workflow.git
cd quot_saspt_workflow
```

### 2. Create a conda environment (recommended)

```bash
conda create -n spt python=3.10
conda activate spt
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install PyTorch for GPU acceleration

The pipeline uses PyTorch to accelerate spot detection (~10–15×) and localization (~20–50×) on CUDA GPUs. CPU-only mode works without this step, but GPU dramatically reduces per-movie runtime.

**Find your CUDA version:**
```bash
nvidia-smi
```
Look for `CUDA Version: XX.X` in the top-right corner.

**Install the matching PyTorch build:**

| CUDA version | Install command |
|---|---|
| 11.8 | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |
| 12.1 | `pip install torch --index-url https://download.pytorch.org/whl/cu121` |
| 12.4 | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| CPU only | `pip install torch` |

> If you're unsure which CUDA version to use, the [PyTorch install selector](https://pytorch.org/get-started/locally/) will generate the right command for your system.

### 5. Verify the installation

```bash
python -c "
import sys; sys.path.insert(0, '.')
import core
print(core._report_gpu_status())
"
```

Expected output on a GPU machine:
```
GPU enabled — NVIDIA Quadro P4000
```

Expected output without a GPU:
```
GPU not available (no CUDA device found) — running on CPU
```

### 6. Install saspt

```bash
pip install saspt
```

> If saspt is not available on PyPI for your platform, install from source:
> ```bash
> git clone https://github.com/alecheckert/saspt.git
> cd saspt && pip install -e .
> ```

---

## Quick start (local)

```bash
python run_local.py --input_directory /path/to/nd2_files/
```

At startup the script prints which GPU (if any) is in use, then runs the full pipeline:
`track → condition → aggregate`

### Options

| Flag | Default | Description |
|---|---|---|
| `--input_directory` | *(required)* | Folder containing `.nd2` files |
| `--set` | *(none)* | Dotted settings override, repeatable — e.g. `--set quot.detect.t=10.0` |
| `--mode` | `all` | `track`, `condition`, `aggregate`, or `all` |
| `--force` | off | Re-run even if output files already exist |

### Run only tracking (detection + localization + SASPT)

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

Settings come from code defaults plus any `--set key.path=value` overrides you pass on the command line — there is no `settings_override.yaml` to hand-edit in the `.nd2` folder. Each run's resolved overrides are frozen into `<analysis_directory>/settings_override.yaml` (e.g. `tracking_output_q=10p0/settings_override.yaml`) purely for provenance/audit — that file lives inside the run's own output directory, never shared with any other run, so sweeping over several thresholds (or anything else) never risks one run's edits landing on another run's already-dispatched jobs.

`--set` is repeatable and accepts any dotted path into the settings tree, with the value parsed the same way a YAML scalar would be (`10.0` → float, `true` → bool, `[1,2,3]` → list, anything else → string):

```bash
python run_local.py --input_directory /path/to/nd2s \
    --set quot.detect.t=10.0 \
    --set quot.track.search_radius=0.1
```

For a cluster sweep, the same `--set` flags go to `prepare_run.py` (see [HPC / SLURM](#hpc--slurm) below).

**Example for fast SPT (~250 Hz, diffraction-limited spots):**

```bash
--set quot.detect.t=20.0 --set quot.detect.w=15 \
--set quot.localize.window_size=15 --set quot.localize.sigma=2.5 --set quot.localize.max_iter=10 \
--set quot.track.method=euclidean --set quot.track.search_radius=1.2 --set quot.track.min_I0=100.0 \
--set "plot.ylim=[0,5]" --set "plot.barGraphBreaks=[0, 0.05, 0.35, 2, 100]" \
--set "plot.barGraphLabels=[Immobile, Confined, Membrane, Free]" --set plot.mult_on_sd=3.0 \
--set saspt.focal_depth=0.7 --set saspt.splitsize=5
```

**Example for slow SPT (~1–10 Hz, single-molecule bleaching):**

```bash
--set quot.detect.t=15.0 --set quot.detect.w=11 \
--set quot.localize.window_size=7 --set quot.localize.max_iter=10 --set quot.localize.damp=1 \
--set quot.track.method=conservative --set quot.track.search_radius=0.15 --set quot.track.min_I0=200.0 \
--set "plot.bleach_xlim=[-125000, 25000]"
```

### All available settings

| Key | Default | Description |
|---|---|---|
| `quot.detect.method` | `llr` | Detection method (`llr`, `dog`, `log`, `gauss`, …) |
| `quot.detect.t` | `20.0` | Detection threshold |
| `quot.detect.k` | `15.0` | Gaussian kernel sigma for LLR filter |
| `quot.detect.w` | `15` | Window size for LLR filter (pixels) |
| `quot.localize.method` | `ls_int_gaussian` | Localization method |
| `quot.localize.window_size` | `11` | Fitting subwindow (pixels) |
| `quot.localize.sigma` | `1.5` | PSF sigma (pixels) |
| `quot.localize.max_iter` | `100` | Max LM iterations |
| `quot.localize.damp` | `1` | LM damping factor |
| `quot.track.method` | `euclidean` | Linking method (`euclidean` or `conservative`) |
| `quot.track.search_radius` | `1.2` | Max linking distance (μm) |
| `quot.track.pixel_size_um` | `0.11` | Pixel size (μm) |
| `quot.track.frame_interval` | `0.006` | Frame interval (s); auto-read from .nd2 if possible |
| `quot.track.min_I0` | `100.0` | Minimum mean spot intensity to include a trajectory |
| `saspt.focal_depth` | `0.1` | Axial depth of field (μm) |
| `saspt.splitsize` | `5` | Rolling-window half-width (frames) |
| `saspt.sample_size` | `100000` | Number of trajectories sampled per SASPT call |
| `plot.ylim` | `[0, 5]` | Y-axis limits for posterior diffusion histogram |
| `plot.barGraphBreaks` | `[-10000, 10000]` | Bin edges for population bar chart (μm²/s) |
| `plot.barGraphLabels` | `['None']` | Labels for each population bin |
| `plot.mult_on_sd` | `5.0` | Outlier rejection: max KS-distance from mean (in SDs) |
| `gpu.use_gpu` | `true` | Set to `false` to force CPU-only execution |

---

## GPU acceleration

When PyTorch is installed and a CUDA GPU is present, two stages are automatically accelerated:

| Stage | Method | Speedup vs CPU |
|---|---|---|
| **Detection** (LLR filter) | `torch.fft` + `avg_pool2d` on GPU | ~10–15× |
| **Localization** (LM Gaussian fit) | Batched `(N, H, W)` tensor ops; all spots in a frame fit simultaneously | ~20–50× on GPU; 1.7× on CPU from vectorization alone |

The fallback is transparent — if PyTorch is absent or no CUDA device is found, the original scipy/numpy code runs unchanged.

**To disable GPU** (e.g. for debugging), add:
```bash
--set gpu.use_gpu=false
```

---

## Output structure

All output is written to a sibling directory named `tracking_output_q=<threshold>/`:

```
tracking_output_q=20p0/
├── trajectories/           # Per-file _traj.csv (localizations + trajectories)
├── posterior/              # Per-file SASPT posterior diffusion-state CSVs
├── MLE/                    # Per-file MLE diffusion-coefficient CSVs
├── rolling_windows/        # Per-file rolling-window MLE CSVs
├── plots/
│   ├── plotting_pkls/      # Serialized per-condition data for aggregate step
│   ├── <condition>_posterior.pdf    # Diffusion coefficient histogram
│   ├── <condition>_bar.pdf          # Population fraction bar chart
│   ├── survival_curves/<condition>_survival.pdf  # Kaplan-Meier trajectory survival curve
│   ├── aggregate_posterior.pdf      # All conditions overlaid
│   └── aggregate_MLE_bar.pdf        # Cross-condition bar chart
├── movies/                 # Overlay MP4s of sampled trajectories
├── settings_override.yaml  # This run's explicit overrides only (provenance)
├── settings.pkl            # Full resolved settings dict (for reproducibility)
└── settings.txt            # Human-readable settings summary
```

---

## HPC / SLURM

`prepare_run.py` resolves settings from `--set` overrides and creates a fresh
`tracking_output_q=<t>/` analysis directory (same convention as `run_local.py`)
*before* anything is submitted to SLURM — that directory, not the `.nd2`
folder, is where this run's `settings_override.yaml` lives, and it's what the
`Snakefile` reads on every job's reparse. Two runs (e.g. sweeping several
thresholds) never share a mutable settings file, so nothing needs to be
serialized between them.

```bash
ANALYSIS_DIR=$(python3 prepare_run.py /path/to/nd2_files \
    --set quot.detect.t=10.0 --set quot.track.search_radius=0.1 \
    --write-controller)
sbatch "$ANALYSIS_DIR/run_snakemake_controller.slurm"
```

`--write-controller` writes a ready-to-submit `run_snakemake_controller.slurm`
into the analysis directory, with `--directory <analysis_directory>` (so this
run's Snakemake lock/metadata never collides with a concurrent run's) and
`--latency-wait 60` (NFS output-visibility latency on `$GROUP_HOME` can make
the default 5s wait fail) already set. Pass `--forceall` through to an
already-submitted controller to intentionally recompute everything after
changing that run's own `settings_override.yaml`:

```bash
sbatch "$ANALYSIS_DIR/run_snakemake_controller.slurm" --forceall
```

The DAG parallelizes the per-file tracking stage as a job array and runs
condition-wise and aggregate steps after all file jobs complete.

---

## Dependencies

- [saspt](https://github.com/alecheckert/saspt) — Bayesian diffusion-state inference (RBME model)
- [PyTorch](https://pytorch.org) — GPU acceleration (optional; CPU fallback if absent)
- numpy, pandas, scipy, matplotlib — core scientific stack
- pims, nd2reader — `.nd2` image file I/O
- imageio, imageio-ffmpeg — trajectory overlay movie writing
- hmmlearn — Gaussian HMM for state assignment
- PyYAML — settings file parsing
- snakemake — HPC workflow DAG (optional for local use)
