# quot_saspt_workflow

Single-particle tracking (SPT) pipeline combining **quot** (detection, localization, linking) with **SASPT** (Bayesian diffusion-state inference). Runs either sequentially on a local/interactive machine or in parallel on a SLURM cluster.

## Workflow

For each `.nd2` movie: **detect** spots → **localize** (sub-pixel Gaussian fit) → **link** into trajectories → run **SASPT** (rolling-window MLE + posterior diffusion-state inference). Per-condition outputs are then aggregated into QC plots (diffusion histograms, population bar charts, Kaplan-Meier survival curves), and finally compared across all conditions in a single aggregate report.

```
quot_saspt_workflow/
├── fastQuot/               # Bundled quot fork (GPU-accelerated), no external quot install needed
├── core/
│   ├── settings.py         # Settings defaults / merging / provenance freeze
│   ├── filewise.py         # Per-file: detect → localize → link → SASPT
│   ├── conditionwise.py    # Per-condition: aggregate, QC, HMM, plots
│   ├── aggregate.py        # Cross-condition comparison plots
│   ├── plots.py            # All plotting functions
│   ├── publish.py          # Push per-run reports to the sptLanding GitHub Pages site
│   └── utils.py
├── calling_scripts/        # Example settings_override.yaml and SLURM caller
├── run_local.py            # Sequential local runner (no SLURM needed)
├── prepare_run.py          # Prepares a SLURM run's analysis directory + controller script
├── Snakefile               # Snakemake DAG for HPC parallel execution
└── requirements.txt
```

Every run gets its own `tracking_output_q=<threshold>/` output directory (trajectories, posterior/MLE CSVs, plots, overlay movies), with its resolved settings frozen into `settings_override.yaml` inside that directory for provenance — sweeps over threshold/search-radius/etc. never share mutable state between runs.

## Dependencies

```bash
pip install -r requirements.txt
```

- [saspt](https://github.com/alecheckert/saspt) — Bayesian diffusion-state inference (RBME model)
- [PyTorch](https://pytorch.org) — GPU acceleration for detection (~10-15x) and localization (~20-50x); optional, falls back to CPU/scipy if absent or no CUDA device is found
- numpy, pandas, scipy, matplotlib — core scientific stack
- pims, nd2reader, tifffile, Pillow — `.nd2` / image I/O
- imageio, imageio-ffmpeg — trajectory overlay movie writing
- hmmlearn — Gaussian HMM for state assignment
- PyYAML — settings file parsing
- snakemake — HPC workflow DAG (optional for local-only use)

The quot/fastQuot package itself is bundled in `fastQuot/`, so it isn't a separate install.

## Example invocation

Local, sequential (runs `track → condition → aggregate`):

```bash
python run_local.py --input_directory /path/to/nd2_files/ \
    --set quot.detect.t=20.0 \
    --set quot.track.search_radius=0.15
```

`--set` is repeatable and takes any dotted path into the settings tree (`core/settings.py`), e.g. `--set quot.detect.t=10.0 --set "plot.ylim=[0,5]"`. `--mode {track,condition,aggregate,all}` runs a single stage; `--force` re-runs even if outputs already exist.

To also emit background-corrected survival curves (a separate `*_survival_corrected.pdf`/`*_survival_overlay_corrected.pdf` output alongside the normal ones, subtracting the implied background population estimated from a control condition), set `--set survival.background_condition=<condition>`. The named condition must contain `000` or `322` (this project's background/control naming convention) or settings resolution raises an error.

HPC / SLURM, for threshold or linking-distance sweeps:

```bash
ANALYSIS_DIR=$(python3 prepare_run.py /path/to/nd2_files \
    --set quot.detect.t=10.0 --set quot.track.search_radius=0.1 \
    --write-controller)
sbatch "$ANALYSIS_DIR/run_snakemake_controller.slurm"
```

`prepare_run.py` creates the run's `tracking_output_q=<t>/` directory and freezes its `settings_override.yaml` before anything is submitted, so concurrent sweep members never share a settings file. The DAG parallelizes per-file tracking as a job array, then runs condition-wise and aggregate steps once all file jobs finish.

## Publishing results

After a run's plots are generated, `core/publish.py` pushes a markdown report (settings + every plot) to the `sptLanding` GitHub Pages repo. To set that up for a new repo or a new machine:

### Add a deploy key

Deploy keys scope push access to a single repo, so the machine running the pipeline never touches your other repos' credentials.

1. Generate a dedicated keypair (no passphrase, since it runs unattended):
   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/gh_sptlanding_deploy -N ""
   ```
2. On GitHub: repo → **Settings → Deploy keys → Add deploy key**, paste the contents of `gh_sptlanding_deploy.pub`, and check **Allow write access**.
3. Add a host alias to `~/.ssh/config` so git uses that key only for this repo:
   ```
   Host github-sptlanding
     HostName github.com
     User git
     IdentityFile ~/.ssh/gh_sptlanding_deploy
     IdentitiesOnly yes
   ```
4. Point the repo URL at the alias instead of `github.com`, e.g. `git@github-sptlanding:<owner>/sptLanding.git`.

### Move a personal GitHub Pages repo under the organization

1. On the personal repo: **Settings → General → Transfer ownership**, enter the organization name, and confirm. An org owner must accept the transfer.
2. Re-check **Settings → Pages** on the transferred repo — the Pages source (branch/folder) carries over but is worth confirming, since transfers occasionally reset it.
3. Re-issue a deploy key scoped to the repo under its new `<org>/<repo>` path (deploy keys don't transfer with the repo) — see above.
4. Grant org teammates who need push access repo membership via the organization's **Teams** settings, rather than adding them as individual collaborators.
