"""
core/publish.py

Publish a per-run markdown report (resolved settings + every plot the run
produced) to the ``sptLanding`` GitHub Pages repo, as the final step after
aggregate plots are generated.

Uses the ``github-results`` SSH host alias (~/.ssh/config), which points at
a deploy key scoped to gschliss/sptLanding only -- this module never touches
any other repository's credentials.

Each call clones the repo fresh into its own temp directory and pushes with
a fetch/rebase retry loop, so concurrent sweep members finishing at
overlapping times race safely on the *remote* branch instead of sharing a
local git working directory (see settings_override.yaml incident notes in
core/settings.py for why shared local mutable state is the thing to avoid
here).
"""

from __future__ import annotations

import datetime
import glob
import io
import os
import shutil
import subprocess
import tempfile

from core.settings import print_nested_dict

_REPO_SSH_ALIAS = "github-results"
_REPO_URL = f"git@{_REPO_SSH_ALIAS}:gschliss/sptLanding.git"
_REPO_BRANCH = "main"
_REPO_DOCS_SUBDIR = "docs"
_MAX_PUSH_RETRIES = 5

# pdftoppm (poppler/0.47.0) is deliberately NOT loaded via `ml system poppler`
# into the caller's shell: that module reloads libjpeg-turbo 2.1.4 -> 1.5.1
# and libtiff 4.5.0 -> 4.0.8, which are exactly the libraries
# py-pillow-simd/matplotlib depend on in load_saspt_modules.sh -- the same
# cascade-downgrade failure mode that file's header comment already warns
# about avoiding for the py-* module set. Instead, pdftoppm is invoked by
# absolute path with its own LD_LIBRARY_PATH passed only to that one
# subprocess, never mutating this process's environment. Captured by diffing
# `ml system poppler/0.47.0`'s LD_LIBRARY_PATH in a clean shell (2026-08-11);
# if Sherlock's poppler/0.47.0 module install ever moves, regenerate this.
_POPPLER_BIN = "/share/software/user/open/poppler/0.47.0/bin/pdftoppm"
_POPPLER_LIB_PATH = ":".join([
    "/share/software/user/open/poppler/0.47.0/lib",
    "/share/software/user/open/fontconfig/2.12.4/lib",
    "/share/software/user/open/x11/7.7/lib",
    "/share/software/user/open/libxkbcommon/0.9.1/lib64",
    "/share/software/user/open/llvm/4.0.0/lib",
    "/share/software/user/open/libxml2/2.9.4/lib",
    "/share/software/user/open/freetype/2.9.1/lib",
    "/share/software/user/open/harfbuzz/1.4.8/lib",
    "/share/software/user/open/icu/59.1/lib",
    "/share/software/user/open/cairo/1.14.10/lib",
    "/share/software/user/open/gobject-introspection/1.52.1/lib",
    "/share/software/user/open/glib/2.52.3/lib",
    "/share/software/user/open/libffi/3.4.5/lib64",
    "/share/software/user/open/libpng/1.2.57/lib",
    "/share/software/user/open/zlib/1.2.11/lib",
    "/share/software/user/open/libtiff/4.0.8/lib",
    "/share/software/user/open/libjpeg-turbo/1.5.1/lib",
])


def _git(*args: str, cwd: str) -> None:
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True)


def _rasterize_pdf_to_png(pdf_path: str, out_png_prefix: str) -> str | None:
    """Render page 1 of *pdf_path* to ``<out_png_prefix>.png`` via pdftoppm.

    Returns None (rather than raising) if the pinned pdftoppm binary is
    missing -- e.g. Sherlock's poppler/0.47.0 install moved -- so a missing
    rasterizer degrades to PDF-only links instead of failing the whole
    publish step.
    """
    if not os.path.exists(_POPPLER_BIN):
        return None
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = _POPPLER_LIB_PATH + ":" + env.get("LD_LIBRARY_PATH", "")
    subprocess.run(
        [_POPPLER_BIN, "-png", "-r", "150", "-singlefile", pdf_path, out_png_prefix],
        check=True, capture_output=True, text=True, env=env,
    )
    png_path = out_png_prefix + ".png"
    return png_path if os.path.exists(png_path) else None


def _settings_markdown_block(settings: dict) -> str:
    display = {k: v for k, v in settings.items() if k in ("quot", "saspt", "plot", "gpu")}
    # diff_coefs/loc_errors are large np.power/np.linspace-derived grids, not
    # values anyone tunes directly -- dumping ~125 numbers here buries the
    # settings a sweep report actually needs to show (t, search_radius, k,
    # w, sigma, ...).
    display["saspt"] = {
        k: v for k, v in display["saspt"].items() if k not in ("diff_coefs", "loc_errors")
    }
    buf = io.StringIO()
    print_nested_dict(display, file=buf)
    return buf.getvalue().rstrip("\n")


def publish_run_report(settings: dict) -> str | None:
    """Render this run's settings + every generated plot into a markdown
    report and push it to the sptLanding repo.

    Never raises: a publish failure (network hiccup, exhausted push
    retries, missing pdftoppm) must not fail a pipeline run that has
    already completed successfully. Returns the report's path inside
    sptLanding on success, or None if publishing was skipped/failed.
    """
    plot_dir = settings["io"]["plot_directory"]
    pdf_paths = sorted(glob.glob(os.path.join(plot_dir, "**", "*.pdf"), recursive=True))
    if not pdf_paths:
        print("publish_run_report: no plots found under plot_directory — skipping publish.")
        return None

    date_dir_name = os.path.basename(settings["io"]["home_directory"].rstrip("/"))
    analysis_tag = os.path.basename(settings["io"]["analysis_directory"].rstrip("/"))
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_stem = f"{timestamp}_{analysis_tag}"
    report_rel_dir = os.path.join(_REPO_DOCS_SUBDIR, date_dir_name, report_stem)

    try:
        with tempfile.TemporaryDirectory(dir=os.environ.get("SCRATCH")) as tmp:
            clone_dir = os.path.join(tmp, "sptLanding")
            _git("clone", "--depth", "1", "--branch", _REPO_BRANCH, _REPO_URL, clone_dir, cwd=tmp)

            report_abs_dir = os.path.join(clone_dir, report_rel_dir)
            pdf_abs_dir = os.path.join(report_abs_dir, "pdfs")
            os.makedirs(pdf_abs_dir, exist_ok=True)

            lines = [
                f"# SPT run report — {date_dir_name} / {analysis_tag}",
                "",
                f"Generated: {timestamp}",
                "",
                f"Analysis directory: `{settings['io']['analysis_directory']}`",
                "",
                "## Settings",
                "",
                "```",
                _settings_markdown_block(settings),
                "```",
                "",
                "## Plots",
                "",
            ]

            for pdf_path in pdf_paths:
                rel = os.path.relpath(pdf_path, plot_dir)
                flat_name = rel.replace(os.sep, "__")

                dest_pdf = os.path.join(pdf_abs_dir, flat_name)
                shutil.copy2(pdf_path, dest_pdf)

                png_prefix = os.path.join(report_abs_dir, os.path.splitext(flat_name)[0])
                png_path = _rasterize_pdf_to_png(pdf_path, png_prefix)

                lines.append(f"### {rel}")
                lines.append("")
                if png_path:
                    # report_abs_dir (and everything under it, incl. pdfs/) is
                    # a subdirectory next to the .md file, not the .md file's
                    # own directory -- links must be prefixed with
                    # report_stem/, a bare basename/"pdfs/..." 404s.
                    lines.append(f"![{rel}]({report_stem}/{os.path.basename(png_path)})")
                else:
                    lines.append("_(PNG preview unavailable — pdftoppm not on PATH for this job)_")
                lines.append("")
                lines.append(f"[Download PDF]({report_stem}/pdfs/{flat_name})")
                lines.append("")

            md_path = report_abs_dir + ".md"
            with open(md_path, "w") as fh:
                fh.write("\n".join(lines))

            _git("add", "-A", cwd=clone_dir)
            _git(
                "-c", "user.email=sherlock-pipeline@noreply.github.com",
                "-c", "user.name=SPT pipeline (Sherlock)",
                "commit", "-m", f"Add run report {date_dir_name}/{report_stem}",
                cwd=clone_dir,
            )

            for attempt in range(_MAX_PUSH_RETRIES):
                try:
                    _git("push", "origin", f"HEAD:{_REPO_BRANCH}", cwd=clone_dir)
                    break
                except subprocess.CalledProcessError:
                    if attempt == _MAX_PUSH_RETRIES - 1:
                        raise
                    _git("fetch", "origin", _REPO_BRANCH, cwd=clone_dir)
                    _git("rebase", f"origin/{_REPO_BRANCH}", cwd=clone_dir)

            print(f"publish_run_report: pushed {report_rel_dir}.md to sptLanding")
            return f"{report_rel_dir}.md"
    except Exception as exc:
        print(f"publish_run_report: FAILED to publish ({exc!r}) — continuing without publish.")
        return None
