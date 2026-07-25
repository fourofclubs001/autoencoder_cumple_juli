#!/usr/bin/env python3
"""
Kaggle notebook runner for RL experiments.

Usage:
  python runner.py push <notebook.ipynb> <kernel-slug>  [--no-gpu] [--accelerator NvidiaTeslaT4] [--github-token TOKEN]
  python runner.py status <kernel-slug>
  python runner.py wait <kernel-slug>                   [--interval 30]
  python runner.py logs <kernel-slug> [--output-dir ./output]
  python runner.py run <notebook.ipynb> <kernel-slug>   [--no-gpu] [--accelerator NvidiaTeslaT4] [--interval 30] [--github-token TOKEN]

  python runner.py queue <notebook.ipynb> <kernel-slug> [--no-gpu] [--accelerator NvidiaTeslaT4]
  python runner.py queue-list
  python runner.py queue-clear
  python runner.py queue-run                            [--wait-for <slug>] [--interval 30] [--github-token TOKEN]

Available accelerators: NvidiaTeslaT4, NvidiaTeslaT4Highmem, NvidiaTeslaP100,
                        NvidiaTeslaA100, NvidiaL4, NvidiaH100

GitHub token injection:
  Notebooks that contain a cell with exactly `token = ""  # injected by runner`
  will have the token substituted before push. Token is read from (in order):
    --github-token flag, GITHUB_TOKEN env var, ~/.kaggle/github_token file.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Kaggle CLI on Windows is installed as a user script, not on the system PATH.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

KAGGLE = Path.home() / "AppData/Roaming/Python/Python313/Scripts/kaggle.exe"
USERNAME = "lucasvitali"

# Exact string that marks the token injection cell inside a notebook.
# The runner replaces the whole cell source with the real token before pushing.
INJECTED_PLACEHOLDER = 'token = ""  # injected by runner'

# Persistent files — kept next to this script so they survive across sessions.
QUEUE_FILE = Path(__file__).parent / "queue.json"   # ordered list of queued experiments
STATE_FILE = Path(__file__).parent / "state.json"   # tracks the last pushed kernel slug


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_github_token(cli_token: str | None) -> str | None:
    """Return a GitHub token from CLI flag, env var, or ~/.kaggle/github_token."""
    if cli_token:
        return cli_token
    if val := os.environ.get("GITHUB_TOKEN"):
        return val
    p = Path.home() / ".kaggle" / "github_token"
    if p.exists():
        return p.read_text().strip()
    return None


def _inject_token(notebook_path: Path, github_token: str) -> Path:
    """Return path to a temp notebook with the token injected into the placeholder cell.

    Kaggle's secret manager (UserSecretsClient) only works in the web UI, not via
    the API. Instead, notebooks expose a placeholder cell that this function replaces
    with the real token before pushing. A temp copy is used so the original file is
    never modified.
    """
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    injected = False
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell["source"])
        if INJECTED_PLACEHOLDER in src:
            cell["source"] = [f'token = "{github_token}"  # injected by runner\n']
            injected = True
            break
    if not injected:
        print("Warning: placeholder not found in notebook — token not injected", file=sys.stderr)
        return notebook_path
    # Write the modified notebook to a temp file in the same directory so that
    # kernel-metadata.json (written next to the notebook) still resolves correctly.
    tmp = Path(tempfile.mktemp(suffix=".ipynb", dir=notebook_path.parent))
    tmp.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return tmp


def _kaggle(*args):
    """Run the Kaggle CLI with the API token from env or ~/.kaggle/access_token.

    PYTHONUTF8=1 prevents cp1252 encoding errors on Windows when the output
    contains ANSI escape codes from the Kaggle CLI.
    """
    token = (
        os.environ.get("KAGGLE_API_TOKEN")
        or (Path.home() / ".kaggle/access_token").read_text().strip()
    )
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        [str(KAGGLE), *args], capture_output=True, text=True, encoding="utf-8", env=env
    )


def _save_state(kernel_slug: str):
    """Persist the last pushed kernel slug so queue-run can auto-detect it."""
    STATE_FILE.write_text(json.dumps({"last_pushed": kernel_slug}, indent=2), encoding="utf-8")


def _last_pushed_slug() -> str | None:
    """Return the slug of the last kernel pushed by this runner, or None."""
    if not STATE_FILE.exists():
        return None
    return json.loads(STATE_FILE.read_text(encoding="utf-8")).get("last_pushed")


# Kaggle's kernel-level status only reflects whether the kernel process itself
# crashed (OOM, timeout, etc). A `!python script.py` cell that raises inside the
# notebook does NOT fail the kernel — Jupyter just moves on to the next cell — so
# `kernels status` still reports COMPLETE. These patterns catch that class of
# silent failure by grepping the downloaded log for tracebacks / argparse errors.
_ERROR_PATTERNS = [
    re.compile(r'Traceback \(most recent call last\)'),
    re.compile(r'\.py: error:'),
]


def _scan_log_for_errors(output_dir: Path) -> list[str]:
    """Return the matching lines from any .log files in output_dir that look like
    a Python traceback or argparse error, even though the kernel itself completed."""
    hits = []
    for f in sorted(output_dir.glob("*.log")):
        text = f.read_text(encoding="utf-8", errors="replace")
        for pattern in _ERROR_PATTERNS:
            for m in pattern.finditer(text):
                line_start = text.rfind("\n", 0, m.start()) + 1
                line_end = text.find("\n", m.end())
                line_end = line_end if line_end != -1 else len(text)
                hits.append(text[line_start:line_end].strip())
    return hits


def _load_queue() -> list:
    """Load the experiment queue from disk. Returns an empty list if not yet created."""
    if not QUEUE_FILE.exists():
        return []
    return json.loads(QUEUE_FILE.read_text(encoding="utf-8"))


def _save_queue(queue: list):
    """Persist the experiment queue to disk."""
    QUEUE_FILE.write_text(json.dumps(queue, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_push(notebook_path: str, kernel_slug: str, enable_gpu: bool = True, accelerator: str = "NvidiaTeslaT4", github_token: str | None = None):
    """Push a notebook to Kaggle.

    Generates kernel-metadata.json next to the notebook (required by the Kaggle CLI),
    injects the GitHub token into the placeholder cell, pushes, then deletes the temp
    copy. Also records the slug in state.json for queue-run auto-detection.
    """
    notebook = Path(notebook_path).resolve()
    if not notebook.exists():
        print(f"Error: notebook not found: {notebook}", file=sys.stderr)
        sys.exit(1)

    token = _resolve_github_token(github_token)
    tmp_notebook = None
    if token:
        tmp_notebook = _inject_token(notebook, token)
        if tmp_notebook != notebook:
            print(f"Injected GitHub token into {tmp_notebook.name}")
            notebook = tmp_notebook
    else:
        print("Warning: no GitHub token found — skipping injection", file=sys.stderr)

    # kernel-metadata.json is what the Kaggle CLI reads when pushing a kernel directory.
    metadata = {
        "id": f"{USERNAME}/{kernel_slug}",
        "title": kernel_slug.replace("-", " ").title(),
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": enable_gpu,
        "enable_tpu": False,
        "enable_internet": True,   # needed to clone the repo and download the model
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }

    meta_path = notebook.parent / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote {meta_path}")

    push_args = ["kernels", "push", "-p", str(notebook.parent)]
    if enable_gpu:
        push_args += ["--accelerator", accelerator]

    result = _kaggle(*push_args)
    print(result.stdout.strip())
    if tmp_notebook and tmp_notebook.exists():
        tmp_notebook.unlink()
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)
    print(f"Pushed: https://www.kaggle.com/code/{USERNAME}/{kernel_slug}")
    _save_state(kernel_slug)


def cmd_status(kernel_slug: str) -> str:
    """Print and return the current status of a kernel (running / complete / error)."""
    result = _kaggle("kernels", "status", f"{USERNAME}/{kernel_slug}")
    text = result.stdout.strip()
    print(text)
    return text


def cmd_wait(kernel_slug: str, interval: int = 30):
    """Block until a kernel reaches a terminal state (complete / error / cancel).

    Polls every `interval` seconds. Kaggle free-tier kernels can run for up to
    ~9 hours, so a long wait is normal for GPU experiments.
    """
    print(f"Waiting for {USERNAME}/{kernel_slug} (polling every {interval}s)...")
    while True:
        result = _kaggle("kernels", "status", f"{USERNAME}/{kernel_slug}")
        status = result.stdout.strip()
        print(f"  [{time.strftime('%H:%M:%S')}] {status}")
        lower = status.lower()
        if any(s in lower for s in ("complete", "error", "cancel")):
            return status
        time.sleep(interval)


def cmd_logs(kernel_slug: str, output_dir: str | None = None):
    """Download kernel output files and print text ones to stdout.

    Files are saved to kaggle_output/<kernel_slug>/ by default (one dir per run).
    .png files are listed but not printed. Text files (.log, .json, .txt) are
    printed in full so the results are visible in the terminal.
    """
    if output_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = str(Path(__file__).parent / "kaggle_output" / f"{kernel_slug}_{timestamp}")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    result = _kaggle("kernels", "output", f"{USERNAME}/{kernel_slug}", "--path", str(out))
    if result.returncode != 0:
        print(result.stderr.strip(), file=sys.stderr)
        sys.exit(1)

    files = sorted(f for f in out.iterdir() if f.is_file() and f.suffix in (".log", ".json", ".txt", ".png"))
    if not files:
        print("No output files found.")
        return out

    for f in files:
        if f.suffix == ".png":
            print(f"  [image] {f.name}")
            continue
        print(f"\n{'='*60}")
        print(f"FILE: {f.name}")
        print("=" * 60)
        print(f.read_text(encoding="utf-8", errors="replace"))

    return out


def cmd_run(notebook_path: str, kernel_slug: str, enable_gpu: bool = True, accelerator: str = "NvidiaTeslaT4", interval: int = 30, github_token: str | None = None, output_dir: str | None = None) -> str:
    """Push a notebook, wait for it to finish, fetch logs, and scan the logs for
    script-level errors. Returns the final status string — 'error' is included in
    it either when Kaggle reports a kernel-level error, or when a script inside
    the notebook raised a traceback that a COMPLETE kernel status would hide."""
    cmd_push(notebook_path, kernel_slug, enable_gpu, accelerator, github_token)
    print()
    final_status = cmd_wait(kernel_slug, interval)
    print()
    out_dir = cmd_logs(kernel_slug, output_dir)

    if out_dir is not None:
        error_lines = _scan_log_for_errors(out_dir)
        if error_lines:
            print(f"\n/!\\ Detected {len(error_lines)} script error(s) in the log "
                  f"despite Kaggle status '{final_status}':")
            for line in error_lines[:10]:
                print(f"  {line}")
            if "error" not in final_status.lower():
                final_status = f"{final_status} (script error detected in log)"

    print(f"\nFinal status: {final_status}")
    return final_status


def cmd_queue_add(notebook_path: str, kernel_slug: str, enable_gpu: bool = True, accelerator: str = "NvidiaTeslaT4", output_dir: str | None = None):
    """Append an experiment to the persistent queue (queue.json).

    The queue is just a JSON array on disk. Entries are processed in order by
    queue-run. You can call this multiple times to build up a list before starting.
    """
    notebook = Path(notebook_path).resolve()
    if not notebook.exists():
        print(f"Error: notebook not found: {notebook}", file=sys.stderr)
        sys.exit(1)
    queue = _load_queue()
    entry = {
        "notebook": str(notebook),
        "kernel_slug": kernel_slug,
        "enable_gpu": enable_gpu,
        "accelerator": accelerator,
        "output_dir": output_dir,
    }
    queue.append(entry)
    _save_queue(queue)
    print(f"Queued [{len(queue)}]: {kernel_slug}  ({notebook.name})")


def cmd_queue_list():
    """Print all experiments currently in the queue."""
    queue = _load_queue()
    if not queue:
        print("Queue is empty.")
        return
    print(f"Queue ({len(queue)} experiment{'s' if len(queue) != 1 else ''}):")
    for i, entry in enumerate(queue, 1):
        gpu_str = f"{entry['accelerator']}" if entry['enable_gpu'] else "no GPU"
        print(f"  {i}. {entry['kernel_slug']}  [{gpu_str}]  ({Path(entry['notebook']).name})")


def cmd_queue_clear():
    """Empty the queue without affecting any currently-running kernel."""
    _save_queue([])
    print("Queue cleared.")


def cmd_queue_run(wait_for: str | None = None, interval: int = 30, github_token: str | None = None):
    """Process the experiment queue one entry at a time.

    If --wait-for is not given, the runner reads state.json to find the last pushed
    kernel and checks whether it is still running. If it is, it waits for it to
    finish before starting the queue. If it has already finished (or there is no
    recorded last-pushed kernel), the queue starts immediately.

    The queue file is re-read before each experiment, so entries added while the
    queue is already running will be picked up automatically.

    Aborts the entire queue if any experiment finishes with an error status.
    """
    if wait_for is None:
        wait_for = _last_pushed_slug()
        if wait_for:
            print(f"Auto-detected last pushed kernel: {wait_for}")

    if wait_for:
        status_result = _kaggle("kernels", "status", f"{USERNAME}/{wait_for}")
        current_status = status_result.stdout.strip().lower()
        if any(s in current_status for s in ("complete", "error", "cancel")):
            print(f"{wait_for} is already done — starting queue immediately.")
        else:
            print(f"Waiting for {USERNAME}/{wait_for} to finish before starting queue...")
            status = cmd_wait(wait_for, interval)
            if "error" in status.lower():
                print(f"\n{wait_for} errored — continuing with queued experiments.", file=sys.stderr)
        print()

    while True:
        queue = _load_queue()
        if not queue:
            print("Queue is empty — done.")
            break
        entry = queue[0]
        _save_queue(queue[1:])  # pop before running so a crash doesn't re-run the same entry
        remaining = len(queue) - 1
        print(f"Starting: {entry['kernel_slug']}  ({remaining} remaining in queue)")
        status = cmd_run(entry["notebook"], entry["kernel_slug"], entry["enable_gpu"], entry["accelerator"], interval, github_token, entry.get("output_dir"))
        if "error" in status.lower():
            print(f"\n{entry['kernel_slug']} errored ({status}) — aborting queue. "
                  f"{remaining} remaining experiment(s) left untouched in queue.json.", file=sys.stderr)
            break
        print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kaggle notebook runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("push", help="Push a notebook to Kaggle")
    p.add_argument("notebook", help="Path to .ipynb file")
    p.add_argument("kernel_slug", help="Kaggle kernel slug (e.g. my-experiment-1)")
    p.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    p.add_argument("--accelerator", default="NvidiaTeslaT4", help="Accelerator type (default: NvidiaTeslaT4)")
    p.add_argument("--github-token", default=None, help="GitHub token to inject (falls back to GITHUB_TOKEN env or ~/.kaggle/github_token)")

    p = sub.add_parser("status", help="Check kernel status")
    p.add_argument("kernel_slug")

    p = sub.add_parser("wait", help="Poll until kernel finishes")
    p.add_argument("kernel_slug")
    p.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")

    p = sub.add_parser("logs", help="Download and print kernel output logs")
    p.add_argument("kernel_slug")
    p.add_argument("--output-dir", default=None, help="Output directory (default: kaggle_output/<slug>/)")

    p = sub.add_parser("run", help="Push, wait, and fetch logs in one step")
    p.add_argument("notebook", help="Path to .ipynb file")
    p.add_argument("kernel_slug", help="Kaggle kernel slug")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--accelerator", default="NvidiaTeslaT4", help="Accelerator type (default: NvidiaTeslaT4)")
    p.add_argument("--interval", type=int, default=30)
    p.add_argument("--github-token", default=None, help="GitHub token to inject (falls back to GITHUB_TOKEN env or ~/.kaggle/github_token)")
    p.add_argument("--output-dir", default=None, help="Directory to save output files (default: kaggle_output/<slug>_<timestamp>/)")

    p = sub.add_parser("queue", help="Add a notebook to the persistent experiment queue")
    p.add_argument("notebook", help="Path to .ipynb file")
    p.add_argument("kernel_slug", help="Kaggle kernel slug")
    p.add_argument("--no-gpu", action="store_true")
    p.add_argument("--accelerator", default="NvidiaTeslaT4", help="Accelerator type (default: NvidiaTeslaT4)")
    p.add_argument("--output-dir", default=None, help="Directory to save output files (default: kaggle_output/<slug>_<timestamp>/)")

    sub.add_parser("queue-list", help="Show all experiments in the queue")

    sub.add_parser("queue-clear", help="Remove all experiments from the queue")

    p = sub.add_parser("queue-run", help="Process the queue: run experiments one by one until empty")
    p.add_argument("--wait-for", default=None, metavar="SLUG", help="Wait for this kernel to finish before starting the queue (default: auto-detect from state.json)")
    p.add_argument("--interval", type=int, default=30, help="Poll interval in seconds")
    p.add_argument("--github-token", default=None, help="GitHub token to inject")

    args = parser.parse_args()

    if args.command == "push":
        cmd_push(args.notebook, args.kernel_slug, not args.no_gpu, args.accelerator, args.github_token)
    elif args.command == "status":
        cmd_status(args.kernel_slug)
    elif args.command == "wait":
        cmd_wait(args.kernel_slug, args.interval)
    elif args.command == "logs":
        cmd_logs(args.kernel_slug, args.output_dir)
    elif args.command == "run":
        status = cmd_run(args.notebook, args.kernel_slug, not args.no_gpu, args.accelerator, args.interval, args.github_token, args.output_dir)
        if "error" in status.lower():
            sys.exit(1)
    elif args.command == "queue":
        cmd_queue_add(args.notebook, args.kernel_slug, not args.no_gpu, args.accelerator, args.output_dir)
    elif args.command == "queue-list":
        cmd_queue_list()
    elif args.command == "queue-clear":
        cmd_queue_clear()
    elif args.command == "queue-run":
        cmd_queue_run(args.wait_for, args.interval, args.github_token)


if __name__ == "__main__":
    main()
