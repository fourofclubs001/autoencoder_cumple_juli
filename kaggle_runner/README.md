# kaggle_runner

Runs 2048-RL experiment notebooks on Kaggle (T4 GPU) and fetches output logs.
Adapted from the same tool used in the `tesis` repo — logic in `runner.py` is
unchanged, only this README and the notebooks under `../notebooks/` are
project-specific.

## Setup

- Kaggle CLI installed, authenticated via `~/.kaggle/access_token`
- Python 3.x

## Prerequisite: push local changes to GitHub first

Kaggle notebooks in this repo `git clone` `autoencoder_cumple_juli` from **GitHub**
at runtime (into `/tmp/`) — they do not see local disk. Any change to `rl2048/*.py`
or the notebooks themselves must be **committed and pushed** before running/queueing,
or the job will run against stale code on Kaggle with no local-vs-remote warning.

## GPU note (why this matters here)

Don't push without `--accelerator` (or via plain `kaggle kernels push`) — an
unspecified GPU can land you on a P100, whose sm_60 kernels this repo's PyTorch
build does not support (`CUDA error: no kernel image is available for execution
on the device`, raised on the first `.forward()` call, i.e. mid-training).
`runner.py`'s default (`NvidiaTeslaT4`) avoids this; always go through `runner.py`
rather than a bare `kaggle kernels push`.

## Usage — single experiment

```bash
# Push, run, wait, and fetch logs in one command
python runner.py run path/to/experiment.ipynb my-experiment-1

# Or step by step:
python runner.py push path/to/experiment.ipynb my-experiment-1
python runner.py wait my-experiment-1
python runner.py logs my-experiment-1

# Without GPU
python runner.py run path/to/experiment.ipynb my-experiment-1 --no-gpu

# Custom poll interval (seconds)
python runner.py run path/to/experiment.ipynb my-experiment-1 --interval 60
```

## Usage — queue (multiple experiments, run sequentially)

Preferred for batches of notebooks (e.g. a concept/template/model sweep), since a failure in one job doesn't block the rest and the queue survives across sessions (`kaggle_runner/queue.json`).

```bash
# Add jobs to the persistent queue
python runner.py queue path/to/experiment_a.ipynb experiment-a
python runner.py queue path/to/experiment_b.ipynb experiment-b

# Inspect / clear the queue
python runner.py queue-list
python runner.py queue-clear

# Process the queue: pushes, waits, downloads logs for each job in order until empty
python runner.py queue-run --interval 30
```

`queue-run` re-reads `queue.json` on every iteration, so jobs can be appended to the queue while it's already running. It also stops the queue (does not continue to the next job) if a run's log shows a traceback or argparse error, even if Kaggle reports the kernel as `COMPLETE` — see research log 2026-07-02 for why kernel-level status alone can't be trusted.

## Notes

- Kernel slugs must be lowercase, hyphens only (e.g. `dqn-baseline-9k`)
- Logs are saved to `kaggle_output/` next to the notebook (single-run commands) or `kaggle_output/<slug>_<timestamp>/` (queue)
- Kaggle free tier: ~30h/week GPU quota
- Notebooks clone the repo from GitHub instead of bundling code so the same `rl2048/*.py` files are used locally, in notebooks, and on Kaggle — no duplicated agent logic anywhere
- Training scripts (e.g. `rl2048/train.py`) take `--output-dir` and write a checkpoint + CSV history + PNG plot there; point it at `/kaggle/working/` so `runner.py logs` fetches them automatically. Checkpoints double as resume points if a session gets interrupted.
