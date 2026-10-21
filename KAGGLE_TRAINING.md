# Training the JEV network on Kaggle (free GPU)

Step-by-step guide to train `configs/full_lora_k12.yaml` (Qwen3-0.6B, first 20
layers, LoRA on layers 8-19) on Kaggle's free GPU. Kaggle gives **30 GPU
hours/week**, sessions up to **12 h**, and **no credit card**.

Best free fit for this job: **GPU P100 (16 GB)** or **GPU T4 x2**. The full run is
roughly 3-6 h, so it fits in one session.

## 0. What you will do

1. Package the repo (the code bugs are already fixed) and upload it as a private
   Kaggle Dataset.
2. Create a notebook with GPU + internet, install `uv`, run a smoke test.
3. Run full training and keep the checkpoints.
4. (Optional) resume if a session times out.

## 1. Code status: already fixed

The bugs that previously blocked a fresh run are fixed in the current repo (the
attention output projection, the `BEVDataset`/`JEVDataset` import mismatch, and
the `network_workers=` DataLoader kwarg). No in-notebook patching is needed.

## 2. Package the repo on your Mac

The repo has no git remote, so ship it as a zip.

```bash
cd ~/Projects/jev-train
rm -f jev-train-code.zip
zip -r jev-train-code.zip . \
  -x ".venv/*" -x "runs/*" -x "export/*" -x ".git/*" \
  -x "**/__pycache__/*" -x "*.pyc" -x ".zed/*" \
  -x "*.zip" -x ".DS_Store"
```

## 3. Upload the code as a Kaggle Dataset

1. kaggle.com -> **Datasets** -> **New Dataset**.
2. Upload `jev-train-code.zip`. Name it `jev-train-code`, set visibility
   **Private**, and create it.
3. It will appear inside notebooks at `/kaggle/input/jev-train-code/`.

## 4. Create the notebook and enable GPU + internet

1. kaggle.com -> **Code** -> **New Notebook**.
2. Right pane -> **Settings**:
   - **Accelerator**: `GPU P100` (or `GPU T4 x2`).
   - **Internet**: `On`. This requires a phone-verified account. Without it, the
     code cannot download Qwen3-0.6B or `avbiswas/bev-decision-150K`.
   - **Persistence**: `Files only` or `Variables and files` (keep `/kaggle/working`).
3. **Add Input** -> your `jev-train-code` dataset.

## 5. Notebook cells (run in order)

Replace the notebook's default boilerplate cell with **Cell 1**, then add
**Cell 2** and run it. Together these replace the old copy + install steps.

**Order matters:** Cell 1 deletes and re-extracts the repo, so run it *before*
Cell 4. Re-running Cell 1 after Cell 4 erases the `configs/kaggle_k12.yaml`
written by Cell 4 (re-run Cell 4 if that happens). To change only the
environment (e.g. the allocator), use a standalone cell rather than re-running
Cell 1.

### Cell 1 - copy the code and set the environment

Gets the repo onto the instance. It first looks for an attached dataset under
`/kaggle/input`; if none is attached, it downloads the dataset with `kagglehub`
(needs internet on). Then it extracts the repo and sets the environment:

```python
import os, glob, zipfile, shutil

!nvidia-smi  # confirm the GPU is attached

DATASET = "neelcontractor/jev-model-zip"   # your Kaggle dataset slug
WORK = "/kaggle/working"
REPO = f"{WORK}/jev-train"
if os.path.exists(REPO):
    shutil.rmtree(REPO)


def find(base):
    return (
        (glob.glob(f"{base}/**/*.zip", recursive=True) or [None])[0],
        (glob.glob(f"{base}/**/pyproject.toml", recursive=True) or [None])[0],
    )


zip_path, proj = find("/kaggle/input")
if zip_path is None and proj is None:
    import kagglehub  # downloads the dataset if it is not attached under "Add Input"
    base = kagglehub.dataset_download(DATASET)
    print("kagglehub path:", base)
    zip_path, proj = find(base)

if zip_path:
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(REPO)
    print("extracted", zip_path)
elif proj:
    shutil.copytree(os.path.dirname(proj), REPO)
else:
    raise FileNotFoundError(f"No .zip or pyproject.toml found in {DATASET}")

# the zip may nest a top-level folder; chdir into wherever pyproject.toml is
root = next((d for d, _, fs in os.walk(REPO) if "pyproject.toml" in fs), None)
if root is None:
    raise FileNotFoundError("pyproject.toml not found - check the uploaded dataset")
os.chdir(root)
print("repo:", root, "->", sorted(os.listdir(root)))

os.environ["HF_HOME"] = "/kaggle/working/hf-cache"
os.environ["PATH"] = os.path.expanduser("~/.local/bin") + ":" + os.environ["PATH"]
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

`DATASET` is your dataset slug (`<owner>/<dataset>`). Attaching it via **Add
Input** is still preferred; the `kagglehub` branch is a fallback so the cell
works either way.

### Cell 2 - install uv and the dependencies

```python
!curl -LsSf https://astral.sh/uv/install.sh | sh
!uv sync
!uv run python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

Notes:
- `uv sync` installs its own Python 3.14 and torch from `pyproject.toml`; the
  Kaggle base image is ignored.
- If `uv sync` fails with `Could not resolve host` or download timeouts, the
  notebook has **Internet off**. Set Session options -> Internet -> On, restart
  the session, and re-run. Internet is required regardless, because training
  downloads Qwen3-0.6B and the dataset from the Hub.
- If the Python 3.14 download from GitHub stalls, pin to Kaggle's Python:
  `!uv python install 3.13` then `!uv sync --python 3.13`.
- If `uv sync` says "command not found", the PATH did not reach the shell - run
  `!~/.local/bin/uv sync` instead.
- If `cuda: False`, the image driver is too old for the torch wheel. Fix with:
  `!uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124`
  (adjust `cu124` to match `nvidia-smi`).

### Cell 3 - verify the code is current

The fixes are already in the repo. Confirm the copy you uploaded has them:

```python
!grep -n "class JEVDataset" dataset.py
!grep -n "num_workers=args.num_workers" inference.py
```

Both lines should print.

### Cell 4 - write the Kaggle training config

Batch size is lowered for the 15 GB T4; effective batch is unchanged
(8 micro-batch x 8 accumulation = 64 questions per optimizer step):

```python
%%writefile configs/kaggle_k12.yaml
seed: 0

model:
  name: Qwen/Qwen3-0.6B
  num_layers: 20

data:
  max_train_questions: null
  max_test_questions: 2000
  max_state_tokens: 512
  max_choice_tokens: 64
  batch_size: 8
  num_workers: 2

lora:
  last_k_layers: 12
  r: 8
  alpha: 16
  dropout: 0.0

head:
  new_dim: 512
  num_layers: 2

optim:
  epochs: 1
  lr_lora: 1.0e-4
  lr_head: 1.0e-3
  weight_decay: 0.01
  warmup_ratio: 0.05
  grad_clip: 1.0
  grad_accum_steps: 8

logging:
  log_every: 20
  eval_every: 1000
  save_every: 250
```

Tuned to avoid OOM on a single 15 GB T4 (`max_state_tokens: 512` is important;
the custom block attention mask makes 1024 cost ~4x the memory). If it still
runs out of memory, set `batch_size: 4` with `grad_accum_steps: 16`, or enable
gradient checkpointing on the backbone in `network.py`.

### Cell 5 - precision on T4/P100 (only if you hit a bf16 error)

T4 and P100 have no native bf16, and `inference.py` uses bf16 autocast on CUDA.
It usually runs (slowly), but if you see a bfloat16 kernel error, make it fall
back to fp32 on those cards:

```python
from pathlib import Path

p = Path("inference.py")
s = p.read_text()
old = '''    return torch.autocast(
        device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps")
    )'''
new = '''    enabled = device.type in ("cuda", "mps")
    if device.type == "cuda" and not torch.cuda.is_bf16_supported():
        enabled = False
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=enabled)'''
assert old in s, "autocast block not found; skip this optional step"
p.write_text(s.replace(old, new))
print("autocast now falls back to fp32 on non-bf16 GPUs")
```

### Cell 6 - smoke test (validates download + pipeline)

This downloads Qwen and the dataset, then overfits 10 questions. If it prints
rising accuracy, the pipeline works.

```python
!uv run python train.py configs/smoke.yaml --name smoke
```

## 6. Full training (Cell 7)

Make sure Cell 4 has been run so `configs/kaggle_k12.yaml` exists. Then run this
in the foreground and let the cell print progress:

```python
!uv run python train.py configs/kaggle_k12.yaml --name full-lora-k12
```

To monitor from another cell without blocking the run, start it in the
background instead. Kaggle's IPython rejects `&`/`nohup` in `!` cells, so use
`subprocess.Popen`:

The `-u` keeps stdout unbuffered so `train.log` updates live (without it, a
redirected stdout is block-buffered and step lines appear only in bursts):

```python
import subprocess
cmd = "uv run python -u train.py configs/kaggle_k12.yaml --name full-lora-k12"
subprocess.Popen(
    cmd, shell=True, cwd="/kaggle/working/jev-train",
    stdout=open("/kaggle/working/jev-train/train.log", "w"),
    stderr=subprocess.STDOUT, start_new_session=True,
)
```

`metrics.jsonl` in the run folder is flushed on every line, so it is also a
reliable live view even with buffered stdout.

```python
!tail -n 20 train.log
```

## 7. Checkpoints: where they are and how they are named

Every run writes to `/kaggle/working/jev-train/runs/<name>-<timestamp>/`:

```
runs/full-lora-k12-YYYYMMDD-HHMMSS/
  config.yaml
  metrics.jsonl        # one JSON record per log line
  summary.json         # written at the end
  checkpoints/
    latest/            # overwritten every save_every steps (250)
    best/              # saved whenever test accuracy improves
    final/             # saved at the end of training
```

Each checkpoint folder is self-contained: `lora/adapter_model.safetensors`
(LoRA only, not Qwen) plus `head.pt` and `meta.json`.

Because `save_every` is 250, you lose at most ~250 optimizer steps if a session
dies.

## 8. Keep the results (Kaggle sessions are ephemeral)

`/kaggle/working` survives only if you save the version. Two reliable options:

Option A - download from the notebook Output (no token needed):

```python
!cd /kaggle/working/jev-train && zip -r /kaggle/working/jev-checkpoints.zip runs
```

Then **Save Version** (top right). When it finishes, the Output tab lets you
download `jev-checkpoints.zip`. Note: Save Version re-runs the notebook - use
**Quick Save** for a version that does not re-execute.

Option B - push to the Hugging Face Hub (best for resuming, needs a **write**
token). This cell creates the repo if it is missing and otherwise just pushes
the latest files, so it is safe to re-run (for example from a watchdog):

```python
import os
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

REPO = "NeelContractor/jev-lora-k12"        # <your-username>/<repo>, exact casing
FOLDER = "/kaggle/working/jev-train/runs"   # absolute; do not use a relative path
token = os.environ.get("HF_TOKEN_WRITE") or UserSecretsClient().get_secret("HF_TOKEN_WRITE")

api = HfApi()
if not api.repo_exists(repo_id=REPO, repo_type="model", token=token):
    api.create_repo(repo_id=REPO, repo_type="model", private=True, token=token)
    print(f"created {REPO}")
else:
    print(f"updating {REPO}")

api.upload_folder(folder_path=FOLDER, repo_id=REPO, repo_type="model", token=token)
print("pushed")
```

Set `private=False` when creating if you want a public repo. `upload_folder`
only commits changed/new files, so re-running is cheap.

## 9. Resume after a session ends

Yes, you can continue in a new session, but understand what `--resume` does:

- It restores the saved **LoRA + head weights only** and starts a **fresh
  optimizer, LR schedule, and step counter**, then runs the config's full
  `epochs` again. It continues the weights, not the exact schedule, so treat it
  as a fine-tune continuation.
- If the session was interrupted, the checkpoint to use is
  **`checkpoints/latest`** (or `best`). `checkpoints/final` only exists if
  training finished, so do not resume from it after a crash.

### 9a. Persist a checkpoint before the session dies

`/kaggle/working` is not guaranteed to survive a stopped session, so get `latest`
off the instance (section 8, Option A or B). A watchdog cell that pushes to the
Hub every 15 minutes is the safest:

```python
import os, time
from huggingface_hub import HfApi
from kaggle_secrets import UserSecretsClient

REPO = "NeelContractor/jev-lora-k12"
FOLDER = "/kaggle/working/jev-train/runs"
token = os.environ.get("HF_TOKEN_WRITE") or UserSecretsClient().get_secret("HF_TOKEN_WRITE")
api = HfApi()

if not api.repo_exists(repo_id=REPO, repo_type="model", token=token):
    api.create_repo(repo_id=REPO, repo_type="model", private=True, token=token)

while True:
    if os.path.exists(FOLDER):
        api.upload_folder(folder_path=FOLDER, repo_id=REPO,
                          repo_type="model", token=token)
        print("pushed", time.strftime("%H:%M:%S"))
    time.sleep(900)
```

Run it in its own cell and stop it when training ends. It creates the repo if
needed, then pushes every 15 minutes; each push only commits changed files.

### 9b. Resume in a new session

1. Get the saved `runs` onto the new notebook - either **Add Input** the Kaggle
   Dataset you uploaded, or download it from the Hub:

   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="<your-username>/jev-lora-k12",
                     local_dir="/kaggle/working/resumed")
   ```

2. Start from `latest` (from the Kaggle Dataset):

   ```python
   !uv run python train.py configs/kaggle_k12.yaml --name continued \
     --resume /kaggle/input/jev-ckpt/runs/<run-id>/checkpoints/latest
   ```

   or from the Hub copy:

   ```python
   !uv run python train.py configs/kaggle_k12.yaml --name continued \
     --resume /kaggle/working/resumed/runs/<run-id>/checkpoints/latest
   ```

The config's backbone, head, and LoRA settings must match the checkpoint, or
`load_weights` raises a mismatch error.

To resume the optimizer/LR state exactly (no schedule restart), `train.py` would
need to also save and load the optimizer and scheduler state - it does not today.

## 10. Limits and gotchas

- **30 GPU-hr/week**, resets weekly (Mon 00:00 UTC). One full run is ~3-6 h.
- **12 h max per session**; interactive sessions also stop after ~60 min idle.
- **Internet off by default on GPU notebooks** - turn it on, or the HF downloads
  fail. If you cannot enable it, pre-download the model and dataset and upload
  them as Kaggle Datasets, then set `HF_HOME` to the mounted path and
  `HF_HUB_OFFLINE=1`.
- **`/kaggle/working` is ~20 GB.** The HF cache plus checkpoints are small
  (Qwen3-0.6B ~1.5 GB, checkpoints a few hundred MB), so this is fine.
- **T4/P100 are slow** for bf16. Expect the run to take several hours.
- `runs/` uses a timestamped folder per run, so re-running the same `--name`
  creates a new folder instead of overwriting.
