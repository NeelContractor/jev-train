# jev-train

Train a **bev-decider-compatible System One model**: the first 20 of Qwen3-0.6B's 28 decoder layers, optional LoRA, and a small attention head that answers `choice`, `noul` (yes/no), and `score` questions in one forward pass, with no text generation.

Choices share the same starting position ids and are read in parallel; the attention mask stops each choice from seeing the others, so the model is order-invariant.

This is a variant of [`avbiswas/bev-train`](https://github.com/avbiswas/bev-train) with bug fixes, Kaggle T4 support (DataParallel and OOM-safe defaults), and an exporter that produces a merged model for [`bev-decider`](https://github.com/avbiswas/bev-decider).

Dataset: [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K). Training downloads it automatically.

## Layout

```
tokenization.py      prompt formatting, choice positions, attention masks
dataset.py           expand dataset rows into typed questions, batching
network.py           Qwen backbone, optional LoRA, choice-scoring head
train.py             training loop, gradient accumulation, validation
logger.py            run metrics and LoRA/head checkpoints
inference.py         evaluation and typed answers
export.py            merge LoRA and write a bev-decider-format model folder
configs/             smoke, head-only, LoRA, and full 20-layer recipes
arch-notebook.ipynb  architecture walkthrough
publish/             model card and Qwen license for the exported model
KAGGLE_TRAINING.md   end-to-end Kaggle T4 notebook guide
```

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
```

The first run downloads Qwen3-0.6B, the tokenizer, and the dataset. Device selection is CUDA, then Apple MPS, then CPU.

## Train

Sanity check that overfits 10 questions:

```bash
uv run python train.py configs/smoke.yaml --name smoke
```

Full run: first 20 layers, LoRA on layers 8-19, trainable head:

```bash
uv run python train.py configs/full_lora_k12.yaml --name full-lora-k12
```

Other configs: `full_head_only.yaml` (frozen backbone), `full_lora.yaml` (LoRA on the last 4 layers), `lora_20k.yaml`, `head_only_20l.yaml`.

`batch_size` is per micro-batch; `optim.grad_accum_steps` sets how many micro-batches make one optimizer step. Logging, validation, and saving intervals all count optimizer steps, and an incomplete final accumulation group is dropped.

On a small GPU, lower `data.max_state_tokens` and `batch_size`; 1,024 tokens does not fit on a 15 GB T4. On Kaggle, follow [`KAGGLE_TRAINING.md`](KAGGLE_TRAINING.md).

## Runs and checkpoints

Each run writes `runs/<name>-<timestamp>/` with `config.yaml`, `run.json`, `metrics.jsonl`, `summary.json`, and `checkpoints/`:

- `latest` — overwritten every `save_every` steps
- `best` — best test accuracy
- `final` — end of training

A checkpoint contains the head (`head.pt`) and, when LoRA is enabled, the adapter (`lora/`) plus `config.yaml` and `meta.json`; the Qwen base weights are loaded separately.

`--resume` loads the LoRA + head weights into a new run. The optimizer, LR schedule, step counter, and data position start fresh, so a resumed run repeats the configured epochs.

## Evaluate

```bash
uv run python inference.py runs/<run-id>/checkpoints/best \
  --max_questions 2000 --batch_size 16 --num_workers 0
```

Reports loss, overall accuracy, accuracy per task type, and yes/no AUC. It uses the checkpoint's state and choice token limits.

## Export and publish

`export.py` merges LoRA into the backbone and writes the exact layout `bev-decider` loads: `model.safetensors` (`backbone.*` in bf16, `head.*` in fp32), `config.json`, `backbone_config.json`, and the Qwen tokenizer files.

```bash
uv run python export.py runs/<run-id>/checkpoints/best export/jev-decider
```

Then upload `export/jev-decider` to a Hugging Face model repo together with `publish/model_card.md` (as `README.md`) and `publish/LICENSE-Qwen`. The result can be run with:

```python
from bev_decider import load
decider = load("<user>/jev-decider-0.4B")
```

## Credits

Architecture and original training code: [`avbiswas/bev-train`](https://github.com/avbiswas/bev-train). Inference library: [`avbiswas/bev-decider`](https://github.com/avbiswas/bev-decider).
