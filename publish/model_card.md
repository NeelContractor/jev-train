---
license: cc-by-nc-4.0
base_model: Qwen/Qwen3-0.6B
library_name: bev-decider
base_model_relation: finetune
datasets:
  - avbiswas/bev-decision-150K
pipeline_tag: text-classification
tags:
  - system-one
  - decision-model
  - jev-compatible
language:
  - en
---

# jev-decider-0.4B

A **0.4B-parameter System One decision model**. It reads a state (text or JSON) and typed questions about it and returns probabilities in a single forward pass, with no text generation:

- **`choice`**: pick one of several named options.
- **`noul`**: yes/no, returned as P(yes).
- **`score`**: an ordinal level, returned as the expected value over the levels.

## What this repo is

- **This repo** is the *model*: the weights, configs and tokenizer in the layout the `bev-decider` package expects.
- [`bev-decider`](https://github.com/avbiswas/bev-decider) is the *inference library* (`pip install bev-decider`). It holds the network definition, the input encoder, `load()`, `decide()`, a CLI and an HTTP server.
- [`bev-train`](https://github.com/avbiswas/bev-train) is the *training code* this model was trained with.

The model and the library are separate: the library defines the architecture, and these weights fill it.

## Architecture

This is a single self-contained model. `model.safetensors` holds the whole network in one file.

| | |
|---|---|
| Backbone | The first **20 of 28 layers** of [`Qwen/Qwen3-0.6B`](https://huggingface.co/Qwen/Qwen3-0.6B), with the fine-tuned LoRA (r=8, alpha=16 on `q/k/v/o_proj` of layers 8-19) **merged** into the weights |
| Decision head | 2-layer attention head, 512-dim, with a task-type embedding |
| Parameters | About 0.4B (0.31B in the transformer layers, 0.16B token embeddings, 7.4M head) |
| Weights | `model.safetensors` (bf16 backbone + fp32 head) |
| Context | Trained with states up to **512** tokens; options are limited to 64 tokens each |

## Files

| File | Contents |
|---|---|
| `model.safetensors` | LoRA-merged 20-layer Qwen3 backbone (`backbone.*`) plus the trained head (`head.*`) |
| `config.json` | bev-decider config: head shape, task types, token limits, source run |
| `backbone_config.json` | Qwen3 config truncated to 20 layers |
| `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` | Qwen tokenizer (needed by the slow `Qwen2Tokenizer` the loader uses) |
| `LICENSE-Qwen` | Apache-2.0 license for the Qwen3-0.6B weights |

## Usage

`bev-decider` is the library; `jev-decider-0.4B` is this model. Install the library, then pass this repo's id to `load`.

~~~bash
pip install bev-decider
~~~

~~~python
from bev_decider import load

decider = load("NeelContractor/jev-decider-0.4B")

state = {
    "message": (
        "URGENT: you charged my card twice this month. "
        "Refund the duplicate within 24 hours or I'm disputing it with my bank."
    )
}

questions = {
    "intent": {
        "type": "choice",
        "instructions": "What does the customer want?",
        "criteria": {
            "refund": "money returned or a duplicate charge reversed",
            "technical_help": "a bug, outage or integration problem",
            "cancellation": "wants to cancel or downgrade",
        },
    },
    "urgent": {
        "type": "noul",
        "instructions": "Does the message communicate time pressure or a deadline?",
    },
    "anger": {
        "type": "score",
        "instructions": "How angry is the customer?",
        "criteria": ["calm", "mildly annoyed", "frustrated", "furious"],
    },
}

answers = decider.decide(state, questions)
~~~

Output:

~~~json
__OUTPUT__
~~~

A local export folder works the same way: `load("/path/to/folder")`.

## Evaluation

Held-out test split of [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K), 2,000 questions, single forward pass on CPU in fp32:

| Metric | Value |
|---|---:|
| Accuracy (all) | **0.7300** |
| Yes/no (`noul`) | 0.8178 |
| Choice | 0.7192 |
| Score | 0.5964 |
| AUC (`noul`) | 0.8859 |

These numbers come from the best checkpoint (step 1000 of the final run), selected by test accuracy. They are not directly comparable to the reference `bev-decider-0.4B` numbers, which are measured on a different split.

## Training

- Base: first 20 of 28 layers of `Qwen/Qwen3-0.6B`; base weights frozen, LoRA trained.
- LoRA: r=8, alpha=16, dropout 0, on `q/k/v/o_proj` of layers 8-19 (12 layers), merged at export.
- Head: 2 attention blocks, 512-dim, `num_task_types=3` (`choice`, `noul`, `score`).
- Data: `avbiswas/bev-decision-150K`, 125,614 rows expanded to **250,262** typed questions.
- Schedule: 1 epoch, effective batch 64 (16 x 4 gradient accumulation), **3,910** optimizer steps.
- Precision: bf16 autocast, fp32 head. Optimizer: AdamW (head lr 1e-3, LoRA lr 1e-4, 5% warmup).
- Hardware: 2 x Tesla T4 on Kaggle with `DataParallel`.
- States capped at 512 tokens (1,024 OOMs on 15 GB T4); options at 64 tokens.

## Limitations

- Trained on 250K questions, whereas the reference model used about 940K; expect lower accuracy than `avbiswas/bev-decider-0.4B`.
- English only, and states are truncated at 512 tokens at inference unless you pass `load(max_state_tokens=...)`.
- `score` questions are the weakest task type.
- Date and arithmetic comparisons are the least reliable question content.

## License

The model is **CC-BY-NC-4.0** (non-commercial), because the training data includes sources with non-commercial terms. It contains weights derived from [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B), which is Apache-2.0; Qwen's license is included as `LICENSE-Qwen`.
