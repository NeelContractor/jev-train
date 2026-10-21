"""Export a training checkpoint as a single self-contained bev-decider-style model folder.

Output layout (same as avbiswas/bev-decider-0.4B):

    out/
      model.safetensors      # LoRA merged into the backbone + head, keys prefixed backbone./head.
      backbone_config.json   # Qwen3 config truncated to the trained layers
      config.json            # bev-decider model config
      tokenizer.json, tokenizer_config.json, vocab.json, merges.txt

Usage:
    uv run python export.py runs/<exp_id>/checkpoints/best export/jev-decider
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from safetensors.torch import save_file

from logger import load_checkpoint
from tokenization import tokenizer


def export(checkpoint_dir, out_dir, max_state_tokens=None):
    checkpoint_dir = Path(checkpoint_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    network, meta = load_checkpoint(checkpoint_dir, torch.device("cpu"))
    backbone = network.backbone
    if isinstance(backbone, PeftModel):
        backbone = backbone.merge_and_unload()  # fold LoRA into the base weights
    backbone = backbone.float()

    state = {
        f"backbone.{k}": v.to(torch.bfloat16).contiguous()
        for k, v in backbone.state_dict().items()
    }
    state.update(
        {
            f"head.{k}": v.to(torch.float32).contiguous()
            for k, v in network.head.state_dict().items()
        }
    )
    save_file(state, out_dir / "model.safetensors")

    backbone_config = backbone.config
    if getattr(backbone_config, "layer_types", None):
        backbone_config.layer_types = backbone_config.layer_types[
            : backbone_config.num_hidden_layers
        ]
    backbone_config.to_json_file(out_dir / "backbone_config.json")

    train_config = yaml.safe_load((checkpoint_dir / "config.yaml").read_text())
    lora = train_config["lora"]
    layers = list(
        range(meta["num_layers"] - lora["last_k_layers"], meta["num_layers"])
    )
    head = meta["head_config"]
    config = {
        "model_type": "bev-decider",
        "version": "0.4.0",
        "format": "merged",
        "backbone_config": "backbone_config.json",
        "backbone_dtype": "bf16",
        "num_layers": meta["num_layers"],
        "head": {
            "new_dim": head["new_dim"],
            "num_layers": head["num_layers"],
            "num_task_types": head["num_task_types"],
        },
        "task_types": {"choice": 0, "noul": 1, "score": 2},
        "max_state_tokens_train": meta["max_state_tokens"],
        "max_state_tokens": max_state_tokens or meta["max_state_tokens"],
        "max_choice_tokens": meta["max_choice_tokens"],
        "source_checkpoint": checkpoint_dir.name,
        "derived_from": {
            "base_model": meta["model_name"],
            "lora": {"r": lora["r"], "alpha": lora["alpha"], "layers": layers},
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    tokenizer.save_pretrained(out_dir)

    size = (out_dir / "model.safetensors").stat().st_size / 1e9
    print(f"exported {out_dir} ({size:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="runs/<exp_id>/checkpoints/<name>")
    parser.add_argument("out", help="output folder, e.g. export/jev-decider")
    parser.add_argument("--max_state_tokens", type=int, default=None)
    args = parser.parse_args()
    export(args.checkpoint, args.out, args.max_state_tokens)


if __name__ == "__main__":
    main()
