"""
eval_val_loss.py — Fast validation loss eval on T4, no LIBERO needed.

Compares your MLA checkpoints against vanilla pi0 baseline.
Runs in ~10 minutes on a T4.

Usage:
    python eval_val_loss.py
    python eval_val_loss.py --checkpoints \
        ./outputs/mixed_layer_attention/checkpoint_006000/model.pt \
        ./outputs/mixed_layer_attention/checkpoint_008000/model.pt \
        ./outputs/mixed_layer_attention/checkpoint_010000/model.pt
"""

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pi0 import PI0Config
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

# ── Config ───────────────────────────────────────────────────────────────────

MODEL_ID     = "lerobot/pi0_libero_finetuned_v044"
DEVICE       = "cuda"
VAL_BATCHES  = 100   # ~10 min on T4, increase for more reliable estimate
BATCH_SIZE   = 2
VAL_FRACTION = 0.05  # 5% of dataset as val set
SEED         = 42

# ── Helpers ──────────────────────────────────────────────────────────────────

def build_base_model():
    print(f"[build] Loading config from {MODEL_ID} ...")
    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict.pop("type", None)
    config_dict["device"] = "cpu"
    config_dict["dtype"] = "bfloat16"

    for key, val in config_dict.get("input_features", {}).items():
        config_dict["input_features"][key] = PolicyFeature(
            type=FeatureType[val["type"]], shape=tuple(val["shape"])
        )
    for key, val in config_dict.get("output_features", {}).items():
        config_dict["output_features"][key] = PolicyFeature(
            type=FeatureType[val["type"]], shape=tuple(val["shape"])
        )

    config = PI0Config(**config_dict)
    policy = PI0PolicyMixedLayerAttention(config)

    print("[build] Loading pretrained weights ...")
    weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
    state_dict   = load_file(weights_path, device="cpu")
    remapped     = {
        (k if k.startswith("model.") else f"model.{k}"): v
        for k, v in state_dict.items()
    }
    missing, unexpected = policy.load_state_dict(remapped, strict=False)
    print(f"[build] Pretrained — missing: {len(missing)}, unexpected: {len(unexpected)}")
    del state_dict, remapped
    gc.collect()

    config.device = DEVICE
    policy = policy.to(DEVICE)
    return policy, config


def load_checkpoint(policy, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    policy.model.load_state_dict(ckpt, strict=False)
    print(f"[ckpt] Loaded {len(ckpt)} tensors from {ckpt_path}")
    return policy


def make_val_loader(config):
    print("[data] Loading dataset ...")
    dataset = LeRobotDataset(
        "lerobot/libero",
        delta_timestamps={"action": [i / 10 for i in range(50)]},
    )
    total    = len(dataset)
    val_size = int(total * VAL_FRACTION)
    train_size = total - val_size
    _, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    print(f"[data] Val set: {len(val_set)} samples ({VAL_FRACTION*100:.0f}% of {total})")
    return DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)


def tokenize_batch(batch, tokenizer, config):
    batch = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
             for k, v in batch.items()}
    tasks  = [t + "\n" for t in batch["task"]]
    tokens = tokenizer(
        tasks,
        return_tensors="pt",
        padding="max_length",
        max_length=config.tokenizer_max_length,
        truncation=True,
    ).to(DEVICE)
    batch["observation.language.tokens"]         = tokens["input_ids"]
    batch["observation.language.attention_mask"] = tokens["attention_mask"].bool()
    return batch


def eval_loss(policy, val_loader, tokenizer, config, label, n_batches=VAL_BATCHES):
    policy.eval()
    losses = []
    t0     = time.time()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            batch = tokenize_batch(batch, tokenizer, config)
            loss, _ = policy.forward(batch)
            losses.append(loss.item())

            if (i + 1) % 10 == 0:
                print(f"  {label}: batch {i+1}/{n_batches} | "
                      f"loss so far: {sum(losses)/len(losses):.4f}")

    mean_loss = sum(losses) / len(losses)
    elapsed   = time.time() - t0
    print(f"  {label}: FINAL loss = {mean_loss:.4f}  ({elapsed:.0f}s)\n")
    return mean_loss


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Fast val loss eval on T4")
    p.add_argument(
        "--checkpoints", nargs="+",
        default=[
            "./outputs/mixed_layer_attention/checkpoint_006000/model.pt",
            "./outputs/mixed_layer_attention/checkpoint_008000/model.pt",
            "./outputs/mixed_layer_attention/checkpoint_010000/model.pt",
        ],
        help="Checkpoint paths to evaluate",
    )
    p.add_argument("--batches", type=int, default=VAL_BATCHES,
                   help=f"Val batches per run (default: {VAL_BATCHES})")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
    val_loader = None  # built once, reused

    results = {}

    # ── Baseline: vanilla pi0, no checkpoint ────────────────────────────────
    print("\n" + "="*50)
    print("BASELINE: vanilla pi0 (no finetuning)")
    print("="*50)
    policy, config = build_base_model()
    if val_loader is None:
        val_loader = make_val_loader(config)
    results["baseline"] = eval_loss(
        policy, val_loader, tokenizer, config,
        label="baseline", n_batches=args.batches
    )
    del policy
    torch.cuda.empty_cache()
    gc.collect()

    # ── Each checkpoint ──────────────────────────────────────────────────────
    for ckpt_path in args.checkpoints:
        if not Path(ckpt_path).exists():
            print(f"[skip] {ckpt_path} not found")
            continue

        label = Path(ckpt_path).parent.name  # e.g. checkpoint_006000
        print("\n" + "="*50)
        print(f"CHECKPOINT: {label}")
        print("="*50)

        policy, config = build_base_model()
        load_checkpoint(policy, ckpt_path)
        results[label] = eval_loss(
            policy, val_loader, tokenizer, config,
            label=label, n_batches=args.batches
        )
        del policy
        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    baseline = results.get("baseline", float("nan"))
    print(f"  {'baseline':<30} {baseline:.4f}")
    for label, loss in results.items():
        if label == "baseline":
            continue
        delta = loss - baseline
        sign  = "+" if delta > 0 else ""
        print(f"  {label:<30} {loss:.4f}  ({sign}{delta:.4f} vs baseline)")
    print("="*50)

    # Save
    out_path = "./outputs/val_loss_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[save] Results → {out_path}")


if __name__ == "__main__":
    main()
