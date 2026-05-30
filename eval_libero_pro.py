"""
eval_libero_pro.py — Full LIBERO-PRO evaluation for PI0 + MixedLayerAttention.

Evaluates BOTH the MLA checkpoint AND vanilla pi0 baseline side by side,
across all 4 perturbation types, 3 episodes per task.
Saves progress after every task — fully resumable after preemption.

Expected runtime on spot A100: ~27 hours. Expected cost: ~$32.

Usage:
    # Sanity check first (always do this before the full run)
    python eval_libero_pro.py \
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt \
        --dry_run

    # Full run — both models, all 4 perturbations, 3 episodes
    python eval_libero_pro.py \
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt

    # Resume after preemption
    python eval_libero_pro.py \
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt \
        --resume

    # Specific perturbations only
    python eval_libero_pro.py \
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt \
        --perturbations position object
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi0 import PI0Config
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID          = "lerobot/pi0_libero_base"
NUM_EPISODES      = 3
ALL_PERTURBATIONS = ["position", "object", "language", "task"]
MAX_EPISODE_STEPS = 600   # hard cap per episode to avoid infinite loops

# ── Model ────────────────────────────────────────────────────────────────────

def build_model(device: str):
    print(f"\n[build] Loading config from {MODEL_ID} ...")
    config_path = hf_hub_download(MODEL_ID, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    config_dict.pop("type", None)
    config_dict["device"] = "cpu"
    config_dict["dtype"]  = "bfloat16"

    for key, val in config_dict.get("input_features", {}).items():
        config_dict["input_features"][key] = PolicyFeature(
            type=FeatureType[val["type"]], shape=tuple(val["shape"])
        )
    for key, val in config_dict.get("output_features", {}).items():
        config_dict["output_features"][key] = PolicyFeature(
            type=FeatureType[val["type"]], shape=tuple(val["shape"])
        )

    config = PI0Config(**config_dict)
    print("[build] Constructing model on CPU ...")
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

    config.device = device
    policy = policy.to(device)
    print(f"[build] Model on {device} | "
          f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return policy, config


def load_mla_checkpoint(policy, ckpt_path: str, device: str):
    print(f"[ckpt] Loading MLA checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    missing, unexpected = policy.model.load_state_dict(ckpt, strict=False)
    print(f"[ckpt] Overlaid {len(ckpt)} tensors | "
          f"missing: {len(missing)} | unexpected: {len(unexpected)}")
    return policy


# ── Tokenizer ────────────────────────────────────────────────────────────────

def make_tokenizer():
    return AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")


def tokenize_task(task_str, tokenizer, config, device):
    tokens = tokenizer(
        [task_str + "\n"],
        return_tensors="pt",
        padding="max_length",
        max_length=config.tokenizer_max_length,
        truncation=True,
    ).to(device)
    return {
        "observation.language.tokens":         tokens["input_ids"],
        "observation.language.attention_mask": tokens["attention_mask"].bool(),
    }


# ── Progress (resume support) ─────────────────────────────────────────────────

def load_progress(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        completed = sum(1 for k in data if not k.startswith("__"))
        print(f"[resume] Found {completed} completed tasks in {path}")
        return data
    return {}


def save_progress(path: str, results: dict):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


# ── Episode rollout ───────────────────────────────────────────────────────────

def run_episode(policy, env, lang_tokens, device, config):
    """Run one episode. Returns True if successful."""
    obs        = env.reset()
    ep_success = False

    for _ in range(MAX_EPISODE_STEPS // config.chunk_size + 1):
        obs_tensor = {
            "observation.image": torch.from_numpy(
                obs["agentview_image"].transpose(2, 0, 1)[None]
            ).float().to(device) / 255.0,

            "observation.image_mask": torch.ones(
                1, 1, dtype=torch.bool, device=device
            ),

            "observation.state": torch.from_numpy(
                obs["robot0_joint_pos_cos"].astype(np.float32)
            ).unsqueeze(0).to(device),

            **lang_tokens,
        }

        with torch.no_grad():
            action = policy.select_action(obs_tensor)

        # Step through the chunk
        action_np = action.cpu().numpy().squeeze(0)
        for step_action in action_np:
            obs, _reward, done, info = env.step(step_action)
            ep_success = ep_success or bool(info.get("success", False))
            if done or ep_success:
                return ep_success

    return ep_success


# ── Perturbation eval ─────────────────────────────────────────────────────────

def run_perturbation(
    policy,
    config,
    tokenizer,
    perturbation: str,
    num_episodes: int,
    device: str,
    seed: int,
    progress_path: str,
    resume: bool,
    label: str,
) -> dict:
    try:
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError(
            "LIBERO-PRO not installed.\n"
            "  git clone https://github.com/Zxy-MLlab/LIBERO-PRO\n"
            "  cd LIBERO-PRO && pip install -e ."
        )

    # Map perturbation name to LIBERO-PRO env flags
    perturbation_flags = {
        "use_swap":     perturbation == "object",
        "use_object":   perturbation == "position",
        "use_language": perturbation == "language",
        "use_task":     perturbation == "task",
    }

    benchmark  = libero_benchmark.get_benchmark_dict()["libero_90"]()
    task_names = benchmark.get_task_names()

    print(f"\n[eval] {label} | perturbation={perturbation} | "
          f"{len(task_names)} tasks × {num_episodes} ep")

    results = load_progress(progress_path) if resume else {}
    policy.eval()

    for task_idx, task_name in enumerate(task_names):
        if task_name in results:
            sr = results[task_name]["success_rate"]
            print(f"  [{task_idx+1:3d}/{len(task_names)}] SKIP {task_name[:50]} "
                  f"(SR={sr*100:.0f}%)")
            continue

        task             = benchmark.get_task(task_idx)
        task_description = task.language
        lang_tokens      = tokenize_task(task_description, tokenizer, config, device)

        env_args = {
            "bddl_file_name": task.bddl_file,
            "camera_heights": 128,
            "camera_widths":  128,
            **perturbation_flags,
        }

        try:
            env = OffScreenRenderEnv(**env_args)
            env.seed(seed)
        except Exception as e:
            print(f"  [{task_idx+1:3d}] ENV ERROR {task_name[:50]}: {e}")
            results[task_name] = {
                "success_rate": 0.0, "successes": 0,
                "episodes": 0, "error": str(e)
            }
            save_progress(progress_path, results)
            continue

        successes = []
        t_task    = time.time()

        for ep in range(num_episodes):
            try:
                success = run_episode(policy, env, lang_tokens, device, config)
                successes.append(float(success))
            except Exception as e:
                print(f"    ep {ep} ERROR: {e}")
                successes.append(0.0)

        env.close()

        sr        = float(np.mean(successes))
        elapsed_s = time.time() - t_task

        results[task_name] = {
            "success_rate": sr,
            "successes":    int(np.sum(successes)),
            "episodes":     num_episodes,
            "elapsed_s":    round(elapsed_s, 1),
        }

        # Save after every task
        save_progress(progress_path, results)

        print(
            f"  [{task_idx+1:3d}/{len(task_names)}] {task_name[:50]:<50} "
            f"SR: {sr*100:5.1f}%  ({int(np.sum(successes))}/{num_episodes})  "
            f"{elapsed_s:.0f}s"
        )

    # Overall
    task_results = {k: v for k, v in results.items() if not k.startswith("__")}
    overall_sr   = float(np.mean([v["success_rate"] for v in task_results.values()]))
    results["__overall__"] = {
        "label":             label,
        "perturbation":      perturbation,
        "success_rate":      overall_sr,
        "num_tasks":         len(task_results),
        "episodes_per_task": num_episodes,
    }
    save_progress(progress_path, results)
    print(f"\n  >> {label} | {perturbation} overall SR: {overall_sr*100:.2f}%\n")
    return results


# ── Dry run ───────────────────────────────────────────────────────────────────

def dry_run(policy, config, device):
    print("\n[dry-run] Sanity forward pass ...")
    B         = 1
    T_img     = 1
    T_lang    = config.tokenizer_max_length
    chunk     = config.chunk_size
    state_dim = list(config.input_features.values())[0].shape[0]

    with torch.no_grad():
        loss = policy.model(
            images      = torch.randn(B, T_img, 3, 224, 224,
                                      device=device, dtype=torch.bfloat16),
            img_masks   = torch.ones(B, T_img, dtype=torch.bool, device=device),
            lang_tokens = torch.randint(0, 256_000, (B, T_lang), device=device),
            lang_masks  = torch.ones(B, T_lang, dtype=torch.bool, device=device),
            state       = torch.randn(B, state_dim,
                                      device=device, dtype=torch.bfloat16),
            actions     = torch.randn(B, chunk, state_dim,
                                      device=device, dtype=torch.bfloat16),
            noise       = torch.randn(B, chunk, state_dim,
                                      device=device, dtype=torch.bfloat16),
            time        = torch.rand(B, device=device, dtype=torch.bfloat16),
        )
    print(f"[dry-run] Forward OK — loss mean: {loss.mean().item():.4f}")
    weights = policy.model.mla.get_layer_weights()
    print(f"[dry-run] MLA weights layer 17: "
          f"{[round(x, 3) for x in weights[-1].tolist()]}")
    print("[dry-run] All good.\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Full LIBERO-PRO eval — MLA checkpoint vs vanilla pi0 baseline"
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to MLA checkpoint model.pt (55 tensors)")
    p.add_argument("--perturbations", nargs="+", default=ALL_PERTURBATIONS,
                   choices=ALL_PERTURBATIONS,
                   help="Perturbation types to eval (default: all 4)")
    p.add_argument("--num_episodes", type=int, default=NUM_EPISODES,
                   help=f"Episodes per task (default: {NUM_EPISODES})")
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save results (default: next to checkpoint)")
    p.add_argument("--resume", action="store_true",
                   help="Resume — skip already-completed tasks")
    p.add_argument("--dry_run", action="store_true",
                   help="One forward pass to verify everything works, then exit")
    p.add_argument("--skip_baseline", action="store_true",
                   help="Skip baseline eval (use if baseline already done)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or str(Path(args.checkpoint).parent / "libero_pro_eval")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[setup] Output dir: {output_dir}")

    tokenizer = make_tokenizer()
    all_results = {}
    t_total = time.time()

    # ── Dry run ──────────────────────────────────────────────────────────────
    if args.dry_run:
        print("\n=== DRY RUN: MLA checkpoint ===")
        policy, config = build_model(args.device)
        load_mla_checkpoint(policy, args.checkpoint, args.device)
        dry_run(policy, config, args.device)
        del policy
        torch.cuda.empty_cache()

        print("=== DRY RUN: baseline ===")
        policy, config = build_model(args.device)
        dry_run(policy, config, args.device)
        del policy
        torch.cuda.empty_cache()
        print("[dry-run] Both models OK. Ready for full eval.")
        return

    # ── Baseline eval ─────────────────────────────────────────────────────────
    if not args.skip_baseline:
        print("\n" + "="*60)
        print("PHASE 1: BASELINE (vanilla pi0, no finetuning)")
        print("="*60)
        policy, config = build_model(args.device)
        # No checkpoint loaded — pure pretrained pi0

        baseline_results = {}
        for perturbation in args.perturbations:
            progress_path = os.path.join(
                output_dir, f"baseline_{perturbation}.json"
            )
            result = run_perturbation(
                policy        = policy,
                config        = config,
                tokenizer     = tokenizer,
                perturbation  = perturbation,
                num_episodes  = args.num_episodes,
                device        = args.device,
                seed          = args.seed,
                progress_path = progress_path,
                resume        = args.resume,
                label         = "baseline",
            )
            baseline_results[perturbation] = result

        all_results["baseline"] = baseline_results
        del policy
        torch.cuda.empty_cache()
        gc.collect()

    # ── MLA checkpoint eval ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2: MLA CHECKPOINT")
    print("="*60)
    policy, config = build_model(args.device)
    load_mla_checkpoint(policy, args.checkpoint, args.device)

    mla_results = {}
    for perturbation in args.perturbations:
        progress_path = os.path.join(
            output_dir, f"mla_{perturbation}.json"
        )
        result = run_perturbation(
            policy        = policy,
            config        = config,
            tokenizer     = tokenizer,
            perturbation  = perturbation,
            num_episodes  = args.num_episodes,
            device        = args.device,
            seed          = args.seed,
            progress_path = progress_path,
            resume        = args.resume,
            label         = "mla",
        )
        mla_results[perturbation] = result

    all_results["mla"] = mla_results
    del policy
    torch.cuda.empty_cache()

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print("\n" + "="*60)
    print(f"LIBERO-PRO RESULTS  ({args.num_episodes} episodes/task)")
    print("="*60)
    print(f"  {'Perturbation':<14} {'Baseline SR':>12} {'MLA SR':>10} {'Delta':>8}")
    print(f"  {'-'*14} {'-'*12} {'-'*10} {'-'*8}")

    for pert in args.perturbations:
        mla_sr  = mla_results[pert]["__overall__"]["success_rate"] * 100
        base_sr = (
            all_results.get("baseline", {})
            .get(pert, {})
            .get("__overall__", {})
            .get("success_rate", float("nan"))
        ) * 100
        delta = mla_sr - base_sr
        sign  = "+" if delta >= 0 else ""
        print(f"  {pert:<14} {base_sr:>11.1f}%  {mla_sr:>9.1f}%  {sign}{delta:>6.1f}%")

    print("="*60)
    print(f"Total time: {elapsed/3600:.1f} hours")

    # Save full results
    final_json = os.path.join(output_dir, "eval_results_final.json")
    with open(final_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] Full results → {final_json}")


if __name__ == "__main__":
    main()
