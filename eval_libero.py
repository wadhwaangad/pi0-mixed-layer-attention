"""
eval_libero.py — Full LIBERO standard benchmark evaluation for PI0 + MixedLayerAttention.

Evaluates on the 4 standard LIBERO suites (libero_goal, libero_spatial, libero_10,
libero_object) without any perturbations. Mirrors eval_libero_pro.py in structure,
output format, and CLI interface.
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
from lerobot.policies.pi0 import PI0Config, PI0Policy
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

MODEL_ID = "lerobot/pi0_libero_base"
NUM_EPISODES = 1

ALL_SUITES = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]

SUITE_MAX_STEPS = {
    "libero_goal":    300,
    "libero_spatial": 220,
    "libero_10":      520,
    "libero_object":  280,
}

# Official published success rates on standard (unperturbed) LIBERO
OFFICIAL_BASELINES = {
    "pi0 (baseline)": {
        "libero_goal":    0.92,
        "libero_spatial": 0.90,
        "libero_10":      0.82,
        "libero_object":  0.98,
    },
    "pi0.5 (baseline)": {
        "libero_goal":    float("nan"),
        "libero_spatial": float("nan"),
        "libero_10":      float("nan"),
        "libero_object":  float("nan"),
    },
    "OpenVLA (baseline)": {
        "libero_goal":    float("nan"),
        "libero_spatial": float("nan"),
        "libero_10":      float("nan"),
        "libero_object":  float("nan"),
    },
}


# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def build_model(device: str, use_mla: bool = True):
    policy_cls = PI0PolicyMixedLayerAttention if use_mla else PI0Policy
    label      = "PI0PolicyMixedLayerAttention" if use_mla else "PI0Policy (vanilla base)"
    print(f"\n[build] Loading config from {MODEL_ID} ...")
    print(f"[build] Policy class: {label}")
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
    print(f"[build] Constructing {label} on CPU ...")
    policy = policy_cls(config)

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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = policy.model.load_state_dict(ckpt, strict=False)
    print(f"[ckpt] Overlaid {len(ckpt)} tensors | "
          f"missing: {len(missing)} | unexpected: {len(unexpected)}")
    return policy


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


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Proprioceptive state
# ---------------------------------------------------------------------------

STATE_DIM: int = 32

_PROPRIO_KEYS = [
    "robot0_joint_pos_cos",
    "robot0_joint_pos_sin",
    "robot0_joint_vel",
    "robot0_gripper_qpos",
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qvel",
    "robot0_eef_vel",
]


def _build_state(obs: dict, state_dim: int, device) -> torch.Tensor:
    parts = []
    for key in _PROPRIO_KEYS:
        if key in obs:
            parts.append(obs[key].astype(np.float32).flatten())
    if not parts:
        raise KeyError(
            f"None of the expected proprioceptive keys found in obs. "
            f"Available keys: {list(obs.keys())}"
        )
    vec = np.concatenate(parts)
    if vec.shape[0] >= state_dim:
        vec = vec[:state_dim]
    else:
        vec = np.pad(vec, (0, state_dim - vec.shape[0]))
    return torch.from_numpy(vec).unsqueeze(0).to(device)


# ---------------------------------------------------------------------------
# Episode / suite runners
# ---------------------------------------------------------------------------

def run_episode(policy, env, lang_tokens, device, config, max_steps: int) -> tuple[bool, int]:
    obs         = env.reset()
    ep_success  = False
    steps_taken = 0

    while steps_taken < max_steps:
        obs_tensor = {
            "observation.images.image": torch.from_numpy(
                obs["agentview_image"].transpose(2, 0, 1)[None]
            ).float().to(device) / 255.0,

            "observation.images.image2": torch.from_numpy(
                obs["robot0_eye_in_hand_image"].transpose(2, 0, 1)[None]
            ).float().to(device) / 255.0,

            "observation.state": _build_state(obs, STATE_DIM, device),

            **lang_tokens,
        }

        with torch.no_grad():
            action = policy.select_action(obs_tensor)

        action_np = action.cpu().numpy().flatten()
        obs, _reward, done, info = env.step(action_np)
        steps_taken += 1
        ep_success = ep_success or bool(info.get("success", False))
        if done or ep_success or steps_taken >= max_steps:
            return ep_success, steps_taken

    return ep_success, steps_taken


def run_suite(
    policy, config, tokenizer, suite, num_episodes,
    device, seed, progress_path, resume,
    model_label="model", num_tasks=10,
):
    try:
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError("LIBERO not installed. pip install -e path/to/LIBERO")

    max_steps = SUITE_MAX_STEPS[suite]
    benchmark  = libero_benchmark.get_benchmark_dict()[suite]()
    task_names = benchmark.get_task_names()[:num_tasks]

    print(f"\n[eval:{model_label}] {suite} | "
          f"{len(task_names)}/{benchmark.get_num_tasks()} tasks × {num_episodes} ep | "
          f"max_steps={max_steps}")

    results = load_progress(progress_path) if resume else {}
    policy.eval()

    total_steps_taken = 0

    for task_idx, task_name in enumerate(task_names):
        if task_name in results:
            sr = results[task_name]["success_rate"]
            print(f"  [{task_idx+1:2d}/{len(task_names)}] SKIP {task_name[:55]} "
                  f"(SR={sr*100:.0f}%)")
            continue

        task             = benchmark.get_task(task_idx)
        task_description = task.language
        lang_tokens      = tokenize_task(task_description, tokenizer, config, device)

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=benchmark.get_task_bddl_file_path(task_idx),
                camera_heights=224,
                camera_widths=224,
            )
            env.seed(seed)
        except Exception as e:
            print(f"  [{task_idx+1:2d}] ENV ERROR {task_name[:55]}: {e}")
            results[task_name] = {
                "success_rate": 0.0, "successes": 0,
                "episodes": 0, "error": str(e),
            }
            save_progress(progress_path, results)
            continue

        successes  = []
        steps_list = []
        t_task     = time.time()

        for ep in range(num_episodes):
            if hasattr(policy, "reset"):
                policy.reset()
            try:
                success, steps = run_episode(
                    policy, env, lang_tokens, device, config, max_steps
                )
                successes.append(float(success))
                steps_list.append(steps)
                total_steps_taken += steps
            except Exception as e:
                print(f"    ep {ep} ERROR: {e}")
                successes.append(0.0)
                steps_list.append(0)
            finally:
                torch.cuda.empty_cache()

        env.close()

        sr        = float(np.mean(successes))
        elapsed_s = time.time() - t_task

        results[task_name] = {
            "success_rate": sr,
            "successes":    int(np.sum(successes)),
            "episodes":     num_episodes,
            "elapsed_s":    round(elapsed_s, 1),
            "avg_steps":    round(float(np.mean(steps_list)), 1),
        }
        save_progress(progress_path, results)

        print(
            f"  [{task_idx+1:2d}/{len(task_names)}] {task_name[:55]:<55} "
            f"SR: {sr*100:5.1f}%  ({int(np.sum(successes))}/{num_episodes})  "
            f"avg_steps={np.mean(steps_list):.0f}  {elapsed_s:.0f}s"
        )

    task_results = {k: v for k, v in results.items() if not k.startswith("__")}
    overall_sr   = float(np.mean([v["success_rate"] for v in task_results.values()]))
    results["__overall__"] = {
        "suite":             suite,
        "success_rate":      overall_sr,
        "num_tasks":         len(task_results),
        "episodes_per_task": num_episodes,
        "total_steps":       total_steps_taken,
        "model_label":       model_label,
    }
    save_progress(progress_path, results)
    print(f"\n  >> [{model_label}] {suite}  SR: {overall_sr*100:.2f}%\n")
    return results


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run(policy, config, device):
    print("\n[dry-run] Sanity forward pass ...")
    B      = 1
    T_lang = config.tokenizer_max_length
    chunk  = config.chunk_size

    state_dim = policy.model.state_proj.weight.shape[1]
    print(f"[dry-run] state_dim={state_dim} (from state_proj.weight)")

    with torch.no_grad():
        loss = policy.model(
            images      = torch.randn(B, 1, 3, 224, 224,
                                      device=device, dtype=torch.bfloat16),
            img_masks   = torch.ones(B, 1, dtype=torch.bool, device=device),
            lang_tokens = torch.randint(0, 256_000, (B, T_lang), device=device),
            lang_masks  = torch.ones(B, T_lang, dtype=torch.bool, device=device),
            state       = torch.randn(B, state_dim,
                                      device=device, dtype=torch.float32),
            actions     = torch.randn(B, chunk, state_dim,
                                      device=device, dtype=torch.float32),
            noise       = torch.randn(B, chunk, state_dim,
                                      device=device, dtype=torch.float32),
            time        = torch.rand(B, device=device, dtype=torch.float32),
        )
    print(f"[dry-run] Forward OK — loss mean: {loss.mean().item():.4f}")
    if hasattr(policy.model, "mla"):
        weights = policy.model.mla.get_layer_weights()
        print(f"[dry-run] MLA weights layer 17: "
              f"{[round(x, 3) for x in weights[-1].tolist()]}")
    print("[dry-run] All good.\n")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(val: float) -> str:
    if val != val:
        return "  N/A"
    return f"{val*100:5.1f}%"


def print_comparison_table(all_results, args, elapsed, ckpt_label="MLA (ours)", base_results=None):
    suites = args.suites
    col_w  = 14

    baseline_models = [m for m in OFFICIAL_BASELINES if
                       any(not np.isnan(OFFICIAL_BASELINES[m].get(s, float("nan")))
                           for s in suites)]

    extra_cols       = (["Base PI0"] if base_results is not None else []) + [ckpt_label]
    all_model_labels = baseline_models + extra_cols

    print("\n" + "=" * 90)
    print(f"LIBERO RESULTS  ({args.num_episodes} episode/task)  —  {ckpt_label}")
    if base_results is not None:
        print("  [compare_base mode: Base PI0 column included for head-to-head comparison]")
    print("=" * 90)

    header = f"  {'Suite':<18}" + "".join(f"{lbl[:col_w]:>{col_w}}" for lbl in all_model_labels)
    sep    = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    col_srs = {lbl: [] for lbl in all_model_labels}

    for suite in suites:
        row = f"  {suite:<18}"

        for lbl in baseline_models:
            v = OFFICIAL_BASELINES[lbl].get(suite, float("nan"))
            col_srs[lbl].append(v if not np.isnan(v) else float("nan"))
            row += f"{_fmt(v):>{col_w}}"

        if base_results is not None:
            base_sr = (
                base_results.get(suite, {})
                .get("__overall__", {})
                .get("success_rate", float("nan"))
            )
            col_srs["Base PI0"].append(base_sr)
            row += f"{_fmt(base_sr):>{col_w}}"

        our_sr = (
            all_results.get(suite, {})
            .get("__overall__", {})
            .get("success_rate", float("nan"))
        )
        col_srs[ckpt_label].append(our_sr)
        row += f"{_fmt(our_sr):>{col_w}}"

        if base_results is not None and not (base_sr != base_sr) and not (our_sr != our_sr):
            delta = our_sr - base_sr
            sign  = "+" if delta >= 0 else ""
            row  += f"  ({sign}{delta*100:.1f}%)"

        print(row)

    print(sep)
    avg_row = f"  {'Avg':<18}"
    for lbl in all_model_labels:
        vals  = [v for v in col_srs[lbl] if not np.isnan(v)]
        avg_v = float(np.mean(vals)) if vals else float("nan")
        avg_row += f"{_fmt(avg_v):>{col_w}}"
    print(avg_row)

    print("\n" + "=" * 90)
    print(f"Total wall-clock time: {elapsed/3600:.1f} hours")


def print_timing_estimate(args, step_time_s: float = 4.5):
    total_eps  = len(args.suites) * args.num_tasks * args.num_episodes
    multiplier = 2 if getattr(args, "compare_base", False) else 1
    label      = " × 2 (base + MLA)" if multiplier == 2 else ""

    avg_max_steps = np.mean([SUITE_MAX_STEPS[s] for s in args.suites])
    optimistic = total_eps * 50  * step_time_s / 3600 * multiplier
    realistic  = total_eps * 100 * step_time_s / 3600 * multiplier
    worst_case = total_eps * avg_max_steps * step_time_s / 3600 * multiplier

    print(f"\n[timing] Estimate for T4 @ {step_time_s}s/step "
          f"({len(args.suites)} suites × {args.num_tasks} tasks × {args.num_episodes} ep{label}):")
    print(f"  Optimistic  (~50 avg steps/ep, fails fast): {optimistic:5.1f} h")
    print(f"  Realistic   (~100 avg steps/ep):            {realistic:5.1f} h")
    print(f"  Worst case  (full max_steps={avg_max_steps:.0f}):          {worst_case:5.1f} h")
    if multiplier == 1:
        print(f"  Add --compare_base to also run base PI0 (doubles runtime).")
    print(f"  Tip: --suites libero_goal --num_tasks 3 for a quick smoke-test.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LIBERO standard eval — MLA vs base PI0 (4 suites, no perturbations)"
    )
    p.add_argument("--checkpoint",   type=str, required=True)
    p.add_argument("--suites",       nargs="+", default=ALL_SUITES, choices=ALL_SUITES)
    p.add_argument("--num_episodes", type=int,  default=NUM_EPISODES)
    p.add_argument("--num_tasks",    type=int,  default=10)
    p.add_argument("--compare_base", action="store_true")
    p.add_argument("--device",       type=str,  default="cuda")
    p.add_argument("--seed",         type=int,  default=42)
    p.add_argument("--output_dir",   type=str,  default=None)
    p.add_argument("--ckpt_label",   type=str,  default="MLA (ours)")
    p.add_argument("--step_time",    type=float, default=4.5)
    p.add_argument("--resume",       action="store_true")
    p.add_argument("--dry_run",      action="store_true")
    p.add_argument("--timing_only",  action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global STATE_DIM

    args = parse_args()

    if args.timing_only:
        print_timing_estimate(args, step_time_s=args.step_time)
        return

    output_dir = args.output_dir or str(
        Path(args.checkpoint).parent / "libero_eval"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[setup] Output dir: {output_dir}")

    total_rollouts = len(args.suites) * args.num_tasks * args.num_episodes
    multiplier     = 2 if args.compare_base else 1
    print(f"[setup] {len(args.suites)} suites | "
          f"{total_rollouts} rollouts per model | "
          f"{total_rollouts * multiplier} total rollouts")

    print_timing_estimate(args, step_time_s=args.step_time)

    tokenizer = make_tokenizer()

    base_results = None
    if args.compare_base:
        base_policy, config = build_model(args.device, use_mla=False)

        STATE_DIM = base_policy.model.state_proj.weight.shape[1]
        print(f"[setup] state_dim={STATE_DIM} (from state_proj.weight)")

        if args.dry_run:
            dry_run(base_policy, config, args.device)
            return

        base_results = {}
        t_base = time.time()

        print("\n" + "=" * 60)
        print("PHASE 1 / 2 — Base PI0 (vanilla, no MLA)")
        print("=" * 60)

        for suite in args.suites:
            progress_path = os.path.join(output_dir, f"base__{suite}.json")
            result = run_suite(
                policy=base_policy, config=config, tokenizer=tokenizer,
                suite=suite, num_episodes=args.num_episodes,
                device=args.device, seed=args.seed, progress_path=progress_path,
                resume=args.resume, model_label="Base PI0", num_tasks=args.num_tasks,
            )
            base_results[suite] = result

        elapsed_base = time.time() - t_base
        base_json = os.path.join(output_dir, "base_pi0_results.json")
        with open(base_json, "w") as f:
            json.dump(base_results, f, indent=2)
        print(f"\n[save] Base PI0 results → {base_json}  ({elapsed_base/3600:.1f}h)")

        del base_policy
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[mem] After base cleanup: "
              f"{torch.cuda.memory_allocated()/1e9:.1f}GB allocated")

    policy, config = build_model(args.device, use_mla=True)
    load_mla_checkpoint(policy, args.checkpoint, args.device)

    STATE_DIM = policy.model.state_proj.weight.shape[1]
    print(f"[setup] state_dim={STATE_DIM} (from state_proj.weight)")

    if args.dry_run and not args.compare_base:
        dry_run(policy, config, args.device)
        return

    all_results = {}
    t_total     = time.time()

    phase_label = "PHASE 2 / 2 — MLA checkpoint" if args.compare_base else "Evaluating MLA checkpoint"
    print("\n" + "=" * 60)
    print(phase_label)
    print("=" * 60)

    for suite in args.suites:
        progress_path = os.path.join(output_dir, f"mla__{suite}.json")
        result = run_suite(
            policy=policy, config=config, tokenizer=tokenizer,
            suite=suite, num_episodes=args.num_episodes,
            device=args.device, seed=args.seed, progress_path=progress_path,
            resume=args.resume, model_label=args.ckpt_label, num_tasks=args.num_tasks,
        )
        all_results[suite] = result

    elapsed = time.time() - t_total

    print_comparison_table(
        all_results, args, elapsed,
        ckpt_label=args.ckpt_label,
        base_results=base_results,
    )

    final_json = os.path.join(output_dir, "mla_results_final.json")
    with open(final_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[save] MLA results → {final_json}")

    if args.compare_base and base_results is not None:
        merged = {
            "base": {k: v.get("__overall__", {}) for k, v in base_results.items()},
            "mla":  {k: v.get("__overall__", {}) for k, v in all_results.items()},
        }
        merged_json = os.path.join(output_dir, "comparison_summary.json")
        with open(merged_json, "w") as f:
            json.dump(merged, f, indent=2)
        print(f"[save] Head-to-head summary → {merged_json}")


if __name__ == "__main__":
    main()
