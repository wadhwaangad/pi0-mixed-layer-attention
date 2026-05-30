"""
eval_libero_pro.py — Full LIBERO-PRO evaluation for PI0 + MixedLayerAttention.

Evaluates the MLA checkpoint across all 4 official task suites and 5 perturbation
types (20 suite×perturbation combinations), 3 episodes per task.

Saves progress after every task — fully resumable after preemption.

Official benchmark structure (Zxy-MLlab/LIBERO-PRO):
  Suites      : libero_goal, libero_spatial, libero_10, libero_object  (10 tasks each)
  Perturbations: object, position, semantic, task, environment         (5 types)
  Combinations: 4 × 5 = 20   (task cannot be combined with others)
  Total tasks : 4 × 10 × 5 = 200 (× 3 episodes = 600 rollouts)

Per-suite max steps (from official TASK_MAX_STEPS):
  libero_goal    : 300
  libero_spatial : 220
  libero_10      : 520
  libero_object  : 280

Perturbation flag → suite suffix mapping (from evaluation_config.yaml):
  use_swap        (position)    → _swap
  use_object      (object)      → _object
  use_language    (semantic)    → _lan
  use_task        (task)        → _task
  use_environment (environment) → _env

Usage:
    # Sanity check first
    python eval_libero_pro.py \\
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt \\
        --dry_run

    # Full benchmark — all 4 suites, all 5 perturbations, 3 episodes
    python eval_libero_pro.py \\
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt

    # Resume after preemption
    python eval_libero_pro.py \\
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt \\
        --resume

    # Specific suites / perturbations only
    python eval_libero_pro.py \\
        --checkpoint ./outputs/mixed_layer_attention/checkpoint_006000/model.pt \\
        --suites libero_goal libero_spatial \\
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

MODEL_ID = "lerobot/pi0_libero_base"
NUM_EPISODES = 3

# Official leaderboard suites (10 tasks each)
ALL_SUITES = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]

# Official per-suite max steps from TASK_MAX_STEPS in run_libero_eval.py
SUITE_MAX_STEPS = {
    "libero_goal":    300,
    "libero_spatial": 220,
    "libero_10":      520,
    "libero_object":  280,
}

# All 5 perturbation types
ALL_PERTURBATIONS = ["position", "object", "semantic", "task", "environment"]

# Official flag mapping (evaluation_config.yaml — Zxy-MLlab/LIBERO-PRO)
#   use_swap        → position generalization
#   use_object      → object generalization
#   use_language    → semantic/language generalization
#   use_task        → task generalization
#   use_environment → environment generalization
PERTURBATION_TO_FLAG = {
    "position":    "use_swap",
    "object":      "use_object",
    "semantic":    "use_language",
    "task":        "use_task",
    "environment": "use_environment",
}

# Official suite-name suffixes (perturbation_mapping in evaluation_config.yaml)
PERTURBATION_TO_SUFFIX = {
    "position":    "swap",
    "object":      "object",
    "semantic":    "lan",
    "task":        "task",
    "environment": "env",
}

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


# ── Perturbed suite setup ─────────────────────────────────────────────────────

def ensure_perturbed_suite(
    suite: str,
    perturbation: str,
    bddl_base: str,
    init_base: str,
) -> str:
    """
    Pre-generate the perturbed task suite via LIBERO-PRO's perturbation.create_env().
    Returns the perturbed suite name e.g. 'libero_goal_swap'.

    Follows the official LIBERO-PRO setup flow: perturbations are baked into
    bddl/init files ahead of time, NOT passed as live flags into OffScreenRenderEnv.
    Note: task cannot be combined with other perturbations (official constraint).
    """
    try:
        from LIBERO_PRO import perturbation as libero_perturbation
    except ImportError:
        raise ImportError(
            "LIBERO-PRO not installed.\n"
            "  git clone https://github.com/Zxy-MLlab/LIBERO-PRO\n"
            "  cd LIBERO-PRO && pip install -e ."
        )

    flag_key        = PERTURBATION_TO_FLAG[perturbation]
    suffix          = PERTURBATION_TO_SUFFIX[perturbation]
    perturbed_suite = f"{suite}_{suffix}"

    init_path = os.path.join(init_base, perturbed_suite)
    bddl_path = os.path.join(bddl_base, perturbed_suite)

    if os.path.exists(init_path) and os.path.exists(bddl_path):
        print(f"[setup] Suite already exists: {perturbed_suite}")
        return perturbed_suite

    print(f"[setup] Generating perturbed suite: {perturbed_suite} ...")
    evaluation_cfg = {
        "bddl_files_path": bddl_base,
        "init_file_dir":   init_base,
        "task_suite_name": suite,
        # Only one flag active at a time (official constraint for single perturbation)
        "use_swap":        flag_key == "use_swap",
        "use_object":      flag_key == "use_object",
        "use_language":    flag_key == "use_language",
        "use_task":        flag_key == "use_task",
        "use_environment": flag_key == "use_environment",
    }
    libero_perturbation.create_env(configs=evaluation_cfg)
    print(f"[setup] Done — suite: {perturbed_suite}")
    return perturbed_suite


# ── Episode rollout ───────────────────────────────────────────────────────────

def run_episode(policy, env, lang_tokens, device, config, max_steps: int) -> bool:
    """Run one episode. Returns True if successful."""
    obs         = env.reset()
    ep_success  = False
    steps_taken = 0

    while steps_taken < max_steps:
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

        action_np = action.cpu().numpy().squeeze(0)
        for step_action in action_np:
            obs, _reward, done, info = env.step(step_action)
            steps_taken += 1
            ep_success = ep_success or bool(info.get("success", False))
            if done or ep_success or steps_taken >= max_steps:
                return ep_success

    return ep_success


# ── Single suite × perturbation eval ─────────────────────────────────────────

def run_combo(
    policy,
    config,
    tokenizer,
    suite: str,
    perturbation: str,
    num_episodes: int,
    device: str,
    seed: int,
    progress_path: str,
    resume: bool,
    bddl_base: str,
    init_base: str,
) -> dict:
    try:
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError(
            "LIBERO not installed. Follow LIBERO-PRO setup instructions."
        )

    max_steps = SUITE_MAX_STEPS[suite]

    # Pre-generate the perturbed suite (official LIBERO-PRO flow)
    perturbed_suite = ensure_perturbed_suite(suite, perturbation, bddl_base, init_base)

    benchmark  = libero_benchmark.get_benchmark_dict()[perturbed_suite]()
    task_names = benchmark.get_task_names()

    print(f"\n[eval] {suite} × {perturbation} (suite={perturbed_suite}) | "
          f"{len(task_names)} tasks × {num_episodes} ep | max_steps={max_steps}")

    results = load_progress(progress_path) if resume else {}
    policy.eval()

    for task_idx, task_name in enumerate(task_names):
        if task_name in results:
            sr = results[task_name]["success_rate"]
            print(f"  [{task_idx+1:2d}/{len(task_names)}] SKIP {task_name[:55]} "
                  f"(SR={sr*100:.0f}%)")
            continue

        task             = benchmark.get_task(task_idx)
        task_description = task.language
        lang_tokens      = tokenize_task(task_description, tokenizer, config, device)

        # Instantiate env WITHOUT perturbation flags — already baked into bddl/init files
        try:
            env = OffScreenRenderEnv(
                bddl_file_name=task.bddl_file,
                camera_heights=128,
                camera_widths=128,
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

        successes = []
        t_task    = time.time()

        for ep in range(num_episodes):
            if hasattr(policy, "reset"):
                policy.reset()
            try:
                success = run_episode(policy, env, lang_tokens, device, config, max_steps)
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
        save_progress(progress_path, results)

        print(
            f"  [{task_idx+1:2d}/{len(task_names)}] {task_name[:55]:<55} "
            f"SR: {sr*100:5.1f}%  ({int(np.sum(successes))}/{num_episodes})  "
            f"{elapsed_s:.0f}s"
        )

    task_results = {k: v for k, v in results.items() if not k.startswith("__")}
    overall_sr   = float(np.mean([v["success_rate"] for v in task_results.values()]))
    results["__overall__"] = {
        "suite":             suite,
        "perturbation":      perturbation,
        "perturbed_suite":   perturbed_suite,
        "success_rate":      overall_sr,
        "num_tasks":         len(task_results),
        "episodes_per_task": num_episodes,
    }
    save_progress(progress_path, results)
    print(f"\n  >> {suite} × {perturbation}  SR: {overall_sr*100:.2f}%\n")
    return results


# ── Dry run ───────────────────────────────────────────────────────────────────

def dry_run(policy, config, device):
    print("\n[dry-run] Sanity forward pass ...")
    B      = 1
    T_img  = 1
    T_lang = config.tokenizer_max_length
    chunk  = config.chunk_size

    state_feature = next(
        (v for k, v in config.input_features.items() if "state" in k.lower()), None
    )
    if state_feature is None:
        raise RuntimeError(
            "[dry-run] Could not find state feature in config.input_features"
        )
    state_dim = state_feature.shape[0]

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
        description="Full LIBERO-PRO benchmark eval — MLA checkpoint "
                    "(4 suites × 5 perturbations)"
    )
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to MLA checkpoint model.pt")
    p.add_argument("--suites", nargs="+", default=ALL_SUITES,
                   choices=ALL_SUITES,
                   help="Task suites to eval (default: all 4)")
    p.add_argument("--perturbations", nargs="+", default=ALL_PERTURBATIONS,
                   choices=ALL_PERTURBATIONS,
                   help="Perturbation types to eval (default: all 5)")
    p.add_argument("--num_episodes", type=int, default=NUM_EPISODES,
                   help=f"Episodes per task (default: {NUM_EPISODES})")
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to save results (default: next to checkpoint)")
    p.add_argument("--bddl_base",  type=str,
                   default="./LIBERO-PRO/libero/libero/bddl_files",
                   help="Path to LIBERO-PRO bddl_files directory")
    p.add_argument("--init_base",  type=str,
                   default="./LIBERO-PRO/libero/libero/init_files",
                   help="Path to LIBERO-PRO init_files directory")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed tasks")
    p.add_argument("--dry_run", action="store_true",
                   help="One forward pass to verify everything works, then exit")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir or str(
        Path(args.checkpoint).parent / "libero_pro_eval"
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"[setup] Output dir: {output_dir}")

    combos = [
        (suite, pert)
        for suite in args.suites
        for pert  in args.perturbations
    ]
    print(f"[setup] {len(combos)} suite×perturbation combinations | "
          f"{len(combos) * 10 * args.num_episodes} total rollouts")

    tokenizer      = make_tokenizer()
    policy, config = build_model(args.device)
    load_mla_checkpoint(policy, args.checkpoint, args.device)

    if args.dry_run:
        dry_run(policy, config, args.device)
        return

    all_results = {}
    t_total     = time.time()

    for suite, perturbation in combos:
        key           = f"{suite}__{perturbation}"
        progress_path = os.path.join(output_dir, f"{key}.json")

        result = run_combo(
            policy        = policy,
            config        = config,
            tokenizer     = tokenizer,
            suite         = suite,
            perturbation  = perturbation,
            num_episodes  = args.num_episodes,
            device        = args.device,
            seed          = args.seed,
            progress_path = progress_path,
            resume        = args.resume,
            bddl_base     = args.bddl_base,
            init_base     = args.init_base,
        )
        all_results[key] = result

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_total

    # Per-suite average across perturbations
    suite_avgs = {}
    for suite in args.suites:
        srs = [
            all_results[f"{suite}__{p}"]["__overall__"]["success_rate"]
            for p in args.perturbations
            if f"{suite}__{p}" in all_results
        ]
        suite_avgs[suite] = float(np.mean(srs)) if srs else float("nan")

    # Per-perturbation average across suites
    pert_avgs = {}
    for pert in args.perturbations:
        srs = [
            all_results[f"{s}__{pert}"]["__overall__"]["success_rate"]
            for s in args.suites
            if f"{s}__{pert}" in all_results
        ]
        pert_avgs[pert] = float(np.mean(srs)) if srs else float("nan")

    overall = float(np.mean(list(suite_avgs.values())))

    col_w = 12
    header = f"  {'Suite':<16}" + "".join(f"{p:>{col_w}}" for p in args.perturbations) + f"{'Avg':>{col_w}}"

    print("\n" + "=" * len(header))
    print(f"LIBERO-PRO RESULTS  ({args.num_episodes} episodes/task)")
    print("=" * len(header))
    print(header)
    print("  " + "-" * (len(header) - 2))

    for suite in args.suites:
        row = f"  {suite:<16}"
        for pert in args.perturbations:
            key = f"{suite}__{pert}"
            sr  = all_results.get(key, {}).get("__overall__", {}).get(
                "success_rate", float("nan")
            ) * 100
            row += f"{sr:>{col_w}.1f}%"[:-1].rjust(col_w)  # strip extra % then pad
            row += "%"
        row += f"{suite_avgs[suite]*100:>{col_w}.1f}%"[:-1].rjust(col_w) + "%"
        print(row)

    print("  " + "-" * (len(header) - 2))
    avg_row = f"  {'Avg':<16}"
    for pert in args.perturbations:
        avg_row += f"{pert_avgs[pert]*100:>{col_w-1}.1f}%".rjust(col_w)
    avg_row += f"{overall*100:>{col_w-1}.1f}%".rjust(col_w)
    print(avg_row)
    print("=" * len(header))
    print(f"Total time: {elapsed/3600:.1f} hours")

    final_json = os.path.join(output_dir, "eval_results_final.json")
    with open(final_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[save] Full results → {final_json}")


if __name__ == "__main__":
    main()
