"""
eval_libero_pro.py — Full LIBERO-PRO evaluation for PI0 + MixedLayerAttention.
[... truncated header ...]
"""

import argparse
import copy
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

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIBERO-PRO"))

MODEL_ID = "lerobot/pi0_libero_base"
NUM_EPISODES = 1

ALL_SUITES = ["libero_goal", "libero_spatial", "libero_10", "libero_object"]

SUITE_MAX_STEPS = {
    "libero_goal":    300,
    "libero_spatial": 220,
    "libero_10":      520,
    "libero_object":  280,
}

ALL_PERTURBATIONS = ["position", "object", "semantic", "task"]

PERTURBATION_TO_FLAG = {
    "position":    "use_swap",
    "object":      "use_object",
    "semantic":    "use_language",
    "task":        "use_task",
    "environment": "use_environment",
}

PERTURBATION_TO_SUFFIX = {
    "position":    "swap",
    "object":      "object",
    "semantic":    "lan",
    "task":        "task",
    "environment": "env",
}

OFFICIAL_BASELINES = {
    "pi0 (baseline)": {
        "position":    {"libero_goal": 0.92, "libero_spatial": 0.90, "libero_10": 0.82, "libero_object": 0.98},
        "object":      {"libero_goal": 0.92, "libero_spatial": 0.90, "libero_10": 0.82, "libero_object": 0.98},
        "semantic":    {"libero_goal": 0.92, "libero_spatial": 0.90, "libero_10": 0.82, "libero_object": 0.98},
        "task":        {"libero_goal": 0.92, "libero_spatial": 0.90, "libero_10": 0.82, "libero_object": 0.98},
        "environment": {"libero_goal": 0.92, "libero_spatial": 0.90, "libero_10": 0.82, "libero_object": 0.98},
    },
    "pi0 (LIBERO-PRO)": {
        "position":    {"libero_goal": 0.00, "libero_spatial": 0.00, "libero_10": 0.00, "libero_object": 0.00},
        "object":      {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "semantic":    {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "task":        {"libero_goal": 0.00, "libero_spatial": 0.00, "libero_10": 0.00, "libero_object": 0.00},
        "environment": {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
    },
    "pi0.5 (LIBERO-PRO)": {
        "position":    {"libero_goal": 0.38, "libero_spatial": 0.20, "libero_10": 0.08, "libero_object": 0.17},
        "object":      {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "semantic":    {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "task":        {"libero_goal": 0.00, "libero_spatial": 0.01, "libero_10": 0.01, "libero_object": 0.01},
        "environment": {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
    },
    "OpenVLA (LIBERO-PRO)": {
        "position":    {"libero_goal": 0.00, "libero_spatial": 0.00, "libero_10": 0.00, "libero_object": 0.00},
        "object":      {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "semantic":    {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "task":        {"libero_goal": 0.00, "libero_spatial": 0.00, "libero_10": 0.00, "libero_object": 0.00},
        "environment": {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
    },
    "UniVLA (LIBERO-PRO)": {
        "position":    {"libero_goal": 0.09, "libero_spatial": 0.05, "libero_10": 0.02, "libero_object": 0.00},
        "object":      {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "semantic":    {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
        "task":        {"libero_goal": 0.00, "libero_spatial": 0.00, "libero_10": 0.00, "libero_object": 0.00},
        "environment": {"libero_goal": float("nan"), "libero_spatial": float("nan"), "libero_10": float("nan"), "libero_object": float("nan")},
    },
}


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


def ensure_perturbed_suite(suite, perturbation, bddl_base, init_base):
    try:
        import perturbation as libero_perturbation
    except ImportError:
        raise ImportError(
            "LIBERO-PRO not found. Clone it next to this script:\n"
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
        "use_swap":        flag_key == "use_swap",
        "use_object":      flag_key == "use_object",
        "use_language":    flag_key == "use_language",
        "use_task":        flag_key == "use_task",
        "use_environment": flag_key == "use_environment",
    }
    libero_perturbation.create_env(configs=evaluation_cfg)
    print(f"[setup] Done — suite: {perturbed_suite}")
    return perturbed_suite


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


def run_episode(policy, env, lang_tokens, device, config, max_steps: int) -> tuple[bool, int]:
    obs        = env.reset()
    ep_success = False
    steps_taken = 0

    while steps_taken < max_steps:
        obs_tensor = {
            "observation.images.image": torch.from_numpy(
                obs["agentview_image"].transpose(2, 0, 1)[None]
            ).float().to(device) / 255.0,
        
            "observation.images.image2": torch.from_numpy(
                obs["robot0_eye_in_hand_image"].transpose(2, 0, 1)[None]  # ← was agentview_image
            ).float().to(device) / 255.0,
        
            "observation.state": _build_state(obs, STATE_DIM, device),
            **lang_tokens,
        }

        with torch.no_grad():
            action = policy.select_action(obs_tensor)

        action_np = action.cpu().numpy().flatten()  # always (action_dim,)
        obs, _reward, done, info = env.step(action_np)
        steps_taken += 1
        ep_success = ep_success or bool(info.get("success", False))
        if done or ep_success or steps_taken >= max_steps:
            return ep_success, steps_taken

    return ep_success, steps_taken


def run_combo(
    policy, config, tokenizer, suite, perturbation, num_episodes,
    device, seed, progress_path, resume, bddl_base, init_base,
    model_label="model", num_tasks=10,
):
    try:
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError("LIBERO not installed. Follow LIBERO-PRO setup instructions.")

    max_steps = SUITE_MAX_STEPS[suite]
    perturbed_suite = ensure_perturbed_suite(suite, perturbation, bddl_base, init_base)

    benchmark  = libero_benchmark.get_benchmark_dict()[perturbed_suite]()
    task_names = benchmark.get_task_names()[:num_tasks]

    print(f"\n[eval:{model_label}] {suite} × {perturbation} | "
          f"{len(task_names)}/{benchmark.get_num_tasks()} tasks × {num_episodes} ep | max_steps={max_steps}")

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
                camera_names=["agentview", "robot0_eye_in_hand"],
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
        "perturbation":      perturbation,
        "perturbed_suite":   perturbed_suite,
        "success_rate":      overall_sr,
        "num_tasks":         len(task_results),
        "episodes_per_task": num_episodes,
        "total_steps":       total_steps_taken,
        "model_label":       model_label,
    }
    save_progress(progress_path, results)
    print(f"\n  >> [{model_label}] {suite} × {perturbation}  SR: {overall_sr*100:.2f}%\n")
    return results


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


def _fmt(val: float) -> str:
    if val != val:
        return "  N/A"
    return f"{val*100:5.1f}%"


def print_comparison_table(all_results, args, elapsed, ckpt_label="MLA (ours)", base_results=None):
    suites        = args.suites
    perturbations = args.perturbations
    col_w         = 12

    baseline_models = list(OFFICIAL_BASELINES.keys())

    print("\n" + "=" * 90)
    print(f"LIBERO-PRO RESULTS  ({args.num_episodes} episode/task)  —  {ckpt_label}")
    if base_results is not None:
        print("  [compare_base mode: Base PI0 column included for head-to-head comparison]")
    print("=" * 90)

    for pert in perturbations:
        active_baselines = [
            m for m in baseline_models
            if any(
                not np.isnan(OFFICIAL_BASELINES[m][pert].get(s, float("nan")))
                for s in suites
            )
        ]

        extra_cols = (["Base PI0"] if base_results is not None else []) + [ckpt_label]
        all_model_labels = active_baselines + extra_cols

        print(f"\n── Perturbation: {pert.upper()} ──")
        header = f"  {'Suite':<16}" + "".join(
            f"{lbl[:col_w]:>{col_w}}" for lbl in all_model_labels
        )
        sep = "  " + "-" * (len(header) - 2)
        print(header)
        print(sep)

        col_srs = {lbl: [] for lbl in all_model_labels}

        for suite in suites:
            row = f"  {suite:<16}"

            for lbl in active_baselines:
                v = OFFICIAL_BASELINES[lbl].get(pert, {}).get(suite, float("nan"))
                col_srs[lbl].append(v if not np.isnan(v) else float("nan"))
                row += f"{_fmt(v):>{col_w}}"

            if base_results is not None:
                key = f"{suite}__{pert}"
                base_sr = (
                    base_results.get(key, {})
                    .get("__overall__", {})
                    .get("success_rate", float("nan"))
                )
                col_srs["Base PI0"].append(base_sr)
                row += f"{_fmt(base_sr):>{col_w}}"

            key    = f"{suite}__{pert}"
            our_sr = (
                all_results.get(key, {})
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
        avg_row = f"  {'Avg':<16}"
        for lbl in all_model_labels:
            vals  = [v for v in col_srs[lbl] if not np.isnan(v)]
            avg_v = float(np.mean(vals)) if vals else float("nan")
            avg_row += f"{_fmt(avg_v):>{col_w}}"
        print(avg_row)

    print("\n" + "=" * 90)
    if base_results is not None:
        print(f"GRAND SUMMARY — Base PI0  vs  {ckpt_label}  (suite × perturbation, delta)")
    else:
        print(f"GRAND SUMMARY — {ckpt_label}  (suite × perturbation)")
    print("=" * 90)

    col_w2 = 14
    if base_results is not None:
        header2 = f"  {'Suite':<16}" + "".join(
            f"{p + ' Δ':>{col_w2}}" for p in perturbations
        ) + f"{'Overall Δ':>{col_w2}}"
    else:
        header2 = f"  {'Suite':<16}" + "".join(
            f"{p:>{col_w2}}" for p in perturbations
        ) + f"{'Avg':>{col_w2}}"

    print(header2)
    print("  " + "-" * (len(header2) - 2))

    suite_avgs = {}
    for suite in suites:
        row      = f"  {suite:<16}"
        pert_srs = []
        for pert in perturbations:
            key    = f"{suite}__{pert}"
            sr     = (
                all_results.get(key, {})
                .get("__overall__", {})
                .get("success_rate", float("nan"))
            )
            pert_srs.append(sr)

            if base_results is not None:
                base_sr = (
                    base_results.get(key, {})
                    .get("__overall__", {})
                    .get("success_rate", float("nan"))
                )
                if not (sr != sr) and not (base_sr != base_sr):
                    delta = sr - base_sr
                    sign  = "+" if delta >= 0 else ""
                    row  += f"{sign}{delta*100:.1f}%".rjust(col_w2)
                else:
                    row += f"{'N/A':>{col_w2}}"
            else:
                row += f"{_fmt(sr):>{col_w2}}"

        suite_avg = float(np.nanmean(pert_srs)) if pert_srs else float("nan")
        suite_avgs[suite] = suite_avg
        row += f"{_fmt(suite_avg):>{col_w2}}"
        print(row)

    print("  " + "-" * (len(header2) - 2))
    avg_row2 = f"  {'Avg':<16}"
    for pert in perturbations:
        srs = [
            all_results.get(f"{s}__{pert}", {})
            .get("__overall__", {})
            .get("success_rate", float("nan"))
            for s in suites
        ]
        pa = float(np.nanmean([v for v in srs if not np.isnan(v)])) if srs else float("nan")
        avg_row2 += f"{_fmt(pa):>{col_w2}}"
    overall = float(np.nanmean(list(suite_avgs.values())))
    avg_row2 += f"{_fmt(overall):>{col_w2}}"
    print(avg_row2)
    print("=" * 90)
    print(f"Total wall-clock time: {elapsed/3600:.1f} hours")


def print_timing_estimate(args, step_time_s: float = 4.5):
    avg_max_steps = np.mean([SUITE_MAX_STEPS[s] for s in args.suites])
    num_combos    = len(args.suites) * len(args.perturbations)
    total_eps     = num_combos * 10 * args.num_episodes

    multiplier = 2 if getattr(args, "compare_base", False) else 1
    label      = " × 2 (base + MLA)" if multiplier == 2 else ""

    optimistic = total_eps * 50  * step_time_s / 3600 * multiplier
    realistic  = total_eps * 100 * step_time_s / 3600 * multiplier
    worst_case = total_eps * avg_max_steps * step_time_s / 3600 * multiplier

    print(f"\n[timing] Estimate for T4 @ {step_time_s}s/step "
          f"({num_combos} combos × 10 tasks × {args.num_episodes} ep{label}):")
    print(f"  Optimistic  (~50 avg steps/ep, fails fast): {optimistic:5.1f} h")
    print(f"  Realistic   (~100 avg steps/ep):            {realistic:5.1f} h")
    print(f"  Worst case  (full max_steps={avg_max_steps:.0f}):          {worst_case:5.1f} h")
    if multiplier == 1:
        print(f"  Add --compare_base to also run base PI0 (doubles runtime).")
    print(f"  Tip: --suites libero_goal --perturbations position for a quick smoke-test.\n")


def parse_args():
    p = argparse.ArgumentParser(
        description="LIBERO-PRO eval — MLA vs base PI0  (4 suites × 5 perturbations, 1 ep)"
    )
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--suites", nargs="+", default=ALL_SUITES, choices=ALL_SUITES)
    p.add_argument("--perturbations", nargs="+", default=ALL_PERTURBATIONS, choices=ALL_PERTURBATIONS)
    p.add_argument("--num_episodes", type=int, default=NUM_EPISODES)
    p.add_argument("--compare_base", action="store_true")
    p.add_argument("--num_tasks", type=int, default=5)
    p.add_argument("--device",     type=str, default="cuda")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--bddl_base",  type=str, default="./LIBERO-PRO/libero/libero/bddl_files")
    p.add_argument("--init_base",  type=str, default="./LIBERO-PRO/libero/libero/init_files")
    p.add_argument("--ckpt_label", type=str, default="MLA (ours)")
    p.add_argument("--step_time",  type=float, default=4.5)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--timing_only", action="store_true")
    return p.parse_args()


def main():
    global STATE_DIM

    args = parse_args()

    if args.timing_only:
        print_timing_estimate(args, step_time_s=args.step_time)
        return

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
    total_rollouts = len(combos) * args.num_tasks * args.num_episodes
    multiplier     = 2 if args.compare_base else 1
    print(f"[setup] {len(combos)} suite×perturbation combos | "
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

        for suite, perturbation in combos:
            key           = f"{suite}__{perturbation}"
            progress_path = os.path.join(output_dir, f"base__{key}.json")
            result = run_combo(
                policy=base_policy, config=config, tokenizer=tokenizer,
                suite=suite, perturbation=perturbation, num_episodes=args.num_episodes,
                device=args.device, seed=args.seed, progress_path=progress_path,
                resume=args.resume, bddl_base=args.bddl_base, init_base=args.init_base,
                model_label="Base PI0", num_tasks=args.num_tasks,
            )
            base_results[key] = result

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

    for suite, perturbation in combos:
        key           = f"{suite}__{perturbation}"
        progress_path = os.path.join(output_dir, f"mla__{key}.json")

        result = run_combo(
            policy=policy, config=config, tokenizer=tokenizer,
            suite=suite, perturbation=perturbation, num_episodes=args.num_episodes,
            device=args.device, seed=args.seed, progress_path=progress_path,
            resume=args.resume, bddl_base=args.bddl_base, init_base=args.init_base,
            model_label=args.ckpt_label, num_tasks=args.num_tasks,
        )
        all_results[key] = result

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
