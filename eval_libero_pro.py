"""
eval_libero_pro.py — Full LIBERO-PRO evaluation for PI0 + MixedLayerAttention.
Observation construction and normalization match eval_libero.py exactly
(make_pre_post_processors + _build_obs_dict pipeline).
Only the perturbation and suite logic differs.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0 import PI0Policy
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "LIBERO-PRO"))

MODEL_ID = "lerobot/pi0_libero_finetuned_v044"

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

CAMERA_RESOLUTION = 256
STATE_DIM: int    = 8

_PROPRIO_KEYS = [
    "robot0_eef_pos",       # 3-dim
    "robot0_eef_quat",      # 4-dim
    "robot0_gripper_qpos",  # 1-dim
]


# ---------------------------------------------------------------------------
# Model helpers — identical to eval_libero.py
# ---------------------------------------------------------------------------

def build_model(device: str, use_mla: bool = True):
    """
    Load policy via from_pretrained (handles weight loading cleanly),
    then build pre/post processors the same way as the reference script.
    For MLA, load base weights into PI0PolicyMixedLayerAttention instead.
    """
    label = "PI0PolicyMixedLayerAttention" if use_mla else "PI0Policy (vanilla base)"
    print(f"\n[build] Loading {label} from {MODEL_ID} ...")

    if use_mla:
        base = PI0Policy.from_pretrained(MODEL_ID)
        policy = PI0PolicyMixedLayerAttention(base.config)
        missing, unexpected = policy.load_state_dict(base.state_dict(), strict=False)
        print(f"[build] MLA weight transfer — missing: {len(missing)}, unexpected: {len(unexpected)}")
        del base
        gc.collect()
    else:
        policy = PI0Policy.from_pretrained(MODEL_ID)

    policy = policy.to(device).eval()

    preprocess, postprocess = make_pre_post_processors(
        policy.config,
        MODEL_ID,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    print(f"[build] Model on {device} | "
          f"GPU mem: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    return policy, policy.config, preprocess, postprocess


def load_mla_checkpoint(policy, ckpt_path: str, device: str):
    print(f"[ckpt] Loading MLA checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    missing, unexpected = policy.model.load_state_dict(ckpt, strict=False)
    print(f"[ckpt] Overlaid {len(ckpt)} tensors | "
          f"missing: {len(missing)} | unexpected: {len(unexpected)}")
    return policy


# ---------------------------------------------------------------------------
# Progress helpers — identical to eval_libero.py
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
# Obs construction — identical to eval_libero.py
# ---------------------------------------------------------------------------

def _build_obs_dict(obs: dict, task_str: str) -> dict:
    """
    Convert a LIBERO env observation into the flat dict that preprocess() expects,
    mirroring the keys used in LeRobotDataset frames.
    """
    eef_pos = obs["robot0_eef_pos"].astype(np.float32)

    eef_rotvec = R.from_quat(
        obs["robot0_eef_quat"]
    ).as_rotvec().astype(np.float32)

    gripper = obs["robot0_gripper_qpos"].astype(np.float32)

    state = np.concatenate([eef_pos, eef_rotvec, gripper])
    if state.shape[0] != STATE_DIM:
        raise ValueError(
            f"State has {state.shape[0]} dims, expected {STATE_DIM}. "
            f"Check _PROPRIO_KEYS."
        )

    return {
        "observation.images.image":  obs["agentview_image"],
        "observation.images.image2": obs["robot0_eye_in_hand_image"],
        "observation.images.empty_camera_0": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation.state": state,
        "task": task_str,
    }


# ---------------------------------------------------------------------------
# Perturbation suite setup — LIBERO-PRO specific
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Episode runner — identical to eval_libero.py
# ---------------------------------------------------------------------------

def run_episode(
    policy, env, task_str, device, config,
    preprocess, postprocess,
    max_steps: int,
) -> tuple[bool, int]:
    obs         = env.reset()
    ep_success  = False
    steps_taken = 0

    while steps_taken < max_steps:
        raw_obs = _build_obs_dict(obs, task_str)
        batch   = preprocess(raw_obs)

        with torch.inference_mode():
            action = policy.select_action(batch)

        action    = postprocess(action)
        action_np = action.cpu().numpy().flatten()

        obs, _reward, done, info = env.step(action_np)
        steps_taken += 1
        ep_success = ep_success or bool(info.get("success", False))
        if done or ep_success or steps_taken >= max_steps:
            return ep_success, steps_taken

    return ep_success, steps_taken


# ---------------------------------------------------------------------------
# Combo runner — run_suite from eval_libero.py extended with perturbation logic
# ---------------------------------------------------------------------------

def run_combo(
    policy, config, preprocess, postprocess,
    suite, perturbation, num_episodes,
    device, seed, progress_path, resume,
    bddl_base, init_base,
    model_label="model", num_tasks=10,
):
    try:
        from libero.libero import benchmark as libero_benchmark
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError:
        raise ImportError("LIBERO not installed. Follow LIBERO-PRO setup instructions.")

    max_steps       = SUITE_MAX_STEPS[suite]
    perturbed_suite = ensure_perturbed_suite(suite, perturbation, bddl_base, init_base)

    benchmark  = libero_benchmark.get_benchmark_dict()[perturbed_suite]()
    task_names = benchmark.get_task_names()[:num_tasks]

    print(f"\n[eval:{model_label}] {suite} × {perturbation} | "
          f"{len(task_names)}/{benchmark.get_num_tasks()} tasks × {num_episodes} ep | "
          f"max_steps={max_steps} | cam_res={CAMERA_RESOLUTION} | state_dim={STATE_DIM}")

    results = load_progress(progress_path) if resume else {}
    policy.eval()

    total_steps_taken = 0

    for task_idx, task_name in enumerate(task_names):
        if task_name in results:
            sr = results[task_name]["success_rate"]
            print(f"  [{task_idx+1:2d}/{len(task_names)}] SKIP {task_name[:55]} "
                  f"(SR={sr*100:.0f}%)")
            continue

        task     = benchmark.get_task(task_idx)
        task_str = task.language

        try:
            env = OffScreenRenderEnv(
                bddl_file_name=benchmark.get_task_bddl_file_path(task_idx),
                camera_names=["agentview", "robot0_eye_in_hand"],
                camera_heights=CAMERA_RESOLUTION,
                camera_widths=CAMERA_RESOLUTION,
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
                    policy, env, task_str, device, config,
                    preprocess, postprocess, max_steps,
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


# ---------------------------------------------------------------------------
# Dry run — identical to eval_libero.py
# ---------------------------------------------------------------------------

def dry_run(policy, config, preprocess, postprocess, device):
    print("\n[dry-run] Sanity check via preprocess + select_action ...")

    fake_obs = {
        "observation.images.image":  np.random.randint(
            0, 255, (CAMERA_RESOLUTION, CAMERA_RESOLUTION, 3), dtype=np.uint8),
        "observation.images.image2": np.random.randint(
            0, 255, (CAMERA_RESOLUTION, CAMERA_RESOLUTION, 3), dtype=np.uint8),
        "observation.state": np.zeros(STATE_DIM, dtype=np.float32),
        "observation.language_instruction": "pick up the red block",
    }

    batch = preprocess(fake_obs)

    with torch.inference_mode():
        action = policy.select_action(batch)

    action = postprocess(action)
    print(f"[dry-run] action shape: {action.shape}  mean: {action.mean().item():.4f}")

    if hasattr(policy.model, "mla"):
        weights = policy.model.mla.get_layer_weights()
        print(f"[dry-run] MLA weights layer 17: "
              f"{[round(x, 3) for x in weights[-1].tolist()]}")
    print("[dry-run] All good.\n")


# ---------------------------------------------------------------------------
# Reporting — LIBERO-PRO specific (perturbation table)
# ---------------------------------------------------------------------------

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

        extra_cols       = (["Base PI0"] if base_results is not None else []) + [ckpt_label]
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
            key = f"{suite}__{pert}"
            sr  = (
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

        suite_avg          = float(np.nanmean(pert_srs)) if pert_srs else float("nan")
        suite_avgs[suite]  = suite_avg
        row               += f"{_fmt(suite_avg):>{col_w2}}"
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
    overall   = float(np.nanmean(list(suite_avgs.values())))
    avg_row2 += f"{_fmt(overall):>{col_w2}}"
    print(avg_row2)
    print("=" * 90)
    print(f"Total wall-clock time: {elapsed/3600:.1f} hours")


def print_timing_estimate(args, step_time_s: float = 4.5):
    avg_max_steps = np.mean([SUITE_MAX_STEPS[s] for s in args.suites])
    num_combos    = len(args.suites) * len(args.perturbations)
    total_eps     = num_combos * args.num_tasks * args.num_episodes

    multiplier = 2 if getattr(args, "compare_base", False) else 1
    label      = " × 2 (base + MLA)" if multiplier == 2 else ""

    optimistic = total_eps * 50  * step_time_s / 3600 * multiplier
    realistic  = total_eps * 100 * step_time_s / 3600 * multiplier
    worst_case = total_eps * avg_max_steps * step_time_s / 3600 * multiplier

    print(f"\n[timing] Estimate for T4 @ {step_time_s}s/step "
          f"({num_combos} combos × {args.num_tasks} tasks × {args.num_episodes} ep{label}):")
    print(f"  Optimistic  (~50 avg steps/ep, fails fast): {optimistic:5.1f} h")
    print(f"  Realistic   (~100 avg steps/ep):            {realistic:5.1f} h")
    print(f"  Worst case  (full max_steps={avg_max_steps:.0f}):          {worst_case:5.1f} h")
    if multiplier == 1:
        print(f"  Add --compare_base to also run base PI0 (doubles runtime).")
    print(f"  Tip: --suites libero_goal --perturbations position for a quick smoke-test.\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="LIBERO-PRO eval — MLA vs base PI0  (4 suites × 5 perturbations)"
    )
    p.add_argument("--checkpoint",    type=str,   required=True)
    p.add_argument("--suites",        nargs="+",  default=ALL_SUITES, choices=ALL_SUITES)
    p.add_argument("--perturbations", nargs="+",  default=ALL_PERTURBATIONS, choices=ALL_PERTURBATIONS)
    p.add_argument("--num_episodes",  type=int,   default=NUM_EPISODES)
    p.add_argument("--num_tasks",     type=int,   default=5)
    p.add_argument("--compare_base",  action="store_true")
    p.add_argument("--device",        type=str,   default="cuda")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output_dir",    type=str,   default=None)
    p.add_argument("--bddl_base",     type=str,   default="./LIBERO-PRO/libero/libero/bddl_files")
    p.add_argument("--init_base",     type=str,   default="./LIBERO-PRO/libero/libero/init_files")
    p.add_argument("--ckpt_label",    type=str,   default="MLA (ours)")
    p.add_argument("--step_time",     type=float, default=4.5)
    p.add_argument("--resume",        action="store_true")
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--timing_only",   action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
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
    print(f"[setup] Camera resolution: {CAMERA_RESOLUTION}x{CAMERA_RESOLUTION}")
    print(f"[setup] State dim: {STATE_DIM}")

    print_timing_estimate(args, step_time_s=args.step_time)

    base_results = None
    if args.compare_base:
        base_policy, config, preprocess, postprocess = build_model(
            args.device, use_mla=False
        )

        if args.dry_run:
            dry_run(base_policy, config, preprocess, postprocess, args.device)
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
                policy=base_policy, config=config,
                preprocess=preprocess, postprocess=postprocess,
                suite=suite, perturbation=perturbation,
                num_episodes=args.num_episodes,
                device=args.device, seed=args.seed,
                progress_path=progress_path, resume=args.resume,
                bddl_base=args.bddl_base, init_base=args.init_base,
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

    policy, config, preprocess, postprocess = build_model(args.device, use_mla=True)
    load_mla_checkpoint(policy, args.checkpoint, args.device)

    if args.dry_run and not args.compare_base:
        dry_run(policy, config, preprocess, postprocess, args.device)
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
            policy=policy, config=config,
            preprocess=preprocess, postprocess=postprocess,
            suite=suite, perturbation=perturbation,
            num_episodes=args.num_episodes,
            device=args.device, seed=args.seed,
            progress_path=progress_path, resume=args.resume,
            bddl_base=args.bddl_base, init_base=args.init_base,
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
