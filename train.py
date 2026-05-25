# train.py

import os
import gc
import json
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from lerobot.policies.pi0 import PI0Config
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

# ── Config ─────────────────────────────────────────────────────────────────

DEVICE = "cuda"
BATCH_SIZE = 1
NUM_STEPS = 20000
LR = 3e-5
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
LOG_EVERY = 100
SAVE_EVERY = 5000
WARMUP_STEPS = 500
OUTPUT_DIR = "./outputs/mixed_layer_attention"
MODEL_ID = "lerobot/pi0_libero_base"

# ── Config ─────────────────────────────────────────────────────────────────

print(f"[setup] Loading config from {MODEL_ID} ...")
t0 = time.time()

config_path = hf_hub_download(MODEL_ID, "config.json")
with open(config_path) as f:
    config_dict = json.load(f)

config_dict.pop("type", None)
config_dict["device"] = "cpu"   # keep on CPU during init
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
print(f"[setup] Config loaded in {time.time()-t0:.1f}s")

# ── Model ──────────────────────────────────────────────────────────────────

print("[setup] Building PI0PolicyMixedLayerAttention on CPU ...")
policy = PI0PolicyMixedLayerAttention(config)

print("[setup] Loading pretrained weights from cache ...")
weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
state_dict = load_file(weights_path, device="cpu")

remapped = {}
for k, v in state_dict.items():
    new_key = k if k.startswith("model.") else f"model.{k}"
    remapped[new_key] = v

missing, unexpected = policy.load_state_dict(remapped, strict=False)
print(f"[setup] Weights loaded — missing: {len(missing)}, unexpected: {len(unexpected)}")
if missing:
    print(f"  Missing (new params, expected): {missing[:5]}")
if unexpected:
    print(f"  Unexpected: {unexpected[:5]}")

del state_dict, remapped
gc.collect()

print(f"[setup] Moving model to {DEVICE} ...")
config.device = DEVICE  # restore before moving
policy = policy.to(DEVICE)
print(f"[setup] Model on {DEVICE}")
print(f"[setup] GPU mem after model load: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Optimizer ──────────────────────────────────────────────────────────────

trainable_params = [p for p in policy.parameters() if p.requires_grad]
total_trainable = sum(p.numel() for p in trainable_params)
total_frozen    = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
print(f"[setup] Trainable: {total_trainable/1e6:.3f}M | Frozen: {total_frozen/1e6:.1f}M")

optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = LinearLR(
    optimizer,
    start_factor=1e-8,
    end_factor=1.0,
    total_iters=WARMUP_STEPS,
)

# ── Tokenizer ──────────────────────────────────────────────────────────────

print("[setup] Loading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
print("[setup] Tokenizer ready")

def tokenize_batch(batch, device):
    """Tokenize task strings and add to batch. Moves tensors to device."""
    batch = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    # Pi0 expects a newline appended to each task string
    tasks = [t + "\n" for t in batch["task"]]
    tokens = tokenizer(
        tasks,
        return_tensors="pt",
        padding="max_length",
        max_length=config.tokenizer_max_length,
        truncation=True,
    ).to(device)
    batch["observation.language.tokens"] = tokens["input_ids"]
    batch["observation.language.attention_mask"] = tokens["attention_mask"].bool()
    return batch

# ── Dataset ────────────────────────────────────────────────────────────────

print("[setup] Loading dataset ...")
dataset = LeRobotDataset(
    "lerobot/libero",
    delta_timestamps={
        "action": [i / 10 for i in range(50)],  
    },
)
print(f"[setup] Dataset size: {len(dataset)} samples")

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# ── Verify batch ───────────────────────────────────────────────────────────

print("[setup] Verifying batch ...")
sample_batch = tokenize_batch(next(iter(dataloader)), DEVICE)
print("Batch keys:", list(sample_batch.keys()))
print("Batch shapes:")
for k, v in sample_batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape} | dtype: {v.dtype}")

# ── Training loop ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainable_names = {n for n, p in policy.model.named_parameters() if p.requires_grad}
print(f"[setup] Trainable param names: {len(trainable_names)}")

policy.train()
step = 0
data_iter = iter(dataloader)
step_times = []

print("[train] Starting training loop ...")

while step < NUM_STEPS:
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)
        print(f"[train] Restarted dataloader at step {step}")

    batch = tokenize_batch(batch, DEVICE)

    t_step = time.time()

    loss, loss_dict = policy.forward(batch)

    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)
    optimizer.step()

    if step < WARMUP_STEPS:
        scheduler.step()

    step_times.append(time.time() - t_step)

    if step % LOG_EVERY == 0:
        lr = optimizer.param_groups[0]["lr"]
        avg_step_ms = (sum(step_times[-LOG_EVERY:]) / max(len(step_times[-LOG_EVERY:]), 1)) * 1000
        mem_gb = torch.cuda.memory_allocated() / 1e9
        mem_res_gb = torch.cuda.memory_reserved() / 1e9
        print(
            f"[train] Step {step:06d}/{NUM_STEPS} | "
            f"Loss: {loss.item():.4f} | "
            f"LR: {lr:.2e} | "
            f"GradNorm: {grad_norm:.3f} | "
            f"StepTime: {avg_step_ms:.0f}ms | "
            f"GPU mem: {mem_gb:.1f}GB alloc / {mem_res_gb:.1f}GB reserved"
        )

    if step % SAVE_EVERY == 0 and step > 0:
        checkpoint_dir = f"{OUTPUT_DIR}/checkpoint_{step:06d}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainable_state = {
            k: v for k, v in policy.model.state_dict().items()
            if k in trainable_names
        }
        torch.save(trainable_state, f"{checkpoint_dir}/model.pt")
        print(f"[save] Checkpoint saved to {checkpoint_dir} ({len(trainable_state)} tensors)")

    step += 1

# ── Final checkpoint ───────────────────────────────────────────────────────

final_dir = f"{OUTPUT_DIR}/final"
os.makedirs(final_dir, exist_ok=True)
trainable_state = {
    k: v for k, v in policy.model.state_dict().items()
    if k in trainable_names
}
torch.save(trainable_state, f"{final_dir}/model.pt")
print(f"[done] Training complete. Final checkpoint saved to {final_dir}")
