# train.py

import os
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from lerobot.policies.pi0 import PI0Config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

# ── Config ─────────────────────────────────────────────────────────────────

DEVICE = "cuda"
BATCH_SIZE = 4
NUM_STEPS = 20000
LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
LOG_EVERY = 100
SAVE_EVERY = 5000
WARMUP_STEPS = 500
OUTPUT_DIR = "./outputs/mixed_layer_attention"
MODEL_ID = "lerobot/pi0_libero_base"

# ── Setup ──────────────────────────────────────────────────────────────────

print(f"[setup] Loading config from {MODEL_ID} ...")
t0 = time.time()
config = PI0Config.from_pretrained(MODEL_ID)
config.device = "cpu"  # keep on CPU during init — move to GPU once at the end
print(f"[setup] Config loaded in {time.time()-t0:.1f}s")

print("[setup] Building PI0PolicyMixedLayerAttention on CPU ...")
policy = PI0PolicyMixedLayerAttention(config)

print("[setup] Loading pretrained weights from cache ...")
weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
state_dict = load_file(weights_path, device="cpu")

# Add "model." prefix to match policy.model.state_dict() keys
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

del state_dict, remapped  # free CPU memory
import gc; gc.collect()

print(f"[setup] Moving model to {DEVICE} ...")
policy = policy.to(DEVICE)
print(f"[setup] Model on {DEVICE}")

# Only pass trainable parameters to optimizer
trainable_params = [p for p in policy.parameters() if p.requires_grad]
total_trainable = sum(p.numel() for p in trainable_params)
total_frozen    = sum(p.numel() for p in policy.parameters() if not p.requires_grad)
print(f"[setup] Trainable: {total_trainable/1e6:.3f}M | Frozen: {total_frozen/1e6:.1f}M")

optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

# Linear warmup from 0 to LR over WARMUP_STEPS
scheduler = LinearLR(
    optimizer,
    start_factor=1e-8,
    end_factor=1.0,
    total_iters=WARMUP_STEPS,
)

# ── Dataset ────────────────────────────────────────────────────────────────

print("[setup] Loading dataset ...")
dataset = LeRobotDataset("lerobot/libero")
print(f"[setup] Dataset size: {len(dataset)} samples")

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# ── Verify batch keys before committing to training ────────────────────────

print("[setup] Verifying batch ...")
sample_batch = next(iter(dataloader))
print("Batch keys:", list(sample_batch.keys()))
print("Batch shapes:")
for k, v in sample_batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# ── Training loop ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build trainable name set once — used for checkpoint saving
trainable_names = {n for n, p in policy.model.named_parameters() if p.requires_grad}
print(f"[setup] Trainable param names: {len(trainable_names)}")

policy.train()
step = 0
data_iter = iter(dataloader)
step_times = []

print("[train] Starting training loop ...")

while step < NUM_STEPS:
    # Restart dataloader if exhausted
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)
        print(f"[train] Restarted dataloader at step {step}")

    # Move batch to GPU
    batch = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    t_step = time.time()

    # Forward pass — PI0Policy.forward returns (loss, loss_dict)
    loss, loss_dict = policy.forward(batch)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()

    # Gradient norm for debugging
    grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)

    optimizer.step()

    # Step scheduler only during warmup
    if step < WARMUP_STEPS:
        scheduler.step()

    step_times.append(time.time() - t_step)

    if step % LOG_EVERY == 0:
        lr = optimizer.param_groups[0]["lr"]
        avg_step_ms = (sum(step_times[-LOG_EVERY:]) / max(len(step_times[-LOG_EVERY:]), 1)) * 1000
        mem_gb = torch.cuda.memory_allocated() / 1e9
        mem_reserved_gb = torch.cuda.memory_reserved() / 1e9
        print(
            f"[train] Step {step:06d}/{NUM_STEPS} | "
            f"Loss: {loss.item():.4f} | "
            f"LR: {lr:.2e} | "
            f"GradNorm: {grad_norm:.3f} | "
            f"StepTime: {avg_step_ms:.0f}ms | "
            f"GPU mem: {mem_gb:.1f}GB alloc / {mem_reserved_gb:.1f}GB reserved"
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

# Save final checkpoint
final_dir = f"{OUTPUT_DIR}/final"
os.makedirs(final_dir, exist_ok=True)
trainable_state = {
    k: v for k, v in policy.model.state_dict().items()
    if k in trainable_names
}
torch.save(trainable_state, f"{final_dir}/model.pt")
print(f"[done] Training complete. Final checkpoint saved to {final_dir}")
