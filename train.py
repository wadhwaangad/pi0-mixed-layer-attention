# train.py

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from lerobot.policies.pi0 import PI0Config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from torch.utils.data import DataLoader
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

# ── Config ─────────────────────────────────────────────────────────────────

DEVICE = "cuda"
BATCH_SIZE = 8
NUM_STEPS = 20000
LR = 1e-4
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0
LOG_EVERY = 100
SAVE_EVERY = 5000
WARMUP_STEPS = 500
OUTPUT_DIR = "./outputs/mixed_layer_attention"

# ── Setup ──────────────────────────────────────────────────────────────────

config = PI0Config("lerobot/pi0_libero_base")
policy = PI0PolicyMixedLayerAttention(config)
policy = policy.to(DEVICE)

# Only pass trainable parameters to optimizer
trainable_params = [p for p in policy.parameters() if p.requires_grad]
print(f"Passing {sum(p.numel() for p in trainable_params)/1e6:.2f}M params to optimizer")

optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

# Linear warmup from 0 to LR over WARMUP_STEPS
scheduler = LinearLR(
    optimizer,
    start_factor=1e-8,
    end_factor=1.0,
    total_iters=WARMUP_STEPS,
)

# ── Dataset ────────────────────────────────────────────────────────────────

dataset = LeRobotDataset("lerobot/libero")
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

# ── Verify batch keys before committing to training ────────────────────────

sample_batch = next(iter(dataloader))
print("Batch keys:", list(sample_batch.keys()))
print("Batch shapes:")
for k, v in sample_batch.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.shape}")

# ── Training loop ──────────────────────────────────────────────────────────

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Build trainable name set once — used for checkpoint saving
trainable_names = {n for n, p in policy.model.named_parameters() if p.requires_grad}

policy.train()
step = 0
data_iter = iter(dataloader)

while step < NUM_STEPS:
    # Restart dataloader if exhausted
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(dataloader)
        batch = next(data_iter)

    # Move batch to GPU
    batch = {
        k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }

    # Forward pass — PI0Policy.forward returns (loss, loss_dict)
    loss, loss_dict = policy.forward(batch)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_CLIP)
    optimizer.step()

    # Step scheduler only during warmup
    if step < WARMUP_STEPS:
        scheduler.step()

    if step % LOG_EVERY == 0:
        lr = optimizer.param_groups[0]["lr"]
        print(f"Step {step:06d}/{NUM_STEPS} | Loss: {loss.item():.4f} | LR: {lr:.2e}")

    if step % SAVE_EVERY == 0 and step > 0:
        checkpoint_dir = f"{OUTPUT_DIR}/checkpoint_{step:06d}"
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Save only trainable parameters — frozen weights are loaded
        # from the pretrained checkpoint at inference time
        trainable_state = {
            k: v for k, v in policy.model.state_dict().items()
            if k in trainable_names
        }
        torch.save(trainable_state, f"{checkpoint_dir}/model.pt")
        print(f"Saved checkpoint to {checkpoint_dir}")

    step += 1

# Save final checkpoint
final_dir = f"{OUTPUT_DIR}/final"
os.makedirs(final_dir, exist_ok=True)
trainable_state = {
    k: v for k, v in policy.model.state_dict().items()
    if k in trainable_names
}
torch.save(trainable_state, f"{final_dir}/model.pt")
print("Training complete.")
