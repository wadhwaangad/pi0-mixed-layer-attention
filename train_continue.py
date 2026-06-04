# train_continue.py

import os
import gc
import json
import time
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from lerobot.policies.pi0 import PI0Config
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from pi0_policy_mixed_layer_attention import PI0PolicyMixedLayerAttention

# ── Config ─────────────────────────────────────────────────────────────────

DEVICE        = "cuda"
BATCH_SIZE    = 2
ACCUM_STEPS   = 2
NUM_STEPS     = 5000     
LR            = 3e-4
LR_MLA        = 3e-3
WEIGHT_DECAY  = 1e-5
GRAD_CLIP     = 1.0
LOG_EVERY     = 10
SAVE_EVERY    = 1000
WARMUP_STEPS  = 0           # set > 0 if you want a fresh warmup after resume

OUTPUT_DIR    = "./outputs/mixed_layer_attention_continued"
MODEL_ID      = "lerobot/pi0_libero_finetuned_v044"      # base architecture weights
RESUME_CKPT   = "./outputs/mixed_layer_attention_continued/checkpoint_008000/model.pt"
RESUME_STEP   = 7000      # step counter offset for logging/saving

# ── Config ─────────────────────────────────────────────────────────────────

print(f"[setup] Loading config from {MODEL_ID} ...")
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

# ── Model ──────────────────────────────────────────────────────────────────

print("[setup] Building model on CPU ...")
policy = PI0PolicyMixedLayerAttention(config)

# Step 1: load base pretrained weights (frozen backbone)
print("[setup] Loading base pretrained weights ...")
weights_path = hf_hub_download(MODEL_ID, "model.safetensors")
base_state   = load_file(weights_path, device="cpu")
remapped = {
    (k if k.startswith("model.") else f"model.{k}"): v
    for k, v in base_state.items()
}
missing, unexpected = policy.load_state_dict(remapped, strict=False)
print(f"[setup] Base weights — missing: {len(missing)}, unexpected: {len(unexpected)}")
del base_state, remapped
gc.collect()

# Step 2: overlay the MLA/LoRA checkpoint on top
print(f"[setup] Loading MLA/LoRA checkpoint from {RESUME_CKPT} ...")
ckpt = torch.load(RESUME_CKPT, map_location="cpu", weights_only=True)

# ckpt only contains trainable params saved as model.* keys
missing_ckpt, unexpected_ckpt = policy.model.load_state_dict(ckpt, strict=False)
print(f"[setup] Checkpoint overlay — missing: {len(missing_ckpt)}, unexpected: {len(unexpected_ckpt)}")
if missing_ckpt:
    print(f"  Missing: {missing_ckpt[:5]}")
if unexpected_ckpt:
    print(f"  Unexpected: {unexpected_ckpt[:5]}")
del ckpt
gc.collect()

print(f"[setup] Moving model to {DEVICE} ...")
config.device = DEVICE
policy = policy.to(DEVICE)
print(f"[setup] GPU mem after load: {torch.cuda.memory_allocated()/1e9:.1f}GB")

# ── Optimizer ──────────────────────────────────────────────────────────────

mla_param_ids = {id(p) for p in policy.model.mla.parameters()}
mla_params    = [p for p in policy.parameters() if p.requires_grad and id(p) in mla_param_ids]
lora_params   = [p for p in policy.parameters() if p.requires_grad and id(p) not in mla_param_ids]

print(f"[setup] MLA params:  {sum(p.numel() for p in mla_params)}")
print(f"[setup] LoRA params: {sum(p.numel() for p in lora_params)/1e6:.3f}M")

optimizer = AdamW(
    [
        {"params": mla_params,  "lr": LR_MLA},
        {"params": lora_params, "lr": LR},
    ],
    weight_decay=WEIGHT_DECAY,
)

if WARMUP_STEPS > 0:
    scheduler = LinearLR(
        optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=WARMUP_STEPS,
    )
else:
    scheduler = None

# ── Tokenizer + Dataset ────────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")

def tokenize_batch(batch, device):
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    tokens = tokenizer(
        [t + "\n" for t in batch["task"]],
        return_tensors="pt",
        padding="max_length",
        max_length=config.tokenizer_max_length,
        truncation=True,
    ).to(device)
    batch["observation.language.tokens"]         = tokens["input_ids"]
    batch["observation.language.attention_mask"] = tokens["attention_mask"].bool()
    return batch

dataset = LeRobotDataset(
    "lerobot/libero",
    delta_timestamps={"action": [i / 10 for i in range(50)]},
)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

print(f"[setup] Dataset: {len(dataset)} samples")

# ── Training loop ──────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)
trainable_names = {n for n, p in policy.model.named_parameters() if p.requires_grad}

policy.train()
optimizer.zero_grad()

global_step = RESUME_STEP   # offset so logs/saves reflect true total steps
local_step  = 0             # counts from 0 up to NUM_STEPS
data_iter   = iter(dataloader)
step_times  = []

print(f"[train] Resuming from step {RESUME_STEP}, running {NUM_STEPS} more steps")
print(f"[train] Effective batch size: {BATCH_SIZE * ACCUM_STEPS}")

while local_step < NUM_STEPS:
    t_step     = time.time()
    accum_loss = 0.0

    for _ in range(ACCUM_STEPS):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch     = next(data_iter)
            print(f"[train] Dataloader restarted at global step {global_step}")

        batch = tokenize_batch(batch, DEVICE)
        loss, _ = policy.forward(batch)
        (loss / ACCUM_STEPS).backward()
        accum_loss += loss.item() / ACCUM_STEPS

    torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], GRAD_CLIP
    )
    optimizer.step()
    optimizer.zero_grad()

    if scheduler is not None and local_step < WARMUP_STEPS:
        scheduler.step()

    step_times.append(time.time() - t_step)

    if local_step % LOG_EVERY == 0:
        avg_ms = (sum(step_times[-LOG_EVERY:]) / max(len(step_times[-LOG_EVERY:]), 1)) * 1000
        with torch.no_grad():
            logit_std = policy.model.mla.layer_logits[-1].float().std().item()
        print(
            f"[train] Step {global_step:06d} (+{local_step}) | "
            f"Loss: {accum_loss:.4f} | "
            f"LR lora: {optimizer.param_groups[1]['lr']:.2e} "
            f"MLA: {optimizer.param_groups[0]['lr']:.2e} | "
            f"LogitStd[17]: {logit_std:.4f} | "
            f"StepTime: {avg_ms:.0f}ms | "
            f"GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB"
        )

    if local_step % SAVE_EVERY == 0 and local_step > 0:
        ckpt_dir = f"{OUTPUT_DIR}/checkpoint_{global_step:06d}"
        os.makedirs(ckpt_dir, exist_ok=True)
        trainable_state = {k: v for k, v in policy.model.state_dict().items()
                           if k in trainable_names}
        torch.save(trainable_state, f"{ckpt_dir}/model.pt")
        print(f"[save] Checkpoint → {ckpt_dir} ({len(trainable_state)} tensors)")

    local_step  += 1
    global_step += 1

# ── Final save ─────────────────────────────────────────────────────────────

final_dir = f"{OUTPUT_DIR}/final"
os.makedirs(final_dir, exist_ok=True)
trainable_state = {k: v for k, v in policy.model.state_dict().items()
                   if k in trainable_names}
torch.save(trainable_state, f"{final_dir}/model.pt")
print(f"[done] Final checkpoint → {final_dir}")
