# pi0_policy_mixed_layer_attention.py

import math
import torch
import torch.nn as nn
from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch
from lerobot.policies.pi0 import PI0Policy
from mixed_layer_attention import MixedLayerAttention

PALIGEMMA_LAYERS = 18
ACTION_EXPERT_LAYERS = 18


class LoRALinear(nn.Module):
    """
    Low-rank adaptation of a frozen linear layer.
    output = frozen_linear(x) + (x @ A @ B)
    A: kaiming initialized, B: zero initialized.
    Both A and B are trainable — per projection that is
    in*rank + rank*out = 2 * in * rank params (when in == out).
    """
    def __init__(self, linear: nn.Linear, rank: int = 16):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
        in_dim = linear.in_features
        out_dim = linear.out_features
        self.A = nn.Parameter(torch.empty(in_dim, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.linear(x) + (x @ self.A @ self.B)


class PI0PytorchMixedLayerAttention(PI0Pytorch):
    """
    π0 with mixed-layer attention + LoRA on Q, K, V projections.

    Trainable parameters:
        MixedLayerAttention: 171 logits + 18 gamma = 189
        LoRA Q/K/V: 18 layers * 3 projections * (in*rank + rank*out)
            = 18 * 3 * 2 * (1024*16) = 1,769,472 ≈ 1.77M
            (assuming in == out == 1024; verify K dim at runtime)
        Total: ~1.77M + 189
    """
    def __init__(self, config, lora_rank: int = 16):
        super().__init__(config)
        self.mla = MixedLayerAttention(
            num_paligemma_layers=PALIGEMMA_LAYERS,
            num_action_expert_layers=ACTION_EXPERT_LAYERS,
        )
        self._freeze_all()
        self._apply_lora(lora_rank)

    def _freeze_all(self):
        for param in self.parameters():
            param.requires_grad_(False)
        for param in self.mla.parameters():
            param.requires_grad_(True)

    def _apply_lora(self, rank: int):
        expert_layers = self.paligemma_with_expert.gemma_expert.model.layers
        for layer in expert_layers:
            layer.self_attn.q_proj = LoRALinear(layer.self_attn.q_proj, rank=rank)
            layer.self_attn.k_proj = LoRALinear(layer.self_attn.k_proj, rank=rank)
            layer.self_attn.v_proj = LoRALinear(layer.self_attn.v_proj, rank=rank)

        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        mla_params = sum(p.numel() for p in self.mla.parameters())
        lora_params = trainable - mla_params

        print(f"Frozen:    {frozen/1e6:.1f}M")
        print(f"Trainable: {trainable/1e6:.2f}M")
        print(f"  → MixedLayerAttention: {mla_params} params")
        print(f"  → LoRA Q/K/V:          {lora_params/1e6:.2f}M params")

        # Print projection dimensions to verify K shape assumption
        k = expert_layers[0].self_attn.k_proj.linear
        q = expert_layers[0].self_attn.q_proj.linear
        v = expert_layers[0].self_attn.v_proj.linear
        print(f"\nProjection dimensions (layer 0):")
        print(f"  Q: {q.in_features} → {q.out_features}")
        print(f"  K: {k.in_features} → {k.out_features}")
        print(f"  V: {v.in_features} → {v.out_features}")


class PI0PolicyMixedLayerAttention(PI0Policy):
    """Drop-in replacement for PI0Policy using mixed-layer attention."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PI0PytorchMixedLayerAttention(config)
