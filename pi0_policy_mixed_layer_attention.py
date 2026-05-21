# pi0_policy_mixed_layer_attention.py

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from lerobot.policies.pi0.modeling_pi0 import (
    PI0Pytorch,
    make_att_2d_masks,
    layernorm_forward,
)
from lerobot.policies.pi0 import PI0Policy
from mixed_layer_attention import MixedLayerAttention

# Import internal dependencies from pi_gemma
from transformers.models.gemma import modeling_gemma
from lerobot.policies.pi_gemma import _gated_residual

PALIGEMMA_LAYERS = 18
ACTION_EXPERT_LAYERS = 18


class LoRALinear(nn.Module):
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


def compute_layer_complete_mla(
    layer_idx,
    inputs_embeds,
    attention_mask,
    position_ids,
    adarms_cond,
    paligemma,
    gemma_expert,
    mla,
    paligemma_hiddens,
):
    """
    Drop-in replacement for compute_layer_complete that injects
    mixed-layer attention into the action expert's K and V.

    Differences from the original:
    - After computing PaliGemma's hidden state for this layer,
      stores it in paligemma_hiddens (a list passed by reference)
    - Computes mixed context via mla(paligemma_hiddens, layer_idx)
    - Uses mixed context for action expert K and V instead of
      the raw PaliGemma hidden state at this depth

    Everything else — rotary embeddings, attention computation,
    MLP, residuals — is identical to the original.
    """
    models = [paligemma.model.language_model, gemma_expert.model]
    query_states = []
    key_states = []
    value_states = []
    gates = []

    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        hidden_states, gate = layernorm_forward(layer.input_layernorm, hidden_states, adarms_cond[i])
        gates.append(gate)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

        if i == 0:
            # PaliGemma stream — standard computation
            # Store pre-layernorm hidden state for MLA
            # Note: hidden_states here is post-layernorm; we store
            # the original input to this layer instead (inputs_embeds[0])
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        else:
            # Action expert stream — use mixed context for K and V
            # Q comes from the action expert's own hidden state (unchanged)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)

            # Compute mixed PaliGemma context up to this layer
            # paligemma_hiddens[layer_idx] was just stored by the i==0 branch above
            mixed_context, _ = mla(paligemma_hiddens, expert_layer_idx=layer_idx)
            # mixed_context: (B, T_prefix + T_suffix, 2048)
            # Slice to prefix length only — action expert only cross-attends to prefix
            prefix_len = inputs_embeds[0].shape[1]
            mixed_prefix = mixed_context[:, :prefix_len, :]

            # Project through action expert's K and V (these have LoRA adapters)
            mixed_hidden_shape = (*mixed_prefix.shape[:-1], -1, layer.self_attn.head_dim)
            key_state = layer.self_attn.k_proj(mixed_prefix).view(mixed_hidden_shape).transpose(1, 2)
            value_state = layer.self_attn.v_proj(mixed_prefix).view(mixed_hidden_shape).transpose(1, 2)

            # Pad key/value to full sequence length to match query length
            # The original code concatenates all Q/K/V and does joint attention
            # so we need K and V to cover the full sequence length
            # Action expert K/V need to cover suffix tokens too (self-attention part)
            suffix_hidden = hidden_states  # post-layernorm suffix hidden
            suffix_hidden_shape = (*suffix_hidden.shape[:-1], -1, layer.self_attn.head_dim)
            key_suffix = layer.self_attn.k_proj(suffix_hidden).view(suffix_hidden_shape).transpose(1, 2)
            value_suffix = layer.self_attn.v_proj(suffix_hidden).view(suffix_hidden_shape).transpose(1, 2)

            # Concatenate: mixed prefix KV + standard suffix KV
            key_state = torch.cat([key_state, key_suffix], dim=2)
            value_state = torch.cat([value_state, value_suffix], dim=2)

        query_states.append(query_state)
        key_states.append(key_state)
        value_states.append(value_state)

    # Store PaliGemma's output hidden state for this layer
    # We store inputs_embeds[0] (pre-layer input) so that layer i
    # stores what goes INTO layer i, giving us the full hierarchy
    paligemma_hiddens.append(inputs_embeds[0])

    # Everything below is identical to the original compute_layer_complete
    query_states = torch.cat(query_states, dim=2)
    key_states = torch.cat(key_states, dim=2)
    value_states = torch.cat(value_states, dim=2)

    dummy_tensor = torch.zeros(
        query_states.shape[0],
        query_states.shape[2],
        query_states.shape[-1],
        device=query_states.device,
        dtype=query_states.dtype,
    )
    cos, sin = paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
    query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
        query_states, key_states, cos, sin, unsqueeze_dim=1
    )

    batch_size = query_states.shape[0]
    scaling = paligemma.model.language_model.layers[layer_idx].self_attn.scaling

    att_output, _ = modeling_gemma.eager_attention_forward(
        paligemma.model.language_model.layers[layer_idx].self_attn,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling,
    )

    head_dim = paligemma.model.language_model.layers[layer_idx].self_attn.head_dim
    att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

    outputs_embeds = []
    start_pos = 0
    for i, hidden_states in enumerate(inputs_embeds):
        layer = models[i].layers[layer_idx]
        end_pos = start_pos + hidden_states.shape[1]
        if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
            att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
        out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])
        out_emb = _gated_residual(hidden_states, out_emb, gates[i])
        after_first_residual = out_emb.clone()
        out_emb, gate = layernorm_forward(layer.post_attention_layernorm, out_emb, adarms_cond[i])
        if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
            out_emb = out_emb.to(dtype=torch.bfloat16)
        out_emb = layer.mlp(out_emb)
        out_emb = _gated_residual(after_first_residual, out_emb, gate)
        outputs_embeds.append(out_emb)
        start_pos = end_pos

    return outputs_embeds


class PI0PytorchMixedLayerAttention(PI0Pytorch):
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

        k = expert_layers[0].self_attn.k_proj.linear
        q = expert_layers[0].self_attn.q_proj.linear
        v = expert_layers[0].self_attn.v_proj.linear
        print(f"\nProjection dimensions (layer 0):")
        print(f"  Q: {q.in_features} → {q.out_features}")
        print(f"  K: {k.in_features} → {k.out_features}")
        print(f"  V: {v.in_features} → {v.out_features}")

    def forward(
        self, images, img_masks, lang_tokens, lang_masks,
        state, actions, noise, time
    ) -> Tensor:
        """
        Training forward with mixed-layer attention.

        Replaces the standard joint forward pass with our custom
        compute_layer_complete_mla which injects mixed PaliGemma
        context into the action expert's K and V at each layer.
        """
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
            state, x_t, time
        )

        dtype = (
            self.paligemma_with_expert.paligemma.model.language_model
            .layers[0].self_attn.q_proj.weight.dtype
        )
        if dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        paligemma = self.paligemma_with_expert.paligemma
        gemma_expert = self.paligemma_with_expert.gemma_expert
        models = [paligemma.model.language_model, gemma_expert.model]
        num_layers = paligemma.config.text_config.num_hidden_layers

        # This list is populated layer by layer inside compute_layer_complete_mla
        # Layer i appends inputs_embeds[0] (PaliGemma input to layer i)
        # so that by the time layer i+1 runs, hiddens[0..i] are available
        paligemma_hiddens = []

        inputs_embeds = [prefix_embs, suffix_embs]
        for layer_idx in range(num_layers):
            inputs_embeds = compute_layer_complete_mla(
                layer_idx=layer_idx,
                inputs_embeds=inputs_embeds,
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                adarms_cond=[None, adarms_cond],
                paligemma=paligemma,
                gemma_expert=gemma_expert,
                mla=self.mla,
                paligemma_hiddens=paligemma_hiddens,
            )

        # Final layer norms
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            out_emb, _ = layernorm_forward(
                models[i].norm, hidden_states, [None, adarms_cond][i]
            )
            outputs_embeds.append(out_emb)

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")


class PI0PolicyMixedLayerAttention(PI0Policy):
    """Drop-in replacement for PI0Policy using mixed-layer attention."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.model = PI0PytorchMixedLayerAttention(config)
