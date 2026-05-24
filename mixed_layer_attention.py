# mixed_layer_attention.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MixedLayerAttention(nn.Module):
    """
    Learned weighted combination of PaliGemma hidden states across depth.
    
    For action expert layer i, mixes PaliGemma hiddens h_1..h_i with
    softmax-normalized weights. Only i weights exist for layer i —
    total parameters: sum(1..L) + L = 171 + 18 = 189.
    """
    def __init__(
        self,
        num_paligemma_layers: int = 18,
        num_action_expert_layers: int = 18,
    ):
        super().__init__()
        assert num_paligemma_layers == num_action_expert_layers
        L = num_paligemma_layers

        # Triangular list: layer_logits[i] has i+1 values (0-indexed)
        # Total: sum(1..18) = 171 logits + 18 gamma = 189 params
        self.layer_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(i + 1)) for i in range(L)
        ])
        self.gamma = nn.Parameter(torch.ones(L))

    def forward(self, paligemma_hiddens: list[Tensor], expert_layer_idx: int):
        i = expert_layer_idx
        if len(paligemma_hiddens) == 0:
            # Nothing to mix yet — return zeros matching expected shape
            # Will be gated by gamma[0] ≈ 1.0 initially, so just return zeros
            dummy = torch.zeros(1, device=self.gamma.device)  # MLA contributes nothing
            return dummy, dummy
        weights = F.softmax(self.layer_logits[i], dim=0)       # shape: (i+1,)
        stacked = torch.stack(paligemma_hiddens[:i + 1], dim=0) # shape: (i+1, B, T, D)
        mixed = (stacked * weights.view(i + 1, 1, 1, 1)).sum(dim=0)
        return self.gamma[i] * mixed, weights.detach()

    def get_layer_weights(self) -> list[Tensor]:
        return [F.softmax(logits, dim=0).detach().cpu() 
                for logits in self.layer_logits]
