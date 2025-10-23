from __future__ import annotations

import torch
from torch import nn, Tensor, stack, einsum
import torch.nn.functional as F
from torch.nn import Module
from torch.jit import ScriptModule, script_method

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from x_mlps_pytorch import create_mlp

from assoc_scan import AssocScan

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def l2norm(t):
    return F.normalize(t, dim = -1)

# path integration

class RNN(ScriptModule):
    def __init__(
        self,
        dim,
    ):
        super().__init__()
        self.init_hidden = nn.Parameter(torch.randn(1, dim) * 1e-2)

    @script_method
    def forward(
        self,
        transitions: Tensor,
        hidden: Tensor | None = None
    ) -> Tensor:

        batch, seq_len = transitions.shape[:2]

        if hidden is None:
            hidden = l2norm(self.init_hidden)
            hidden = hidden.expand(batch, -1)

        hiddens: list[Tensor] = []

        for i in range(seq_len):
            transition = transitions[:, i]

            hidden = einsum('b i, b i j -> b j', hidden, transition)
            hidden = F.relu(hidden)
            hidden = l2norm(hidden)

            hiddens.append(hidden)

        return stack(hiddens, dim = 1)

class PathIntegration(Module):
    def __init__(
        self,
        dim_action,
        dim_structure,
        mlp_hidden_dim = None,
        mlp_depth = 2
    ):
        # they use the same approach from Ruiqi Gao's paper from 2021
        super().__init__()

        self.init_structure = nn.Parameter(torch.randn(dim_structure))

        self.to_transitions = create_mlp(
            default(mlp_hidden_dim,  dim_action * 4),
            dim_in = dim_action,
            dim_out = dim_structure * dim_structure,
            depth = mlp_depth
        )

        self.mlp_out_to_weights = Rearrange('... (i j) -> ... i j', j = dim_structure)

        self.rnn = RNN(dim_structure)

    def forward(
        self,
        actions # (b n d)
    ):
        batch = actions.shape[0]

        transitions = self.to_transitions(actions)
        transitions = self.mlp_out_to_weights(transitions)

        return self.rnn(transitions)

# proposed mmTEM

class mmTEM(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

    def forward(
        self,
        data
    ):
        raise NotImplementedError
