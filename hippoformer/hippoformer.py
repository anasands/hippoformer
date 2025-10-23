import torch
from torch import nn, stack
from torch.nn import Module
import torch.nn.functional as F

from einops import einsum, repeat, rearrange
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

    def forward(
        self,
        actions # (b n d)
    ):
        batch = actions.shape[0]

        transitions = self.to_transitions(actions)
        transitions = self.mlp_out_to_weights(transitions)

        # do it the slow way for now

        structure_out = []

        structure = l2norm(self.init_structure)
        structure = repeat(structure, 'd -> b d', b = batch)

        for transition in transitions.unbind(dim = 1):

            structure = einsum(structure, transition, 'b i, b i j -> b j')
            structure = F.relu(structure)
            structure = l2norm(structure)

            structure_out.append(structure)

        return stack(structure_out, dim = 1)

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
