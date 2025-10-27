from __future__ import annotations

import torch
from torch import nn, Tensor, stack, einsum, tensor
import torch.nn.functional as F
from torch.nn import Module
from torch.jit import ScriptModule, script_method
from torch.func import vmap, grad, functional_call

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
        actions,                 # (b n d)
        prev_structural = None   # (b n d) | (b d)
    ):
        batch = actions.shape[0]

        transitions = self.to_transitions(actions)
        transitions = self.mlp_out_to_weights(transitions)

        if exists(prev_structural) and prev_structural.ndim == 3:
            prev_structural = prev_structural[:, -1]

        return self.rnn(transitions, prev_structural)

# proposed mmTEM

class mmTEM(Module):
    def __init__(
        self,
        dim,
        *,
        sensory_encoder: Module,
        sensory_decoder: Module,
        dim_sensory,
        dim_action,
        dim_encoded_sensory,
        dim_structure,
        meta_mlp_depth = 2,
        decoder_mlp_depth = 2,
        structure_variance_pred_mlp_depth = 2,
        path_integrate_kwargs: dict = dict(),
        loss_weight_generative = 1.,
        loss_weight_inference = 1.,
        loss_weight_consistency = 1.,
        loss_weight_relational = 1.,
    ):
        super().__init__()

        # sensory

        self.sensory_encoder = sensory_encoder
        self.sensory_decoder = sensory_decoder

        dim_joint_rep = dim_encoded_sensory + dim_structure

        self.dim_encoded_sensory = dim_encoded_sensory
        self.dim_structure = dim_structure
        self.joint_dims = (dim_structure, dim_encoded_sensory)

        # path integrator

        self.path_integrator = PathIntegration(
            dim_action = dim_action,
            dim_structure = dim_structure,
            **path_integrate_kwargs
        )

        # meta mlp related

        self.to_queries = nn.Linear(dim_joint_rep, dim, bias = False)
        self.to_keys = nn.Linear(dim_joint_rep, dim, bias = False)
        self.to_values = nn.Linear(dim_joint_rep, dim, bias = False)

        self.meta_memory_mlp = create_mlp(
            dim = dim * 2,
            depth = meta_mlp_depth,
            dim_in = dim,
            dim_out = dim,
            activation = nn.ReLU()
        )

        # mlp decoder (from meta mlp output to joint)

        self.memory_output_decoder = create_mlp(
            dim = dim * 2,
            dim_in = dim,
            dim_out = dim_joint_rep,
            depth = decoder_mlp_depth,
            activation = nn.ReLU()
        )

        # the mlp that predicts the variance for the structural code
        # for correcting the generated structural code modeling the feedback from HC to MEC

        self.structure_variance_pred_mlp_depth = create_mlp(
            dim = dim_structure * 2,
            dim_in = dim_structure * 2 + 1,
            dim_out = dim_structure,
            depth = structure_variance_pred_mlp_depth
        )

        # loss related

        self.loss_weight_generative = loss_weight_generative
        self.loss_weight_inference = loss_weight_inference
        self.loss_weight_relational = loss_weight_relational
        self.loss_weight_consistency = loss_weight_consistency
        self.register_buffer('zero', tensor(0.), persistent = False)

    def forward(
        self,
        sensory,
        actions
    ):
        structural_codes = self.path_integrator(actions)

        # first have the structure code be able to fetch from the meta memory mlp

        structure_codes_with_zero_sensory = F.pad(structural_codes, (0, self.dim_encoded_sensory))

        queries = self.to_queries(structure_codes_with_zero_sensory)

        retrieved = self.meta_memory_mlp(queries)

        decoded_structure, decoded_encoded_sensory = self.memory_output_decoder(retrieved).split(self.joint_dims, dim = -1)

        decoded_sensory = self.sensory_decoder(decoded_encoded_sensory)

        generative_pred_loss = F.mse_loss(sensory, decoded_sensory)

        # losses

        total_loss = (
            generative_pred_loss * self.loss_weight_generative
        )

        losses = (
            generative_pred_loss,
        )

        return total_loss, losses
