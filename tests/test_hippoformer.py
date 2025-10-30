import pytest

import torch

def test_path_integrate():
    from hippoformer.hippoformer import PathIntegration

    path_integrator = PathIntegration(32, 64)

    actions = torch.randn(2, 16, 32)

    structure_codes = path_integrator(actions)
    structure_codes = path_integrator(actions, structure_codes) # pass in previous structure codes, it will auto use the last one as hidden and pass it to the RNN

    assert structure_codes.shape == (2, 16, 64)

def test_mm_tem():
    import torch
    from hippoformer.hippoformer import mmTEM

    from torch.nn import Linear

    model = mmTEM(
        dim = 32,
        sensory_encoder = Linear(11, 32),
        sensory_decoder = Linear(32, 11),
        dim_sensory = 11,
        dim_action = 7,
        dim_structure = 32,
        dim_encoded_sensory = 32
    )

    actions = torch.randn(2, 16, 7)
    sensory = torch.randn(2, 16, 11)

    actions = torch.randn(2, 16, 7)
    sensory = torch.randn(2, 16, 11)

    next_params = model(sensory, actions, return_memory_mlp_params = True)
    next_params = model(sensory, actions, memory_mlp_params = next_params, return_memory_mlp_params = True)

    loss = model(sensory, actions, memory_mlp_params = next_params)
    loss.backward()
