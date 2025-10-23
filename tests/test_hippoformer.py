import pytest

import torch

def test_path_integrate():
    from hippoformer.hippoformer import PathIntegration

    path_integrator = PathIntegration(32, 64)

    actions = torch.randn(2, 16, 32)

    structure_codes = path_integrator(actions)

    assert structure_codes.shape == (2, 16, 64)
