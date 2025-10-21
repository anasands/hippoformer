import torch
from torch import nn

from einops import rearrange

from x_mlps_pytorch import MLP

from assoc_scan import AssocScan

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

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
