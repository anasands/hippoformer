import torch
from torch import nn

from einops import rearrange

from x_mlps_pytorch import MLP

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

