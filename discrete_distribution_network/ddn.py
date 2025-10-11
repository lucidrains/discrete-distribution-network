import torch
from torch.nn import Module

from einops import rearrange

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class GuidedSampler(Module):
    def __init__(
        self
    ):
        super().__init__()

    def split_and_prune_(
        self
    ):
        raise NotImplementedError
