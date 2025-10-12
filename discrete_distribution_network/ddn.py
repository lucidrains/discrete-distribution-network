from __future__ import annotations
from typing import Callable

import torch
from torch import nn, arange
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, einsum

from x_mlps_pytorch.ensemble import Ensemble

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

# classes

class GuidedSampler(Module):
    def __init__(
        self,
        dim,                            # input feature dimension
        dim_query = 3,                  # channels of image (default 3 for rgb)
        codebook_size = 10,             # K in paper
        network: Module | None = None,
        split_thres = 2.,
        prune_thres = 0.5,
        distance_fn: Callable | None = None
    ):
        super().__init__()

        if not exists(network):
            network = nn.Conv2d(dim, dim_query, 1, bias = False)

        self.to_key_values = Ensemble(network, ensemble_size = codebook_size)
        self.distance_fn = default(distance_fn, torch.cdist)

        self.register_buffer('counts', torch.zeros(codebook_size).long())

    def split_and_prune_(
        self
    ):
        raise NotImplementedError

    def forward(
        self,
        features, # (b d h w)
        query     # (b c h w)
    ):
        batch, device = query.shape[0], query.device

        key_values = self.to_key_values(features)

        # get the l2 distance

        distance = self.distance_fn(
            rearrange(query, 'b c h w -> b 1 (c h w)'),
            rearrange(key_values, 'k b c h w -> b k (c h w)')
        )

        distance = rearrange(distance, 'b 1 k -> b k')

        # select the code parameters that produced the image that is closest to the query

        codes = distance.argmin(dim = -1)

        if self.training:
            self.counts.scatter_add_(0, codes, torch.ones_like(codes))

        # some tensor gymnastics to select out the image across batch

        key_values = rearrange(key_values, 'k b ... -> b k ...')

        codes_for_indexing = rearrange(codes, 'b -> b 1')
        batch_for_indexing = arange(batch, device = device)[:, None]

        sel_key_values = key_values[batch_for_indexing, codes_for_indexing]
        sel_key_values = rearrange(sel_key_values, 'b 1 ... -> b ...')

        # commit loss

        commit_loss = F.mse_loss(sel_key_values, query)

        return sel_key_values, codes, commit_loss

class Network(Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()

class DDN(Module):
    def __init__(
        self
    ):
        super().__init__()

# trainer

class Trainer(Module):
    def __init__(
        self,
    ):
        super().__init__()
