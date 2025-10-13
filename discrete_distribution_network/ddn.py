from __future__ import annotations
from typing import Callable
from random import random

import torch
from torch import nn, arange, tensor, Tensor
import torch.nn.functional as F
from torch.nn import Module

from einops import rearrange, repeat, einsum, pack, unpack
from einops.layers.torch import Rearrange

from x_mlps_pytorch.ensemble import Ensemble

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def sample_prob(prob):
    return random() < prob

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def pack_one(t, pattern):
    packed, ps = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        unpacked, = unpack(out, ps, inv_pattern)
        return unpacked

    return packed, inverse

# classes

def split_and_prune(network: Module):
    # given some parent network, calls split and prune for all guided samplers

    for m in network.modules():
        if isinstance(m, GuidedSampler):
            m.split_and_prune_()

class GuidedSampler(Module):
    def __init__(
        self,
        dim,                            # input feature dimension
        dim_query = 3,                  # channels of image (default 3 for rgb)
        codebook_size = 10,             # K in paper
        network: Module | None = None,
        distance_fn: Callable | None = None,
        split_thres = 2.,
        prune_thres = 0.5,
        min_total_count_before_split_prune = 100,
        crossover_top2_prob = 0.,
        straight_through_distance_logits = False,
        stochastic = False,
        gumbel_noise_scale = 1.,
        patch_size = None,               # facilitate the future work where the guided sampler is done on patches
        transform: Callable[[Tensor], Tensor] | None = None
    ):
        super().__init__()

        if not exists(network):
            network = nn.Conv2d(dim, dim_query, 1, bias = False)

        self.codebook_size = codebook_size
        self.to_key_values = Ensemble(network, ensemble_size = codebook_size)
        self.distance_fn = default(distance_fn, torch.cdist)

        # split and prune related

        self.register_buffer('counts', torch.zeros(codebook_size).long())

        self.split_thres = split_thres / codebook_size
        self.prune_thres = prune_thres / codebook_size
        self.min_total_count_before_split_prune = min_total_count_before_split_prune

        # improvisations

        self.crossover_top2_prob = crossover_top2_prob

        self.stochastic = stochastic
        self.gumbel_noise_scale = gumbel_noise_scale
        self.straight_through_distance_logits = straight_through_distance_logits

        # acting on patches instead of whole image, mentioned by author

        self.patch_size = patch_size
        self.acts_on_patches = exists(patch_size)

        if self.acts_on_patches:
            self.image_to_patches = Rearrange('b c (h p1) (w p2) -> b h w c p1 p2', p1 = patch_size, p2 = patch_size)
            self.patches_to_image = Rearrange('b h w c p1 p2 -> b c (h p1) (w p2)')

        # transform features and query image into some other space
        # say vgg features

        self.transform = transform

    @torch.no_grad()
    def split_and_prune_(
        self
    ):
        # following Algorithm 1 in the paper

        counts = self.counts
        total_count = counts.sum()

        if total_count < self.min_total_count_before_split_prune:
            return

        top2_values, top2_indices = counts.topk(2, dim = -1)

        count_max, count_max_index = top2_values[0], top2_indices[0]
        count_min, count_min_index = counts.min(dim = -1)

        if (
            ((count_max / total_count) <= self.split_thres) &
            ((count_min / total_count) >= self.prune_thres)
        ).all():
            return

        codebook_params = self.to_key_values.param_values
        half_count_max = count_max // 2

        # update the counts

        self.counts[count_max_index] = half_count_max
        self.counts[count_min_index] = half_count_max + 1 # adds 1 to k_new for some reason

        # whether to crossover top 2

        should_crossover = sample_prob(self.crossover_top2_prob)

        # update the params

        for codebook_param in codebook_params:

            split = codebook_param[count_max]

            # whether to crossover
            if should_crossover:
                second_index = top2_indices[1]
                second_split = codebook_param[second_index]
                split = (split + second_split) / 2. # naive average for now

            # prune by replacement
            codebook_param[count_min].copy_(split)

            # take care of grad if present
            if exists(codebook_param.grad):
                codebook_param.grad[count_min].zero_()

    def forward_for_codes(
        self,
        features,      # (b d h w)
        codes          # (b) | ()
    ):
        batch = features.shape[0]

        # maybe transform

        if exists(self.transform):
            features = self.transform(features)

        # handle patches

        if self.acts_on_patches:

            if codes.numel() == 1:
                codes = repeat(codes, ' -> b', b = features.shape[0])

            features = self.image_to_patches(features)
            features, inverse_pack = pack_one(features, '* c h w')

            codes = repeat(codes, 'b h w -> (b h w)')

        # if one code, just forward the selected network for all features
        # else each batch is matched with the corresponding code

        if codes.numel() == 1:
            sel_key_values = self.to_key_values.forward_one(features, id = codes.item())
        else:
            sel_key_values =  self.to_key_values(features, ids = codes, each_batch_sample = True)

        # handle patches

        if self.acts_on_patches:
            sel_key_values = inverse_pack(sel_key_values)
            sel_key_values = self.patches_to_image(sel_key_values)

        return sel_key_values

    def forward(
        self,
        features,       # (b d h w)
        query,          # (b c h w)
    ):

        # maybe transform

        if exists(self.transform):
            features, query = map(self.transform, (features, query))

        # take care of maybe patching

        if self.acts_on_patches:
            assert features.ndim == 4 and query.ndim == 4


            features = self.image_to_patches(features)
            query = self.image_to_patches(query)

            features, _ = pack_one(features, '* c h w')
            query, inverse_pack = pack_one(query, '* c h w')

        # variables

        batch, device = query.shape[0], query.device

        key_values = self.to_key_values(features)

        # get the l2 distance

        distance = self.distance_fn(
            rearrange(query, 'b c h w -> b 1 (c h w)'),
            rearrange(key_values, 'k b c h w -> b k (c h w)')
        )

        distance = rearrange(distance, 'b 1 k -> b k')

        logits = -distance

        # allow for a bit of stochasticity

        if self.stochastic:
            logits = logits + gumbel_noise(logits) * self.gumbel_noise_scale

        # select the code parameters that produced the image that is closest to the query

        codes = logits.argmax(dim = -1)

        if self.training:
            self.counts.scatter_add_(0, codes, torch.ones_like(codes))

        # some tensor gymnastics to select out the image across batch

        if not self.straight_through_distance_logits:
            key_values = rearrange(key_values, 'k b ... -> b k ...')

            codes_for_indexing = rearrange(codes, 'b -> b 1')
            batch_for_indexing = arange(batch, device = device)[:, None]

            sel_key_values = key_values[batch_for_indexing, codes_for_indexing]
            sel_key_values = rearrange(sel_key_values, 'b 1 ... -> b ...')
        else:
            # variant treating the distance as attention logits

            attn = logits.softmax(dim = -1)
            one_hot = F.one_hot(codes, num_classes = self.codebook_size)

            st_one_hot = one_hot + attn - attn.detach()
            sel_key_values = einsum(key_values, st_one_hot, 'k b ..., b k -> b ...')

        # commit loss

        commit_loss = F.mse_loss(sel_key_values, query)

        # maybe reconstitute patch dimensions

        if self.acts_on_patches:
            sel_key_values = inverse_pack(sel_key_values, '* c p1 p2')
            sel_key_values = self.patches_to_image(sel_key_values)

            codes = inverse_pack(codes, '*')

        # return the chosen feature, the code indices, and commit loss

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
