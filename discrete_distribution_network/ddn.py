from __future__ import annotations

from math import log2
from typing import Callable
from random import random
from collections import namedtuple

import torch
from torch import nn, arange, tensor, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, einsum, pack, unpack
from einops.layers.torch import Rearrange

from x_mlps_pytorch.ensemble import Ensemble

# constants

GuidedSamplerOutput = namedtuple('GuidedSamplerOutput', ('output', 'codes', 'commit_loss'))

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

def split_and_prune_(network: Module):
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
        patch_size = None               # facilitate the future work where the guided sampler is done on patches
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

            split = codebook_param[count_max_index]

            # whether to crossover
            if should_crossover:
                second_index = top2_indices[1]
                second_split = codebook_param[second_index]
                split = (split + second_split) / 2. # naive average for now

            # prune by replacement
            codebook_param[count_min_index].copy_(split)

            # take care of grad if present
            if exists(codebook_param.grad):
                codebook_param.grad[count_min_index].zero_()

    def forward_for_codes(
        self,
        features,      # (b d h w)
        codes          # (b) | ()
    ):
        batch = features.shape[0]

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
        return_distances = False
    ):

        # take care of maybe patching

        if self.acts_on_patches:
            features = self.image_to_patches(features)
            query = self.image_to_patches(query)

            features, _ = pack_one(features, '* c h w')
            query, inverse_pack = pack_one(query, '* c h w')

        # variables

        batch, device = query.shape[0], query.device

        key_values = self.to_key_values(features)

        # get the l2 distance

        distance = self.distance_fn(
            rearrange(query, 'b ... -> b 1 (...)'),
            rearrange(key_values, 'k b ... -> b k (...)')
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

        output = GuidedSamplerOutput(sel_key_values, codes, commit_loss)

        if not return_distances:
            return output

        return output, distance

# ddn

class ChanRMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * (self.gamma + 1.) * self.scale

class Block(Module):
    def __init__(
        self,
        dim,
        dim_out,
        dropout = 0.
    ):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = ChanRMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.contiguous()
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return self.dropout(x)

class ResnetBlock(Module):
    def __init__(
        self,
        dim,
        dim_out,
        dropout = 0.
    ):
        super().__init__()
        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h

class DDN(Module):
    def __init__(
        self,
        dim,
        dim_max = 1024,
        image_size = 256,
        channels = 3,
        codebook_size = 10,
        dropout = 0.1,
        guided_sampled_kwargs: dict = dict(),
    ):
        super().__init__()
        assert log2(image_size).is_integer()

        self.input_image_shape = (channels, image_size, image_size)

        # number of stages from 2x2 features

        stages = int(log2(image_size))

        self.num_stages = stages
        self.codebook_size = codebook_size

        # dimensions

        dim_mults = [2 ** stage for stage in range(stages)]

        dims = [min(dim_max, dim * dim_mult) for dim_mult in dim_mults]

        dim_first = dims[0]

        dim_pairs = tuple(zip(dims[:-1], dims[1:]))

        # initial 2x2 features

        self.init_features = nn.Parameter(torch.randn(dim_first, 2, 2) * 1e-2)

        # layers

        self.layers = ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(dim_pairs):

            has_prev_sampler_output = ind != 0

            prev_sampled_dim = channels if has_prev_sampler_output else 0

            upsampler = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                nn.Conv2d(dim_in + prev_sampled_dim, dim_out, 3, padding = 1)
            )

            resnet_block = ResnetBlock(dim_out, dim_out, dropout = dropout)

            guided_sampler = GuidedSampler(
                dim = dim_out,
                dim_query = channels,
                codebook_size = codebook_size,
                **guided_sampled_kwargs
            )

            self.layers.append(ModuleList([
                upsampler,
                resnet_block,
                guided_sampler
            ]))

    def split_and_prune_(self):
        split_and_prune_(self)

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(
        self,
        batch_size = None,
        codes = None  # (b stages)
    ):
        assert exists(batch_size) ^ exists(codes)

        # if only batch size sent in, random codes

        if not exists(codes):
            codes = torch.randint(0, self.codebook_size, (batch_size, self.num_stages), device = self.device)

        # init features

        features = repeat(self.init_features, '... -> b ...', b = batch_size)

        # sampled output of a stage

        sampled_output = None

        for (upsampler, resnet_block, guided_sampler), layer_codes in zip(self.layers, codes.unbind(dim = 1)):

            if exists(sampled_output):
                features = cat((sampled_output, features), dim = 1)

            features = upsampler(features)

            features = resnet_block(features)

            sampled_output = guided_sampler.forward_for_codes(features, layer_codes)

        # last sampled output 

        return sampled_output

    def forward(
        self,
        images
    ):
        assert images.shape[1:] == self.input_image_shape
        batch = images.shape[0]

        # init features

        features = repeat(self.init_features, '... -> b ...', b = batch)

        losses = []
        codes = []
        sampled_outputs = []

        for upsampler, resnet_block, guided_sampler in self.layers:

            if len(sampled_outputs) > 0:
                features = cat((sampled_outputs[-1], features), dim = 1)

            features = upsampler(features)

            features = resnet_block(features)

            # query image for guiding is just input images resized

            height, width = features.shape[-2:]
            query_images = F.interpolate(images, (height, width), mode = 'bilinear')

            # guided sampler

            sampled_output, layer_code, layer_loss = guided_sampler(features, query_images)

            # losses, codes, outputs

            sampled_outputs.append(sampled_output)
            codes.append(layer_code)
            losses.append(layer_loss)

        # losses summed across layers

        total_loss = sum(losses)
        return total_loss

# trainer

class Trainer(Module):
    def __init__(
        self,
    ):
        super().__init__()
