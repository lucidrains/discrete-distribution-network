from __future__ import annotations

import math
from math import log2
from typing import Callable
from pathlib import Path
from random import random
from collections import namedtuple

import torch
from torch import nn, arange, tensor, cat
import torch.nn.functional as F
from torch.nn import Module, ModuleList, Sequential

from einops import rearrange, repeat, einsum, pack, unpack
from einops.layers.torch import Rearrange, Reduce

from x_mlps_pytorch.ensemble import Ensemble

# constants

GuidedSamplerOutput = namedtuple('GuidedSamplerOutput', ('output', 'codes', 'commit_loss'))

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def divisible_by(num, den):
    return (num % den) == 0

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

# norms

class ChanRMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim = 1) * (self.gamma + 1.) * self.scale

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
        chain_dropout_prob = 0.05,
        split_thres = 2.,
        prune_thres = 0.5,
        network_activation: Module | None = None,
        prenorm = False,
        norm_module: Module | None = None,
        min_total_count_before_split_prune = 100,
        crossover_top2_prob = 0.,
        straight_through_distance_logits = False,
        stochastic = False,
        gumbel_noise_scale = 1.,
        patch_size = None               # facilitate the future work where the guided sampler is done on patches
    ):
        super().__init__()

        if prenorm and not exists(norm_module):
            norm_module = ChanRMSNorm(dim)

        self.norm = default(norm_module, nn.Identity())

        if not exists(network):
            network = nn.Conv2d(dim, dim_query, 1, bias = False)

            if exists(network_activation):
                network = Sequential(network, nn.Sigmoid())

        self.codebook_size = codebook_size
        self.to_key_values = Ensemble(network, ensemble_size = codebook_size)
        self.distance_fn = default(distance_fn, torch.cdist)

        # chain dropout

        self.chain_dropout_prob = chain_dropout_prob

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
        codes,         # (b) | ()
        residual = None
    ):
        batch = features.shape[0]

        features = self.norm(features)

        # handle patches

        if self.acts_on_patches:

            if codes.numel() == 1:
                codes = repeat(codes, ' -> b', b = features.shape[0])

            features = self.image_to_patches(features)
            features, inverse_pack = pack_one(features, '* c h w')

            if exists(residual):
                residual = self.image_to_patches(residual)
                residual, _ = pack_one(residual, '* c h w')

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

        if exists(residual):
            sel_key_values = sel_key_values + residual

        return sel_key_values

    def forward(
        self,
        features,       # (b d h w)
        query,          # (b c h w)
        return_distances = False,
        residual = None
    ):

        features = self.norm(features)

        # take care of maybe patching

        if self.acts_on_patches:
            features = self.image_to_patches(features)
            query = self.image_to_patches(query)

            features, _ = pack_one(features, '* c h w')
            query, inverse_pack = pack_one(query, '* c h w')

            if exists(residual):
                residual = self.image_to_patches(residual)
                residual, _ = pack_one(residual, '* c h w')

        # variables

        batch, device = query.shape[0], query.device

        key_values = self.to_key_values(features)

        # handle residual

        if exists(residual):
            key_values = key_values + residual

        # get the l2 distance

        distance = self.distance_fn(
            rearrange(query, 'b ... -> b 1 (...)'),
            rearrange(key_values, 'k b ... -> b k (...)')
        )

        distance = rearrange(distance, 'b 1 k -> b k')

        logits = -distance

        # allow for a bit of stochasticity

        if self.stochastic and self.training:
            logits = logits + gumbel_noise(logits) * self.gumbel_noise_scale

        # select the code parameters that produced the image that is closest to the query

        if self.training and sample_prob(self.chain_dropout_prob):
            # handle the chain dropout

            codes = torch.randint(0, self.codebook_size, (batch,), device = device)

        else:
            codes = logits.argmax(dim = -1)

            if self.training:
                self.counts.scatter_add_(0, codes, torch.ones_like(codes))

        # some tensor gymnastics to select out the image across batch

        if not self.straight_through_distance_logits or not self.training:
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

class Conv2dCroppedResidual(Module):
    # used in alphagenome

    def __init__(
        self,
        dim,
        dim_out,
        kernel_size,
        **kwargs
    ):
        super().__init__()
        assert dim >= dim_out
        self.pad = dim - dim_out
        self.conv = nn.Conv2d(dim, dim_out, kernel_size, **kwargs)

    def forward(self, x):
        residual, length = x, x.shape[1]
        return self.conv(x) + residual[:, :(length - self.pad)]

class SqueezeExcite(Module):
    def __init__(
        self,
        dim,
        squeeze_factor = 4.
    ):
        super().__init__()
        dim_squeezed = int(max(32, dim // squeeze_factor))

        self.squeeze = Sequential(
            Reduce('b c h w -> b c 1 1', 'mean'),
            nn.Conv2d(dim, dim_squeezed, 1),
            nn.ReLU(),
            nn.Conv2d(dim_squeezed, dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.squeeze(x)

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
        dim_out = None,
        dropout = 0.
    ):
        super().__init__()
        dim_out = default(dim_out, dim)

        self.block1 = Block(dim, dim_out, dropout = dropout)
        self.block2 = Block(dim_out, dim_out)
        self.squeeze_excite = SqueezeExcite(dim_out)

        self.layerscale = nn.Parameter(torch.randn(dim_out, 1, 1) * 1e-6)

    def forward(self, x):
        res = x
        h = self.block1(x)
        h = self.block2(h)
        h = self.squeeze_excite(h)
        return h * self.layerscale + res

class DDN(Module):
    def __init__(
        self,
        dim,
        dim_max = 1024,
        image_size = 256,
        channels = 3,
        codebook_size = 10,
        dropout = 0.,
        num_resnet_blocks = 2,
        guided_sampler_kwargs: dict = dict(),
    ):
        super().__init__()
        assert log2(image_size).is_integer()

        self.input_image_shape = (channels, image_size, image_size)

        # number of stages from 2x2 features

        stages = int(log2(image_size))

        self.num_stages = stages
        self.codebook_size = codebook_size

        # dimensions

        dim_mults = reversed([2 ** stage for stage in range(stages)])

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

            dim_in_with_maybe_prev = dim_in + prev_sampled_dim

            upsampler = nn.Sequential(
                nn.Upsample(scale_factor = 2, mode = 'bilinear'),
                Conv2dCroppedResidual(dim_in_with_maybe_prev, dim_out, 1)
            )

            resnet_block = nn.Sequential(*[
                ResnetBlock(dim_out, dropout = dropout) for _ in range(num_resnet_blocks)
            ])

            guided_sampler = GuidedSampler(
                dim = dim_out,
                dim_query = channels,
                codebook_size = codebook_size,
                **guided_sampler_kwargs
            )

            self.layers.append(ModuleList([
                upsampler,
                resnet_block,
                guided_sampler
            ]))


    def guided_sampler_codes_param_names(self):

        names = []

        for name, _ in self.named_parameters():
            sub_names = set(name.split('.'))

            if 'to_key_values' not in sub_names:
                continue

            names.append(name)

        return set(names)

    def split_and_prune_(self):
        split_and_prune_(self)

    @property
    def device(self):
        return next(self.parameters()).device

    def sample(
        self,
        batch_size = None,
        codes = None,  # (b stages)
        return_codes = False
    ):
        was_training = self.training
        self.eval()

        assert exists(batch_size) ^ exists(codes)

        # if only batch size sent in, random codes

        if not exists(codes):
            codes = torch.randint(0, self.codebook_size, (batch_size, self.num_stages), device = self.device)

        # init features

        features = repeat(self.init_features, '... -> b ...', b = batch_size)

        # sampled output of a stage

        sampled_output = None
        rgb_residual = None

        for (upsampler, resnet_block, guided_sampler), layer_codes in zip(self.layers, codes.unbind(dim = 1)):

            if exists(sampled_output):
                features = cat((sampled_output, features), dim = 1)

            features = upsampler(features)

            features = resnet_block(features)

            if exists(rgb_residual):
                height, width = features.shape[-2:]
                rgb_residual = F.interpolate(rgb_residual, (height, width), mode = 'bilinear')

            sampled_output = guided_sampler.forward_for_codes(features, layer_codes, residual = rgb_residual)

            rgb_residual = sampled_output

        self.train(was_training)

        # last sampled output 

        if not return_codes:
            return sampled_output

        return sampled_output, codes

    def forward(
        self,
        images,
        return_intermediates = False
    ):
        assert images.shape[1:] == self.input_image_shape
        batch = images.shape[0]

        # init features

        features = repeat(self.init_features, '... -> b ...', b = batch)

        losses = []
        codes = []
        sampled_outputs = []

        rgb_residual = None

        for upsampler, resnet_block, guided_sampler in self.layers:

            if len(sampled_outputs) > 0:
                features = cat((sampled_outputs[-1], features), dim = 1)

            features = upsampler(features)

            features = resnet_block(features)

            # query image for guiding is just input images resized

            height, width = features.shape[-2:]
            query_images = F.interpolate(images, (height, width), mode = 'bilinear')

            # handle rgb residual

            if exists(rgb_residual):
                rgb_residual = F.interpolate(rgb_residual, (height, width), mode = 'bilinear')

            # guided sampler

            sampled_output, layer_code, layer_loss = guided_sampler(features, query_images, residual = rgb_residual)

            rgb_residual = sampled_output

            # losses, codes, outputs

            sampled_outputs.append(sampled_output)
            codes.append(layer_code)
            losses.append(layer_loss)

        # losses summed across layers

        total_loss = sum(losses)

        if not return_intermediates:
            return total_loss

        codes = stack(codes, dim = -1)

        return total_loss, (codes, sampled_outputs)

# trainer

from torch.optim import AdamW
from accelerate import Accelerator
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import save_image
from torchvision.models import VGG16_Weights

from ema_pytorch import EMA

def cycle(dl):
    while True:
        for batch in dl:
            yield batch

class Trainer(Module):
    def __init__(
        self,
        ddn: dict | DDN,
        *,
        dataset: dict | Dataset,
        num_train_steps = 70_000,
        learning_rate = 3e-4,
        weight_decay = 1e-3,
        batch_size = 32,
        checkpoints_folder: str = './checkpoints',
        results_folder: str = './results',
        save_results_every: int = 100,
        checkpoint_every: int = 1000,
        num_samples: int = 16,
        adam_kwargs: dict = dict(),
        accelerate_kwargs: dict = dict(),
        ema_kwargs: dict = dict(),
        use_ema = True,
        max_grad_norm = 0.5
    ):
        super().__init__()
        self.accelerator = Accelerator(**accelerate_kwargs)

        if isinstance(dataset, dict):
            dataset = ImageDataset(**dataset)

        if isinstance(ddn, dict):
            ddn = DDN(**ddn)

        self.model = ddn

        self.use_ema = use_ema
        self.ema_model = None

        if self.is_main and use_ema:
            self.ema_model = EMA(
                self.model,
                forward_method_names = ('sample',),
                param_or_buffer_names_no_ema = ddn.guided_sampler_codes_param_names(),
                **ema_kwargs
            )

            self.ema_model.to(self.accelerator.device)

        # optimizer, dataloader, and all that

        self.optimizer = AdamW(self.model.parameters(), lr = learning_rate, weight_decay = weight_decay, **adam_kwargs)
        self.dl = DataLoader(dataset, batch_size = batch_size, shuffle = True, drop_last = True)

        self.model, self.optimizer, self.dl = self.accelerator.prepare(self.model, self.optimizer, self.dl)

        self.num_train_steps = num_train_steps

        # folders

        self.checkpoints_folder = Path(checkpoints_folder)
        self.results_folder = Path(results_folder)

        self.checkpoints_folder.mkdir(exist_ok = True, parents = True)
        self.results_folder.mkdir(exist_ok = True, parents = True)

        self.checkpoint_every = checkpoint_every
        self.save_results_every = save_results_every

        self.num_sample_rows = int(math.sqrt(num_samples))
        assert (self.num_sample_rows ** 2) == num_samples, f'{num_samples} must be a square'
        self.num_samples = num_samples

        assert self.checkpoints_folder.is_dir()
        assert self.results_folder.is_dir()

        self.max_grad_norm = max_grad_norm

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save(self, path):
        if not self.is_main:
            return

        save_package = dict(
            model = self.accelerator.unwrap_model(self.model).state_dict(),
            ema_model = self.ema_model.state_dict(),
            optimizer = self.optimizer.state_dict(),
        )

        torch.save(save_package, str(self.checkpoints_folder / path))

    def load(self, path):
        if not self.is_main:
            return
        
        load_package = torch.load(path)
        
        self.model.load_state_dict(load_package["model"])
        self.ema_model.load_state_dict(load_package["ema_model"])
        self.optimizer.load_state_dict(load_package["optimizer"])

    def log(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    def log_images(self, *args, **kwargs):
        return self.accelerator.log(*args, **kwargs)

    @torch.no_grad()
    def sample(self, fname):
        eval_model = default(self.ema_model, self.model)

        sampled = eval_model.sample(batch_size = self.num_samples)
      
        sampled = rearrange(sampled, '(row col) c h w -> c (row h) (col w)', row = self.num_sample_rows)
        sampled.clamp_(0., 1.)

        save_image(sampled, fname)
        return sampled

    def forward(self):

        dl = cycle(self.dl)

        for ind in range(self.num_train_steps):
            step = ind + 1

            self.model.train()

            data = next(dl)

            loss = self.model(data)
            self.log(dict(loss = loss.item()), step = step)

            self.accelerator.print(f'[{step}] loss: {loss.item():.3f}')
            self.accelerator.backward(loss)

            self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.model.split_and_prune_() # call split and prune after update

            if self.is_main and self.use_ema:
                self.ema_model.update()

            self.accelerator.wait_for_everyone()

            if self.is_main:

                if divisible_by(step, self.save_results_every):

                    sampled = self.sample(fname = str(self.results_folder / f'results.{step}.png'))

                    self.log_images(sampled, step = step)

                if divisible_by(step, self.checkpoint_every):
                    self.save(f'checkpoint.{step}.pt')

            self.accelerator.wait_for_everyone()

        print('training complete')
