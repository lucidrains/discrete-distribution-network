import pytest
param = pytest.mark.parametrize

import torch
from torch import tensor, nn

@param('use_mlp', (False, True))
@param('straight_through', (False, True))
@param('prenorm', (False, True))
@param('chain_dropout_prob', (0., 1.))
def test_ddn(
    use_mlp,
    straight_through,
    prenorm,
    chain_dropout_prob
):
    from discrete_distribution_network.ddn import GuidedSampler

    network = None
    if use_mlp:
        network = nn.Sequential(
            nn.Conv2d(16, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 3, 1)
        )

    sampler = GuidedSampler(
        dim = 16,
        dim_query = 3,
        codebook_size = 10,
        network = network,
        prenorm = prenorm,
        min_total_count_before_split_prune = 1,
        crossover_top2_prob = 1.,
        chain_dropout_prob = chain_dropout_prob,
        straight_through_distance_logits = straight_through
    )

    features = torch.randn(10, 16, 32, 32)
    query_image = torch.randn(10, 3, 32, 32)

    out, codes, commit_loss = sampler(features, query_image)

    assert out.shape == query_image.shape
    assert codes.shape == (10,)
    assert commit_loss.numel() == 1

    sampler.split_and_prune_()

    # after much training

    assert sampler.forward_for_codes(features[:3], tensor([3, 5, 2])).shape == (3, 3, 32, 32)
    assert sampler.forward_for_codes(features[:3], tensor(7)).shape == (3, 3, 32, 32)

@param('add_feature_residual', (False, True))
def test_patches(
    add_feature_residual
):
    from discrete_distribution_network.ddn import GuidedSampler

    sampler = GuidedSampler(
        dim = 16,
        dim_query = 3,
        codebook_size = 10,
        min_total_count_before_split_prune = 1,
        crossover_top2_prob = 1.,
        patch_size = 16
    )

    features = torch.randn(10, 16, 32, 32)
    query_image = torch.randn(10, 3, 32, 32)

    residual = None
    if add_feature_residual:
        residual = torch.randn(10, 3, 32, 32)

    out, codes, commit_loss = sampler(features, query_image, residual = residual)

    assert out.shape == query_image.shape
    assert codes.shape == (10, 2, 2) # (2, 2) since each batch has 4 patches
    assert commit_loss.numel() == 1

    sampler.split_and_prune_()

    # after much training

    assert sampler.forward_for_codes(features[:3], codes[:3]).shape == (3, 3, 32, 32)

def test_non_image():
    from discrete_distribution_network.ddn import GuidedSampler

    sampler = GuidedSampler(
        dim = 32,
        codebook_size = 10,
        network = nn.Linear(32, 16),
        norm_module = nn.RMSNorm(32),
        prenorm = True,
        min_total_count_before_split_prune = 1,
        crossover_top2_prob = 1.,
    )

    features = torch.randn(10, 32)
    query = torch.randn(10, 16)

    out, codes, commit_loss = sampler(features, query)

    assert out.shape == query.shape
    assert codes.shape == (10,)
    assert commit_loss.numel() == 1

    sampler.split_and_prune_()

    # after much training

    assert sampler.forward_for_codes(features[:3], codes[:3]).shape == (3, 16)

def test_ddn():
    from discrete_distribution_network.ddn import DDN

    ddn = DDN(
        dim = 32,
        image_size = 64
    )

    images = torch.randn(2, 3, 64, 64)
    loss = ddn(images)
    loss.backward()
    ddn.split_and_prune_()

    sampled = ddn.sample(batch_size = 1)

    assert sampled.shape == (1, 3, 64, 64)

