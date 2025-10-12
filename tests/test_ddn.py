import pytest
param = pytest.mark.parametrize

import torch
from torch import tensor, nn

@param('use_mlp', (False, True))
@param('straight_through', (False, True))
def test_ddn(
    use_mlp,
    straight_through
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
        min_total_count_before_split_prune = 1,
        crossover_top2_prob = 1.,
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
