import pytest

import torch
from torch import tensor

def test_ddn():
    from discrete_distribution_network.ddn import GuidedSampler

    sampler = GuidedSampler(
        dim = 16,
        dim_query = 3,
        codebook_size = 10
    )

    features = torch.randn(2, 16, 32, 32)
    query_image = torch.randn(2, 3, 32, 32)

    out, codes, commit_loss = sampler(features, query_image)

    assert out.shape == query_image.shape
    assert codes.shape == (2,)
    assert commit_loss.numel() == 1

    sampler.split_and_prune_()

    # after much training

    assert sampler.forward_for_codes(features, tensor([3, 5])).shape == (2, 3, 32, 32)
    assert sampler.forward_for_codes(features, tensor(7)).shape == (2, 3, 32, 32)