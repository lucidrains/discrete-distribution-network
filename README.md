<img src="./ddn.png" width="400px"></img>

## Discrete Distribution Network

Exploration into [Discrete Distribution Network](https://discrete-distribution-networks.github.io/), by Lei Yang out of Beijing

Besides the split-and-prune, may also throw in an option for crossover (mixing of top 2 nodes to replace the pruned)

## Install

```bash
$ discrete-distribution-network
```

## Usage

```python
import torch
from discrete_distribution_network import DDN

ddn = DDN(
    dim = 32,
    image_size = 256
)

images = torch.randn(2, 3, 256, 256)

loss = ddn(images)
loss.backward()

# after much training

sampled = ddn.sample(batch_size = 1)

assert sampled.shape == (1, 3, 256, 256)
```

The proposed `GuidedSampler` in the paper

```python
import torch
from discrete_distribution_network import GuidedSampler

sampler = GuidedSampler(
    dim = 16,              # feature dimension
    dim_query = 3,         # the query image dimension
    codebook_size = 10,    # the codebook size K in the paper, which is K separate projections of the features, which is then measured distance wise to the query image guide
)

features = torch.randn(20, 16, 32, 32)
query_image = torch.randn(20, 3, 32, 32)

out, codes, commit_loss = sampler(features, query_image)

# (20, 3, 32, 32), (20,), ()

assert torch.allclose(sampler.forward_for_codes(features, codes), out, atol = 1e-5)

# after optimizer step, this needs to be called
# there is also a helper function by the same name that can take in your module and will invoke all of the guided samplers

sampler.split_and_prune_()
```

## Oxford flowers

Install `uv`, which will probably become the default in the near future

```shell
$ pip install uv
```

Then

```shell
$ uv run train_oxford_flowers.py
```

## Citations

```bibtex
@misc{yang2025discretedistributionnetworks,
    title   = {Discrete Distribution Networks}, 
    author  = {Lei Yang},
    year    = {2025},
    eprint  = {2401.00036},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url     = {https://arxiv.org/abs/2401.00036}, 
}
```
