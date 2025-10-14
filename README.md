<img src="./ddn.png" width="400px"></img>

## Discrete Distribution Network (wip)

Exploration into [Discrete Distribution Network](https://discrete-distribution-networks.github.io/), by Lei Yang out of Beijing

Besides the split-and-prune, may also throw in an option for crossover (mixing of top 2 nodes to replace the pruned)

## Install

```bash
$ discrete-distribution-network
```

## Usage

```python
import torch
from discrete_distribution_network.ddn import DDN

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
