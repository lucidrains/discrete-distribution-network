# /// script
# dependencies = [
#   "discrete-distribution-network",
#   "datasets",
#   "wandb"
# ]
# ///

import torch

# hf datasets for easy oxford flowers training

import torchvision.transforms as T
from torch.utils.data import Dataset
from datasets import load_dataset

IMAGE_SIZE = 64

class OxfordFlowersDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        self.ds = load_dataset('nelorth/oxford-flowers')['train']

        self.transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.Resize((image_size, image_size)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        tensor = self.transform(pil)
        return tensor / 255.

flowers_dataset = OxfordFlowersDataset(
    image_size = IMAGE_SIZE
)

# models and trainer

from discrete_distribution_network import DDN, Trainer

ddn = DDN(
    dim = 64,
    image_size = IMAGE_SIZE,
    dropout = 0.05,
    guided_sampler_kwargs = dict(
        crossover_top2_prob = 0.1,
        straight_through_distance_logits = True
    )
)

trainer = Trainer(
    ddn,
    dataset = flowers_dataset,
    num_train_steps = 70_000,
    use_ema = True,
    batch_size = 32,
    results_folder = './results'   # samples will be saved periodically to this folder
)

trainer()
