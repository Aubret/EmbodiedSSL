import numpy as np
import torch


class ProbabilisticDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_a, dataset_b, p=0.3):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.p = p
        self.subsample_size = p * len(dataset_a) / (1 - self.p)
        np.random.seed(42)
        self.subsample = np.random.permutation(len(self.dataset_b))[:self.subsample_size]

    def __len__(self):
        return self.subsample_size + len(self.dataset_a) #max(len(self.dataset_a), len(self.dataset_b))

    def __getitem__(self, idx):
        if torch.rand(1) < self.p_a:
            j = torch.randint(len(self.dataset_a), (1,)).item()
            return self.dataset_a[j]
        else:
            j = torch.randint(len(self.dataset_b), (1,)).item()
            return self.dataset_b[j]