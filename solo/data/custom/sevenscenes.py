import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class SevenScenesDataset(Dataset):
    """Dataset loader for 7-Scenes"""

    def __init__(self, root_dir, scene, split='test', transform=None):
        self.root_dir = Path(root_dir) / scene
        self.split = split
        self.transform = transform

        # Load pose data
        self.samples = []
        split_file = self.root_dir / f"{split}.txt"

        with open(split_file, 'r') as f:
            for line in f:
                img_path, *pose_data = line.strip().split()
                pose = np.array([float(x) for x in pose_data])
                self.samples.append({
                    'image': self.root_dir / img_path,
                    'pose': pose  # [tx, ty, tz, qw, qx, qy, qz]
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['image']).convert('RGB')

        if self.transform:
            img = self.transform(img)

        pose = torch.from_numpy(sample['pose']).float()
        return img, pose