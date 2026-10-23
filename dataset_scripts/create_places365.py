from pathlib import Path

import h5py
import numpy as np
from torchvision.datasets import Places365
from tqdm import tqdm

if __name__ == '__main__':
    dataset_root = Path("/pfss/mlde/workspaces/mlde_wsp_PI_Roig/shared/datasets/")
    output_dir = dataset_root / "Places365" / "h5"
    output_dir.mkdir(exist_ok=True, parents=True)

    for split, t_split in [("train", "train-standard"), ("val", "val")]:
        with h5py.File(output_dir / f"{split}.h5", "w") as h5_file:
            dataset = Places365(
                dataset_root,
                split=t_split,
                download=False,
            )

            image_dataset = h5_file.create_dataset(f"images",
                                                   shape=(len(dataset),),
                                                   )
            target_dataset = h5_file.create_dataset(f"targets",
                                                    shape=(len(dataset),),
                                                    dtype=np.int32)

            bar = tqdm(range(len(dataset)), total=len(dataset), desc=f"{split} split")

            for i in bar:
                file, target = dataset.imgs[i]
                image_dataset[i] = np.fromfile(file, dtype=np.uint8)
                target_dataset[i] = target
