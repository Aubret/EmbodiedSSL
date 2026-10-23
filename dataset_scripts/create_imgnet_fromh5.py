import io
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from torchvision.datasets import SUN397
from tqdm import tqdm
from sklearn.utils import shuffle
from PIL import Image

if __name__ == '__main__':
    dataset_root = Path("/scratch/autolearn/aubret/imgnet/")
    output_dir = dataset_root
    output_dir.mkdir(exist_ok=True, parents=True)



    data_dir = Path(dataset_root) / "data"
    dataset = pd.read_csv(data_dir / f'dataset2_train.csv', header=None)
    dataset.columns.astype(str)
    dataset.columns = ["index", "0", "1", "2", "label", "4"]
    train_size = len(dataset)
    train_h5 = h5py.File(output_dir / "train.h5", "w")

    train_images = train_h5.create_dataset(f"images",shape=(train_size,), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    train_targets = train_h5.create_dataset(f"targets",shape=(train_size,),dtype=np.int32)
    with h5py.File(str(data_dir / f'data2_train.h5'), "r") as h5:
        bar = tqdm(range(len(dataset)), total=len(dataset), desc=f"process")
        for i in bar:
            h5_index, target, begin = dataset.loc[i, "0"], dataset.loc[i, "label"], dataset.loc[i, '4']
            h5_index = h5_index % 50000
            img = h5.get(str(begin)).get("data")[h5_index]
            img = np.frombuffer(img, dtype=np.uint8)
            # print(img.shape, img.dtype)
            # print(img[0:8])
            # i = Image.open(io.BytesIO(img)).convert("RGB")
            # print(i.size)
            train_images[i] = img
            train_targets[i] = target

    train_h5.close()




