import io
from typing import Tuple, Callable, Optional
from pathlib import Path
import h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Core50(Dataset):
    def __init__(self,
                 h5_path: str,
                 backgrounds: Optional[Tuple[str, ...]] = None,
                 transform: Optional[Callable] = None,
                 return_bg: bool = False,
                 ):
        self.transform = transform
        self.return_bg = return_bg

        self.h5_file = h5py.File(h5_path, "r")

        avail_bgs = set(self.h5_file.keys())
        self.backgrounds = backgrounds if backgrounds is not None else avail_bgs

        df = []
        for bg in self.backgrounds:
            if bg not in avail_bgs:
                raise ValueError(f"Background class {bg} not found. Use {avail_bgs}.")
            for i in range(self.h5_file.get(bg).get("images").shape[0]):
                df.append({'h5_index': i, 'bg': bg})
        self.df = pd.DataFrame(df)
        print(f"Using {self.backgrounds} with {len(df)} images.")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        dp = self.df.iloc[idx]
        bg = self.h5_file.get(dp.bg)

        image = Image.fromarray(bg.get("images")[dp.h5_index])
        target = bg.get("targets")[dp.h5_index]

        if self.transform is not None:
            image = self.transform(image)

        if self.return_bg:
            bg_label = dp.bg
            return image, target, bg_label

        return image, target


class Core50ForBGClassification(Dataset):
    def __init__(self,
                 h5_path: str,
                 split: str,
                 transform: Optional[Callable] = None,
                 ):
        self.transform = transform
        print(h5_path, Path(h5_path).exists())

        self.h5_file = h5py.File(h5_path, "r")
        self.mapper = pd.read_parquet(Path(h5_path).parent / f"bg_per_instance_{split}.parquet")

    def __len__(self) -> int:
        return len(self.mapper)

    def __getitem__(self, idx: int) -> Tuple[Image.Image, int]:
        dp = self.mapper.iloc[idx]
        session = self.h5_file.get(dp.session)
        session_label = int(dp.session[1:]) - 1

        image = Image.fromarray(session.get("images")[dp.h5_index])

        if self.transform is not None:
            image = self.transform(image)

        return image, session_label
