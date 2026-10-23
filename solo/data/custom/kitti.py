import csv
import io
import os
import random

import scipy
import h5py
import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from PIL import Image

#
# class Kitti(Dataset):
#     def __init__(self, data_root, transform, split, time_window=0, **kwargs):
#         super().__init__()
#         self.time_window = time_window
#         self.split = split
#         self.transform = transform
#
#         self.hdf5_file = h5py.File(os.path.join(data_root, "kitti.h5"), "r")
#         self.dataset = pd.read_parquet(os.path.join(data_root, f"kitti_mapper.parquet"))
#         print(len(self.dataset))
#
#     def __len__(self):
#         return len(self.dataset)
#
#
#     def __getitem__(self, idx):
#         # idx=530
#         index, dn = self.dataset.loc[idx, ["h5_index", "dirname"]]
#         image = Image.open(io.BytesIO(self.hdf5_file[index]))
#
#         if not self.time_window:
#             # return self.transform(image), label
#             return self.transform(image, image), label
#
#         new_idx = (idx + random.randint(-self.time_window, self.time_window))
#         new_index, new_dn = self.dataset.loc[new_idx, ["h5_index", "dirname"]]
#         while dn != new_dn:
#             new_idx = (idx + random.randint(-self.time_window, self.time_window))
#             new_index, new_dn = self.dataset.loc[new_idx, ["h5_index", "dirname"]]
#
#         new_image = Image.open(io.BytesIO(self.hdf5_file[new_index]))
#         return self.transform(image, new_image), -1
#

import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

class Kitti(Dataset):
    def __init__(self, data_root, transform, output_mode='error', time_window=5,
                 sequence_start_mode='all', N_seq=None, data_format='channels_first', **kwargs):
        """
        Args:
            data_file: Path to .hkl file with image data (n_images, H, W, C) by default.
            source_file: Path to .hkl file with source IDs for each frame.
            output_mode: 'error' (y=0) or 'prediction' (y=x).
            sequence_start_mode: 'all' (allow overlapping pairs) or 'unique' (non-overlapping).
            N_seq: Maximum number of pairs to include.
            data_format: 'channels_first' or 'channels_last' (Keras-style).
        """
        data_file, source_file = os.path.join(data_root, "X_train.hkl"), os.path.join(data_root, "sources_train.hkl")
        with h5py.File(data_file, 'r') as f:
            print(f.keys())
            self.X = np.array(f['data_0'])

        with h5py.File(source_file, 'r') as f:
            print(f.keys())
            self.sources = np.array(f['data_0'])
        # self.X = hkl.load(data_file)
        # self.sources = hkl.load(source_file)
        self.output_mode = output_mode
        self.transform = transform
        assert output_mode in {'error', 'prediction'}, 'output_mode must be in {error, prediction}'
        assert sequence_start_mode in {'all', 'unique'}, 'sequence_start_mode must be in {all, unique}'

        # Handle channel ordering (mimics original Keras logic)
        if data_format == 'channels_first':
            self.X = np.transpose(self.X, (0, 3, 1, 2))  # NHWC -> NCHW
        self.im_shape = self.X[0].shape  # (C, H, W) or (H, W, C)
        self.time_window = time_window
        # Find valid indices for adjacent pairs (same source)
        # if sequence_start_mode == 'all':
        #     self.possible_starts = np.array([
        #         i for i in range(self.X.shape[0] - 1)
        #         if self.sources[i] == self.sources[i + 1]
        #     ])
        # elif sequence_start_mode == 'unique':
        #     curr_location = 0
        #     possible_starts = []
        #     while curr_location < self.X.shape[0] - 1:
        #         if self.sources[curr_location] == self.sources[curr_location + 1]:
        #             possible_starts.append(curr_location)
        #             curr_location += 2  # Skip next frame to avoid overlap
        #         else:
        #             curr_location += 1
        #     self.possible_starts = possible_starts
        #
        # if N_seq is not None and len(self.possible_starts) > N_seq:
        #     self.possible_starts = self.possible_starts[:N_seq]
        print(len(self))

    def __len__(self):
        # return len(self.possible_starts)
        return len(self.X)

    def __getitem__(self, idx):
        # start_idx = self.possible_starts[idx]
        # Get adjacent pair: shape (2, ...) + im_shape

        new_idx = idx + random.randint(-self.time_window, self.time_window)
        while new_idx == idx or new_idx >= len(self) or self.sources[idx] != self.sources[new_idx] :
            new_idx = idx + random.randint(-self.time_window, self.time_window)

        p1, p2 = torch.from_numpy(self.X[idx]).to(torch.float) / 255,\
                torch.from_numpy(self.X[new_idx]).to(torch.float) / 255
        return self.transform(p1, p2, None, None, None), -1

    # def preprocess(self, X):
    #     return X.astype(np.float32) / 255