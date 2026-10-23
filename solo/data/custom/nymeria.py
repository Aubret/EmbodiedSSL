import io
import os
import random

import h5py
import numpy as np
import pandas as pd
import scipy
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.multiprocessing as mp
from solo.data.cortical_magnification import radial_quad_isotrop_gridfun, img_cortical_magnif_tsr, \
    radial_exp_isotrop_gridfun
from solo.data.foveation import foveation


class Nymeria(Dataset):
    def __init__(self, data_root, transform,gaze_size=224, time_window=3, resolution=512, fps=5,  normalize=True,
                 mask=[], min_time_window=None, swap_actions=False,  delayed_actions=0, **kwargs):
        super().__init__()
        self.data_root = data_root
        self.transform = transform
        self.time_window = time_window
        self.gaze_size = gaze_size
        self.resolution = resolution
        self.normalize = normalize
        self.adj_resolution = int(self.resolution*0.9)
        self.mask = np.array(mask, dtype=bool)
        self.min_time_window = -time_window if min_time_window is None else min_time_window
        self.current_epoch = mp.Value('i', 0)
        self.swap_actions = swap_actions
        self.delayed_actions = delayed_actions

        v = f"v3"
        self.hdf5_file = h5py.File(os.path.join(self.data_root, f"data{v}_fps{fps}_res{resolution}.h5"), "r")
        self.dataset = pd.read_csv(os.path.join(self.data_root, f"egodata{v}_fps{fps}_res{resolution}.csv"))

        self.action_size = 9
        self.size = len(self.dataset)

        self.means = np.zeros((9,), dtype=np.float32)
        self.stds = np.array([50.698, 40.,0.30,0.14, 0.14, 0.30,0.22,0.57,0.63],dtype=np.float32)

        if self.swap_actions:
            shuffled_indices = np.random.permutation(len(self.dataset))
            self.shuffled_dataset = self.dataset.iloc[shuffled_indices].reset_index(drop=True)

        print("Length:", self.size)


    def __len__(self):
        return self.size

    def check_boundaries(self, adj_gaze_x, adj_gaze_y, gaze_size):
        if adj_gaze_x + gaze_size // 2 > self.adj_resolution:
            adj_gaze_x += - min(max(0, adj_gaze_x - gaze_size // 2), adj_gaze_x + gaze_size // 2 - self.adj_resolution)
        if adj_gaze_y + gaze_size // 2 > self.adj_resolution:
            adj_gaze_y += - min(max(0, adj_gaze_y - gaze_size // 2), adj_gaze_y + gaze_size // 2 - self.adj_resolution)
        if adj_gaze_x - gaze_size // 2 < 0:
            adj_gaze_x += min(max(0, adj_gaze_x + gaze_size // 2 - self.adj_resolution), - adj_gaze_x + gaze_size // 2)
        if adj_gaze_y - gaze_size // 2 < 0:
            adj_gaze_y += min(max(0, adj_gaze_y + gaze_size // 2 - self.adj_resolution), - adj_gaze_y + gaze_size // 2)
        return adj_gaze_x, adj_gaze_y

    def open_image(self, row, gaze_size):
        img = Image.open(io.BytesIO(self.hdf5_file.get("data")[int(row["new_index"])]))
        gap_resolution = self.resolution - self.adj_resolution

        adj_gaze_x, adj_gaze_y = row["gaze_x"], row["gaze_y"]
        #Because the images were cropped during preprocessing, we need to fix the original gaze positions
        adj_gaze_x, adj_gaze_y = adj_gaze_x - gap_resolution//2, adj_gaze_y - gap_resolution//2

        ### We control the boundaries of the gaze to not go beyond the image boundaries
        adj_gaze_x, adj_gaze_y = self.check_boundaries(adj_gaze_x, adj_gaze_y, gaze_size)
        img = torchvision.transforms.functional.crop(img, adj_gaze_y - gaze_size//2, adj_gaze_x - gaze_size//2, gaze_size,gaze_size)
        return img, (adj_gaze_x, adj_gaze_y)


    def get_action(self, row_before, row_after, gaze_before, gaze_after):
        bef_rot = row_before.loc[["quatx", "quaty", "quatz","quatw"]].values
        aft_rot = row_after.loc[["quatx", "quaty", "quatz","quatw"]].values

        bef_trans = row_before.loc[["tx", "ty", "tz"]].values
        aft_trans = row_after.loc[["tx", "ty", "tz"]].values

        assert np.abs(np.linalg.norm(bef_rot) - 1) < 0.05
        assert np.abs(np.linalg.norm(aft_rot) -1) < 0.05

        return self.get_action_v2(bef_rot, aft_rot, bef_trans, aft_trans, gaze_before, gaze_after)

    def compute_relative_pose(self, R1, t1, R2, t2):
        # Compute relative rotation
         # or np.dot(R2, R1.T)
        # Compute relative translation
        R_rel = R1.T @ R2
        t_rel = R1.T @ (t2 - t1)
        return R_rel, t_rel

    def get_action_v2(self, r1, r2, t1, t2, g1, g2):
        r1 = np.asarray(r1, dtype=np.float64)
        r2 = np.asarray(r2, dtype=np.float64)
        r1 = scipy.spatial.transform.Rotation.from_quat(r1).as_matrix()
        r2 = scipy.spatial.transform.Rotation.from_quat(r2).as_matrix()

        r_rel, t_rel = self.compute_relative_pose(r1, t1, r2, t2)

        gaze_movement = np.array(g2) - np.array(g1)
        r_rel = scipy.spatial.transform.Rotation.from_matrix(r_rel).as_quat()

        action = torch.tensor(np.concatenate((gaze_movement.astype(np.float32), r_rel.astype(np.float32), t_rel.astype(np.float32)), axis=0))
        action = (action - self.means)/self.stds
        if len(self.mask):
            action = action[self.mask]
        if self.current_epoch.value < self.delayed_actions:
            action[2:] = torch.randn((7,))
        return action


    def search(self, dataset, video_name, idx, min_time_window, max_time_window):
        new_video_name, new_idx, try_cpt, try_threshold = "", idx, 0, 10
        keep_searching=True
        while keep_searching:
            new_idx = idx + random.randint(min_time_window, max_time_window)
            while new_idx == idx:
                new_idx = idx + random.randint(min_time_window, max_time_window)
            new_idx = max(0,min(new_idx, self.size-1))
            if try_cpt > try_threshold:
                new_idx = idx
            rn = dataset.iloc[new_idx]
            new_video_name = rn["file_id"]

            if try_cpt > try_threshold:
                break
            if try_cpt % 10 == 0:
                min_time_window = min_time_window //2
                max_time_window = max_time_window //2

            try_cpt += 1
            keep_searching = (video_name != new_video_name)
        return rn, new_idx


    def __getitem__(self, idx):
        r = self.dataset.iloc[idx]
        video_name = r["file_id"]

        image, adj_gaze_before = self.open_image(r, self.gaze_size)

        if self.time_window == 0:
            return self.transform(image, image), -1

        rn, new_idx = self.search(self.dataset, video_name, idx, self.min_time_window, self.time_window)
        image_pair, adj_gaze_after = self.open_image(rn, self.gaze_size) if new_idx != idx else (image, adj_gaze_before)

        if self.swap_actions:
            r = self.shuffled_dataset.iloc[idx]
            rn, new_idx = self.search(self.shuffled_dataset, r["file_id"], idx, self.min_time_window, self.time_window)
            gap_resolution = self.resolution - self.adj_resolution
            adj_gaze_before = self.check_boundaries(r["gaze_x"] - gap_resolution//2, r["gaze_y"] - gap_resolution//2, g1)
            adj_gaze_after = self.check_boundaries(rn["gaze_x"] - gap_resolution//2, rn["gaze_y"] - gap_resolution//2, g2)


        action = self.get_action(r, rn, adj_gaze_before, adj_gaze_after)
        return self.transform(image, image_pair, None, None, action), -1


