
import os
from glob import glob

import PIL.Image
import torch
import torchvision

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

class SunRGBD(Dataset):

    def __init__(self, data_root, split, transform, transform_depth, max_depth=10000, **kwargs):
        self.max_depth = max_depth
        self.transform = transform
        self.transform_depth = transform_depth

        self.images = []
        self.depths = []
        # for data_source in ["kv1", "kv2","realsense","xtion"]:
        #     ds = os.path.join(data_root, data_source)
        #     for ds2 in sorted(os.listdir(ds)):
        #         if ds2.startswith("."):
        #             continue
        #         dataset = os.path.join(ds, ds2)
        #         if ds2 == "sun3ddata":
        #             for n in sorted(os.listdir(dataset)):
        #                 if img_dir.startswith("."):
        #                     continue
        #                 n2 = os.listdir(os.path.join(dataset, n))
        #                 for img_dir in sorted(os.listdir(n2)):
        #                     if img_dir.startswith("."):
        #                         continue
        #         else:
        #             for img_dir in sorted(os.listdir(dataset)):
        #                 if img_dir.startswith("."):
        #                     continue
        #                 scene_dir = os.path.join(dataset, img_dir)
        #                 images.append(glob(f"{scene_dir}/image/*")[0])
        #                 ## Get depth map file path from scene directory
        #                 depths.append(glob(f"{scene_dir}/depth_bfx/*")[0])

        for dirpath, dirnames, filenames in sorted(os.walk(data_root)):
            dir_depth = os.path.join(os.path.dirname(dirpath), "depth")
            if os.path.basename(dirpath) == "image":
                for filename in sorted(filenames):
                    self.images.append(os.path.join(dirpath, filename))
                for filename in sorted(os.listdir(dir_depth)):
                    self.depths.append(os.path.join(dir_depth, filename))


        # train_images, test_images, train_depths, test_depths = train_test_split(images, depths, test_size=0.25, random_state=0)
        # if split == "train":
        #     self.images = train_images
        #     self.depths = train_depths
        # else:
        #     self.images = test_images
        #     self.depths = test_depths

        print("Dataset size:", len(self))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        x = self.transform(PIL.Image.open(self.images[idx]))
        # d = torchvision.transforms.functional.to_tensor(PIL.Image.open(self.depths[idx]))
        d = self.transform_depth(PIL.Image.open(self.depths[idx]))
        d = torchvision.transforms.functional.pil_to_tensor(d)
        # d = torch.clamp(d.to(torch.float32), 0, self.max_depth) / self.max_depth
        d = torch.clamp(d.to(torch.float32), 0, self.max_depth) / 1000
        return x, d
