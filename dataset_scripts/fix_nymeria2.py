import csv
import os

import argparse
import h5py
import numpy as np
import pandas as pd
import torchvision
from tqdm import tqdm

from PIL import Image
import numpy as np
import io


parser = argparse.ArgumentParser()
parser.add_argument('--data_root',default="/scratch/autolearn/aubret/nymeria/", type=str)
parser.add_argument('--fps',default=1, type=int)
parser.add_argument('--resolution',default=512, type=int)
parser.add_argument('--data_dest',default="", type=str)
args = parser.parse_args()

def encode_np_image(np_img: np.ndarray, format="JPEG") -> bytes:
    """
    Encode a NumPy image (HWC, uint8) into bytes (JPEG or PNG).
    """
    pil_img = Image.fromarray(np_img.astype(np.uint8))  # assumes RGB, dtype=uint8
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format)
    return buffer.getvalue()  # returns raw bytes


if __name__ == '__main__':
    data_root = args.data_root
    data_dest = args.data_dest
    if not data_dest:
        data_dest = data_root
    fps=args.fps
    resolution=args.resolution

    hdf5_file = h5py.File(os.path.join(data_root, f"data_fps{fps}_res{resolution}.h5"), "r")
    dataset = pd.read_csv(os.path.join(data_root, f"egodata_fps{fps}_res{resolution}.csv"))

    hdf5_file_w = h5py.File(os.path.join(data_dest, f"datav3_fps{fps}_res{resolution}.h5"), "w")
    dataset_w = csv.writer(open(os.path.join(data_dest, f"egodatav3_fps{fps}_res{resolution}.csv"), "w"))
    img_dataset = hdf5_file_w.create_dataset("data", shape=(len(dataset),), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    dataset_w.writerow(["file_id", "device_time_ns", "index", "gaze_x", "gaze_y", "quatw", "quatx", "quaty", "quatz", "tx", "ty", "tz", 'new_index'])
    adj_resolution = int(resolution*0.9)
    for index, row in tqdm(dataset.iterrows()):
        r = list(row.to_dict().values())
        r.append(index)
        recording, index2 = row.loc["file_id"], row.loc["index"]

        img = Image.fromarray(hdf5_file.get(recording)[int(index2)].reshape(resolution,resolution,3))
        img = torchvision.transforms.functional.center_crop(img, (adj_resolution, adj_resolution))
        img = np.array(img)

        img_bytes = encode_np_image(img, format="JPEG")
        img_dataset[index] = np.frombuffer(img_bytes, dtype=np.uint8)

        # img_dataset[index] = np.frombuffer(img_bytes, dtype=np.uint8)
        dataset_w.writerow(r)

    hdf5_file_w.close()
    hdf5_file.close()
