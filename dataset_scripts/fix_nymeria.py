import csv
import os

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm


if __name__ == '__main__':
    data_root = "/scratch/autolearn/aubret/nymeria/"
    fps=1
    resolution=512

    hdf5_file = h5py.File(os.path.join(data_root, f"data_fps{fps}_res{resolution}.h5"), "r")
    dataset = pd.read_csv(os.path.join(data_root, f"egodata_fps{fps}_res{resolution}.csv"))

    hdf5_file_w = h5py.File(os.path.join(data_root, f"datav2_fps{fps}_res{resolution}.h5"), "w")
    dataset_w = csv.writer(open(os.path.join(data_root, f"egodatav2_fps{fps}_res{resolution}.csv"), "w"))
    img_dataset = hdf5_file_w.create_dataset("data", shape=(len(dataset),), dtype=h5py.vlen_dtype(np.dtype('uint8')))
    dataset_w.writerow(["file_id", "device_time_ns", "index", "gaze_x", "gaze_y", "quatw", "quatx", "quaty", "quatz", "tx", "ty", "tz", 'new_index'])

    for index, row in tqdm(dataset.iterrows()):
        r = list(row.to_dict().values())
        r.append(index)
        recording, index2 = row.loc["file_id"], row.loc["index"]
        img_dataset[index] = hdf5_file.get(recording)[index2]
        dataset_w.writerow(r)

    hdf5_file_w.close()
    hdf5_file.close()
