import argparse
import csv
import os
import pandas as pd
from projectaria_tools.core import mps, data_provider
import json

from tqdm import tqdm



def str2table(v):
    return v.split(',')



def bisection_timestamp_search(timed_data, query_timestamp_ns: int) -> int:
    """
    Binary search helper function, assuming that timed_data is sorted by the field names 'tracking_timestamp'
    Returns index of the element closest to the query timestamp else returns None if not found (out of time range)
    """
    # Deal with border case
    if timed_data and len(timed_data) > 1:
        first_timestamp = timed_data[0].tracking_timestamp.total_seconds() * 1e9
        last_timestamp = timed_data[-1].tracking_timestamp.total_seconds() * 1e9
        if query_timestamp_ns <= first_timestamp:
            return None
        elif query_timestamp_ns >= last_timestamp:
            return None
    # If this is safe we perform the Bisection search
    start = 0
    end = len(timed_data) - 1
    while start < end:
        mid = (start + end) // 2
        mid_timestamp = timed_data[mid].tracking_timestamp.total_seconds() * 1e9
        if mid_timestamp == query_timestamp_ns:
            return mid
        if mid_timestamp < query_timestamp_ns:
            start = mid + 1
        else:
            end = mid - 1
    return start

def get_nearest_eye_gaze(eye_gazes, query_timestamp_ns):
    """
    Helper function to get nearest eye gaze for a timestamp (ns)
    Return the closest or equal timestamp eye_gaze information that can be found, returns None if not found (out of time range)
    """
    bisection_index = bisection_timestamp_search(eye_gazes, query_timestamp_ns)
    # print(eye_gazes, bisection_index, len(eye_gazes))
    if bisection_index is None:
        return None
    return eye_gazes[bisection_index]


def get_nearest_pose(
    mps_trajectory, query_timestamp_ns
):
    """
    Helper function to get nearest pose for a timestamp (ns)
    Return the closest or equal timestamp pose information that can be found, returns None if not found (out of time range)
    """
    bisection_index = bisection_timestamp_search(mps_trajectory, query_timestamp_ns)
    if bisection_index is None:
        return None
    return mps_trajectory[bisection_index]


parser = argparse.ArgumentParser()
parser.add_argument('--data_root',default='/home/fias/postdoc/datasets/nymeria/', type=str)
parser.add_argument('--id_files', default=[] , type=str2table)#"Nymeria_v0.0_20230607_s0_james_johnson_act0_e72nhq"
parser.add_argument('--id_exceptions', default=[] , type=str2table)#"Nymeria_v0.0_20230607_s0_james_johnson_act0_e72nhq"
parser.add_argument('--fps', default=5, type=int)
parser.add_argument('--resolution', default=512 , type=int)
parser.add_argument('--mode', default="w" , type=str)
args = parser.parse_args()


init_csv_file = pd.read_csv(os.path.join(args.data_root, f"egodatav3_fps{args.fps}_res{args.resolution}.csv"))
csv_file = csv.writer(open(os.path.join(args.data_root, f"egodatav4_fps{args.fps}_res{args.resolution}.csv"),args.mode))

if args.mode == "w":
    csv_file.writerow(["file_id", "device_time_ns", "index", "gaze_x", "gaze_y", "quatw", "quatx", "quaty", "quatz", "tx", "ty", "tz","new_index","cquatw", "cquatx", "cquaty", "cquatz", "ctx", "cty", "ctz"])

if args.id_files:
    list_files = args.id_files
else:
    json_list_sequence = json.load(open(os.path.join(args.data_root, "Nymeria_download_urls.json" ),"r"))
    list_files = ["Nymeria_v0.0_"+k for k in json_list_sequence["sequences"].keys()]

args.id_exceptions = ["Nymeria_v0.0_"+k for k in args.id_exceptions]
os.makedirs(os.path.join(args.data_root, "tests"), exist_ok=True)

all_index = 0
for id_file in tqdm(list_files):
    gaze_path = os.path.join(args.data_root,f"{id_file}_recording_head/recording_head/mps/eye_gaze/general_eye_gaze.csv")
    f = init_csv_file[init_csv_file["file_id"] == id_file]

    closed_loop_path = os.path.join(args.data_root,f"{id_file}_recording_head/recording_head/mps/slam/closed_loop_trajectory.csv")
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    vrs_camera = os.path.join(args.data_root, f"{id_file}_recording_head_data_data.vrs")
    vrs_data_provider = data_provider.create_vrs_data_provider(vrs_camera)
    stream_id = vrs_data_provider.get_stream_id_from_label("camera-rgb")

    stream_id = vrs_data_provider.get_stream_id_from_label("camera-rgb")

    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(stream_id)
    device_calibration = vrs_data_provider.get_device_calibration()
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    T_device_rgb_camera = rgb_camera_calibration.get_transform_device_camera()
    for _, row in f.iterrows():
        device_time_ns = row["device_time_ns"]
        pose_info = get_nearest_pose(closed_loop_traj, int(device_time_ns))

        T_world_rgb_camera = (pose_info.transform_world_device @ T_device_rgb_camera).to_quat_and_translation()
        csv_file.writerow(row.values.tolist() + list(T_world_rgb_camera[0]))
        all_index += 1



