import sys
import glob
import pandas as pd
import re
from pathlib import Path
import numpy as np

WORKSPACE_PATH = "/root/workspace/"

def format_pose_files():
    PATH = WORKSPACE_PATH + 'json/*.json'
    files = glob.glob(PATH, recursive=True)

    df = pd.DataFrame(
    filename = ''
    list_data = []
    for f in track(files):
        filename = re.search(r'/root/workspace/json/.+_(\d+)_keypoints.json', f)
        with open(Path(filename.group(0))) as json_file:
            json_data = json.load(json_file)

        for p in json_data['people']:
            data_pose = np.array(p['pose_keypoints_2d'])
            data_pose = np.insert(data_pose, 0, p['person_id'])
            data_pose = np.insert(data_pose, 0, filename.group(1))
            list_data.append(data_pose)

    df = pd.DataFrame(list_data, columns=['frame','person_id',
                                          'X_0', 'Y_0', 'P_0', 'X_1', 'Y_1', 'P_1',
                                          'X_2', 'Y_2', 'P_2', 'X_3', 'Y_3', 'P_3',
                                          'X_4', 'Y_4', 'P_4', 'X_5', 'Y_5', 'P_5',
                                          'X_6', 'Y_6', 'P_6', 'X_7', 'Y_7', 'P_7',
                                          'X_8', 'Y_8', 'P_8', 'X_9', 'Y_9', 'P_9',
                                          'X_10', 'Y_10', 'P_10', 'X_11', 'Y_11', 'P_11', 
                                          'X_12', 'Y_12', 'P_12', 'X_13', 'Y_13', 'P_13',
                                          'X_14', 'Y_14', 'P_14', 'X_15', 'Y_15', 'P_15',
                                          'X_16', 'Y_16', 'P_16', 'X_17', 'Y_17', 'P_17'])
    df = df.sort_values(by='frame')
    df = df.reset_index(drop=True)
    df.to_pickle(Path(WORKSPACE_PATH + '.pkl'))
    df.to_csv(Path(WORKSPACE_PATH + '.csv'))

if __main__ == "__main__":
    format_pose_files()

