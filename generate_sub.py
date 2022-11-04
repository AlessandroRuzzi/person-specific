import json
import numpy as np
import configparser
import os
import h5py
import cv2
import math


data_dir = '/disks/data1/xucong/eth_xgaze'

refer_list_file = os.path.join(data_dir, 'train_split.json')
print('load the train file list from: ', refer_list_file)

with open(refer_list_file, 'r') as f:
    datastore = json.load(f)

keys_to_use = datastore["test_s"]
selected_keys = [k for k in keys_to_use]
assert len(selected_keys) > 0


for num_i in range(0, len(selected_keys)):  #
    train_list = []
    test_list = []
    file_path = os.path.join(data_dir, selected_keys[num_i])
    hdfs = h5py.File(file_path, 'r', swmr=True)

    num_sample = hdfs["face_patch"].shape[0]
    index_picked = np.random.choice(num_sample, 200, replace=False)
    num_s = 0
    while num_s < num_sample:
        if num_s in index_picked:
            train_list.append([num_i, num_s])
        else:
            test_list.append([num_i, num_s])
        num_s = num_s + 1
    hdfs.close()

    train_list = np.asarray(train_list)
    np.savetxt('sub_list/calibration_'+str(num_i)+'.txt', train_list, fmt='%d')
    test_list = np.asarray(test_list)
    np.savetxt('sub_list/test_' + str(num_i) + '.txt', test_list, fmt='%d')

