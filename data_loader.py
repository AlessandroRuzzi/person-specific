import numpy as np
from utils import plot_images
from typing import List
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')
import h5py
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset, DataLoader
import os
import json
import csv
import pandas as pd
from skimage import io, transform
import math
import scipy.stats as ss
import random
from skimage import exposure
import cv2

# two changes:
# (1) pixel range is [0, 1] instead of [-1, 1]
# (2) shuffle the list

label_std = 1
range_angle_set = 76.0
range_angle = range_angle_set / 180.0 * math.pi  # the angle we set to be maxium

trans_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.RandomAffine(degrees=0, translate=(0.3, 0.3)),
        # transforms.Resize(224),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

trans = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize(224),
        transforms.ToTensor(),  # this also convert pixel value from [0,255] to [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_train_test_loader(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=False,
                           subject_id=0):
    # load dataset
    refer_list_file = os.path.join('data/train_test_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    all_subjects = datastore["test_specific"]
    # load dataset
    folder_path = 'data/subjects'
    file_path = os.path.join(folder_path, all_subjects[subject_id][:-3] +'_calibration.txt')
    train_set = GazeDataset(dataset_path="/data/aruzzi/person_specific", keys_to_use=datastore["test_specific"],
                            transform=trans, is_shuffle=is_shuffle, index_file=file_path, subject_id=subject_id, is_train=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    file_path = os.path.join(folder_path, all_subjects[subject_id][:-3] + '_test.txt')
    test_set = GazeDataset(dataset_path="/data/aruzzi/person_specific", keys_to_use=datastore["test_specific"],
                           transform=trans, is_shuffle=is_shuffle, index_file=file_path, subject_id=subject_id, is_train=False)
    test_loader = DataLoader(test_set, batch_size=20, num_workers=num_workers)  #batch_size could not be much bigger

    return (train_loader, test_loader)


class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, transform=None, is_shuffle=True,
                 index_file=None, subject_id=0, is_train=True):
        self.path = dataset_path
        self.hdfs = {}
        self.is_train = is_train

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [keys_to_use[subject_id]]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
            if is_train:
                file_path = os.path.join(self.path, self.selected_keys[num_i][:-3] + "_nsample_1_iter_0.h5")
            else:
                file_path = os.path.join(self.path, self.selected_keys[num_i])
            self.hdfs[num_i] = h5py.File(file_path, 'r', swmr=True)
            # print('read file: ', os.path.join(self.path, self.selected_keys[num_i]))
            assert self.hdfs[num_i].swmr_mode

        # Construct mapping from full-data index to key and person-specific index
        if index_file is None:
            self.idx_to_kv = []
            for num_i in range(0, len(self.selected_keys)):
                n = self.hdfs[num_i]["face_patch"].shape[0]
                self.idx_to_kv += [(num_i, i) for i in range(n)]
        else:
            print('load the file: ', index_file)
            if is_train:
                content = np.loadtxt(index_file, dtype=np.float)
                #self.idx_to_kv = content[:, 0].astype(np.int)
                self.idx_to_kv += [i for i in len(content)]
                self.gaze_labels_train = content[:, 1:3]
            else:
                self.idx_to_kv = np.loadtxt(index_file, dtype=np.int)

        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        # num_sample = len(self.idx_to_kv)
        # pick_index = [i for i in range(0, num_sample, 18)]
        # self.idx_to_kv = [self.idx_to_kv[index_i] for index_i in pick_index]
        if is_shuffle:
            random.shuffle(self.idx_to_kv)  # random the order to stable the training

        self.hdf = None
        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        for num_i in range(0, len(self.hdfs)):
            if self.hdfs[num_i]:
                self.hdfs[num_i].close()
                self.hdfs[num_i] = None

        if self.hdf:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        # # Change to expected format and values
        # # image = image[:, :, ::-1] - np.zeros_like(image) # BGR to RGB
        # image = image.transpose(2, 0, 1)
        # # image = image.astype(np.uint8)
        # image = image.astype(np.float32)
        # # image *= 1.0 / 255.0
        # # image -= 1.0

        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        # equalize the histogram of the Y channel
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        # convert the YUV image back to RGB format
        image = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        image = image.transpose(2, 0, 1)

        return image

    def __getitem__(self, idx):
        # for num_i in range(0, len(self.selected_keys)):
        #     if self.hdfs[num_i] is None:
        #         self.hdfs[num_i] = h5py.File(os.path.join(self.path, self.selected_keys[num_i]), 'r', swmr=True)
        #         assert self.hdfs[num_i].swmr_mode

        idx_current = self.idx_to_kv[idx]

        # if self.hdf is None:
        self.hdf = h5py.File(os.path.join(self.path, self.selected_keys[0]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        # Get face image
        pixels = self.hdf['face_patch'][idx_current, :]

        image = pixels
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB

        image = self.transform(image)

        # head = self.hdfs[key]['face_head_pose'][idx, :]

        if self.is_train:
            gaze_label = self.gaze_labels_train[idx, :]
        else:
            gaze_label = np.zeros((2))
        gaze_label = gaze_label.astype('float')

        return image, gaze_label
