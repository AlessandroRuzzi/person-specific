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
                           is_shuffle=True):

    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore["train"], transform=trans, is_shuffle=is_shuffle)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore["test_1"],
                           transform=trans, is_shuffle=is_shuffle) # , index_file='../ggd_test_list.txt'
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return (train_loader, test_loader)


def get_train_test_loader_rtgene(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):

    # load dataset
    refer_list_file = os.path.join(data_dir, 'train_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    train_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore["train"],
                            transform=trans, is_shuffle=is_shuffle)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    test_set = GazeDataset(dataset_path=data_dir, keys_to_use=datastore["test"],
                           transform=trans, is_shuffle=is_shuffle)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return (train_loader, test_loader)


def get_test_loader_gazecapture(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'gazecapture_split.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    data_list_file = os.path.join(data_dir, 'face_zhang.h5')

    train_set = GazeDataset_gazecapture(dataset_path=data_list_file, keys_to_use=datastore["train"], transform=trans)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    test_set = GazeDataset_gazecapture(dataset_path=data_list_file, keys_to_use=datastore["test"], transform=trans)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return (train_loader, test_loader)

def get_test_loader_mpiigaze(data_dir,
                           batch_size,
                           num_workers=4,
                           is_shuffle=True):
    # load dataset
    refer_list_file = os.path.join(data_dir, 'all.json')
    print('load the train file list from: ', refer_list_file)

    with open(refer_list_file, 'r') as f:
        datastore = json.load(f)

    data_list_file = os.path.join(data_dir, 'face_zhang.h5')

    train_set = GazeDataset_gazecapture(dataset_path=data_list_file, keys_to_use=datastore["train"], transform=trans)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=num_workers)

    test_set = GazeDataset_gazecapture(dataset_path=data_list_file, keys_to_use=datastore["test"], transform=trans)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)

    return (train_loader, test_loader)

class GazeDataset_gazecapture(Dataset):

    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, transform=None):
        self.path = dataset_path
        self.hdf = None
        hdf = h5py.File(self.path, 'r')

        # Grab all available keys
        all_keys = list(hdf.keys())
        if keys_to_use is None:
            keys_to_use = all_keys
        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use if k in all_keys]
        assert len(self.selected_keys) > 0

        # Construct mapping from full-data index to key and person-specific index
        self.idx_to_kv = []
        for person_id in self.selected_keys:
            n = hdf[person_id]["images"].shape[0]
            self.idx_to_kv += [(person_id, i) for i in range(n)]

        random.shuffle(self.idx_to_kv) # random the order to stable the training

        self.transform = transform

    def __len__(self):
        return len(self.idx_to_kv)

    def __del__(self):
        if self.hdf:
            self.hdf.close()
            self.hdf = None

    def preprocess_image(self, image):
        # Change to expected format and values
        # image = image[:, :, ::-1] - np.zeros_like(image) # BGR to RGB
        image = image.transpose(2, 0, 1)
        # image = image.astype(np.uint8)
        # image = image.astype(np.float32)
        # image *= 1.0 / 255.0
        # image -= 1.0
        return image

    def __getitem__(self, idx):
        self.hdf = h5py.File(self.path, 'r')

        key, idx = self.idx_to_kv[idx]

        # Get face (448x448) image
        pixels = np.copy(self.hdf[key]['images'][idx, :])

        # Get labels and other data
        vals = np.copy(self.hdf[key]['metas'][idx, :])

        # entry = {
        #     'face': self.preprocess_image(pixels),
        #     'head_pose': vals[:2],
        #     'gaze_direction': vals[2:4],
        #     # 'gaze_origin': vals[4:7],
        #     # 'gaze_target': vals[7:10],
        #     # 'inverse_M': vals[10:19],
        #     # 'left_eye_gaze': vals[19:21],
        #     # 'right_eye_gaze': vals[21:23],
        # }

        # image = self.preprocess_image(pixels)
        image = pixels
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB

        image = self.transform(image)

        gaze_label = vals[2:4]
        gaze_label = gaze_label.astype('float')

        head_pose = vals[:2]
        head_pose = head_pose.astype('float')

        return image, gaze_label, head_pose

class GazeDataset(Dataset):
    def __init__(self, dataset_path: str, keys_to_use: List[str] = None, transform=None, is_shuffle=True,
                 index_file=None):
        self.path = dataset_path
        self.hdfs = {}

        # assert len(set(keys_to_use) - set(all_keys)) == 0
        # Select keys
        # TODO: select only people with sufficient entries?
        self.selected_keys = [k for k in keys_to_use]
        assert len(self.selected_keys) > 0

        for num_i in range(0, len(self.selected_keys)):
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

        key, idx = self.idx_to_kv[idx]

        self.hdf = h5py.File(os.path.join(self.path, self.selected_keys[key]), 'r', swmr=True)
        assert self.hdf.swmr_mode

        # Get face (448x448) image
        pixels = self.hdf['face_patch'][idx, :]

        # Get labels and other data
        vals = self.hdf['face_gaze'][idx, :]

        # entry = {
        #     'face': self.preprocess_image(pixels),
        #     'head_pose': vals[:2],
        #     'gaze_direction': vals[2:4],
        #     # 'gaze_origin': vals[4:7],
        #     # 'gaze_target': vals[7:10],
        #     # 'inverse_M': vals[10:19],
        #     # 'left_eye_gaze': vals[19:21],
        #     # 'right_eye_gaze': vals[21:23],
        # }

        image = pixels
        image = image[:, :, [2, 1, 0]]  # from BGR to RGB

        image = self.transform(image)

        head = self.hdf['face_head_pose'][idx, :]

        gaze_label = vals
        gaze_label = gaze_label.astype('float')

        return image, gaze_label, head

# class GazeDataset(Dataset):
#     """Gaze estimation dataset."""
#     def __init__(self, input_file, transform=None):
#         """
#         Args:
#             input_file (string): Path to the csv file with annotations.
#             transform (callable, optional): Optional transform to be applied
#                 on a sample.
#         """
#         self.content = pd.read_csv(input_file, header=None)
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.content)
#
#     def __getitem__(self, idx):
#         img_name = self.content.iloc[idx, 0]
#         image = io.imread(img_name)
#         gaze_label = self.content.iloc[idx, 1:].values
#         gaze_label = gaze_label.astype('float')
#
#         if self.transform:
#             image = self.transform(image)
#
#         return image, gaze_label