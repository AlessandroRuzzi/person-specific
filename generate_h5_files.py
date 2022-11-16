from h5_dataloader import get_train_loader
from new_model import gaze_net
import torch
import os
import h5py
import numpy as np
import json

with open("data/train_test_split.json", "r") as f:
        datastore = json.load(f)

train_keys = datastore["train"]

if torch.cuda.is_available():
    use_gpu = True
    device = "cuda:0" 
else:
    use_gpu = False
    device = "cpu"

model =  gaze_net()
if use_gpu:
    model.cuda()

print("[*] Loading model from {}".format('ckpt/epoch_24_resnet_correct_ckpt.pth.tar'))

ckpt_path = 'ckpt/epoch_24_resnet_correct_ckpt.pth.tar'

print('load the pre-trained model: ', ckpt_path)
ckpt = torch.load(ckpt_path)

# load variables from checkpoint
model.load_state_dict(ckpt['model_state'], strict=False)

print(
    "[*] Loaded {} checkpoint @ epoch {}".format(
        ckpt_path, ckpt['epoch'])
)

model.eval()

for subjects in train_keys:

    train_data_loader = get_train_loader(
            "/data/aruzzi/xgaze_subjects", 1, 0, evaluate="landmark", is_shuffle=False, subject= subjects
        )

    print(len(train_data_loader.dataset))

    output_h5_id = h5py.File("/data/aruzzi/xgaze_meta/" + subjects, "w")
    print("output file save to ", "/data/aruzzi/xgaze_meta/" + subjects)

    output_code = []
    output_gaze = []

    for i, (image,gaze_direction) in enumerate(train_data_loader):
        print(i)

        model.eval()

        if not output_code:
                total_data = len(train_data_loader.dataset)
                
                output_code = output_h5_id.create_dataset(
                    "code", 
                    shape=(total_data, 2048),
                    compression="lzf",
                    dtype=np.float, 
                    chunks=(1, 2048))
                output_gaze = output_h5_id.create_dataset(
                    "gaze", 
                    shape=(total_data, 2),
                    compression="lzf",
                    dtype=np.float, 
                    chunks=(1, 2))

        with torch.set_grad_enabled(False):
            code = model(image.to(device))

        output_code[i] = code[0,:]
        output_gaze[i] = gaze_direction

    output_h5_id.close()
    print("close the h5 file")

    print("finish the subject: ", subjects)
            


