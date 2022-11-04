import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.distributions as dis
import numpy as np
import math
import random

from models import alexnet, resnet18, inception_v3, resnet50, resnet101

class gaze_network(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_network, self).__init__()
        # self.fc = nn.Linear(input_size, output_size)
        # self.gaze_network = alexnet(pretrained=True, stride_first=4)
        self.gaze_network = resnet50(pretrained=True)
        # self.gaze_network = inception_v3(pretrained=True)
        self.gaze_network = nn.DataParallel(self.gaze_network)

        # self.gaze_fc = nn.Sequential(
        #     nn.Dropout(),
        #     nn.Linear(256 * 6 * 6, 4096),
        #     # nn.Linear(2048*7*7, 4096),
        #     nn.LeakyReLU(),
        #     nn.Dropout(),
        #     nn.Linear(4096, 4096),
        #     nn.LeakyReLU(),
        #     nn.Linear(4096, 2),
        # )

        self.gaze_fc = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(2048, 2),
        )

    def forward(self, x):
        # feature = self.gaze_network.features(x)
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)

        gaze = self.gaze_fc(feature)

        return gaze
