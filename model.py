import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions import Normal
from torch.autograd import Variable

from modules import gaze_network
import cv2
import os
import numpy as np


class gaze_net(nn.Module):
    """
    A Recurrent Model of Visual Attention (RAM) [1].

    RAM is a recurrent neural network that processes
    inputs sequentially, attending to different locations
    within the image one at a time, and incrementally
    combining information from these fixations to build
    up a dynamic internal representation of the image.

    References
    ----------
    - Minh et. al., https://arxiv.org/abs/1406.6247
    """
    def __init__(self):

        super(gaze_net, self).__init__()

        self.gze_net = gaze_network()

    def resize2d(self, img, size):
        with torch.no_grad():
            return (F.adaptive_avg_pool2d(Variable(img), size)).data

    def forward(self, x):
        l_t = []
        b_t = []
        log_pi = []
        pred_gaze = []
        log_pi_base = []
        pred_gaze_base = []
        state_value = []
        action = []

        middle_x = self.resize2d(x, (224, 224))

        pred_gaze = self.gze_net(middle_x)

        return pred_gaze




