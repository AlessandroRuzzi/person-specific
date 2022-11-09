import torch.nn as nn

from modules import resnet50, resnet18

class gaze_net(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_net, self).__init__()
        self.gaze_fc = nn.Sequential(
            nn.Linear(2, 2),
        )

    def forward(self, x):
        feature = self.gaze_fc(x)

        return feature