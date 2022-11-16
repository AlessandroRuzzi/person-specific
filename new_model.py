import torch.nn as nn

from modules import resnet50, resnet18
from torch.autograd import Variable as V

class gaze_net(nn.Module):
    def __init__(self, use_face=False, num_glimpses=1):
        super(gaze_net, self).__init__()
        self.gaze_network = resnet50(pretrained=True)

        self.gaze_fc = nn.Sequential(
            nn.Linear(2048, 4),
        )

    def named_submodules(self):
        return []

    def set_param(self,name, param):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param)
                    break
        else:
            assert hasattr(self, name)
            setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_parameters():
            self.set_param(name, param)
            

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        return feature
        feature = self.gaze_fc(feature)
        gaze = feature[:,:2]
        head = feature [:,2:]

        return gaze, head