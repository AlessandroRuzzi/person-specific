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
    
    def params(self):
            return [p for _, p in self.named_params()]

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self):
        subparams = []
        for name, mod in self.named_submodules():
            for subname, param in mod.named_params():
                subparams.append((name + '.' + subname, param))
        return self.named_leaves() + subparams

    def set_param(self, name, param, copy=False):
        if '.' in name:
            n = name.split('.')
            module_name = n[0]
            rest = '.'.join(n[1:])
            for name, mod in self.named_submodules():
                if module_name == name:
                    mod.set_param(rest, param, copy=copy)
                    break
        else:
            if copy is True:
                setattr(self, name, V(param.data.clone(), requires_grad=True))
            else:
                assert hasattr(self, name)
                setattr(self, name, param)

    def copy(self, other, same_var=False):
        for name, param in other.named_params():
            self.set_param(name, param, copy=not same_var)

    def clone(self, make_alpha=None):
        new_model = self.__class__()
        new_model.copy(self)
        return new_model

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        feature = self.gaze_fc(feature)
        gaze = feature[:,:2]
        head = feature [:,2:]

        return gaze, head