from new_model import gaze_net

model = gaze_net()

for name,p in model.named_submodules():
    print(name)