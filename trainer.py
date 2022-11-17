import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import torch.nn as nn

import math
import os
import time
import shutil
import pickle
import numpy as np
import cv2
import json

from tqdm import tqdm
from utils import AverageMeter
from new_model import gaze_net
from linear_model import gaze_net as linear_gaze_net 
#from vgg_model import gaze_network as gaze_net
from new_meta_learning import MAML
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from new_meta_learning import Tasks, GazeEstimationModelPreExtended, GazeEstimationModel


from util.error_calculation import mean_angular_error, classFeature2value, angular_error

label_std = 1
range_angle_set = 76.0
range_angle = range_angle_set / 180.0 * math.pi  # the angle we set to be maxium

# ratio loss, softmax, no index_better_var

def draw_gaze(image_in, pos, pitchyaw, length=40.0, thickness=1, color=(0, 0, 255)):
    """Draw gaze angle on given image with a given eye positions."""
    image_out = image_in
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv2.cvtColor(image_out, cv2.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)),
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)), color,
                   thickness, cv2.LINE_AA, tipLength=0.2)
    return image_out

def torch_equalize(image):
    """Implements Equalize function from PIL using PyTorch ops based on:
    https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/autoaugment.py#L352"""

    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = torch.histc(im, bins=256, min=0, max=255)  # .type(torch.int32)
        # For the purposes of computing the step, filter out the nonzeros.
        nonzero_histo = torch.reshape(histo[histo != 0], [-1])
        step = (torch.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (torch.cumsum(histo, 0) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = torch.cat([torch.zeros(1), lut[:-1]])
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return torch.clamp(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        if step == 0:
            result = im
        else:
            # can't index using 2d index. Have to flatten and then reshape
            result = torch.gather(build_lut(histo, step), 0, im.flatten().long())
            result = result.reshape_as(im)

        return result.type(torch.uint8)

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = torch.stack([s1, s2, s3], 2)
    return image

class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader, subject_id):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")

        log_dir = '../logs/' + os.path.basename(os.getcwd())
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

        self.config = config

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            # self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.dataset)
            # self.num_valid = len(self.valid_loader.dataset)
            self.test_loader = data_loader[1]
            self.num_test = len(self.test_loader.dataset)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 2
        self.num_channels = 3

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.lr_decay_factor = config.lr_decay_factor
        self.lr_decay_interval = config.lr_decay_interval

        data_dir = '/media/xucong/external/xgaze_224/test_person_specific'
        refer_list_file = os.path.join('data/train_test_split.json')
        print('load the train file list from: ', refer_list_file)

        with open(refer_list_file, 'r') as f:
            datastore = json.load(f)

        all_subjects = datastore["test_specific"]
        self.subject_id = all_subjects[subject_id][:-3]

        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 99999999
        self.counter = 0
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = 'ram'

        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # build RAM model
        self.model = gaze_net()
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # # initialize optimizer and scheduler
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=self.momentum,
        # )
        #self.linear_model = linear_gaze_net()
        #if self.use_gpu:
        #    self.linear_model.cuda()
        
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, #weight_decay=1.0  # ,  # betas=(0.9, 0.95), weight_decay=0.1
        )
        self.scheduler = StepLR(
            self.optimizer, step_size=self.lr_decay_interval, gamma=self.lr_decay_factor)

        self.train_loc = True
        self.train_rand = True
        self.train_gaze = True
        self.first_time = True
        self.train_iter = 0


    def reset(self, batch_size):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        h_t = torch.zeros(batch_size, self.hidden_size)
        h_t = Variable(h_t).type(dtype)

        # l_t = torch.Tensor(self.batch_size, 2).uniform_(-1, 1)
        l_t = torch.zeros(batch_size, 4)
        l_t.fill_(0.0)  #-1
        l_t[:, 2:4] = 1.0
        l_t = Variable(l_t).type(dtype)

        return h_t, l_t

    def train_func(self):
        for epoch in range(0, self.epochs):
            print(
                '\nEpoch: {}/{} - base LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )

            for param_group in self.optimizer.param_groups:
                print('Learning rate: ', param_group['lr'])

            # train for 1 epoch
            print('Now go to training')
            self.model.train()
            train_acc, loss_gaze = \
                self.train_one_epoch(epoch, self.train_loader)

            # msg = "train acc: {:.3f} - loss_gaze: {:.5f}"
            # print(msg.format(train_acc, loss_gaze))

            add_file_name = None
            add_file_name = 'epoch_'+str(epoch)
            # if self.train_rand:
            #     add_file_name = 'random'
            # elif self.train_loc:
            #     add_file_name = 'location'
            # elif self.train_gaze:
            #     add_file_name = 'gaze'
            #self.scheduler.step()  # update learning rate

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        #if self.resume:
        #    self.load_checkpoint(best=True, is_strict=False,
                                 #input_file_name='ckpt/epoch_24_resnet_correct_ckpt.pth.tar')
                                 #input_file_name='ckpt/epoch_24_VGG_80_subj_ckpt.pth.tar')
                                 #input_file_name='ckpt/epoch_24_VGG_correct_small_ckpt.pth.tar')
                                 #input_file_name='ckpt/ram_epoch_24_ckpt.pth.tar')
            # self.model.locator.gaze_network.load_state_dict(self.model.sensor.gaze_network.state_dict())
            #for param in self.model.parameters():
            #    param.requires_grad = False
            #for param in self.model.gaze_fc.parameters():
            #    param.requires_grad = True
        
        """
        self.model.eval()
        input_lg = []
        target_lg = []

        for i, (input_img, target) in enumerate(self.train_loader):
            input_var = torch.autograd.Variable(input_img.float().cuda())
            target_var = torch.autograd.Variable(target.float().cuda())
            self.batch_size = input_var.shape[0]
            # train gaze net
            pred_gaze, pred_head= self.model(input_var)
            #print(pred_gaze, target_var)
            input_lg.append(pred_gaze[0,:].cpu().data.numpy())
            target_lg.append(target_var[0,:].cpu().data.numpy())

        self.poly = PolynomialFeatures(3)
        #input_lg = self.poly.fit_transform(input_lg)

        #print(input_lg)
        print(input_lg)
        print(target_lg)
        self.reg_gt = LinearRegression().fit(input_lg, target_lg)
        print(self.reg_gt.score(input_lg, target_lg))
        """

        #self.model.train()
        self.train_task = Tasks()
        sample_train, _ = self.train_task.sample(num_train=1, num_test=0)
        #self.gaze_estimator = GazeEstimationModelPreExtended()
        layer_num_features = [int(f) for f in "64".split(',')]
        layer_num_features = [sample_train[0].shape[1]] + layer_num_features + [3]
    
        self.gaze_estimator = GazeEstimationModel(activation_type='selu',
                                    layer_num_features=layer_num_features)
        self.meta_model = MAML(model = self.gaze_estimator, k = 3, train_tasks=self.train_task, valid_tasks=self.train_task )       
        self.meta_model.train(steps_outer=100000,steps_inner=5, lr_inner=1e-5, lr_outer=1e-3)
        #self.meta_model.test(lr_outer=1e-5)

        #self.model.train()
        #self.linear_model.train()
        #self.train_func()

        #print('We are now doing the final test')
        #self.model.eval()
        #self.linear_model.eval()
        #self.test(is_final=True)
        #self.best_valid_acc = 0


    def train_one_epoch(self, epoch, data_loader, is_train=True):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        accs = AverageMeter()
        losses_gaze = AverageMeter()

        show_interval = 100
        save_interval = 3000

        tic = time.time()
        for i, (input_img, target) in enumerate(data_loader):
            input_var = torch.autograd.Variable(input_img.float().cuda())
            target_var = torch.autograd.Variable(target.float().cuda())
            self.batch_size = input_var.shape[0]
            # train gaze net
            pred_gaze, pred_head= self.model(input_var)

            #pred_gaze = self.linear_model(pred_gaze)

            error_each_gaze = angular_error(pred_gaze.cpu().data.numpy(), target_var.cpu().data.numpy())
            error = np.mean(error_each_gaze)
            acc = error
            accs.update(acc.item(), input_var.size()[0])

            loss_gaze = F.l1_loss(pred_gaze, target_var)
            self.optimizer.zero_grad()
            loss_gaze.backward()
            self.optimizer.step()
            losses_gaze.update(loss_gaze.item(), input_var.size()[0])

        toc = time.time()
        batch_time.update(toc-tic)

        print('running time is ', batch_time.avg, 'gaze loss is: ', losses_gaze.avg)
        return accs.avg, losses_gaze.avg

    def test(self, is_final=False):
        if self.resume:
            self.load_checkpoint(best=True, is_strict=False,
                                 input_file_name='ckpt/epoch_24_resnet_correct_ckpt.pth.tar')
                                 #input_file_name='ckpt/epoch_24_VGG_80_subj_ckpt.pth.tar')
                                 #input_file_name='ckpt/epoch_24_VGG_correct_small_ckpt.pth.tar')
                                 #input_file_name='ckpt/ram_epoch_24_ckpt.pth.tar')
            self.test_task = Tasks(subject=self.subject_id)
            sample_test, _ = self.test_task.sample(num_train=1, num_test=0)
            #self.gaze_estimator = GazeEstimationModelPreExtended()
            layer_num_features = [int(f) for f in "64".split(',')]
            layer_num_features = [sample_test[0].shape[1]] + layer_num_features + [3]
        
            self.gaze_estimator = GazeEstimationModel(activation_type='selu',
                                        layer_num_features=layer_num_features)
            check_path = "outputs/MAML03/meta_learned_parameters_90000.pth.tar"
            weights = torch.load(check_path)
            try:
                state_dict = {
                    'layer01.weights': weights['module.gaze1.weight'],
                    'layer01.bias': weights['module.gaze1.bias'],
                    'layer02.weights': weights['module.gaze2.weight'],
                    'layer02.bias': weights['module.gaze2.bias'],
                }
            except:  # noqa
                state_dict = {
                    'layer01.weights': weights['gaze1.weight'],
                    'layer01.bias': weights['gaze1.bias'],
                    'layer02.weights': weights['gaze2.weight'],
                    'layer02.bias': weights['gaze2.bias'],
                }
            for key, values in state_dict.items():
                self.gaze_estimator.set_param(key, values, copy=True)
            del state_dict
            print('Loaded %s' % check_path)
            self.meta_model = MAML(model = self.gaze_estimator, k = 3, train_tasks=self.train_task, valid_tasks=self.train_task ) 
            self.meta_model.lr_inner = 1e-5
            self.meta_model.test(self.test_task)
        error_all = []

        prediction_all = []
        show_interval = 10

        mean_error = []
        index_all = 0
        is_save = True

        for i, (input_img, target) in enumerate(self.test_loader):
            input_var = torch.autograd.Variable(input_img.float().cuda())
            target_var = torch.autograd.Variable(target.float().cuda())

            self.batch_size = input_var.shape[0]

            with torch.set_grad_enabled(False):
                latent_code = self.model(input_var)
                pred_gaze = self.meta_model.model(latent_code)
            #pred_gaze = self.reg_gt.predict(pred_gaze.cpu().data.numpy())
            #pred_gaze = self.reg_gt.predict(self.poly.transform(pred_gaze.cpu().data.numpy()))
            #pred_gaze = self.linear_model(pred_gaze)
            #pred_gaze, pred_head = self.meta_model.model(input_var)
            pred_gaze_np = pred_gaze.cpu().data.numpy()
            #pred_gaze_np = pred_gaze
            prediction_all.append(pred_gaze_np)
            target_gaze_np = target_var.cpu().data.numpy()

            error_each = angular_error(pred_gaze_np, target_gaze_np)
            error = np.mean(error_each)
            error_all.append(error)

            if not is_final:
                if i == show_interval:
                    mean_error = sum(error_all) / float(len(error_all))
                    print('Now go for test. I want this line to be Test error {0:.3f}\t'.format(mean_error))
                    break

        if is_final:
            print('Here we print the final test error')
            mean_error = sum(error_all) / float(len(error_all))
            print('This is the final test. I want this line to be Test error {0:.3f}\t'.format(mean_error))

            save_path = '/local/home/aruzzi/submission_specific_eva'
            save_file_path = os.path.join(save_path, self.subject_id+'_test.txt')
            print('save the file:  ', save_file_path)
            prediction_all = np.array([x for x in prediction_all])
            prediction_all = np.vstack(prediction_all)
            print('shape: ', prediction_all.shape)
            np.savetxt(save_file_path, prediction_all)
        return mean_error

    def save_checkpoint(self, state, add=None):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        if add is not None:
            filename = self.model_name + '_' + add + '_ckpt.pth.tar'
        else:
            filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        print('save file to: ', filename)

    def load_checkpoint(self, best=False, add='', input_file_name='', is_strict=True):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        if not input_file_name:
            filename = add + self.model_name + '_ckpt.pth.tar'
            if best:
                filename = self.model_name + '_model_best.pth.tar'
            ckpt_path = os.path.join(self.ckpt_dir, filename)
        else:
            ckpt_path = input_file_name

        print('load the pre-trained model: ', ckpt_path)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.model.load_state_dict(ckpt['model_state'], strict=is_strict)
        # self.optimizer.load_state_dict(ckpt['optim_state'])
        # self.scheduler.load_state_dict(ckpt['scheule_state'])
        # self.start_epoch = ckpt['epoch'] - 1

        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )
