import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

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

from tqdm import tqdm
from utils import AverageMeter
from model import gaze_net
from util.error_calculation import mean_angular_error, classFeature2value, angular_error, angular_error_zhang

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

class Tester(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        log_dir = '../logs/' + os.path.basename(os.getcwd())
        if os.path.exists(log_dir) and os.path.isdir(log_dir):
            shutil.rmtree(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)
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
        self.model_name = 'ram_'

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

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr,  # betas=(0.9, 0.95), weight_decay=0.1
        )
        self.scheduler = StepLR(
            self.optimizer, step_size=self.lr_decay_interval, gamma=self.lr_decay_factor)

        self.train_loc = True
        self.train_rand = True
        self.train_gaze = True
        self.first_time = True
        self.train_iter = 0

    def test_fun(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        model_path = './ckpt/ram_epoch_24_ckpt.pth.tar'
        self.load_checkpoint(best=True, is_strict=False,
                             input_file_name=model_path)
                                # input_file_name='../ckpt/reg_1/ram_1_100x2_0_random_ckpt.pth.tar')
            # self.model.locator.gaze_network.load_state_dict(self.model.sensor.gaze_network.state_dict())

        print('We are now doing the final test')
        self.model.eval()
        self.test(is_final=True)
        self.best_valid_acc = 0

    def test(self, is_final=False):
        error_all = []
        head_pose_all = []
        gaze_all = []

        show_interval = 5

        mean_error = []
        index_all = 0
        is_save = True

        for i, (input_img, target, head) in enumerate(self.test_loader):
            input_var = torch.autograd.Variable(input_img.float().cuda())
            target_var = torch.autograd.Variable(target.float().cuda())
            head_var = torch.autograd.Variable(head.float().cuda())
            head_pose_np = head_var.cpu().data.numpy()
            head_pose_all.extend(head_var.cpu().data.numpy())

            self.batch_size = input_var.shape[0]

            pred_gaze = self.model(input_var)
            pred_gaze_np = pred_gaze.cpu().data.numpy()
            target_gaze_np = target_var.cpu().data.numpy()
            gaze_all.extend(target_gaze_np)

            error_each = angular_error(pred_gaze_np, target_gaze_np)
            # print('error_each: ', error_each)

            # error_zhang = angular_error_zhang(pred_gaze_np[:, 0], pred_gaze_np[:, 1], target_gaze_np[:, 0], target_gaze_np[:, 1])
            # print('error_zhang: ', error_zhang)

            error_all.extend(error_each)

            # index_current = 0
            # B, C, H, W = input_var.size()
            # if error_each[index_current] > 6:
            #     img = input_var[index_current].cpu().data.numpy().transpose(1, 2, 0)
            #     img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #     img = img.astype(np.uint8)
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #
            #     # draw the gaze on the image
            #     gaze_theta = pred_gaze_np[index_current, 0]
            #     gaze_phi = pred_gaze_np[index_current, 1]
            #     (h, w) = img.shape[:2]
            #     img = draw_gaze(img, (int(h/2.0), int(w/2.0)), (gaze_theta, gaze_phi),
            #                                length=120.0, thickness=5, color=(0, 0, 255))
            #
            #     gaze_theta = target_gaze_np[index_current, 0]
            #     gaze_phi = target_gaze_np[index_current, 1]
            #     (h, w) = img.shape[:2]
            #     img = draw_gaze(img, (int(h / 2.0), int(w / 2.0)), (gaze_theta, gaze_phi),
            #                                length=120.0, thickness=5, color=(0, 255, 0))
            #
            #     font = cv2.FONT_HERSHEY_SIMPLEX
            #     cv2.putText(img, 'Error: {:.2f}'.format(error_each[index_current])+' degrees', (10, 30), font, 1.2, (0, 0, 255), 2,
            #                 cv2.LINE_AA)
            #
            #     # head = head_pose_np[i, :]
            #     # M = cv2.Rodrigues(head)[0]
            #     # Zv = M[:, 2]
            #     # head_2d = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2])])
            #     # cv2.putText(img, 'Head: {:.1f} {:.1f}'.format(head_2d[0], head_2d[1]) + ' degrees', (10, 100), font, 1.2,
            #     #             (0, 0, 255), 2,
            #     #             cv2.LINE_AA)
            #     if is_save:
            #         # print('i: ', i)
            #         index_sub = int(i / 615)
            #         # print('index_sub: ', index_sub)
            #         index_sample = int((i-index_sub*615) * 18)
            #         # print('index_sample: ', index_sample)
            #         save_name = os.path.join(
            #             'temp/test_sample/img_' + str(index_sub).zfill(3) + '_' + str(index_sample).zfill(5) + '.png')
            #         cv2.imwrite(save_name, img)

            # if i == 0:  # 0, 30, 50
            #     for index in range(0, 50):
            #         img = input_var[index].cpu().data.numpy().transpose(1, 2, 0)
            #         img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #         img = img.astype(np.uint8)
            #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #
            #         # draw the gaze on the image
            #         gaze_theta = pred_gaze_np[index, 0]
            #         gaze_phi = pred_gaze_np[index, 1]
            #         (h, w) = img.shape[:2]
            #         img = draw_gaze(img, (int(h/2.0), int(w/2.0)), (gaze_theta, gaze_phi),
            #                                    length=120.0, thickness=5, color=(0, 0, 255))
            #
            #         gaze_theta = target_gaze_np[index, 0]
            #         gaze_phi = target_gaze_np[index, 1]
            #         (h, w) = img.shape[:2]
            #         img = draw_gaze(img, (int(h / 2.0), int(w / 2.0)), (gaze_theta, gaze_phi),
            #                                    length=120.0, thickness=5, color=(0, 255, 0))
            #
            #         font = cv2.FONT_HERSHEY_SIMPLEX
            #         cv2.putText(img, 'Error: {:.2f}'.format(error_each[index])+' degrees', (10, 30), font, 1.2, (0, 0, 255), 2,
            #                     cv2.LINE_AA)
            #
            #         head = head_pose_np[index, :]
            #         M = cv2.Rodrigues(head)[0]
            #         Zv = M[:, 2]
            #         head_2d = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2])])
            #         cv2.putText(img, 'Head: {:.1f} {:.1f}'.format(head_2d[0], head_2d[1]) + ' degrees', (10, 100), font, 1.2,
            #                     (0, 0, 255), 2,
            #                     cv2.LINE_AA)
            #
            #         if is_save:
            #             save_name = os.path.join(
            #                 'temp/test_sample/img_' +  str(i) + '_' + str(index).zfill(2) + '.png')
            #             cv2.imwrite(save_name, img)
            #         # print('save image to ', save_name)
            #         # img = cv2.resize(img, (224, 224))
            #         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #         img = img.astype(np.float32)
            #         img = cv2.normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            #         img = img.transpose(2, 0, 1)
            #
            #         self.writer.add_image('test/test_' + str(index), img, global_step=self.train_iter)

            if not is_final:
                if i == show_interval:
                    break

        if is_final:
            print('Here we print the final test error')
            error_all = np.asarray(error_all)
            error_all = error_all.reshape(-1, 1)
            print('error_all shape: ', error_all.shape)
            mean_error = sum(error_all) / float(error_all.shape[0])
            mean_error = mean_error[0]
            print('This is the final test. I want this line to be Test error {0:.3f}\t'.format(mean_error))
            
            head_pose_all = np.asarray(head_pose_all)
            if head_pose_all.shape[1] == 3:
                head_pose_all_2d = []
                for num_i in range(0, head_pose_all.shape[0]):
                    head = head_pose_all[num_i, :]
                    M = cv2.Rodrigues(head)[0]
                    Zv = M[:, 2]
                    head_2d = np.array([math.asin(Zv[1]), math.atan2(Zv[0], Zv[2])])
                    head_pose_all_2d.append(head_2d)
                head_pose_all_2d = np.asarray(head_pose_all_2d)
            else:
                head_pose_all_2d = head_pose_all
            head_pose_all_2d = head_pose_all_2d.reshape(-1, 2)
            gaze_all = np.asarray(gaze_all)
            gaze_all = gaze_all.reshape(-1, 2)
            print('gaze_all shape: ', gaze_all.shape)

            print('head_pose_all_2d shape: ', head_pose_all_2d.shape)

            content = np.concatenate((error_all, head_pose_all_2d, gaze_all), axis=1)
            np.savetxt('test_error_all.txt', content, delimiter=',')

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
        # self.model.locator.check_table = ckpt['self.model.locator.check_table']

        print(
            "[*] Loaded {} checkpoint @ epoch {}".format(
                ckpt_path, ckpt['epoch'])
        )
