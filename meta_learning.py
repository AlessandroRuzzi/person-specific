import argparse
import os
import pickle
import random
from collections import OrderedDict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as V
from tqdm import tqdm
import copy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
    Tasks class for grabbing training/testing samples
"""

"""
    Utility functions
"""


def angular_error(a, b):
    """Calculate angular error (via cosine similarity)."""
    a = pitchyaw_to_vector(a) if a.shape[1] == 2 else a
    b = pitchyaw_to_vector(b) if b.shape[1] == 2 else b

    ab = np.sum(np.multiply(a, b), axis=1)
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)

    # Avoid zero-values (to avoid NaNs)
    a_norm = np.clip(a_norm, a_min=1e-6, a_max=None)
    b_norm = np.clip(b_norm, a_min=1e-6, a_max=None)

    similarity = np.divide(ab, np.multiply(a_norm, b_norm))
    similarity = np.clip(similarity, a_min=-1.0 + 1e-6, a_max=1.0 - 1e-6)

    return np.degrees(np.arccos(similarity))


def nn_angular_error(y, y_hat):
    sim = F.cosine_similarity(y, y_hat, eps=1e-6)
    sim = F.hardtanh(sim, -1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(sim) * (180 / np.pi)


def nn_mean_angular_loss(y, y_hat):
    return torch.mean(nn_angular_error(y, y_hat))


def nn_mean_asimilarity(y, y_hat):
    return torch.mean(1.0 - F.cosine_similarity(y, y_hat, eps=1e-6))


def pitchyaw_to_vector(pitchyaws):
    n = pitchyaws.shape[0]
    sin = np.sin(pitchyaws)
    cos = np.cos(pitchyaws)
    out = np.empty((n, 3))
    out[:, 0] = np.multiply(cos[:, 0], sin[:, 1])
    out[:, 1] = sin[:, 0]
    out[:, 2] = np.multiply(cos[:, 0], cos[:, 1])
    return out

"""
    Meta-learning utility functions.
"""


def forward_and_backward(model, input, target, optim=None, create_graph=False,
                         train_data=None, loss_function=nn_mean_angular_loss):
    model.train()
    if optim is not None:
        optim.zero_grad()
    loss = forward(model, input,target, train_data=train_data, for_backward=True,
                   loss_function=loss_function)
    
    loss.backward(create_graph=create_graph, retain_graph=(optim is None))
   
    if optim is not None:
        # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
    return loss.data.cpu().numpy()


def forward(model, input,target, return_predictions=False, train_data=None,
            for_backward=False, loss_function=nn_mean_angular_loss):
    model.train()
    x, y = input, target
    y_hat, _ = model(x)
    loss = loss_function(y_hat, y)
    if return_predictions:
        return y_hat.data.cpu().numpy()
    elif for_backward:
        return loss
    else:
        return loss.data.cpu().numpy()


"""
    Inference through model (with/without gradient calculation)
"""


class MAML(object):
    def __init__(self, model, k, output_dir='./outputs/',
                 train_tasks=None, valid_tasks=None, no_tensorboard=False):
        self.model = model
        self.meta_model = copy.deepcopy(model)

        self.train_tasks = train_tasks
        self.valid_tasks = valid_tasks
        self.k = k

        self.output_dir = None
        self.tensorboard = None
        if output_dir is not None:
            self.output_dir = '%s/%s_%02d' % (output_dir, self.__class__.__name__, k)
            if not os.path.isdir(self.output_dir):
                os.makedirs(self.output_dir)

    @property
    def model_parameters_path(self):
        return '%s/meta_learned_parameters.pth.tar' % self.output_dir

    def save_model_parameters(self):
        if self.output_dir is not None:
            torch.save(self.model.state_dict(), self.model_parameters_path)

    def load_model_parameters(self):
        if os.path.isfile(self.model_parameters_path):
            weights = torch.load(self.model_parameters_path)
            self.model.load_state_dict(weights)
            print('> Loaded weights from %s' % self.model_parameters_path)

    def train(self, steps_outer, steps_inner=1, lr_inner=0.01, lr_outer=0.001,
              disable_tqdm=False):
        self.lr_inner = lr_inner
        print('\nBeginning meta-learning for k = %d' % self.k)
        print('> Please check tensorboard logs for progress.\n')

        # Outer loop optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr_outer)

        # Model and optimizer for validation
        valid_model = copy.deepcopy(self.model)
        valid_optim = torch.optim.SGD(valid_model.parameters(), lr=self.lr_inner)

        for i in tqdm(range(steps_outer), disable=disable_tqdm):
            for j in range(steps_inner):
                # Make copy of main model
                #self.meta_model = copy.deepcopy(self.model)
                self.meta_model = copy.copy(self.model)
                
                # Get a task
                for i, (input_img, target) in enumerate(self.train_tasks):
                    input_var = torch.autograd.Variable(input_img.float().cuda())
                    target_var = torch.autograd.Variable(target.float().cuda())
                    break
                #train_data, test_data = self.train_tasks.dataset.sample(num_train=self.k, train = True)
                train_input,train_target, test_input, test_target = input_var[:self.k,:],target_var[:self.k,:] , input_var[self.k:,:],target_var[self.k:,:]
                task_loss = self.inner_loop(train_input,train_target, self.lr_inner)

            # Calculate gradients on a held-out set
            new_task_loss = forward_and_backward(
                self.meta_model, test_input, test_target,
            )

            # Update the main model
            with torch.no_grad():
                for name,p in self.model.named_parameters():
                    if p.grad is None:
                        print(name,"None")
                        #print(lr_inner)
                        #print(p.grad)
                        continue
                        #print(lr_inner*p.grad)
                    if(p.grad is not None):
                        print(name,p.requires_grad)
            optimizer.step()
            optimizer.zero_grad()

            if (i + 1) % 1 == 0:
                # Validation
                losses = []
                valid_model = copy.deepcopy(self.model)
                for i, (input_img, target) in enumerate(self.train_tasks):
                    input_var = torch.autograd.Variable(input_img.float().cuda())
                    target_var = torch.autograd.Variable(target.float().cuda())
                    break
                #train_data, test_data = self.train_tasks.dataset.sample(num_train=self.k, train = True)
                train_input,train_target, test_input, test_target = input_var[:self.k,:],target_var[:self.k,:] , input_var[self.k:,:],target_var[self.k:,:]
                train_loss = forward_and_backward(valid_model, train_input,train_target, valid_optim)
                valid_loss = forward(valid_model, test_input, test_target)
                losses.append((train_loss, valid_loss))
                train_losses, valid_losses = zip(*losses)
                print("train losses: ", train_losses)
                print("valid losses: ", valid_losses)

        # Save MAML initial parameters
        #self.save_model_parameters()

    def test(self, test_tasks_list, num_iterations=[1, 5, 10], num_repeats=20):
        print('\nBeginning testing for meta-learned model with k = %d\n' % self.k)
        model = self.model.clone()

        # IMPORTANT
        #
        # Sets consistent seed such that as long as --num-test-repeats is the
        # same, experiment results from multiple invocations of this script can
        # yield the same calibration samples.

        """
        random.seed(4089213955)

        for test_set_name, test_tasks in test_tasks_list.items():
            predictions = OrderedDict()
            losses = OrderedDict([(n, []) for n in num_iterations])
            for i, task_name in enumerate(test_tasks.selected_tasks):
                predictions[task_name] = []
                for t in range(num_repeats):
                    model.copy(self.model)
                    optim = torch.optim.SGD(model.params(), lr=self.lr_inner)

                    train_data, test_data = test_tasks.sample_for_task(i, num_train=self.k)
                    if num_iterations[0] == 0:
                        train_loss = forward(model, train_data)
                        test_loss = forward(model, test_data, train_data=train_data)
                        losses[0].append((train_loss, test_loss))
                    for j in range(np.amax(num_iterations)):
                        train_loss = forward_and_backward(model, train_data, optim)
                        if (j + 1) in num_iterations:
                            test_loss = forward(model, test_data, train_data=train_data)
                            losses[j + 1].append((train_loss, test_loss))

                    # Register ground truth and prediction
                    predictions[task_name].append({
                        'groundtruth': test_data[1].cpu().numpy(),
                        'predictions': forward(model, test_data,
                                               return_predictions=True,
                                               train_data=train_data),
                    })
                    predictions[task_name][-1]['errors'] = angular_error(
                        predictions[task_name][-1]['groundtruth'],
                        predictions[task_name][-1]['predictions'],
                    )

                print('Done for k = %3d, %s/%s... train: %.3f, test: %.3f' % (
                    self.k, test_set_name, task_name,
                    np.mean([both[0] for both in losses[num_iterations[-1]][-num_repeats:]]),
                    np.mean([both[1] for both in losses[num_iterations[-1]][-num_repeats:]]),
                ))

            if self.output_dir is not None:
                # Save predictions to file
                pkl_path = '%s/predictions_%s.pkl' % (self.output_dir, test_set_name)
                with open(pkl_path, 'wb') as f:
                    pickle.dump(predictions, f)

                # Finally, log values to tensorboard
                if self.tensorboard is not None:
                    for n, v in losses.items():
                        train_losses, test_losses = zip(*v)
                        stem = 'meta-test/%s/' % test_set_name
                        self.tensorboard.add_scalar(stem + 'train-loss', np.mean(train_losses), n)
                        self.tensorboard.add_scalar(stem + 'valid-loss', np.mean(test_losses), n)

                # Write loss values as plain text too
                np.savetxt('%s/losses_%s_train.txt' % (self.output_dir, test_set_name),
                           [[n, np.mean(list(zip(*v))[0])] for n, v in losses.items()])
                np.savetxt('%s/losses_%s_valid.txt' % (self.output_dir, test_set_name),
                           [[n, np.mean(list(zip(*v))[1])] for n, v in losses.items()])

            out_msg = '> Completed test on %s for k = %d' % (test_set_name, self.k)
            final_n = sorted(num_iterations)[-1]
            final_train_losses, final_test_losses = zip(*(losses[final_n]))
            out_msg += ('\n  at %d steps losses were... train: %.3f, test: %.3f +/- %.3f' %
                        (final_n, np.mean(final_train_losses),
                         np.mean(final_test_losses),
                         np.mean([
                             np.std([
                                 data['errors'] for data in person_data
                             ], axis=0)
                             for person_data in predictions.values()
                         ])))
            print(out_msg)
        """

    def inner_loop(self, train_input,train_target, lr_inner=0.01):
        # Forward-pass and calculate gradients on meta model
        loss = forward_and_backward(self.meta_model, train_input,train_target,
                                    create_graph=True)

        # Apply gradients
        with torch.no_grad():
            for name,p in self.meta_model.named_parameters():
                if p.grad is None:
                    #print(name,"None")
                    #print(lr_inner)
                    #print(p.grad)
                    continue
                    #print(lr_inner*p.grad)
                if(p.grad is not None):
                    #print(name,p.requires_grad)
                    p.copy_(p-lr_inner*p.grad)
        #for name, param in self.meta_model.named_params():
        #    self.meta_model.set_param(name, param - lr_inner * param.grad)
        return loss