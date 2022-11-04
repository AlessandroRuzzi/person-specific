import torch

from tester import Tester
from config import get_config
from utils import prepare_dirs, save_config
from data_loader_test import get_train_test_loader, get_test_loader_gazecapture, get_test_loader_mpiigaze, get_train_test_loader_rtgene

import numpy as np

import configparser

def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # Load the configuration file
    config_ini = configparser.ConfigParser()
    config_ini.sections()
    config_ini.read('config.ini')

    # ggd_folder mpiigaze_folder gazecapture_folder eyediap_folder rtgene_folder gaze360_folder
    test_set = 'ggd_folder' #

    data_dir = config_ini['path'][test_set]

    # ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': config.num_workers, 'is_shuffle': False}

    # instantiate data loaders
    if test_set == 'gazecapture_folder':
        if config.is_train:
            data_loader = get_test_loader_gazecapture(
                data_dir, config.batch_size,
                **kwargs
            )
        else:
            data_loader = get_test_loader_gazecapture(
                data_dir, config.batch_size,
                **kwargs
            )
    elif test_set == 'ggd_folder':
        if config.is_train:
            data_loader = get_train_test_loader(
                data_dir, config.batch_size,
                **kwargs
            )
        else:
            data_loader = get_train_test_loader(
                data_dir, config.batch_size,
                **kwargs
            )
    elif test_set == 'mpiigaze_folder' or test_set == 'eyediap_folder':
        if config.is_train:
            data_loader = get_test_loader_mpiigaze(
                data_dir, config.batch_size,
                **kwargs
            )
        else:
            data_loader = get_test_loader_mpiigaze(
                data_dir, config.batch_size,
                **kwargs
            )
    elif test_set == 'rtgene_folder' or test_set == 'gaze360_folder':
        if config.is_train:
            data_loader = get_train_test_loader_rtgene(
                data_dir, config.batch_size,
                **kwargs
            )
        else:
            data_loader = get_train_test_loader_rtgene(
                data_dir, config.batch_size,
                **kwargs
            )

    # instantiate trainer
    tester = Tester(config, data_loader)

    tester.test_fun()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
