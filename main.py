import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from data_loader import get_train_test_loader

import numpy as np

import configparser

def main(config, subject_id=0):
    # ensure directories are setup
    prepare_dirs(config)

    # Load the configuration file
    config_ini = configparser.ConfigParser()
    config_ini.sections()
    config_ini.read('config.ini')

    data_dir = config_ini['path']['data_folder']

    # ensure reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': config.num_workers, 'is_shuffle': config.shuffle, 'fold': config.fold}

    # instantiate data loaders
    if config.is_train:
        data_loader = get_train_test_loader(
            data_dir=data_dir,
            batch_size=config.batch_size,
            num_workers=4,
            is_shuffle=False,
            subject_id=subject_id
        )
    # else:
    #     data_loader = get_test_loader(
    #         data_dir, config.batch_size, config.shuffle,
    #         **kwargs
    #     )

    # instantiate trainer
    trainer = Trainer(config, data_loader, subject_id)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    for num_s in range(0, 15):
        main(config, subject_id=num_s)
