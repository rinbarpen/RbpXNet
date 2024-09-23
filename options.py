import logging
from argparse import ArgumentParser

import torch
from typing import Literal
from enum import Enum

from utils.utils import create_dirs, fix_dir_tail

def check_args(args):
    if args.config:
        ext = args.config.splitext(args.config)[1]
        if ext not in ['json', 'yaml', 'yml', 'toml']:
            raise ValueError(f'Unsupported config file format: {ext}')

def parse_args():
    """
    This function parses command-line arguments for training, testing, and predicting a model.
    It handles various configurations, including model parameters, training settings, and prediction options.

    Parameters:
    None

    Returns:
    args: argparse.Namespace
        An object containing the parsed command-line arguments.
    """
    from config import CONFIG
    parser = ArgumentParser(description='Training Configuration')
    parser.add_argument('-c', '--config', type=str, help='Train configuration file (JSON format)')

    general_group = parser.add_argument_group('General Settings')
    model_group = parser.add_argument_group('Model Configuration')
    training_group = parser.add_argument_group('Training Configuration')
    predict_group = parser.add_argument_group('Predict Configuration')

    general_group.add_argument('--project', type=str, help='Project Name')
    general_group.add_argument('--entity', type=str, help='Entity Name')
    general_group.add_argument('--test', action='store_true', help='Test the model')
    general_group.add_argument('--predict', action='store_true', help='Predict the model')
    general_group.add_argument('--train', action='store_true', help='Train the model')
    general_group.add_argument('--wandb', action='store_true', help='Launch Wandb instance')

    model_group.add_argument('-m', '--model', type=str, help='Model to train')
    model_group.add_argument('--in_channels', type=int, help='Number of input channels')
    model_group.add_argument('--n_classes', type=int, help='Number of output classes')
    # --classes "category1,category2,..."
    model_group.add_argument('--classes', type=str, help='predicted classes group')

    training_group.add_argument('-b', '--batch_size', type=int, help='Number of samples loaded at one time')
    training_group.add_argument('-e', '--epochs', type=int, help='Number of training epochs')
    training_group.add_argument('-lr', '--learning_rate', type=float, help='Learning rate for training the model')
    training_group.add_argument('--weight_decay', default=1e-8, type=float, help='Weight decay for training the model')
    training_group.add_argument('--data_dir', type=str, help='The directory of datasets')
    training_group.add_argument('--dataset', type=str, help='Training and testing dataset')
    training_group.add_argument('--augment_boost', action='store_true', help='Use augment of data')
    training_group.add_argument('--gpu', action='store_true', help='GPU to train')
    training_group.add_argument('--amp', action='store_true', help='Use half precision mode')
    training_group.add_argument('--save_every_n_epoch', type=int, default=1, help='Use half precision mode')

    predict_group.add_argument('-l', '--load', type=str, help='The model used to predict')
    predict_group.add_argument('-i', '--input', type=str, help='The input data to predict')

    args = parser.parse_args()
    check_args(args)
    
    if not torch.cuda.is_available() and args.gpu:
        logging.warning('No GPU found, training will be performed on CPU.')
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    if args.data_dir and not args.data_dir.endswith('/'):
        args.data_dir += '/'

    if args.dataset:
        args.dataset = args.dataset.upper()

    if args.config:
        ext = args.config.splitext(args.config)[1]
        match ext:
            case 'json':
                import json
                with open(args.config, 'r') as f:
                    CONFIG = json.load(f)
            case 'yaml'|'yml':
                import yaml
                with open(args.config, 'r') as f:
                    CONFIG = yaml.safe_load(f)
            case 'toml':
                import toml
                with open(args.config, 'r') as f:
                    CONFIG = toml.load(f)
    else:
        classes = args.classes.split(',')
        classes = [c.strip() for c in classes]

        CONFIG["model"] = args.model
        CONFIG["private"] = {
            "in_channels": args.in_channels,
            "n_classes": args.n_classes,
            "classes": classes,
        }
        CONFIG["device"] = device
        CONFIG["amp"] = args.amp
        if args.predict:
            CONFIG["predict"] = True
            CONFIG["load"] = args.load
            CONFIG["input"] = args.input
        else:
            CONFIG["batch_size"] = args.batch_size
            CONFIG["dataset"] = args.dataset
            CONFIG["data_dir"] = args.data_dir
            if args.test:
                CONFIG["test"] = True
            else:
                CONFIG["train"] = True
                CONFIG["learning_rate"] = args.learning_rate
                CONFIG["epochs"] = args.epochs
                CONFIG["augment_boost"] = args.augment_boost
                CONFIG["save_every_n_epoch"] = args.save_every_n_epoch
                if args.weight_decay:
                    CONFIG["weight_decay"] = args.weight_decay

    # supply the tail '/' of the directory path
    CONFIG["save"]["predict_dir"] = fix_dir_tail(CONFIG["save"]["predict_dir"])
    CONFIG["save"]["train_dir"] = fix_dir_tail(CONFIG["save"]["train_dir"])
    CONFIG["save"]["valid_dir"] = fix_dir_tail(CONFIG["save"]["valid_dir"])
    CONFIG["save"]["test_dir"] = fix_dir_tail(CONFIG["save"]["test_dir"])
    CONFIG["save"]["model_dir"] = fix_dir_tail(CONFIG["save"]["model_dir"])

    if args.wandb:
        import wandb
        CONFIG["wandb"] = True
        wandb.init(project=args.project,
                   config=CONFIG)
    
    create_dirs(CONFIG["save"]["train_dir"])
    create_dirs(CONFIG["save"]["valid_dir"])
    create_dirs(CONFIG["save"]["test_dir"])
    create_dirs(CONFIG["save"]["predict_dir"])
    create_dirs(CONFIG["save"]["model_dir"])


def dump_config(filename, file_type: Literal['json', 'yaml', 'yml', 'toml']):
    from config import CONFIG
    match file_type:
        case 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(CONFIG, f)
        case 'yaml'|'yml':
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(CONFIG, f)
        case 'toml':
            import toml
            with open(filename, 'w') as f:
                toml.dump(CONFIG, f)
