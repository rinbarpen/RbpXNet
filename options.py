import logging
from argparse import ArgumentParser
import os.path
from typing import Literal

import torch
import sys
import config
from utils.utils import create_file_parents, file_suffix_name

ConfigFileType = Literal['json', 'yaml', 'yml']

def check_args(args):
    if not args.project:
        raise ValueError('You must specify a project name')
    if not args.author:
        raise ValueError('There should have less than an author with this project')
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

    parser = ArgumentParser(description='AI Configuration')
    parser.add_argument('-c', '--config', type=str, help='Running configuration file (JSON format)')

    general_group = parser.add_argument_group('General Settings')
    model_group = parser.add_argument_group('Model Configuration')
    training_group = parser.add_argument_group('Training Configuration')
    predict_group = parser.add_argument_group('Predict Configuration')

    general_group.add_argument('--project', type=str, help='Project Name')
    general_group.add_argument('--author', type=str, help='Author Name')
    general_group.add_argument('--wandb', action='store_true', help='Launch Wandb instance')
    general_group.add_argument('--train', action='store_true', help='Train the model')
    general_group.add_argument('--test', action='store_true', help='Test the model')
    general_group.add_argument('--predict', action='store_true', help='Predict the model')
    general_group.add_argument('--print', action='store_true', help='Model Information')
    general_group.add_argument('--export', type=str, default='toml', help='Export Configurations with file format')
    # general_group.add_argument('--transfer', action='store_true', help='Transfer Configurations')

    
    model_group.add_argument('-m', '--model', type=str, help='Model to train')
    model_group.add_argument('--n_channels', type=int, help='Number of input channels')
    model_group.add_argument('--n_classes', type=int, help='Number of output classes')
    # --classes "category1,category2,..."
    model_group.add_argument('--classes', type=str, help='predicted classes group')
    model_group.add_argument('--seed', type=int, help='seed of initial model')

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

    if args.config:
        ext = file_suffix_name(args.config)
        match ext:
            case '.json':
                import json
                with open(args.config, 'r') as f:
                    CONFIG = json.load(f)
            case '.yaml'|'.yml':
                import yaml
                with open(args.config, 'r') as f:
                    CONFIG = yaml.safe_load(f)
            case _:
                raise ValueError('Unsupported config file: %s' % args.config)

        if args.train or args.test or args.predict:
            CONFIG['train'] = CONFIG['test'] = CONFIG['predict'] = False
            if args.train:
                CONFIG['train'] = True
            elif args.test:
                CONFIG['test'] = True
            elif args.predict:
                CONFIG['predict'] = True

        # Rewrite settings
        if args.load:
            CONFIG['load'] = args.load
        if args.input:
            CONFIG['input'] = args.input
        if args.dataset:
            CONFIG['dataset'] = args.dataset
        if args.data_dir:
            CONFIG['data_dir'] = args.data_dir
        if args.weight_decay:
            CONFIG['weight_decay'] = args.weight_decay
        if args.learning_rate:
            CONFIG['learning_rate'] = args.learning_rate
        if args.batch_size:
            CONFIG['batch_size'] = args.batch_size
        if args.epochs:
            CONFIG['epochs'] = args.epochs
        if args.save_every_n_epoch:
            CONFIG['save_every_n_epoch'] = args.save_every_n_epoch

    else:
        check_args(args)
        
        CONFIG['project'] = args.project
        CONFIG['author'] = args.author

        classes = args.classes.split(',')
        classes = [c.strip() for c in classes]

        CONFIG["model"] = args.model
        CONFIG["private"] = {
            "n_channels": args.n_channels,
            "n_classes": args.n_classes,
            "classes": classes,
        }
        CONFIG["device"] = 'cuda' if args.gpu else 'cpu'
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

    if not torch.cuda.is_available() and CONFIG['device']:
        logging.warning('No GPU found, training and inferring will be performed on CPU.')
    CONFIG['device'] = 'cuda' if CONFIG['device'] == 'cuda' and torch.cuda.is_available() else 'cpu'

    CONFIG['dataset'] = CONFIG['dataset'].upper()

    if args.seed:
        CONFIG['seed'] = args.seed
        config.USE_SEED = True

    if args.wandb:
        import wandb
        CONFIG["wandb"] = True
        wandb.init(project=args.project,
                   entity=args.author,
                   config=CONFIG)

    if args.export:
        filename = os.path.join('configs', CONFIG['project'], CONFIG['model'] + '_' + CONFIG['dataset'] + '.' + args.export)
        dump_config(filename, args.export)
        sys.exit(0)

def dump_config(filename, file_type: ConfigFileType):
    from config import CONFIG

    create_file_parents(filename)
    match file_type:
        case 'json':
            import json
            with open(filename, 'w') as f:
                json.dump(CONFIG, f)
        case 'yaml'|'yml':
            import yaml
            with open(filename, 'w') as f:
                yaml.dump(CONFIG, f)

    logging.info('Dumping config to %s' % filename)
