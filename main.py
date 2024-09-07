import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import sys
import yaml
import wandb
from train import train
from test import test
from predict import predict
from utils.datasets.dataset import get_train_valid_and_test_loader

from utils.utils import *

from argparse import ArgumentParser

logging.basicConfig(
    level=logging.INFO,  # 设置日志级别
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]  # 设置输出到标准输出
)

# logging.basicConfig(level=logging.INFO, format='%(asctime) - %(name)s - %(levelname)s - %(message)s',
#                     filename=f'logs/{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}.log',
#                     filemode='w')

# neuron_logger = logging.getLogger("neuron")

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

    if not torch.cuda.is_available() and args.gpu:
        logging.warning('No GPU found, training will be performed on CPU.')
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    if args.data_dir and not args.data_dir.endswith('/'):
        args.data_dir += '/'

    if args.dataset:
        args.dataset = args.dataset.upper()

    if args.config:
        ext = args.config.splitext(args.config)[1]
        if ext in ['json']:
            with open(args.config, 'r') as f:
                CONFIG = json.load(f)
        elif ext in ['yaml' 'yml']:
            with open(args.config, 'r') as f:
                CONFIG = yaml.safe_load(f)
        else:
            raise ValueError(f'Unsupported config file format: {ext}')
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
                CONFIG["batch_size"] = args.batch_size
                CONFIG["learning_rate"] = args.learning_rate
                CONFIG["epochs"] = args.epochs
                CONFIG["augment_boost"] = args.augment_boost
                CONFIG["save_every_n_epoch"] = args.save_every_n_epoch
                if args.weight_decay:
                    CONFIG["weight_decay"] = args.weight_decay

    if args.wandb:
        CONFIG["wandb"] = True
        wandb.init(project=args.project,
                   config=CONFIG)


if __name__ == '__main__':
    from config import CONFIG
    parse_args()

    create_dirs(CONFIG["save"]["train_dir"])
    create_dirs(CONFIG["save"]["valid_dir"])
    create_dirs(CONFIG["save"]["predict_dir"])
    create_dirs(CONFIG["save"]["model_dir"])

    net = select_model(CONFIG["model"],
                       in_channels=CONFIG["private"]["in_channels"],
                       n_classes=CONFIG["private"]["n_classes"],
                       use_bilinear=True)
    if CONFIG["predict"]:
        if os.path.isdir(CONFIG["input"]):
            input_dir = Path(CONFIG["input"])
            inputs = [input_dir / input for input in input_dir.glob("*.*")
                      if input.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".gif"]]
        else:
            inputs = [CONFIG["input"]]
        predict(net, inputs, classes=CONFIG["private"]["classes"], device=CONFIG["device"])
        sys.exit(0)

    data_dir = CONFIG["data_dir"]+CONFIG["dataset"]
    train_loader, valid_loader, test_loader = \
        get_train_valid_and_test_loader(CONFIG["dataset"], data_dir,
                                        batch_size=CONFIG["batch_size"],
                                        train_valid_test=[0.9, 0, 0.1],
                                        use_augment_enhance=CONFIG["augment_boost"],
                                        resize=(512, 512),
                                        num_workers=0)  # type: ignore

    if CONFIG["test"]:
        metrics = ['mIoU', 'accuracy', 'f1', 'recall', 'dice']
        test(net, test_loader, device=CONFIG["device"], classes=CONFIG["private"]["classes"], selected_metrics=metrics)
        sys.exit(0)

    train(net, train_loader, valid_loader if len(valid_loader) > 0 else None,
          device=CONFIG["device"],
          n_classes=CONFIG["private"]["n_classes"])
