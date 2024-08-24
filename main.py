import os
import sys

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

def parse_args():
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

  model_group.add_argument('-m', '--model', type=str, help='Model to train')
  model_group.add_argument('--in_channels', type=int, help='Number of input channels')
  model_group.add_argument('--n_classes', type=int, help='Number of output classes')

  training_group.add_argument('-b', '--batch_size', type=int, help='Number of samples loaded at one time')
  training_group.add_argument('-e', '--epochs', type=int, help='Number of training epochs')
  training_group.add_argument('-lr', '--learning_rate', type=float, help='Learning rate for training the model')
  training_group.add_argument('--data_dir', type=str, help='The directory of datasets')
  training_group.add_argument('--dataset', type=str, help='Training and testing dataset')
  training_group.add_argument('--augment_boost', action='store_true', help='Use augment of data')
  training_group.add_argument('--gpu', action='store_true', help='GPU to train')
  training_group.add_argument('--amp', action='store_true', help='Use half precision mode')
  training_group.add_argument('--save_n_epoch', type=int, default=1, help='Use half precision mode')

  predict_group.add_argument('-l', '--load', type=str, help='The model used to predict')
  predict_group.add_argument('-i', '--input', type=str, help='The input data to predict')

  args = parser.parse_args()

  if args.predict:
    wandb.init(project=args.proj,
               config={
                'predict': True,
                'model': args.model,
                'load': args.load,
                'input': args.input,
                'device': 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu',
                'amp': args.amp,
                'private': {
                    'in_channels': args.in_channels,
                    'n_classes': args.n_classes,
                },
              })

    if not torch.cuda.is_available() and args.gpu:
      print('Warning: No GPU found, training will be performed on CPU.')
      logging.warning('No GPU found, training will be performed on CPU.')
  
    return args

  if not args.data_dir.endswith('/'):
    args.data_dir += '/'
  
  if args.dataset:
    args.dataset = args.dataset.upper()

  if args.config:
    ext = args.config.splitext(args.config)[1]
    if ext in ['json']:
      with open(args.config, 'r') as f:
        config_data = json.load(f)
    elif ext in ['yaml' 'yml']:
      with open(args.config, 'r') as f:
        config_data = yaml.safe_load(f)
    else:
      raise ValueError(f'Unsupported config file format: {ext}')
  
    wandb.init(project=args.project, 
               config=config_data)
  else:
    db_config = {
                'model': args.model,
                'data_dir': args.data_dir,
                'dataset': args.dataset,
                'batch_size': args.batch_size,
                'device': 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu',
                'amp': args.amp,
                'augment_boost': args.augment_boost,
                'save_n_epoch': args.save_n_epoch,
              }
    db_config['private'] = {
      'in_channels': args.in_channels,
      'n_classes': args.n_classes,
    }
    if args.test:
      db_config['test'] = True
    else:
      db_config['train'] = True
      db_config['epochs'] = args.epochs
      db_config['learning_rate'] = args.learning_rate
      db_config['augment_boost'] = args.augment_boost
      db_config['save_n_epoch'] = args.save_n_epoch

    wandb.init(project=args.project,
               config=db_config)

  if not torch.cuda.is_available() and args.gpu:
    print('Warning: No GPU found, training will be performed on CPU.')
    logging.warning('No GPU found, training will be performed on CPU.')
  
  return args


if __name__ == '__main__':
  args = parse_args()
  
  model = args.model
  device = wandb.config['device']
  epochs = args.epochs
  learning_rate = args.learning_rate
  in_channels = args.in_channels
  n_classes = args.n_classes
  save_n_epoch = args.save_n_epoch
  
  net = select_model(model, in_channels=in_channels, n_classes=n_classes)
  if args.predict:
    if os.path.isdir(args.input):
      input_dir = Path(args.input)
      inputs = [input_dir / input for input in input_dir.glob("*.*") 
                if input.suffix.lower() in [".png", ".jpg", ".jpeg", ".tif", ".gif"]]
    else:
      inputs = [args.input]
    predict(net, inputs, device=device)
    sys.exit(0) 

  dataset = args.dataset
  batch_size = args.batch_size
  augment_boost = args.augment_boost
  dataset_dir = args.data_dir + dataset
  train_loader, valid_loader, test_loader = \
    get_train_valid_and_test_loader(dataset, dataset_dir,
                                    batch_size=batch_size,
                                    train_valid_test=[0.9, 0, 0.1], 
                                    use_augment_enhance=augment_boost, 
                                    resize=(512, 512),
                                    num_workers=0) # type: ignore

  if args.test:
    test(net, test_loader, device=device, n_classes=n_classes)
    sys.exit(0)
  
  train(net, train_loader, valid_loader if len(valid_loader) > 0 else None, device=device, 
        epochs=epochs, 
        save_n_epoch=save_n_epoch,
        learning_rate=learning_rate,
        n_classes=n_classes,
        weight_decay=1e-8)
