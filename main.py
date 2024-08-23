import json
import yaml
import wandb
from models.unet.unet import UNet, UNetOriginal
from multiprocessing import Process
from threading import Thread
from train import *
from test import *
from predict import *
from utils.Transforms import TransformBuilder
from utils.datasets.dataset import get_train_valid_and_test

import torch
from torch import nn
from utils.utils import *
from utils.visualization import draw_loss_graph, draw_metrics

from argparse import ArgumentParser
from utils.writer import CSVWriter

def parse_args():
  parser = ArgumentParser(description='Training Configuration')
  parser.add_argument('-c', '--config', type=str, help='Train configuration file (JSON format)')

  general_group = parser.add_argument_group('General Settings')
  model_group = parser.add_argument_group('Model Configuration')
  training_group = parser.add_argument_group('Training Configuration')
  predict_group = parser.add_argument_group('Predict Configuration')

  general_group.add_argument('--proj', type=str, help='Project Name')
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
                'model': args.model,
                'load': args.load,
                'input': args.input,
                'device': 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu',
                'in_channels': args.in_channels,
                'n_classes': args.n_classes,
                'amp': args.amp,
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
  
    wandb.init(project=args.proj, 
               config=config_data)
  else:
    wandb.init(project=args.proj,
               config={
                'model': args.model,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'learning_rate': args.learning_rate,
                'data_dir': args.data_dir,
                'dataset': args.dataset,
                'device': 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu',
                'amp': args.amp,
                'augment_boost': args.augment_boost,
                'save_n_epoch': args.save_n_epoch,
                'in_channels': args.in_channels,
                'n_classes': args.n_classes,
                'test': args.test,
                'predict': args.predict, 
              })

  if not torch.cuda.is_available() and args.gpu:
    print('Warning: No GPU found, training will be performed on CPU.')
    logging.warning('No GPU found, training will be performed on CPU.')
  
  return args


def select_model(model: str, *args, **kwargs):
  if model == 'UNet':
    return UNet(kwargs['in_channels'], kwargs['n_classes'])
  elif model == 'UNetOriginal':
    return UNetOriginal(kwargs['in_channels'], kwargs['n_classes'])
  elif model == 'UNet++':
    pass
  else: 
    raise ValueError(f'Not supported model: {model}')


def test(net, test_dataset):  
  net.load_state_dict(load_model('./output/best_model.pth'))
  metrics = test_model(net, 
                       device=wandb.config.device, 
                       test_dataset=test_dataset, 
                       batch_size=wandb.config.batch_size, 
                       n_classes=wandb.config.n_classes,
                       average='macro')
  

  # writer = CSVWriter('output/test.csv')
  # writer.write_headers(['mIoU', 'accuracy', 'f1', 'precision', 'recall']).write('mIoU', metrics['mIoU']).write('accuracy', metrics['accuracy']).write('f1', metrics['f1']).write('precision', metrics['precision']).write('recall', metrics['recall']).flush()

  # test_loss_image_path = './output/metrics.png'
  # create_file_path_or_not(test_loss_image_path)

  # colors = ['red', 'green', 'blue', 'yellow', 'purple']
  # draw_metrics(metrics, title='Metrics', colors=colors, save_data=True, filename=test_loss_image_path)
  # wandb.log({'metrics': metrics, 'metrics_image': test_loss_image_path})

def train(net, train_dataset, valid_dataset):  
  train_losses, valid_losses = \
    train_model(net, 
                device=wandb.config.device, 
                train_dataset=train_dataset, 
                valid_dataset=valid_dataset, 
                n_classes=wandb.config.n_classes,
                batch_size=wandb.config.batch_size, 
                epochs=wandb.config.epochs, 
                lr=wandb.config.learning_rate,
                average='macro')

  writer = CSVWriter('output/train.csv')
  writer.write_headers(['loss']).write('loss', train_losses).flush()
  writer = CSVWriter('output/valid.csv')
  writer.write_headers(['loss']).write('loss', valid_losses).flush()

  train_loss_image_path = './output/train_loss.png'
  create_file_path_or_not(train_loss_image_path)
  draw_loss_graph(losses=train_losses, title='Train Losses', save_data=True, 
                  filename=train_loss_image_path)
  
  if valid_dataset is not None:
    valid_loss_image_path = './output/valid_loss.png'
    create_file_path_or_not(valid_loss_image_path)
    draw_loss_graph(losses=valid_losses, title='Validation Losses', save_data=True, 
                  filename=valid_loss_image_path)

  # wandb.log({'train_losses': train_losses, 'valid_losses': valid_losses, 'train_loss_image': train_loss_image_path, 'valid_loss_image': valid_loss_image_path})


def predict(net, input):
  model_state_dict = load_model(wandb.config.load)
  net.load_state_dict(model_state_dict)
  input = Image.open(input).convert('RGB')
  original_size = input.size
  input = input.resize((512,512))
  
  input = torch.from_numpy(np.array(input).astype(np.float32))
  input = input.expand(1, -1, -1, -1).permute(0, 3, 1, 2)
  
  output = predict_one(net, input)
  
  img = Image.fromarray(output, 'L')
  img = img.resize(original_size)
  img.save('./output/predict.png')
  
  # plt.imshow(np.array(input.squeeze(0).permute(1, 2, 0)), alpha=0.5)  # 原始图像
  plt.imshow(img)  # 叠加预测结果
  plt.axis('off')  # 不显示坐标轴
  plt.title('Model Prediction')
  plt.show()

def main():  
  args = parse_args()
  
  net = select_model(wandb.config.model, in_channels=wandb.config.in_channels, n_classes=wandb.config.n_classes)
  if args.predict:
    predict(net, args.input)
    return 
  

  dataset_dir = wandb.config.data_dir + wandb.config.dataset
  train_dataset, valid_dataset, test_dataset = \
    get_train_valid_and_test(wandb.config.dataset, dataset_dir,
                             train_valid_test=[0.9, 0, 0.1], 
                             use_augment_enhance=wandb.config.augment_boost) # type: ignore

  if args.test:
    test(net, test_dataset)
    return
  
  train(net, train_dataset, None)
  # test(net, test_dataset)


main()
