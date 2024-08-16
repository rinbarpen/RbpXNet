import json
from models.unet.unet import UNet, UNetOrignal
from multiprocessing import Process
from threading import Thread
from train import *
from test import *
from predict import *
from utils.datasets.dataset import *

import torch
from torch import nn
from utils.utils import *
from utils.visualization import draw_loss_graph

from argparse import ArgumentParser
from utils.datasets.dataset_isic import ISIC2018Dataset

def Seeding():
  torch.seed()
  

CONFIG = dict(
  in_channels = 3,
  n_classes = 1,
  model = 'UNet',
  epochs = 10,
  batch_size = 10,
  lr = 1e-2,
  device = 'cuda' if torch.cuda.is_available else 'cpu',
  amp = True, 
  data_dir = 'I:/AI/Data/',
  output_dir = './output/',
  dataset_name = 'ISIC2017',
  average='weighted',
)

def parse_args():
  global CONFIG
  parser = ArgumentParser
  parser.add_argument('--gpu', default=True, help='GPU to train')
  parser.add_argument('-m', '--model', type='str', help='Our model to train')
  parser.add_argument('-c', '--config', type='str', help='Train configuration')
  parser.add_argument('-b', '--batch_size', type=int, help='The number of sample loaded in one time')
  parser.add_argument('-e', '--epochs', type=int, help='The number of training turn')
  parser.add_argument('-lr', '--learning_rate', type=float, help='The learning rate for training model')
  parser.add_argument('-d', '--data', type=str, help='The training and testing dataset')
  parser.add_argument('--amp', default=False, help='Use half precision mode')
  parser.add_argument('-h', '--help', description='Show this help message')
  
  if parser.config:
    CONFIG = json.load(parser.config) 

  if parser.model:
    if parser.model == '':
      print('Error: Model is not specified.')
      return
    CONFIG['model'] = parser.model
  if parser.epochs:
    CONFIG['epochs'] = int(parser.epochs)
  if parser.batch_size:
    CONFIG['batch_size'] = int(parser.batch_size)
  if parser.data:
    CONFIG['dataset_name'] = parser.data
  if parser.learning_rate:
    CONFIG['learning_rate'] = float(parser.learning_rate)
  if parser.amp:
    CONFIG['amp'] = parser.amp

  if torch.cuda.is_available:
    CONFIG['device'] = 'cuda' if parser.gpu else 'cpu'
  else:
    print('Warning: No GPU found, training will be performed on CPU.')  # 若没有GPU，则使用CPU进行训练
    logging.warn('No GPU found, training will be performed on CPU.')
    CONFIG['device'] = 'cpu'
    
def select_model(model: str):
  match model:
    case 'UNet':
      return UNet(CONFIG['in_channels'], CONFIG['n_classes'])
    
    case _:
      raise ValueError(f'Not supported model: {model}')


def isic2018():
  unet = UNet(CONFIG['in_channels'], CONFIG['n_classes'])
  # R2UNet = ResUNet(CONFIG['in_channels'], CONFIG['n_classes'], 2)

  isic2018_dir = f'{CONFIG["data_dir"]}ISIC2018/'
  isic2018_train_dataset, isic2018_valid_dataset, isic2018_test_dataset = \
    ISIC2018Dataset.get_train_valid_and_test(isic2018_dir, transformers=CustomTransform(resize=(512, 512), rotation=0))

  train_losses, valid_losses = train_model(unet, device=CONFIG['device'], 
                          train_dataset=isic2018_train_dataset, valid_dataset=isic2018_valid_dataset, 
                          n_classes=CONFIG['n_classes'],
                          batch_size=CONFIG['batch_size'], 
                          epochs=CONFIG['epochs'], 
                          lr=CONFIG['lr'],
                          average=CONFIG['average'])

  train_loss_image_path = './output/isic2018_train_loss.png'
  valid_loss_image_path = './output/isic2018_valid_loss.png'
  draw_loss_graph(losses=train_losses, title='Train Losses', save_data=True, 
                  filename=train_loss_image_path)
  draw_loss_graph(losses=valid_losses, title='Validation Losses', save_data=True, 
                  filename=valid_loss_image_path)

  metrics = test_model(UNet, device=CONFIG['device'], 
                       test_dataset=isic2018_test_dataset, 
                       batch_size=CONFIG['batch_size'], 
                       n_classes=CONFIG['n_classes'],
                       average=CONFIG['average'])
  draw_metrics(metrics, colors='red green blue yellow purple'.split(), save_data=True, filename='output/UNet/metrics.png')


def main():  
  unet = UNetOrignal(CONFIG['in_channels'], CONFIG['n_classes'])
  # R2UNet = unet.ResUNet(CONFIG['in_channels'], CONFIG['n_classes'], 2)
  transforms = CustomTransform(resize=(512, 512), rotation=0)

  isic2017_dir = f'{CONFIG["data_dir"]}ISIC2017/'
  ISIC2017_train_dataset, ISIC2017_valid_dataset, ISIC2017_test_dataset = \
    ISIC2017Dataset.get_train_valid_and_test(isic2017_dir, transforms)
  
  train_losses, valid_losses = train_model(unet, device=CONFIG['device'], 
                          train_dataset=ISIC2017_train_dataset, valid_dataset=ISIC2017_valid_dataset, 
                          n_classes=CONFIG['n_classes'],
                          batch_size=CONFIG['batch_size'], 
                          epochs=CONFIG['epochs'], 
                          lr=CONFIG['lr'],
                          average=CONFIG['average'])

  train_loss_image_path = './output/train_loss.png'
  valid_loss_image_path = './output/valid_loss.png'
  draw_loss_graph(losses=train_losses, title='Train Losses', save_data=True, 
                  filename=train_loss_image_path)
  draw_loss_graph(losses=valid_losses, title='Validation Losses', save_data=True, 
                  filename=valid_loss_image_path)

  metrics = test_model(UNet, device=CONFIG['device'], 
                       test_dataset=ISIC2017_test_dataset, 
                       batch_size=CONFIG['batch_size'], 
                       n_classes=CONFIG['n_classes'],
                       average=CONFIG['average'])
  draw_metrics(metrics, colors='red green blue yellow purple'.split(), save_data=True, filename='output/UNet/metrics.png')

main()
