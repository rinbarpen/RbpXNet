import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb
from utils.metrics.dice_score import dice_loss
from utils.metrics import get_metrics
from typing import *

import matplotlib.pyplot as plt
import cv2

from utils.utils import create_file_if_not_exist, load_model
from utils.visualization import draw_metrics
from utils.writer import CSVWriter


def test_model(model, device, test_loader,
  classes: List[str], average: str='marco'):  
  assert len(classes) >= 2, 'predict the number of classes should be greater than 0'
  # labels = ["background", "vein"]

  model.to(device)
  model.eval()
  
  mean_metric = dict()
  n_step = len(test_loader)
  with tqdm(total=n_step, desc=f'Testing') as pbar:
    with torch.no_grad():
      for inputs, labels in test_loader:
        inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        
        outputs = model(inputs)
        threshold = 0.5
        outputs[outputs >= threshold] = 255
        outputs[outputs < threshold] = 0
                
        metric = get_metrics(
          outputs.cpu().detach().numpy(), 
          labels.cpu().detach().numpy(), 
          labels=classes,
          average=average
        )
        
        for k, v in metric.items():
          if k in mean_metric:
            mean_metric[k] += v
          else:
            mean_metric[k] = v
        
        pbar.update()
        pbar.set_postfix(**{'metrics(batch)': repr(metric)})
  
  for k in mean_metric.keys():
    mean_metric[k] /= len(test_loader)

  return mean_metric


def test(net, test_loader, device, classes: List[str]):
  """
  This function tests a deep learning model using a given test dataset.

  Parameters:
  - net: The deep learning model to be tested.
  - test_loader: A DataLoader object for the test dataset.
  - device: The device (CPU or GPU) to run the model on.
  - classes: A list of class names for the dataset.

  Returns:
  - metrics: A dictionary containing the evaluation metrics (mIoU, accuracy, f1, dice, roc, auc) of the model on the test dataset.
  """

  net.to(device)
  net.load_state_dict(load_model('./output/best_model.pth', device))
  metrics = test_model(net, 
                       device=device, 
                       test_loader=test_loader,
                       classes=classes,
                       average='macro')
  
  writer = CSVWriter('output/test.csv')
  writer.write_headers(['mIoU', 'accuracy', 'f1', 'dice', 'roc', 'auc']).write('mIoU', metrics['mIoU']).write('accuracy', metrics['accuracy']).write('f1', metrics['f1']).write('dice', metrics['dice']).write('roc', metrics['roc']).write(['auc'], metrics['auc']).flush()

  test_loss_image_path = './output/metrics.png'
  colors = ['red', 'green', 'blue', 'yellow', 'purple']
  draw_metrics(metrics, title='Metrics', colors=colors, filename=test_loss_image_path)
  # wandb.log({'metrics': metrics, 'metrics_image': test_loss_image_path})
