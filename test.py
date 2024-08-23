import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb
from utils.loss.dice_score import dice_loss
from utils.metrics import get_metrics, calculate_average_metrics
from typing import *

import matplotlib.pyplot as plt
import cv2

from utils.utils import create_file_path_or_not, load_model
from utils.visualization import draw_metrics


def test_model(model, device, test_loader, 
  n_classes: int, average: str='marco'):  
  assert n_classes > 0, 'n_classes must be greater than 0'
  
  model.to(device)
  model.eval()
  
  n_step = len(test_loader)
  metrics = []
  with tqdm(total=n_step, desc=f'Testing') as pbar:
    with torch.no_grad():
      for inputs, labels, filenames, original_sizes in test_loader:
        inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
        
        outputs = model(inputs)
        threshold = 0.5
        outputs[outputs >= threshold] = 255
        outputs[outputs < threshold] = 0
        
        for i in range(0, len(outputs)):
          output = outputs[i].squeeze()
          output = output.cpu().detach().numpy()
          width, height = original_sizes[i][2], original_sizes[i][1]
          output = cv2.resize(output, (width, height), interpolation=cv2.INTER_NEAREST)
          cv2.imwrite(f'./output/test/{filenames[i]}.png', output)
        
        metric = get_metrics(
          outputs.cpu().detach().numpy(), 
          labels.cpu().detach().numpy(), 
          n_classes=n_classes + 1,
          average=average,
          selected=['mIoU', 'accuracy']
        )
        metrics.append(metric)
        
        pbar.update()
        pbar.set_postfix(**{'metrics(batch)': repr(metric)})
  
  return calculate_average_metrics(metrics)
  # return None


def test(net, test_loader, device, n_classes):  
  net.to(device)
  net.load_state_dict(load_model('./output/models/UNetOriginal-10of10-DRIVE.pth', device))
  # net.load_state_dict(load_model('./output/best_model.pth', device))
  metrics = test_model(net, 
                       device=device, 
                       test_loader=test_loader,
                       n_classes=n_classes,
                       average='macro')
  

  # writer = CSVWriter('output/test.csv')
  # writer.write_headers(['mIoU', 'accuracy', 'f1', 'precision', 'recall']).write('mIoU', metrics['mIoU']).write('accuracy', metrics['accuracy']).write('f1', metrics['f1']).write('precision', metrics['precision']).write('recall', metrics['recall']).flush()

  test_loss_image_path = './output/metrics.png'
  create_file_path_or_not(test_loss_image_path)

  colors = ['red', 'green', 'blue', 'yellow', 'purple']
  draw_metrics(metrics, title='Metrics', colors=colors, save_data=True, filename=test_loss_image_path)
  wandb.log({'metrics': metrics, 'metrics_image': test_loss_image_path})
