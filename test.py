import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from utils.loss.dice_score import dice_loss
from utils.metrics import get_metrics, calculate_average_metrics
from typing import *

import matplotlib.pyplot as plt
import cv2

def test_model(model, device, 
  test_dataset, batch_size: int, 
  n_classes: int, num_workers: int=0, 
  average: str='marco'):  
  assert n_classes > 0, 'n_classes must be greater than 0'
  assert batch_size > 0, 'batch_size must be greater than 0'
  
  model.to(device)
  model.eval()
  
  def custom_collate_fn(batch):
    images = torch.stack([item[0] for item in batch]) 
    masks = torch.stack([item[1] for item in batch])
    filenames = [item[2] for item in batch]
    original_sizes = [item[3] for item in batch]
    return images, masks, filenames, original_sizes
  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=custom_collate_fn
  )
  
  n_step = len(test_loader)
  metrics = []
  with tqdm(total=n_step, desc=f'Testing') as pbar:
    with torch.no_grad():
      for inputs, labels, filenames, original_sizes in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
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
        
        # metric = get_metrics(
        #   outputs.cpu().detach().numpy(), 
        #   labels.cpu().detach().numpy(), 
        #   n_classes=n_classes, 
        #   average=average,
        #   selected=['mIoU', 'accuracy', 'f1']
        # )
        # metrics.append(metric)
        
        pbar.update()
        # pbar.set_postfix(**{'metrics(batch)': repr(metric)})
  
  # return calculate_average_metrics(metrics)
  return None
