import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from utils.loss.dice_score import dice_loss
from utils.metrics import get_metrics, calculate_average_metrics
from typing import *


def test_model(model, device, 
  test_dataset, batch_size: int, 
  n_classes: int, num_workers: int=0, 
  average: str='weighted'):  
  assert n_classes > 0, 'n_classes must be greater than 0'
  assert batch_size > 0, 'batch_size must be greater than 0'
  
  model.to(device)
  model.eval()
  test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
  )
  
  n_step = len(test_loader)
  metrics = []
  with tqdm(total=n_step, desc=f'Testing'):
    with torch.no_grad():
      for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        
        labels = (labels * 256).type(dtype=torch.int8)
        outputs = (outputs * 256).type(dtype=torch.int8)
        metrics.append(
          get_metrics(
            outputs.cpu().detach().numpy(), 
            labels.cpu().detach().numpy(), 
            n_classes=n_classes, 
            average=average
          )) 
  
  return calculate_average_metrics(metrics)
