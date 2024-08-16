import torch
from torch import nn, optim
import numpy as np
import logging 
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from typing import *
from utils.metrics import get_metrics, calculate_average_metrics
from utils.visualization import draw_loss_graph


def train_one_epoch(model, device, epoch, train_loader, optimizer, criterion):
  model.train()
  train_loss = 0.0
  
  step = 0
  n_step = len(train_loader)

  with tqdm(total=n_step, desc=f'Training... {step}/{n_step}') as pbar:
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = criterion(outputs, labels)

      loss.backward()
      train_loss += loss.item()
      
      step = step + 1

      pbar.set_postfix(**{'loss(batch)': loss.item()})
  
  train_loss /= len(train_loader.dataset)
  
  return train_loss


def valid_one_epoch(model, device, epoch, valid_loader, criterion, n_classes, average):
  model.eval()
  val_loss = 0.0

  step = 0
  n_step = len(valid_loader)

  with tqdm(total=n_step, desc=f'Validating... {step}/{n_step}') as pbar:
    with torch.no_grad():
      for inputs, labels in valid_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        val_loss += loss.item() * inputs.size(0)

        step = step + 1

        pbar.set_postfix(**{'loss(batch)': loss.item()})

  val_loss /= len(valid_loader.dataset)

  return val_loss


def train_model(model, device, 
                train_dataset, valid_dataset, batch_size: int, 
                n_epoch: int, 
                lr: float,
                n_classes: int,
                num_workers: int=0, 
                weight_decay: float=0.1,
                average: str='weighted',
                valid_per_n_epoch: int=1):  
  model.to(device)
  train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=num_workers, 
    pin_memory=True
  )
  valid_loader = DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
  ) if valid_dataset is not None else None  

  optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
  criterion = nn.CrossEntropyLoss()

  best_val_loss = float('inf')
  saved_model_filename = './output/best_model.pth'

  train_losses = []
  valid_losses = []
  for epoch in trange(n_epoch, desc='Epoch: '):
    train_loss = train_one_epoch(model, device, epoch, 
      train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    val_loss = valid_one_epoch(model, device, epoch, 
      valid_loader=valid_loader, criterion=criterion, 
      n_classes=n_classes, average=average)

    print(f"Epoch {epoch+1}/{n_epoch}, Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}")

    train_losses.append(train_loss)
    valid_losses.append(val_loss)

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), saved_model_filename)
      logging.info(f'save model to {saved_model_filename} when epoch={epoch}, loss={val_loss}')

  return train_losses, valid_losses 
