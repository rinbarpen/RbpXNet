import torch
from torch import nn, optim
import numpy as np
import logging 
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from typing import *
from utils.visualization import *
from evaluate import *
import wandb


def train_one_epoch(model, device, epoch, train_loader, optimizer, criterion):
  model.train()
  train_loss = 0.0
  
  with tqdm(total=len(train_loader), desc=f'Training') as pbar:
    for inputs, labels in train_loader:
      optimizer.zero_grad()
      inputs, labels = inputs.to(device, dtype=torch.float32), labels.to(device, dtype=torch.float32)
      
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      
      optimizer.step()
      train_loss += loss.item()
      
      pbar.update()
      pbar.set_postfix(**{'loss(batch)': loss.item()})
  
  train_loss /= len(train_loader)
  
  return train_loss


def train_model(model, device, 
                train_dataset, valid_dataset, batch_size: int, 
                epochs: int, 
                lr: float,
                n_classes: int,
                num_workers: int=0, 
                weight_decay: float=0.1,
                average: str='marco'):  
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
  criterion = nn.BCEWithLogitsLoss() if wandb.config.n_classes == 1 else nn.CrossEntropyLoss()

  best_val_loss = float('inf')

  train_losses = []
  valid_losses = []
  for epoch in trange(epochs, desc='Epoch: '):
    train_loss = train_one_epoch(model, device, epoch, 
      train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    val_loss = valid_one_epoch(model, device, epoch, 
      valid_loader=valid_loader, criterion=criterion, 
      n_classes=n_classes, average=average)

    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
          f"Validation Loss: {val_loss:.4f}")

    train_losses.append(train_loss)
    valid_losses.append(val_loss)
    
    if wandb.config.save_n_epoch % (epoch+1) == 0:
      saved_model_filename = \
        f'./output/models/{wandb.config.model}-{epoch+1}of{wandb.config.epochs}-{wandb.config.dataset}.pth'
      torch.save(model.state_dict(), saved_model_filename)
      logging.info(f'save model to {saved_model_filename} when epoch={epoch}, loss={val_loss}')
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      torch.save(model.state_dict(), './output/best_model.pth')
      logging.info(f'save model to best_model.pth when epoch={epoch}, loss={val_loss}')

  return train_losses, valid_losses 
