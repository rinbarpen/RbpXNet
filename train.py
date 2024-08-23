import torch
from torch import nn, optim
import numpy as np
import logging 
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from typing import *
from utils.visualization import *
from evaluate import *
from utils.writer import CSVWriter
from utils.utils import *


def train_one_epoch(model, device, epoch, train_loader, optimizer, criterion):
  model.train()
  train_loss = 0.0
  
  with tqdm(total=len(train_loader), desc=f'Training') as pbar:
    for inputs, labels, _, _ in train_loader:
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
                train_loader, valid_loader,
                epochs: int, 
                learning_rate: float,
                n_classes: int,
                save_n_epoch: int,
                weight_decay: float=1e-8,
                average: str='marco'):  
  model.to(device)

  optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=weight_decay)
  criterion = nn.BCEWithLogitsLoss() if n_classes == 1 else nn.CrossEntropyLoss()

  best_train_loss = float('inf')

  train_losses = []
  valid_losses = []
  for epoch in trange(epochs, desc='Epoch: '):
    train_loss = train_one_epoch(model, device, epoch, 
      train_loader=train_loader, optimizer=optimizer, criterion=criterion)
    
    if valid_loader is not None:
      val_loss = valid_one_epoch(model, device, epoch, 
        valid_loader=valid_loader, criterion=criterion, 
        n_classes=n_classes, average=average)

      print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}")
      valid_losses.append(val_loss)
    else:
      print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
      
    train_losses.append(train_loss)

    if (epoch+1) % save_n_epoch == 0:
      saved_model_filename = \
        f'./output/models/{wandb.config.model}-{epoch+1}of{epochs}-{wandb.config.dataset}.pth'
      torch.save(model.state_dict(), saved_model_filename)
      logging.info(f'save model to {saved_model_filename} when epoch={epoch}, loss={train_loss}')
    if train_loss < best_train_loss:
      best_train_loss = train_loss
      torch.save(model.state_dict(), './output/best_model.pth')
      logging.info(f'save model to best_model.pth when epoch={epoch}, loss={train_loss}')

  return train_losses, valid_losses 


def train(net, train_loader, valid_loader, device, epochs, learning_rate, n_classes, save_n_epoch, weight_decay=1e-8):
  train_losses, valid_losses = \
    train_model(net, 
                device=device, 
                train_loader=train_loader, 
                valid_loader=valid_loader, 
                n_classes=n_classes,
                epochs=epochs, 
                save_n_epoch=save_n_epoch,
                learning_rate=learning_rate,
                average='macro')

  writer = CSVWriter('output/train.csv')
  writer.write_headers(['loss']).write('loss', train_losses).flush()
  writer = CSVWriter('output/valid.csv')
  writer.write_headers(['loss']).write('loss', valid_losses).flush()

  train_loss_image_path = './output/train_loss.png'
  create_file_path_or_not(train_loss_image_path)
  draw_loss_graph(losses=train_losses, title='Train Losses', save_data=True, 
                  filename=train_loss_image_path)
  
  if valid_loader is not None:
    valid_loss_image_path = './output/valid_loss.png'
    create_file_path_or_not(valid_loss_image_path)
    draw_loss_graph(losses=valid_losses, title='Validation Losses', save_data=True, 
                  filename=valid_loss_image_path)

  # wandb.log({'train_losses': train_losses, 'valid_losses': valid_losses, 'train_loss_image': train_loss_image_path, 'valid_loss_image': valid_loss_image_path})
